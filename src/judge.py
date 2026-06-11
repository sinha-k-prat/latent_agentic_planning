"""Judge interface: score (instruction, response) -> reward in [0, 1].

Three backends behind one interface (config `judge.kind`):
  local : Qwen2.5-7B-Instruct with a rubric prompt, 4-bit if bitsandbytes is available.
  stub  : length-based heuristic, no model — for --smoke-test and CI.
  api   : Anthropic Messages API (off by default).

Scores are cached on disk keyed by sha256(instruction + "\x00" + response). Parsing
never raises: an unparseable judge output -> reward 0.0 and a counter increment.
"""
import hashlib
import json
import os
import re

_RATING_RE = re.compile(r"\b(10|[1-9])\b")


def _key(instruction, response):
    h = hashlib.sha256()
    h.update(instruction.encode("utf-8"))
    h.update(b"\x00")
    h.update(response.encode("utf-8"))
    return h.hexdigest()


def parse_rating(text):
    """Parse the first integer in [1, 10] from judge output -> (reward in [0,1], ok)."""
    if not text:
        return 0.0, False
    m = _RATING_RE.search(text)
    if not m:
        return 0.0, False
    n = int(m.group(1))
    if n < 1 or n > 10:
        return 0.0, False
    return (n - 1) / 9.0, True  # 1 -> 0.0, 10 -> 1.0


RUBRIC = (
    "You are a strict evaluator. Rate the assistant response to the instruction on a "
    "scale of 1 to 10 for helpfulness, correctness, and instruction-following. "
    "Output ONLY the integer.\n\n"
    "Instruction:\n{instruction}\n\nResponse:\n{response}\n\nRating (1-10):"
)


class JudgeCache:
    def __init__(self, path):
        self.path = path
        self.data = {}
        if path and os.path.exists(path):
            try:
                with open(path) as f:
                    self.data = json.load(f)
            except Exception:
                self.data = {}

    def get(self, k):
        return self.data.get(k)

    def put(self, k, v):
        self.data[k] = v

    def flush(self):
        if not self.path:
            return
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.data, f)


class BaseJudge:
    def __init__(self, cfg):
        self.cfg = cfg
        self.cache = JudgeCache(cfg.judge.cache_path)
        self.n_unparsable = 0

    def _score_uncached(self, pairs):
        raise NotImplementedError

    def score_batch(self, pairs):
        """pairs: list[(instruction, response)] -> list[float]. Uses + fills the cache."""
        keys = [_key(i, r) for i, r in pairs]
        out = [None] * len(pairs)
        todo, todo_idx = [], []
        for j, k in enumerate(keys):
            c = self.cache.get(k)
            if c is None:
                todo.append(pairs[j])
                todo_idx.append(j)
            else:
                out[j] = c
        if todo:
            fresh = self._score_uncached(todo)
            for j, v in zip(todo_idx, fresh):
                out[j] = v
                self.cache.put(keys[j], v)
            self.cache.flush()
        return out

    def score(self, instruction, response):
        return self.score_batch([(instruction, response)])[0]


class StubJudge(BaseJudge):
    """Length/structure heuristic in [0,1]; deterministic, no model. For smoke tests."""

    def _score_uncached(self, pairs):
        out = []
        for _, r in pairs:
            words = len(r.split())
            length_score = min(words, 120) / 120.0
            has_structure = 0.1 if ("\n" in r or any(c.isdigit() for c in r)) else 0.0
            out.append(round(min(1.0, 0.9 * length_score + has_structure), 4))
        return out


class LocalJudge(BaseJudge):
    """Qwen2.5-7B-Instruct rubric judge, 4-bit when bitsandbytes is present."""

    def __init__(self, cfg, device):
        super().__init__(cfg)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(cfg.model.judge)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "left"

        kwargs = dict(torch_dtype=torch.bfloat16)
        if cfg.judge.load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_quant_type="nf4",
                )
            except Exception:
                pass  # fall back to bf16 if bitsandbytes is unavailable
        self.model = AutoModelForCausalLM.from_pretrained(cfg.model.judge, **kwargs)
        if "quantization_config" not in kwargs:
            self.model.to(device)
        self.model.eval()

    def _prompt(self, instruction, response):
        msg = [{"role": "user", "content": RUBRIC.format(
            instruction=instruction[:4000], response=response[:4000])}]
        return self.tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)

    def _score_uncached(self, pairs):
        torch = self.torch
        out = []
        bs = self.cfg.judge.batch_size
        for i in range(0, len(pairs), bs):
            chunk = pairs[i:i + bs]
            prompts = [self._prompt(ins, res) for ins, res in chunk]
            enc = self.tok(prompts, return_tensors="pt", padding=True,
                           add_special_tokens=False).to(self.model.device)
            with torch.no_grad():
                gen = self.model.generate(
                    **enc, max_new_tokens=self.cfg.judge.max_new_tokens, do_sample=False,
                    pad_token_id=self.tok.pad_token_id,
                )
            new = gen[:, enc["input_ids"].shape[1]:]
            for row in new:
                text = self.tok.decode(row, skip_special_tokens=True)
                r, ok = parse_rating(text)
                if not ok:
                    self.n_unparsable += 1
                out.append(r)
        return out


class APIJudge(BaseJudge):
    """Anthropic Messages API judge (off by default). Same interface and parsing."""

    def __init__(self, cfg):
        super().__init__(cfg)
        import anthropic
        self.client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
        self.model = cfg.judge.api.model
        self.max_tokens = cfg.judge.api.max_tokens

    def _score_uncached(self, pairs):
        out = []
        for ins, res in pairs:
            try:
                resp = self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    messages=[{"role": "user", "content": RUBRIC.format(
                        instruction=ins[:6000], response=res[:6000])}],
                )
                text = next((b.text for b in resp.content if b.type == "text"), "")
            except Exception:
                text = ""
            r, ok = parse_rating(text)
            if not ok:
                self.n_unparsable += 1
            out.append(r)
        return out


def build_judge(cfg, device=None):
    kind = cfg.judge.kind
    if kind == "stub":
        return StubJudge(cfg)
    if kind == "api":
        return APIJudge(cfg)
    if kind == "local":
        return LocalJudge(cfg, device)
    raise ValueError(f"unknown judge.kind {kind}")
