"""Distill RICH, plan-guided responses for the plan corpus.

For each (instruction, plan_tokens, brief correct answer), a capable TEACHER rewrites the brief
answer into a thorough, detailed one that explicitly carries out each plan step — keeping facts
and any computed numbers exactly correct. This is the "first the natural/brief answer, then 10x
better through the plan" recipe: the brief gold is the correctness seed, the plan is the scaffold,
the teacher fills it in richly.

Teachers (pluggable):
  --teacher local  : a HuggingFace instruct model (default Qwen2.5-7B-Instruct; 4-bit on GPU)
  --teacher api    : Anthropic Messages API (needs ANTHROPIC_API_KEY)

Output: dataset/plan_dataset_rich.jsonl (same instruction/plan/plan_tokens, richer response).

  # real run (GPU): 7B teacher in 4-bit
  python dataset/distill_rich.py --teacher local --model Qwen/Qwen2.5-7B-Instruct --load_4bit --device cuda
  # API teacher
  python dataset/distill_rich.py --teacher api --model claude-opus-4-8
  # tiny local sanity (0.5B on CPU)
  python dataset/distill_rich.py --teacher local --model Qwen/Qwen2.5-0.5B-Instruct --device cpu --limit 3
"""
import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def build_prompt(instruction, plan_names, brief):
    steps = "\n".join(f"  {i+1}. {n.lower().replace('_', ' ')}" for i, n in enumerate(plan_names))
    return (
        "Rewrite a brief, correct answer into a thorough, genuinely helpful one.\n\n"
        f"Question:\n{instruction}\n\n"
        f"Carry out these steps in order (this is your reasoning scaffold):\n{steps}\n\n"
        f"A correct but brief answer (keep all of its facts and any computed numbers EXACTLY):\n{brief}\n\n"
        "Now write the full answer: detailed, specific, well-organized, and tailored to the question. "
        "Work through each step's content, add useful explanation and caveats where helpful, and keep "
        "every fact and number correct. Do NOT mention the words 'plan', 'step', or the step names — "
        "just produce the polished answer."
    )


class LocalTeacher:
    def __init__(self, model, load_4bit, device, max_tokens):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch, self.max_tokens = torch, max_tokens
        self.tok = AutoTokenizer.from_pretrained(model)
        kw = dict(torch_dtype=torch.bfloat16)
        if load_4bit:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4")
            kw["device_map"] = "auto"   # 4-bit must place at load time; cannot .to() afterwards
        self.model = AutoModelForCausalLM.from_pretrained(model, **kw)
        if "quantization_config" not in kw:
            self.model.to(device)
        self.model.eval()

    def __call__(self, prompt):
        text = self.tok.apply_chat_template([{"role": "user", "content": prompt}],
                                            tokenize=False, add_generation_prompt=True)
        enc = self.tok(text, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        with self.torch.no_grad():
            g = self.model.generate(**enc, max_new_tokens=self.max_tokens, do_sample=True,
                                    temperature=0.7, top_p=0.9, pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(g[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()


class APITeacher:
    def __init__(self, model, max_tokens):
        import anthropic
        self.client, self.model, self.max_tokens = anthropic.Anthropic(), model, max_tokens

    def __call__(self, prompt):
        r = self.client.messages.create(model=self.model, max_tokens=self.max_tokens,
                                        messages=[{"role": "user", "content": prompt}])
        return "".join(b.text for b in r.content if b.type == "text").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", choices=["local", "api"], default="local")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--in_path", default=os.path.join(HERE, "plan_dataset.jsonl"))
    ap.add_argument("--out_path", default=os.path.join(HERE, "plan_dataset_rich.jsonl"))
    ap.add_argument("--load_4bit", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max_tokens", type=int, default=384)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.in_path)]
    if args.limit:
        rows = rows[:args.limit]
    teacher = (LocalTeacher(args.model, args.load_4bit, args.device, args.max_tokens)
               if args.teacher == "local" else APITeacher(args.model, args.max_tokens))

    n = 0
    with open(args.out_path, "w") as f:
        for r in rows:
            plan_names = [t for t in r["plan_tokens"] if t != "EOP"]
            try:
                rich = teacher(build_prompt(r["instruction"], plan_names, r["response"]))
            except Exception as e:
                rich = r["response"]  # fall back to the brief answer on any teacher error
                print(f"[warn] teacher failed on row {n}: {e}")
            out = dict(r)
            out["response_brief"] = r["response"]
            out["response"] = rich or r["response"]
            f.write(json.dumps(out) + "\n")
            f.flush()
            n += 1
            if n % 25 == 0 or args.limit:
                print(f"{n}/{len(rows)}  e.g. -> {out['response'][:90]}...")
    print(f"wrote {args.out_path} ({n} rows, teacher={args.teacher}:{args.model})")


if __name__ == "__main__":
    main()
