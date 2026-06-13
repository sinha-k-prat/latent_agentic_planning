"""Path-B planning trainer: instruction -> plan-token sequence (target-supervised).

Qwen-0.5B encodes each instruction (frozen; encodings precomputed once for speed). A small
autoregressive decoder (GRU + learned token embeddings) predicts the plan-token sequence,
teacher-forced on dataset/plan_dataset.jsonl. We evaluate on TWO held-out sets:

  held_inst : random held-out INSTANCES (seen compositions, new content)
  held_comp : whole held-out COMPOSITIONS (plan-token sequences never seen in training)
              -> the real test of composing planning tokens in a NEW way at inference.

Metrics: exact-sequence match and per-token accuracy. This trains the PLANNING half and the
plan-token embeddings. (Execution = steering the frozen executor with these tokens is the next
step; it reuses the CE-through-frozen-executor path and is GPU-scale.)

  python dataset/train_planb.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models import ModelBundle
from src.utils import get_device, load_config, set_seed

HERE = os.path.dirname(os.path.abspath(__file__))
EOP = 64
N_CLASS = 65            # ops 0..63 + EOP(64)
BOS, PAD = 65, 66       # decoder-input specials
EPOCHS, H, E, LR, BS = 60, 256, 128, 1.5e-3, 32
# whole compositions held out (never trained) -> "new composition" generalization test
HELD_COMPS = [
    ("EXTRACT_NUMBERS", "SORT", "TOP_K", "AGGREGATE", "WRITE_SENTENCE", "EOP"),
    ("PARSE_INPUT", "SELECT_RELEVANT", "FORMAT_LIST", "EOP"),
    ("IDENTIFY_TASK", "INFER_CAUSE", "WRITE_SENTENCE", "EOP"),
    ("EXTRACT_NUMBERS", "COMPUTE_ARITHMETIC", "VALIDATE_NUMBER", "CONCLUDE", "EOP"),
]


class PlanDecoder(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.emb = nn.Embedding(PAD + 1, E)
        self.init = nn.Linear(d_model, H)
        self.gru = nn.GRU(E, H, batch_first=True)
        self.out = nn.Linear(H, N_CLASS)

    def forward(self, enc_h, dec_in):
        h0 = torch.tanh(self.init(enc_h)).unsqueeze(0)          # [1,B,H]
        o, _ = self.gru(self.emb(dec_in), h0)                  # [B,L,H]
        return self.out(o)                                     # [B,L,N_CLASS]

    @torch.no_grad()
    def greedy(self, enc_h, max_len=10):
        h = torch.tanh(self.init(enc_h)).unsqueeze(0)
        tok = torch.full((enc_h.shape[0], 1), BOS, dtype=torch.long, device=enc_h.device)
        out = []
        for _ in range(max_len):
            o, h = self.gru(self.emb(tok), h)
            tok = self.out(o[:, -1]).argmax(-1, keepdim=True)
            out.append(tok)
            if (tok == EOP).all():
                break
        return torch.cat(out, 1)                               # [B, T]


@torch.no_grad()
def encode_all(bundle, instructions):
    tok = bundle.tokenizer
    hs = []
    for i in range(0, len(instructions), 16):
        sub = [f"<|im_start|>user\n{x}<|im_end|>" for x in instructions[i:i + 16]]
        enc = tok(sub, return_tensors="pt", padding=True, add_special_tokens=False).to(bundle.device)
        out = bundle.planner_forward(input_ids=enc.input_ids, attention_mask=enc.attention_mask,
                                     output_hidden_states=True)
        hs.append(out.hidden_states[-1][:, -1, :].float().cpu())
    return torch.cat(hs, 0)


def pad(seqs, val):
    L = max(len(s) for s in seqs)
    return torch.tensor([s + [val] * (L - len(s)) for s in seqs], dtype=torch.long)


def seq_metrics(decoder, enc_h, gold):
    pred = decoder.greedy(enc_h)
    exact = tokc = tokn = 0
    for i, g in enumerate(gold):
        p = pred[i].tolist()
        p = p[: p.index(EOP) + 1] if EOP in p else p           # trim at first EOP
        exact += int(p == g)
        for a, b in zip(p, g):
            tokc += int(a == b)
        tokn += len(g)
    return exact / len(gold), tokc / max(1, tokn)


def main():
    cfg = load_config(os.path.join(os.path.dirname(HERE), "configs/default.yaml"), overrides=[
        "model.base=Qwen/Qwen2.5-0.5B-Instruct", "model.gradient_checkpointing=false",
        "train.bf16=false", "device=cpu",
    ])
    set_seed(cfg.seed)
    device = get_device(cfg)
    rows = [json.loads(l) for l in open(os.path.join(HERE, "plan_dataset.jsonl"))]

    import random
    rng = random.Random(0)
    rng.shuffle(rows)
    held_comp, pool = [], []
    held_set = set(HELD_COMPS)
    for r in rows:
        (held_comp if tuple(r["plan_tokens"]) in held_set else pool).append(r)
    n_inst = max(40, len(pool) // 10)
    held_inst, train = pool[:n_inst], pool[n_inst:]
    print(f"train {len(train)} | held_inst {len(held_inst)} | held_comp {len(held_comp)} "
          f"({len(HELD_COMPS)} unseen compositions)")

    print("Encoding instructions with Qwen-0.5B (once) ...")
    bundle = ModelBundle(cfg, device)
    def enc(split):
        return encode_all(bundle, [r["instruction"] for r in split]).to(device)
    H_tr, H_hi, H_hc = enc(train), enc(held_inst), enc(held_comp)
    G_tr = [r["plan_token_ids"] for r in train]
    G_hi = [r["plan_token_ids"] for r in held_inst]
    G_hc = [r["plan_token_ids"] for r in held_comp]

    dec = PlanDecoder(bundle.hidden_size).to(device)
    opt = torch.optim.Adam(dec.parameters(), lr=LR)
    idx = list(range(len(train)))
    for ep in range(1, EPOCHS + 1):
        rng.shuffle(idx)
        dec.train()
        tot = 0.0
        for i in range(0, len(idx), BS):
            b = idx[i:i + BS]
            gold = [G_tr[j] for j in b]
            din = pad([[BOS] + g[:-1] for g in gold], PAD).to(device)
            tgt = pad([g for g in gold], -100).to(device)
            logits = dec(H_tr[b], din)
            loss = F.cross_entropy(logits.reshape(-1, N_CLASS), tgt.reshape(-1), ignore_index=-100)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.item()) * len(b)
        if ep % 5 == 0 or ep == 1:
            dec.eval()
            tr = seq_metrics(dec, H_tr, G_tr)
            hi = seq_metrics(dec, H_hi, G_hi)
            hc = seq_metrics(dec, H_hc, G_hc) if held_comp else (0, 0)
            print(f"ep {ep:3d} loss {tot/len(train):.3f} | "
                  f"train exact {tr[0]:.2f} | held_inst exact {hi[0]:.2f} tok {hi[1]:.2f} | "
                  f"held_comp exact {hc[0]:.2f} tok {hc[1]:.2f}")

    torch.save({"decoder": dec.state_dict()}, os.path.join(HERE, "plan_head_b.pt"))
    print("\n--- held-out predictions (instruction -> predicted plan-token ids) ---")
    dec.eval()
    pred = dec.greedy(H_hi[:5])
    for r, p in zip(held_inst[:5], pred.tolist()):
        p = p[: p.index(EOP) + 1] if EOP in p else p
        print(f"\n{r['instruction'][:70]}")
        print(f"  gold: {r['plan_token_ids']}")
        print(f"  pred: {p}")
    print("\nsaved plan_head_b.pt")


if __name__ == "__main__":
    main()
