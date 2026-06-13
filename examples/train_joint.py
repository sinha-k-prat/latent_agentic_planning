"""One-model joint planner+executor (single Qwen, LoRA).

A single Qwen emits ONE autoregressive trajectory with a phase switch:

    [system, user(instruction), assistant]  ->  p1..pN EOP   ->  y1..yM
                                                (PLAN head,        (LANGUAGE head,
                                                 64+EOP vocab,      Qwen vocab,
                                                 latent thinking)   readable answer)

Same backbone, two output heads; the active head depends on the phase. Plan tokens use a NEW
input-embedding table (plan_emb); response tokens use Qwen's word embeddings. Trained by
teacher-forcing the whole x->plan->response sequence in one forward pass:

    L = CE_plan(p* | x)  +  lambda * CE_response(y* | x, p*)

Trainable: planner LoRA + plan_head + plan_emb. Qwen's language head/embeddings stay frozen
(base answering preserved). Reports TRAIN loss and HELD-OUT loss (both should fall).

  python examples/train_joint.py                       # small CPU sanity check
  python examples/train_joint.py --train 6000 --held 600 --epochs 3 --device cuda   # real (GPU)
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models import ModelBundle
from src.utils import get_device, load_config, set_seed

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "dataset", "plan_dataset.jsonl")
N_PLAN = 65   # 64 ops + EOP
SYS = "You are a helpful assistant."


class Joint(nn.Module):
    """Holds the two NEW trainable pieces; the Qwen backbone (LoRA) lives in `bundle`."""

    def __init__(self, bundle):
        super().__init__()
        object.__setattr__(self, "bundle", bundle)         # keep backbone out of self.parameters()
        d = bundle.hidden_size
        self.plan_head = nn.Linear(d, N_PLAN)              # hidden -> plan-token logits
        self.plan_emb = nn.Embedding(N_PLAN, d)            # plan-token id -> input vector
        self.to(bundle.device)
        self.plan_head.to(bundle.dtype)
        self.plan_emb.to(bundle.dtype)

    def loss(self, prompt_ids, plan_ids, resp_ids, lam):
        b = self.bundle
        dev = b.device
        p_emb = b.embed_tokens(torch.tensor(prompt_ids, device=dev))      # [P, d]
        pl = torch.tensor(plan_ids, device=dev)
        pl_emb = self.plan_emb(pl).to(b.dtype)                            # [L, d]
        r_emb = b.embed_tokens(torch.tensor(resp_ids, device=dev))       # [M, d]
        full = torch.cat([p_emb, pl_emb, r_emb], 0).unsqueeze(0)         # [1, T, d]
        mask = torch.ones(1, full.shape[1], dtype=torch.long, device=dev)
        out = b.planner_forward(inputs_embeds=full, attention_mask=mask, output_hidden_states=True)
        hidden = out.hidden_states[-1][0]                                # [T, d]
        logits = out.logits[0]                                          # [T, V] (Qwen language head)

        P, L, M = len(prompt_ids), len(plan_ids), len(resp_ids)
        # PLAN head predicts plan token j from the hidden at position (P-1 .. P+L-2)
        plan_logits = self.plan_head(hidden[P - 1: P + L - 1].float())  # [L, 65]
        loss_plan = F.cross_entropy(plan_logits, pl)
        # LANGUAGE head predicts response token k from hidden at (P+L-1 .. P+L+M-2)
        resp_logits = logits[P + L - 1: P + L + M - 1].float()          # [M, V]
        loss_resp = F.cross_entropy(resp_logits, torch.tensor(resp_ids, device=dev))
        return loss_plan, loss_resp, loss_plan + lam * loss_resp


def make_example(tok, r, max_resp):
    prompt = tok(f"<|im_start|>system\n{SYS}<|im_end|>\n<|im_start|>user\n{r['instruction']}"
                 f"<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False).input_ids
    resp = tok(r["response"], add_special_tokens=False).input_ids[:max_resp] + [tok.eos_token_id]
    return prompt, r["plan_token_ids"], resp


@torch.no_grad()
def eval_loss(joint, tok, rows, lam, max_resp):
    joint.eval()
    tp = tr = n = 0.0
    for r in rows:
        pr, pl, rs = make_example(tok, r, max_resp)
        lp, lr, _ = joint.loss(pr, pl, rs, lam)
        tp += float(lp); tr += float(lr); n += 1
    return tp / n, tr / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=160)
    ap.add_argument("--held", type=int, default=48)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--max_resp", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--base", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    cfg = load_config(os.path.join(os.path.dirname(HERE), "configs/default.yaml"), overrides=[
        f"model.base={args.base}", "model.gradient_checkpointing=false",
        "train.bf16=false", f"device={args.device}",
    ])
    set_seed(cfg.seed)
    get_device(cfg)
    rows = [json.loads(l) for l in open(DATA)]
    random.Random(0).shuffle(rows)
    train, held = rows[:args.train], rows[args.train:args.train + args.held]
    print(f"train {len(train)} | held {len(held)} | base {args.base} | device {args.device}")

    bundle = ModelBundle(cfg, get_device(cfg))
    tok = bundle.tokenizer
    joint = Joint(bundle)
    params = bundle.trainable_backbone_params() + list(joint.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr)

    print(f"{'ep':>3} {'train_total':>11} {'train_plan':>10} {'train_resp':>10} "
          f"{'held_total':>10} {'held_plan':>9} {'held_resp':>9}")
    hp0, hr0 = eval_loss(joint, tok, held, args.lam, args.max_resp)
    print(f"{0:>3} {'-':>11} {'-':>10} {'-':>10} {hp0 + args.lam * hr0:>10.3f} {hp0:>9.3f} {hr0:>9.3f}")
    hist = [(0, "", "", "", hp0 + args.lam * hr0, hp0, hr0)]

    idx = list(range(len(train)))
    for ep in range(1, args.epochs + 1):
        random.Random(ep).shuffle(idx)
        joint.train()
        tp = tr = tt = 0.0
        opt.zero_grad(set_to_none=True)
        for step, j in enumerate(idx, 1):
            pr, pl, rs = make_example(tok, train[j], args.max_resp)
            lp, lr, loss = joint.loss(pr, pl, rs, args.lam)
            (loss / args.accum).backward()
            tp += float(lp); tr += float(lr); tt += float(loss)
            if step % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); opt.zero_grad(set_to_none=True)
        n = len(train)
        hp, hr = eval_loss(joint, tok, held, args.lam, args.max_resp)
        print(f"{ep:>3} {tt/n:>11.3f} {tp/n:>10.3f} {tr/n:>10.3f} "
              f"{hp + args.lam * hr:>10.3f} {hp:>9.3f} {hr:>9.3f}")
        hist.append((ep, tt / n, tp / n, tr / n, hp + args.lam * hr, hp, hr))

    import csv
    with open(os.path.join(HERE, "joint_metrics.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_total", "train_plan", "train_resp",
                    "held_total", "held_plan", "held_resp"])
        w.writerows(hist)

    os.makedirs(os.path.join(HERE, "joint_ckpt"), exist_ok=True)
    bundle.planner.save_pretrained(os.path.join(HERE, "joint_ckpt"))
    torch.save(joint.state_dict(), os.path.join(HERE, "joint_ckpt", "heads.pt"))
    print("saved joint_ckpt/")


if __name__ == "__main__":
    main()
