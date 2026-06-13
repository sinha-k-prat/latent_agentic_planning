"""One-model joint planner+executor (single Qwen, LoRA) — with KL anchor, plan ablation, early-stop.

A single Qwen emits ONE autoregressive trajectory with a phase switch:

    [system, user(instruction), assistant]  ->  p1..pN EOP   ->  y1..yM
                                                (PLAN head)        (LANGUAGE head)

Trained by teacher-forcing x->plan->response:  L = CE_plan + lambda*CE_response.

Three additions (all motivated by the OOD failure where the planned answer collapsed to a terse
template):
  * RESPONSE KL ANCHOR: on a SEPARATE calibration set (general prompts, no plan), pull the
    LoRA model's response distribution toward the FROZEN base (KL(base || adapted)). Preserves
    fluent general answering without fighting the task CE. (lambda_kl)
  * PLAN-vs-NO-PLAN ABLATION in eval: held-out response NLL WITH plan vs with plan ablated.
    The plan only earns its keep if planned < unplanned (gap > 0).
  * EARLY-STOP on the held-out planned loss; best checkpoint saved.

Trainable: planner LoRA + plan_head + plan_emb. Frozen: Qwen language head/embeddings.

  python examples/train_joint.py                                  # CPU sanity
  python examples/train_joint.py --train 700 --held 100 --device cuda   # full (GPU)
"""
import argparse
import csv
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
    def __init__(self, bundle):
        super().__init__()
        object.__setattr__(self, "bundle", bundle)
        d = bundle.hidden_size
        self.plan_head = nn.Linear(d, N_PLAN)
        self.plan_emb = nn.Embedding(N_PLAN, d)
        self.to(bundle.device)
        self.plan_head.to(bundle.dtype)
        self.plan_emb.to(bundle.dtype)

    def _seq(self, prompt_ids, plan_ids, resp_ids, use_base=False):
        b, dev = self.bundle, self.bundle.device
        parts = [b.embed_tokens(torch.tensor(prompt_ids, device=dev))]
        if plan_ids:
            parts.append(self.plan_emb(torch.tensor(plan_ids, device=dev)).to(b.dtype))
        parts.append(b.embed_tokens(torch.tensor(resp_ids, device=dev)))
        full = torch.cat(parts, 0).unsqueeze(0)
        mask = torch.ones(1, full.shape[1], dtype=torch.long, device=dev)
        if use_base:
            with b.planner.disable_adapter():
                out = b.planner(inputs_embeds=full, attention_mask=mask,
                                output_hidden_states=True, use_cache=False)
        else:
            out = b.planner_forward(inputs_embeds=full, attention_mask=mask, output_hidden_states=True)
        return out.hidden_states[-1][0], out.logits[0], len(prompt_ids), len(plan_ids), len(resp_ids)

    def loss(self, prompt_ids, plan_ids, resp_ids, lam):
        dev = self.bundle.device
        hidden, logits, P, L, M = self._seq(prompt_ids, plan_ids, resp_ids)
        plan_logits = self.plan_head(hidden[P - 1: P + L - 1].float())
        loss_plan = F.cross_entropy(plan_logits, torch.tensor(plan_ids, device=dev))
        resp_logits = logits[P + L - 1: P + L + M - 1].float()
        loss_resp = F.cross_entropy(resp_logits, torch.tensor(resp_ids, device=dev))
        return loss_plan, loss_resp, loss_plan + lam * loss_resp

    @torch.no_grad()
    def resp_nll(self, prompt_ids, resp_ids, plan_ids):
        dev = self.bundle.device
        _, logits, P, L, M = self._seq(prompt_ids, plan_ids, resp_ids)
        rl = logits[P + L - 1: P + L + M - 1].float()
        return float(F.cross_entropy(rl, torch.tensor(resp_ids, device=dev)))

    def kl_resp(self, prompt_ids, base_resp_ids):
        """KL(base || adapted) over a base-generated response span, NO plan -> anti-forgetting."""
        dev = self.bundle.device
        _, la_logits, P, L, M = self._seq(prompt_ids, [], base_resp_ids, use_base=False)
        span = slice(P - 1, P - 1 + M)
        la = F.log_softmax(la_logits[span].float(), -1)
        with torch.no_grad():
            _, lb_logits, *_ = self._seq(prompt_ids, [], base_resp_ids, use_base=True)
            lb = F.log_softmax(lb_logits[span].float(), -1)
        return (lb.exp() * (lb - la)).sum(-1).mean()


def make_example(tok, r, max_resp):
    prompt = tok(f"<|im_start|>system\n{SYS}<|im_end|>\n<|im_start|>user\n{r['instruction']}"
                 f"<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False).input_ids
    resp = tok(r["response"], add_special_tokens=False).input_ids[:max_resp] + [tok.eos_token_id]
    return prompt, r["plan_token_ids"], resp


@torch.no_grad()
def base_responses(bundle, instructions, max_resp):
    """Generate the FROZEN base model's response for each calibration instruction (once)."""
    tok = bundle.tokenizer
    out = []
    for instr in instructions:
        pids = tok(f"<|im_start|>system\n{SYS}<|im_end|>\n<|im_start|>user\n{instr}"
                   f"<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False).input_ids
        emb = bundle.embed_tokens(torch.tensor(pids, device=bundle.device)).unsqueeze(0)
        with bundle.planner.disable_adapter():
            g = bundle.planner.generate(inputs_embeds=emb,
                                        attention_mask=torch.ones(1, len(pids), dtype=torch.long),
                                        max_new_tokens=max_resp, do_sample=False,
                                        pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        ids = g[0].tolist()
        if tok.eos_token_id in ids:
            ids = ids[: ids.index(tok.eos_token_id) + 1]
        out.append((pids, [t for t in ids if t != tok.pad_token_id] or [tok.eos_token_id]))
    return out


@torch.no_grad()
def eval_metrics(joint, tok, rows, lam, max_resp):
    joint.eval()
    tp = trp = tru = 0.0
    for r in rows:
        pr, pl, rs = make_example(tok, r, max_resp)
        lp, lr, _ = joint.loss(pr, pl, rs, lam)
        tp += float(lp)
        trp += float(lr)                              # response NLL WITH plan
        tru += joint.resp_nll(pr, rs, [])             # response NLL with plan ABLATED
    n = len(rows)
    return tp / n, trp / n, tru / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=160)
    ap.add_argument("--held", type=int, default=48)
    ap.add_argument("--calib", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--max_resp", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--lam_kl", type=float, default=0.3)
    ap.add_argument("--kl_batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--base", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--data", default=DATA, help="jsonl corpus (e.g. dataset/plan_dataset_rich.jsonl)")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    cfg = load_config(os.path.join(os.path.dirname(HERE), "configs/default.yaml"), overrides=[
        f"model.base={args.base}", "model.gradient_checkpointing=false",
        "train.bf16=false", f"device={args.device}",
    ])
    set_seed(cfg.seed)
    rows = [json.loads(l) for l in open(args.data)]
    random.Random(0).shuffle(rows)
    train = rows[:args.train]
    held = rows[args.train:args.train + args.held]
    calib_rows = rows[args.train + args.held:args.train + args.held + args.calib]
    print(f"train {len(train)} | held {len(held)} | calib {len(calib_rows)} | "
          f"lam_kl {args.lam_kl} | base {args.base} | device {args.device}")
    if args.lam_kl > 0 and not calib_rows:
        print("[warn] calibration set is EMPTY (train+held+calib exceeds dataset size) -> "
              "KL anchor is DISABLED. Lower --train/--held/--calib so they sum to <= dataset size.")

    bundle = ModelBundle(cfg, get_device(cfg))
    tok = bundle.tokenizer
    joint = Joint(bundle)
    params = bundle.trainable_backbone_params() + list(joint.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr)

    calib = []
    if args.lam_kl > 0:
        print(f"Precomputing {len(calib_rows)} base responses for the KL anchor ...")
        calib = base_responses(bundle, [r["instruction"] for r in calib_rows], args.max_resp)

    cols = ["epoch", "train_total", "train_plan", "train_resp",
            "held_plan", "held_resp_planned", "held_resp_unplanned", "gap_unplanned_minus_planned"]
    print(" ".join(f"{c:>12}" for c in cols))

    def report(ep, tt, tplan, tresp):
        hp, hrp, hru = eval_metrics(joint, tok, held, args.lam, args.max_resp)
        gap = hru - hrp
        vals = [ep, tt, tplan, tresp, hp, hrp, hru, gap]
        print(" ".join(f"{v:>12.3f}" if isinstance(v, float) else f"{v:>12}" for v in vals))
        return vals, hp + args.lam * hrp

    hist = []
    v0, _ = report(0, float("nan"), float("nan"), float("nan"))
    hist.append(v0)
    best, best_ep, bad = float("inf"), 0, 0
    idx = list(range(len(train)))
    for ep in range(1, args.epochs + 1):
        random.Random(ep).shuffle(idx)
        joint.train()
        tt = tp = tr = 0.0
        opt.zero_grad(set_to_none=True)
        for step, j in enumerate(idx, 1):
            pr, pl, rs = make_example(tok, train[j], args.max_resp)
            lp, lr, loss = joint.loss(pr, pl, rs, args.lam)
            (loss / args.accum).backward()
            tt += float(loss); tp += float(lp); tr += float(lr)
            if step % args.accum == 0:
                if args.lam_kl > 0 and calib:
                    kb = random.sample(calib, min(args.kl_batch, len(calib)))
                    kl = sum(joint.kl_resp(p, r) for p, r in kb) / len(kb)
                    (args.lam_kl * kl).backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); opt.zero_grad(set_to_none=True)
        n = len(train)
        vals, score = report(ep, tt / n, tp / n, tr / n)
        hist.append(vals)
        if score < best - 1e-3:
            best, best_ep, bad = score, ep, 0
            os.makedirs(os.path.join(HERE, "joint_ckpt"), exist_ok=True)
            bundle.planner.save_pretrained(os.path.join(HERE, "joint_ckpt"))
            torch.save(joint.state_dict(), os.path.join(HERE, "joint_ckpt", "heads.pt"))
        else:
            bad += 1
            if bad >= args.patience:
                print(f"early stop @ ep {ep} (best held planned loss @ ep {best_ep})")
                break

    with open(os.path.join(HERE, "joint_metrics.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(cols); w.writerows(hist)
    print(f"best epoch {best_ep} | saved joint_ckpt/ + joint_metrics.csv")


if __name__ == "__main__":
    main()
