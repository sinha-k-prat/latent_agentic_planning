"""Held-out generalization metric: NLL of UNSEEN-topic gold targets through the frozen Qwen,
with no plan (base) vs the trained latent plan. The planner was trained on 8 *other* topics;
these 4 targets were never trained on. If the trained plan LOWERS held-out NLL vs base, the
latent codes encode a transferable decompose-then-execute strategy (generalization), not memo.

  python examples/heldout_ce.py --ckpt runs/ce_demo/checkpoints/final
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.inference import load_for_inference
from src.injection import build_one_prefix
from src.losses import _response_logprob
from src.utils import get_device, load_config

HERE = os.path.dirname(os.path.abspath(__file__))


@torch.no_grad()
def nll(bundle, q, plan_vec, y_ids, sysp):
    prefix, plen = build_one_prefix(bundle, q, plan_vec, sysp)
    return -float(_response_logprob(bundle, prefix, plen, y_ids).item())  # mean per-token NLL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/ce_demo/checkpoints/final")
    ap.add_argument("--splits", default=os.path.join(HERE, "ce_demo_splits.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "heldout_ce.md"))
    args = ap.parse_args()

    cfg = load_config(os.path.join(os.path.dirname(HERE), "configs/default.yaml"), overrides=[
        "model.base=Qwen/Qwen2.5-0.5B-Instruct", "plan.N=8", "plan.K=64",
        "train.max_new_tokens=96", "model.gradient_checkpointing=false",
        "train.bf16=false", "device=cpu",
    ])
    device = get_device(cfg)
    bundle, plan_gen = load_for_inference(cfg, args.ckpt, device)
    sysp = cfg.plan.system_prompt
    tok = bundle.tokenizer
    eos = tok.eos_token_id
    held = [d for d in json.load(open(args.splits))["eval"] if d.get("target")]

    rows, base_all, plan_all = [], [], []
    for d in held:
        q, tgt = d["instruction"], d["target"]
        ids = tok(tgt, add_special_tokens=False).input_ids[: cfg.train.max_new_tokens]
        y = torch.tensor((ids or [eos]) + [eos], dtype=torch.long)
        base = nll(bundle, q, None, y, sysp)
        p, _ = plan_gen.compute_plans([q], tau=cfg.plan.gumbel.tau_end, hard=True)
        plan = nll(bundle, q, p[0], y, sysp)
        base_all.append(base)
        plan_all.append(plan)
        rows.append((q, base, plan))

    mb = sum(base_all) / len(base_all)
    mp = sum(plan_all) / len(plan_all)
    out = [
        "# Held-out generalization: NLL of unseen-topic targets (lower = better)",
        "",
        "Planner trained on 8 *different* topics; the 4 targets below were never trained on. "
        "`base` = frozen Qwen, no plan. `planner` = same frozen Qwen + trained latent plan.",
        "",
        "| held-out prompt | base NLL | planner NLL | Δ |",
        "|---|---|---|---|",
    ]
    for q, b, p in rows:
        out.append(f"| {q[:60]}… | {b:.3f} | {p:.3f} | {p - b:+.3f} |")
    out += ["", f"**Mean held-out NLL — base {mb:.3f} · planner {mp:.3f} · Δ {mp - mb:+.3f}**",
            "", ("Planner LOWERS held-out NLL → the latent plan transfers (generalization)."
                 if mp < mb else
                 "Planner does NOT lower held-out NLL → no transfer at this scale (honest negative).")]
    with open(args.out, "w") as f:
        f.write("\n".join(out) + "\n")
    print("\n".join(out[-3:]))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
