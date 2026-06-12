"""Side-by-side: bare frozen Qwen vs (trained latent planner + the SAME frozen Qwen).

Loads a CE-distilled checkpoint and runs each multi-step prompt twice — once with no plan
(the original Qwen executor) and once with the trained latent plan prepended (hardened codes).
The executor weights are identical in both columns; only the learned latent prefix differs.
Writes examples/planner_vs_base.md.

  python examples/planner_vs_base.py --ckpt runs/ce_demo/checkpoints/final
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference import answer, load_for_inference
from src.utils import get_device, load_config

HERE = os.path.dirname(os.path.abspath(__file__))


def lists_then_selects(text):
    """Crude structure check: does the answer enumerate items before honing in on one?"""
    enumerated = bool(re.search(r"(?:^|\s)(?:1[\).]|Plan:|first[, ]|steps?:)", text, re.I))
    multi = len(re.findall(r"\d[\).]", text)) >= 2 or text.count(",") >= 2
    return enumerated and multi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/ce_demo/checkpoints/final")
    ap.add_argument("--splits", default=os.path.join(HERE, "ce_demo_splits.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "planner_vs_base.md"))
    args = ap.parse_args()

    cfg = load_config(os.path.join(os.path.dirname(HERE), "configs/default.yaml"), overrides=[
        "model.base=Qwen/Qwen2.5-0.5B-Instruct", "plan.N=8", "plan.K=64",
        "train.exec_temperature=0.3", "train.max_new_tokens=96",
        "model.gradient_checkpointing=false", "train.bf16=false", "device=cpu",
    ])
    device = get_device(cfg)
    bundle, plan_gen = load_for_inference(cfg, args.ckpt, device)
    prompts = [d["instruction"] for d in json.load(open(args.splits))["eval"]]

    out = [
        "# Bare Qwen vs (trained latent planner + frozen Qwen)",
        "",
        "Model: **Qwen2.5-0.5B-Instruct**. *Base* = the frozen executor, no plan. *Planner* = a "
        "trained latent plan (hardened codebook entries) prepended as prefix `inputs_embeds` to "
        "the **same frozen** Qwen — only the learned latent prefix differs between columns.",
        "",
        "_**Generalization test**: the planner was CE-distilled on 8 *different* 'list → select → "
        "describe' prompts and is evaluated here on these **held-out, unseen topics**. If the latent "
        "plan steers the frozen Qwen to decompose-then-execute on a topic it never trained on, the "
        "codes encode a transferable thinking strategy in latent space — not memorized text._",
        "",
    ]
    base_struct = plan_struct = 0
    for i, q in enumerate(prompts, 1):
        _, base = answer(bundle, plan_gen, q, cfg, use_plan=False)
        codes, plan = answer(bundle, plan_gen, q, cfg, use_plan=True)
        b1, p1 = base.strip().replace("\n", " "), plan.strip().replace("\n", " ")
        base_struct += lists_then_selects(b1)
        plan_struct += lists_then_selects(p1)
        out += [
            f"## {i}. {q}", "",
            f"**Base Qwen (no plan):**", "", f"> {b1}", "",
            f"**Planner + frozen Qwen** (codes {codes}):", "", f"> {p1}", "",
        ]
    out += [
        "---", "",
        f"**Decompose-then-execute structure detected:** base {base_struct}/{len(prompts)}  ·  "
        f"planner {plan_struct}/{len(prompts)}",
    ]
    with open(args.out, "w") as f:
        f.write("\n".join(out) + "\n")
    print("wrote", args.out, f"| structure base={base_struct} planner={plan_struct}")


if __name__ == "__main__":
    main()
