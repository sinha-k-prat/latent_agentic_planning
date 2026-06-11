"""Inference CLI.

  python -m src.inference --ckpt runs/default/checkpoints/best --question "..."
  python -m src.inference --ckpt ... --question "..." --judge
  python -m src.inference --ckpt ... --question "..." --no-plan   # bare executor
"""
import argparse
import os

import torch
from peft import PeftModel

from .injection import build_batch_prefix
from .judge import build_judge
from .models import ModelBundle
from .plan_head import PlanGenerator
from .utils import get_device, load_config


def load_for_inference(cfg, ckpt, device):
    bundle = ModelBundle(cfg, device)
    if not cfg.model.full_finetune and os.path.isdir(ckpt):
        # Reload LoRA adapters from the checkpoint on top of the shared base.
        try:
            bundle.planner = PeftModel.from_pretrained(bundle.planner.get_base_model(), ckpt)
            bundle.planner.to(device)
            bundle.embed_module = bundle.planner.get_input_embeddings()
        except Exception as e:
            print(f"[warn] could not reload adapters ({e}); using freshly-initialised planner")
    plan_gen = PlanGenerator(bundle, cfg)
    pg_path = os.path.join(ckpt, "plan_gen.pt")
    if os.path.exists(pg_path):
        plan_gen.load_state_dict(torch.load(pg_path, map_location=device))
    return bundle, plan_gen


@torch.no_grad()
def answer(bundle, plan_gen, question, cfg, use_plan=True):
    sysp = cfg.plan.system_prompt
    codes = None
    plan_vectors = None
    if use_plan and cfg.plan.plan_mode in ("gumbel_codebook", "vq_codebook"):
        p, aux = plan_gen.compute_plans([question], tau=cfg.plan.gumbel.tau_end, hard=True)
        plan_vectors = p
        codes = aux["indices"].tolist()[0]
    emb, mask = build_batch_prefix(bundle, [question], plan_vectors, sysp, pad_side="left")
    gen = bundle.executor_generate(
        emb, mask, cfg.train.max_new_tokens, cfg.train.exec_temperature, cfg.train.exec_top_p,
    )
    row = gen[0].tolist()
    eos = bundle.tokenizer.eos_token_id
    if eos in row:
        row = row[: row.index(eos)]
    text = bundle.tokenizer.decode(row, skip_special_tokens=True)
    return codes, text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--ckpt", default="runs/default/checkpoints/best")
    ap.add_argument("--question", required=True)
    ap.add_argument("--judge", action="store_true", help="also print the judge score")
    ap.add_argument("--no-plan", action="store_true", help="bare executor, no plan prefix")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    cfg = load_config(args.config, overrides=args.overrides)
    device = get_device(cfg)
    bundle, plan_gen = load_for_inference(cfg, args.ckpt, device)

    codes, text = answer(bundle, plan_gen, args.question, cfg, use_plan=not args.no_plan)
    if codes is not None:
        print("selected code indices:", codes)
    print("\n--- answer ---\n" + text)

    if args.judge:
        judge = build_judge(cfg, device)
        r = judge.score(args.question, text)
        print(f"\njudge score: {r:.3f}")


if __name__ == "__main__":
    main()
