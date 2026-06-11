"""Main training loop for the latent Planner–Executor RL pipeline.

  python -m src.train                 # full run from configs/default.yaml
  python -m src.train --smoke-test    # 0.5B base, stub judge, 10 steps
  python -m src.train plan.K=256 train.steps=200   # dotlist overrides
"""
import argparse
import os
import random

import torch
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup

from .data import build_or_load_splits
from .evaluate import run_eval
from .injection import build_batch_prefix
from .judge import build_judge
from .losses import kl_anchor_loss, rl_loss_and_backward
from .models import ModelBundle
from .plan_head import PlanGenerator
from .rollout import run_rollout
from .utils import CSVLogger, JsonlLogger, get_device, gumbel_tau, load_config, microbatches, set_seed


@torch.no_grad()
def precompute_calibration(bundle, calib_instructions, cfg):
    """Sample one frozen-executor response per calibration instruction (no plan), once."""
    items = []
    eos, pad = bundle.tokenizer.eos_token_id, bundle.tokenizer.pad_token_id
    for mb in microbatches(range(len(calib_instructions)), cfg.train.gen_microbatch):
        sub = [calib_instructions[j] for j in mb]
        emb, mask = build_batch_prefix(bundle, sub, None, cfg.plan.system_prompt, pad_side="left")
        gen = bundle.executor_generate(
            emb, mask, cfg.train.max_new_tokens, cfg.train.exec_temperature, cfg.train.exec_top_p,
        )
        for instr, row in zip(sub, gen):
            ids = row.tolist()
            if eos in ids:
                ids = ids[: ids.index(eos)]
            ids = [t for t in ids if t != pad] or [eos]
            items.append((instr, ids))
    return items


def save_checkpoint(out_dir, bundle, plan_gen, tag):
    d = os.path.join(out_dir, "checkpoints", tag)
    os.makedirs(d, exist_ok=True)
    bundle.planner.save_pretrained(d)  # LoRA adapters (or full model if full_finetune)
    torch.save(plan_gen.state_dict(), os.path.join(d, "plan_gen.pt"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--smoke-test", action="store_true")
    ap.add_argument("overrides", nargs="*", help="dotlist overrides, e.g. train.steps=50")
    args = ap.parse_args()

    cfg = load_config(args.config, overrides=args.overrides, smoke=args.smoke_test)
    set_seed(cfg.seed)
    device = get_device(cfg)
    out_dir = cfg.logging.out_dir
    os.makedirs(out_dir, exist_ok=True)

    csv = CSVLogger(os.path.join(out_dir, "metrics.csv")) if cfg.logging.csv else None
    jsonl = JsonlLogger(os.path.join(out_dir, "rollouts.jsonl"))
    wb = None
    if cfg.logging.wandb:
        import wandb
        wb = wandb.init(project=cfg.logging.project, config=dict(cfg))

    print("Loading models ...")
    bundle = ModelBundle(cfg, device)
    plan_gen = PlanGenerator(bundle, cfg)
    judge = build_judge(cfg, device)

    print("Building data splits ...")
    splits = build_or_load_splits(cfg, bundle.tokenizer)
    train_set, eval_set, calib_set = splits["train"], splits["eval"], splits["calib"]

    print(f"Precomputing {len(calib_set)} calibration responses ...")
    calib_items = precompute_calibration(bundle, calib_set, cfg)

    params = bundle.trainable_backbone_params() + list(plan_gen.parameters())
    opt = AdamW(params, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    sched = get_cosine_schedule_with_warmup(opt, cfg.train.warmup_steps, cfg.train.steps)

    base_sig = bundle.weight_signature()  # frozen-executor weight hash (must never change)
    rng = random.Random(cfg.seed)
    best_eval = -1.0

    for step in range(1, cfg.train.steps + 1):
        instructions = rng.sample(train_set, min(cfg.train.batch_size, len(train_set)))
        tau = gumbel_tau(step, cfg)

        batch = run_rollout(bundle, plan_gen, judge, instructions, cfg, tau)
        if cfg.plan.plan_mode == "vq_codebook":
            plan_gen.vq_update(batch.aux)

        opt.zero_grad(set_to_none=True)
        rl_stats = rl_loss_and_backward(bundle, plan_gen, batch, cfg, tau, optimizer_scale=1.0)

        # Calibration KL anchor on a fresh minibatch.
        calib_mb = random.sample(calib_items, min(cfg.train.kl_batch, len(calib_items)))
        kl = kl_anchor_loss(bundle, calib_mb, cfg)
        (cfg.train.lambda_kl * kl).backward()

        gnorm = torch.nn.utils.clip_grad_norm_(params, cfg.train.grad_clip)
        opt.step()
        sched.step()

        row = {
            "step": step,
            "reward_mean": float(batch.rewards.mean().item()),
            "reward_p10": float(batch.rewards.flatten().quantile(0.1).item()),
            "reward_p90": float(batch.rewards.flatten().quantile(0.9).item()),
            "adv_abs_mean": float(batch.advantages.abs().mean().item()),
            "kl": float(kl.item()),
            "rl_loss": rl_stats["rl_loss"],
            "plan_grad_norm": rl_stats["plan_grad_norm"],
            "grad_norm": float(gnorm),
            "tau": tau,
            "kept_groups": int(batch.keep.sum().item()),
            "judge_unparsable": judge.n_unparsable,
            "lr": sched.get_last_lr()[0],
        }
        if cfg.plan.plan_mode == "gumbel_codebook":
            row["peakedness"] = batch.aux.get("peakedness", 0.0)
        if cfg.plan.plan_mode == "vq_codebook":
            row["vq_perplexity"] = batch.aux.get("perplexity", 0.0)
            row["vq_usage"] = batch.aux.get("usage", 0)
            row["commit_loss"] = rl_stats["commit_loss"]

        if step % cfg.logging.log_every == 0:
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})
            if csv:
                csv.log(row)
            if wb:
                wb.log(row, step=step)

        if step % cfg.logging.jsonl_every == 0:
            for j in range(min(len(batch.responses), cfg.train.group_size)):
                jsonl.log({
                    "step": step, "x": batch.flat_instructions[j],
                    "codes": (batch.aux.get("indices").tolist()[j]
                              if batch.aux.get("indices") is not None else None),
                    "y": batch.response_texts[j],
                    "R": float(batch.rewards.flatten()[j].item()),
                })

        # The executor must be frozen forever.
        if step % 50 == 0:
            assert abs(bundle.weight_signature() - base_sig) < 1e-3, "executor weights changed!"

        if step % cfg.logging.ckpt_every == 0:
            save_checkpoint(out_dir, bundle, plan_gen, f"step{step}")

        if step % cfg.logging.eval_every == 0:
            ev = run_eval(bundle, plan_gen, judge, eval_set[: min(64, len(eval_set))], cfg)
            print("EVAL", {k: round(v, 4) for k, v in ev.items()})
            if csv:
                csv.log({"step": step, **{f"eval_{k}": v for k, v in ev.items()}})
            key = ev.get("plan_hard", ev.get("executor_only", -1))
            if key > best_eval:
                best_eval = key
                save_checkpoint(out_dir, bundle, plan_gen, "best")

    save_checkpoint(out_dir, bundle, plan_gen, "final")
    if csv:
        csv.close()
    jsonl.close()
    print("done. checkpoints in", os.path.join(out_dir, "checkpoints"))


if __name__ == "__main__":
    main()
