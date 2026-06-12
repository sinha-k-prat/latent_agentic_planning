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

from .data import build_or_load_splits, split_instructions, split_pairs
from .evaluate import run_eval
from .injection import build_batch_prefix
from .judge import build_judge
from .losses import ce_loss_and_backward, kl_anchor_loss, rl_loss_and_backward
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

    mode = cfg.train.loss_mode  # "rl" | "ce"
    use_kl = cfg.train.lambda_kl > 0
    will_eval = cfg.logging.eval_every <= cfg.train.steps

    print(f"Loading models ... (loss_mode={mode})")
    bundle = ModelBundle(cfg, device)
    plan_gen = PlanGenerator(bundle, cfg)
    # The judge is only needed for RL training and for the (judge-scored) eval harness.
    judge = build_judge(cfg, device) if (mode == "rl" or will_eval) else None

    print("Building data splits ...")
    splits = build_or_load_splits(cfg, bundle.tokenizer)
    train_set, eval_set, calib_set = splits["train"], splits["eval"], splits["calib"]
    eval_instr = split_instructions(eval_set)

    ce_pool = None
    if mode == "ce":
        ce_pool = [d for d in split_pairs(train_set) if d["target"]]
        if not ce_pool:
            raise ValueError("loss_mode=ce needs targets, but the dataset has no "
                             "output/response field. Use e.g. data.dataset=Open-Orca/OpenOrca.")
        print(f"CE distillation: {len(ce_pool)} (instruction, target) pairs.")

    calib_items = []
    if use_kl:
        print(f"Precomputing {len(calib_set)} calibration responses ...")
        calib_items = precompute_calibration(bundle, split_instructions(calib_set), cfg)

    params = bundle.trainable_backbone_params() + list(plan_gen.parameters())
    opt = AdamW(params, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    sched = get_cosine_schedule_with_warmup(opt, cfg.train.warmup_steps, cfg.train.steps)

    base_sig = bundle.weight_signature()  # frozen-executor weight hash (must never change)
    rng = random.Random(cfg.seed)
    best_eval = -1.0

    for step in range(1, cfg.train.steps + 1):
        tau = gumbel_tau(step, cfg)
        opt.zero_grad(set_to_none=True)
        row = {"step": step, "tau": tau}
        batch = None

        if mode == "rl":
            instructions = split_instructions(rng.sample(train_set, min(cfg.train.batch_size, len(train_set))))
            batch = run_rollout(bundle, plan_gen, judge, instructions, cfg, tau)
            if cfg.plan.plan_mode == "vq_codebook":
                plan_gen.vq_update(batch.aux)
            stats = rl_loss_and_backward(bundle, plan_gen, batch, cfg, tau, optimizer_scale=1.0)
            aux = batch.aux
            row.update({
                "reward_mean": float(batch.rewards.mean().item()),
                "reward_p10": float(batch.rewards.flatten().quantile(0.1).item()),
                "reward_p90": float(batch.rewards.flatten().quantile(0.9).item()),
                "adv_abs_mean": float(batch.advantages.abs().mean().item()),
                "rl_loss": stats["rl_loss"], "kept_groups": int(batch.keep.sum().item()),
                "judge_unparsable": judge.n_unparsable,
            })
        else:  # ce
            items = rng.sample(ce_pool, min(cfg.train.batch_size, len(ce_pool)))
            instructions = [d["instruction"] for d in items]
            targets = [d["target"] for d in items]
            stats = ce_loss_and_backward(bundle, plan_gen, instructions, targets, cfg, tau)
            aux = stats["aux"]
            if cfg.plan.plan_mode == "vq_codebook" and aux is not None:
                plan_gen.vq_update(aux)
            row["ce_loss"] = stats["ce_loss"]

        # Shared: calibration KL anchor (skipped when lambda_kl == 0).
        kl_val = 0.0
        if use_kl and calib_items:
            calib_mb = rng.sample(calib_items, min(cfg.train.kl_batch, len(calib_items)))
            kl = kl_anchor_loss(bundle, calib_mb, cfg)
            (cfg.train.lambda_kl * kl).backward()
            kl_val = float(kl.item())

        gnorm = torch.nn.utils.clip_grad_norm_(params, cfg.train.grad_clip)
        opt.step()
        sched.step()

        row.update({
            "kl": kl_val, "plan_grad_norm": stats["plan_grad_norm"],
            "grad_norm": float(gnorm), "lr": sched.get_last_lr()[0],
        })
        if aux is not None and cfg.plan.plan_mode == "gumbel_codebook":
            row["peakedness"] = aux.get("peakedness", 0.0)
        if aux is not None and cfg.plan.plan_mode == "vq_codebook":
            row["vq_perplexity"] = aux.get("perplexity", 0.0)
            row["vq_usage"] = aux.get("usage", 0)
            row["commit_loss"] = stats.get("commit_loss", 0.0)

        if step % cfg.logging.log_every == 0:
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})
            if csv:
                csv.log(row)
            if wb:
                wb.log(row, step=step)

        if step % cfg.logging.jsonl_every == 0 and batch is not None:
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

        if judge is not None and step % cfg.logging.eval_every == 0:
            ev = run_eval(bundle, plan_gen, judge, eval_instr[: min(64, len(eval_instr))], cfg)
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
