"""Three-way evaluation harness.

  (1) executor alone (no plan prefix)                      -- baseline
  (2) executor + fixed strategy prompt ("Think step ...")  -- prompt-engineering baseline
  (3) planner + executor, HARDENED plans (gumbel: argmax over codes; vq: nearest)

Reports mean judge score and head-to-head win rate vs baseline (1). For gumbel we also
report the SOFT plan score to quantify the hardening gap. Note (README): same-judge eval
favours reward hacking — report a second judge when available.
"""
import torch

from .injection import build_batch_prefix
from .utils import microbatches


def _decode_trim(bundle, gen_rows):
    eos, pad = bundle.tokenizer.eos_token_id, bundle.tokenizer.pad_token_id
    out = []
    for row in gen_rows:
        ids = row.tolist()
        if eos in ids:
            ids = ids[: ids.index(eos)]
        ids = [t for t in ids if t != pad]
        out.append(bundle.tokenizer.decode(ids, skip_special_tokens=True))
    return out


@torch.no_grad()
def _generate(bundle, plan_gen, instructions, cfg, mode, hard=True):
    """mode in {'none','strategy','plan_hard','plan_soft'}. Returns (texts, all_code_indices)."""
    sysp = cfg.plan.system_prompt
    texts, indices = [], []
    for mb in microbatches(range(len(instructions)), cfg.train.gen_microbatch):
        sub = [instructions[j] for j in mb]
        plan_vectors = None
        if mode == "strategy":
            sub = [f"Think step by step, then answer.\n\n{x}" for x in sub]
        elif mode in ("plan_hard", "plan_soft"):
            p, aux = plan_gen.compute_plans(sub, tau=cfg.plan.gumbel.tau_end,
                                            hard=(mode == "plan_hard"))
            plan_vectors = p
            indices.extend(aux["indices"].tolist())
        emb, mask = build_batch_prefix(bundle, sub, plan_vectors, sysp, pad_side="left")
        gen = bundle.executor_generate(
            emb, mask, cfg.train.max_new_tokens, cfg.train.exec_temperature, cfg.train.exec_top_p,
        )
        texts.extend(_decode_trim(bundle, gen))
    return texts, indices


def _winrate(scores, base):
    w = sum((s > b) + 0.5 * (s == b) for s, b in zip(scores, base))
    return w / max(1, len(scores))


@torch.no_grad()
def run_eval(bundle, plan_gen, judge, instructions, cfg):
    mode = cfg.plan.plan_mode

    def score(texts):
        return judge.score_batch(list(zip(instructions, texts)))

    results = {}

    base_texts, _ = _generate(bundle, plan_gen, instructions, cfg, "none")
    base_scores = score(base_texts)
    results["executor_only"] = sum(base_scores) / len(base_scores)

    strat_texts, _ = _generate(bundle, plan_gen, instructions, cfg, "strategy")
    strat_scores = score(strat_texts)
    results["strategy_prompt"] = sum(strat_scores) / len(strat_scores)
    results["strategy_winrate_vs_base"] = _winrate(strat_scores, base_scores)

    if mode in ("gumbel_codebook", "vq_codebook"):
        hard_texts, _ = _generate(bundle, plan_gen, instructions, cfg, "plan_hard")
        hard_scores = score(hard_texts)
        results["plan_hard"] = sum(hard_scores) / len(hard_scores)
        results["plan_hard_winrate_vs_base"] = _winrate(hard_scores, base_scores)
        if mode == "gumbel_codebook":
            soft_texts, _ = _generate(bundle, plan_gen, instructions, cfg, "plan_soft", hard=False)
            soft_scores = score(soft_texts)
            results["plan_soft"] = sum(soft_scores) / len(soft_scores)
            results["hardening_gap"] = results["plan_soft"] - results["plan_hard"]
    return results


def _markdown(res, cfg, n):
    rows = [
        ("Qwen executor — original, no plan (baseline)", res.get("executor_only"), None),
        ("Qwen + fixed strategy prompt", res.get("strategy_prompt"),
         res.get("strategy_winrate_vs_base")),
        ("Qwen planner + latent plan (hardened codes)", res.get("plan_hard"),
         res.get("plan_hard_winrate_vs_base")),
    ]
    if "plan_soft" in res:
        rows.append(("Qwen planner + latent plan (soft)", res.get("plan_soft"), None))
    out = [
        f"# Comparison — plan_mode={cfg.plan.plan_mode}, n={n}, judge={cfg.judge.kind}",
        "",
        "| Condition | Mean judge score | Win-rate vs executor |",
        "|---|---|---|",
    ]
    for name, score, wr in rows:
        if score is None:
            continue
        out.append(f"| {name} | {score:.4f} | {('%.3f' % wr) if wr is not None else '—'} |")
    if "hardening_gap" in res:
        out += ["", f"Soft→hard hardening gap: {res['hardening_gap']:.4f}"]
    out += ["", "_Caveat: same-judge eval favours reward hacking; report a second judge "
            "(judge.kind=api) for a sanity cross-check._"]
    return "\n".join(out) + "\n"


def main():
    import argparse
    import os

    from .data import build_or_load_splits, split_instructions
    from .inference import load_for_inference
    from .judge import build_judge
    from .utils import get_device, load_config

    ap = argparse.ArgumentParser(description="Three-way comparison from a trained checkpoint.")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--ckpt", default="runs/default/checkpoints/best")
    ap.add_argument("--n", type=int, default=200, help="held-out eval instructions")
    ap.add_argument("--out", default="results/comparison.md")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    cfg = load_config(args.config, overrides=args.overrides)
    device = get_device(cfg)
    bundle, plan_gen = load_for_inference(cfg, args.ckpt, device)
    judge = build_judge(cfg, device)
    instructions = split_instructions(build_or_load_splits(cfg, bundle.tokenizer)["eval"])[: args.n]

    res = run_eval(bundle, plan_gen, judge, instructions, cfg)
    md = _markdown(res, cfg, len(instructions))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write(md)
    print(md)
    print("saved", args.out)


if __name__ == "__main__":
    main()
