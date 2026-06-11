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
