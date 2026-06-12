"""Losses: grad-through-executor RL loss, calibration KL anchor, VQ commitment.

RL loss (stochastic computation graph estimator):
    L_RL = - A * (1/|y|) * sum_t log P_E(y_t | x, p, y_<t)
where y was SAMPLED from the frozen executor and is now teacher-forced back through it
WITH gradients enabled. The executor's weights are frozen (requires_grad=False), so the
backward pass flows only into the plan vectors p (and thence the planner).

Microbatching: to keep graphs independent and memory bounded, the plan is RECOMPUTED for
each microbatch from the stored (instructions, gumbel-noise) so we can call backward() per
microbatch and accumulate gradients.
"""
import torch
import torch.nn.functional as F

from .injection import build_one_prefix, prompt_ids
from .utils import microbatches


def _response_logprob(bundle, prefix_emb, prefix_len, y_ids, sysp=None):
    """Mean log P_E(y_t | prefix, y_<t), teacher-forced. Grad flows via prefix_emb."""
    y_ids = y_ids.to(bundle.device)
    y_emb = bundle.embed_tokens(y_ids)                         # [T, d] (no grad)
    full = torch.cat([prefix_emb, y_emb], dim=0).unsqueeze(0)  # [1, Lp+T, d]
    mask = torch.ones(1, full.shape[1], dtype=torch.long, device=bundle.device)
    logits = bundle.executor_logits(inputs_embeds=full, attention_mask=mask)[0]
    T = y_ids.shape[0]
    # logits at position (Lp-1 .. Lp-1+T-1) predict y_0 .. y_{T-1}
    sel = logits[prefix_len - 1: prefix_len - 1 + T].float()
    logp = F.log_softmax(sel, dim=-1)
    tok_lp = logp[torch.arange(T, device=bundle.device), y_ids]
    return tok_lp.mean()


def rl_loss_and_backward(bundle, plan_gen, batch, cfg, tau, optimizer_scale):
    """Compute L_RL over kept rollouts, calling backward() per microbatch (grad accumulation).

    Returns dict of scalars for logging, including the grad-norm arriving at the plan vectors
    (`plan_grad_norm`) — the signal from the executor. If it is ~0 the estimator is broken.
    """
    B, G = len(batch.instructions), cfg.train.group_size
    mode = cfg.plan.plan_mode
    sysp = cfg.plan.system_prompt
    keep_flat = batch.keep.repeat_interleave(G)  # [B*G]
    idxs = [j for j in range(B * G) if bool(keep_flat[j])]
    if not idxs:
        return {"rl_loss": 0.0, "n_rollouts": 0, "plan_grad_norm": 0.0, "commit_loss": 0.0}

    total_rl, total_commit, n, plan_gnorm = 0.0, 0.0, 0, 0.0

    for mb in microbatches(idxs, cfg.train.micro_rollouts):
        sub_instr = [batch.flat_instructions[j] for j in mb]
        adv = batch.advantages[mb].to(bundle.device)

        if mode == "hard_text":
            # REINFORCE on planner plan-token log-probs (baseline = group advantage).
            plan_ids = [batch.plan_ids[j] for j in mb]
            logp = plan_gen.plan_token_logprobs(sub_instr, plan_ids)
            loss = -(adv * logp).mean()
            loss.backward()
            total_rl += float(loss.item()) * len(mb)
            n += len(mb)
            continue

        # Recompute the exact plan (grad on) from stored noise.
        if mode == "gumbel_codebook":
            noise = batch.noise[mb]
            p, aux = plan_gen.compute_plans(sub_instr, noise=noise, tau=tau)
        else:  # vq_codebook
            p, aux = plan_gen.compute_plans(sub_instr, tau=tau)
        p.retain_grad()

        commit = aux.get("commit_loss")
        loss = torch.zeros((), device=bundle.device)
        for k, j in enumerate(mb):
            prefix, plen = build_one_prefix(bundle, batch.flat_instructions[j], p[k], sysp)
            mean_lp = _response_logprob(bundle, prefix, plen, batch.responses[j])
            loss = loss + (-adv[k] * mean_lp)
        loss = loss / len(mb)
        if commit is not None:
            loss = loss + commit
            total_commit += float(commit.item()) * len(mb)

        loss.backward()
        if p.grad is not None:
            plan_gnorm += float(p.grad.detach().float().norm().item())
        total_rl += float(loss.item()) * len(mb)
        n += len(mb)

    return {
        "rl_loss": total_rl / max(1, n),
        "commit_loss": total_commit / max(1, n),
        "n_rollouts": n,
        "plan_grad_norm": plan_gnorm / max(1, len(list(microbatches(idxs, cfg.train.micro_rollouts)))),
    }


def kl_anchor_loss(bundle, calib_items, cfg):
    """token-level KL(P_E || P_P) on plain (no-plan) calibration prompts, fp32.

    calib_items: list of (instruction, y_ids). Grad flows into the planner (LoRA); the
    executor distribution is detached. Returns a scalar loss tensor (with grad).
    """
    sysp = cfg.plan.system_prompt
    tok = bundle.tokenizer
    total = torch.zeros((), device=bundle.device)
    n = 0
    for instr, y_ids in calib_items:
        prompt = prompt_ids(tok, instr, sysp)
        full = torch.tensor(prompt + list(y_ids), device=bundle.device)[None]
        lp_logits = bundle.planner_forward(input_ids=full).logits[0]
        with torch.no_grad():
            le_logits = bundle.executor_logits(input_ids=full)[0]
        Lp, T = len(prompt), len(y_ids)
        sl = slice(Lp - 1, Lp - 1 + T)
        logP = F.log_softmax(lp_logits[sl].float(), dim=-1)
        logE = F.log_softmax(le_logits[sl].float(), dim=-1)
        kl = (logE.exp() * (logE - logP)).sum(-1).mean()  # KL(E || P)
        total = total + kl
        n += 1
    return total / max(1, n)


def ce_loss_and_backward(bundle, plan_gen, instructions, targets, cfg, tau):
    """Cross-entropy (distillation) loss: teacher-force the GOLD target y* through the
    frozen executor and maximize log P_E(y* | x, p). Same grad-through-frozen-executor
    path as the RL loss, but with gold tokens and weight 1 (no sampling, no judge, no
    advantage). Backprops into the plan vectors / codebook / planner; executor frozen.

    instructions: list[str], targets: list[str] (same length). Returns logging scalars.
    """
    tok = bundle.tokenizer
    eos = tok.eos_token_id
    mode = cfg.plan.plan_mode
    sysp = cfg.plan.system_prompt

    # Pre-tokenize targets (truncate; append EOS so the model learns to stop).
    tgt_ids = []
    for t in targets:
        ids = tok(t, add_special_tokens=False).input_ids[: cfg.train.max_new_tokens]
        tgt_ids.append(torch.tensor((ids or [eos]) + [eos], dtype=torch.long))

    total, n, plan_gnorm, nbatches, total_commit = 0.0, 0, 0.0, 0, 0.0
    last_aux = None
    for mb in microbatches(range(len(instructions)), cfg.train.micro_rollouts):
        sub_instr = [instructions[j] for j in mb]
        noise = plan_gen.sample_gumbel(len(mb)) if mode == "gumbel_codebook" else None
        p, aux = plan_gen.compute_plans(sub_instr, noise=noise, tau=tau)
        p.retain_grad()
        last_aux = aux
        commit = aux.get("commit_loss")
        loss = torch.zeros((), device=bundle.device)
        for k, j in enumerate(mb):
            prefix, plen = build_one_prefix(bundle, instructions[j], p[k], sysp)
            loss = loss - _response_logprob(bundle, prefix, plen, tgt_ids[j])  # NLL of gold y*
        loss = loss / len(mb)
        if commit is not None:
            loss = loss + commit
            total_commit += float(commit.item()) * len(mb)
        loss.backward()
        if p.grad is not None:
            plan_gnorm += float(p.grad.detach().float().norm().item())
        total += float(loss.item()) * len(mb)
        n += len(mb)
        nbatches += 1

    return {
        "ce_loss": total / max(1, n),
        "commit_loss": total_commit / max(1, n),
        "plan_grad_norm": plan_gnorm / max(1, nbatches),
        "n": n,
        "aux": last_aux,
    }
