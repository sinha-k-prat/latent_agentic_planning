"""Losses for the latent Planner–Executor.

The RL and CE objectives are the *same* computation — teacher-force a target token sequence
through the FROZEN executor and accumulate a weighted negative log-likelihood, letting the
gradient flow through the frozen executor's forward into the plan vectors / codebook / planner
(executor weights are `requires_grad=False`, so the backward is pure chain rule into the inputs).

They differ only in **which tokens** are fed and **the per-item weight**:

    RL  :  L = Σ_i  A_i      · (− mean_t log P_E( y_i_t      | x_i, p_i ))   # y sampled, A = advantage
    CE  :  L = Σ_i  1        · (− mean_t log P_E( y*_i_t     | x_i, p_i ))   # y* gold,    weight = 1

i.e. REINFORCE is "cross-entropy on self-sampled tokens, reweighted by advantage." Both route
through `_weighted_tf_loss_and_backward`. (`hard_text` RL is the one exception — it does REINFORCE
on the planner's *plan-token* log-probs, not on the response, so it keeps a separate path.)

Microbatching: plans are recomputed WITH gradients per microbatch (from the stored per-item
Gumbel noise when given) so `backward()` runs per microbatch with gradient accumulation, keeping
graphs independent and memory bounded.
"""
import torch
import torch.nn.functional as F

from .injection import build_one_prefix, prompt_ids
from .utils import microbatches


def _response_logprob(bundle, prefix_emb, prefix_len, y_ids):
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


def _weighted_tf_loss_and_backward(bundle, plan_gen, instructions, targets, weights, noises, cfg, tau):
    """Shared RL/CE core: minimize  Σ_i weight_i · (− mean_t log P_E(target_i_t | x_i, p_i)).

    instructions : list[str]                       length M
    targets      : list[LongTensor]  token ids to teacher-force (sampled y for RL, gold y* for CE)
    weights      : list[float]       advantage for RL, 1.0 for CE
    noises       : list[Tensor[N,K]] per-item Gumbel noise (RL, gumbel), or None to sample fresh (CE)

    Recomputes plans with grad per microbatch, backward()s per microbatch (grad accumulation).
    Returns {loss, plan_grad_norm, commit_loss, n, aux}. `plan_grad_norm` is the signal arriving
    at the plan vectors through the frozen executor — ~0 means the estimator is broken.
    """
    M = len(instructions)
    if M == 0:
        return {"loss": 0.0, "plan_grad_norm": 0.0, "commit_loss": 0.0, "n": 0, "aux": None}
    mode = cfg.plan.plan_mode
    sysp = cfg.plan.system_prompt
    w = torch.as_tensor(weights, dtype=torch.float32, device=bundle.device)

    total, total_commit, n, plan_gnorm, nbatches = 0.0, 0.0, 0, 0.0, 0
    last_aux = None
    for mb in microbatches(range(M), cfg.train.micro_rollouts):
        sub_instr = [instructions[j] for j in mb]
        if mode == "gumbel_codebook":
            noise = (torch.stack([noises[j] for j in mb]) if noises is not None
                     else plan_gen.sample_gumbel(len(mb)))
        else:  # vq_codebook: encoder is deterministic, no noise
            noise = None
        p, aux = plan_gen.compute_plans(sub_instr, noise=noise, tau=tau)
        p.retain_grad()
        last_aux = aux

        commit = aux.get("commit_loss")
        loss = torch.zeros((), device=bundle.device)
        for k, j in enumerate(mb):
            prefix, plen = build_one_prefix(bundle, instructions[j], p[k], sysp)
            loss = loss - w[j] * _response_logprob(bundle, prefix, plen, targets[j])
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
        "loss": total / max(1, n),
        "plan_grad_norm": plan_gnorm / max(1, nbatches),
        "commit_loss": total_commit / max(1, n),
        "n": n,
        "aux": last_aux,
    }


def rl_loss_and_backward(bundle, plan_gen, batch, cfg, tau, optimizer_scale=1.0):
    """RL objective over kept rollouts: feed the SAMPLED responses, weight by advantage.

    Delegates to the shared core (gumbel/vq). `hard_text` keeps its own REINFORCE path because
    it differentiates the planner's plan-token log-probs, not the response.
    """
    G = cfg.train.group_size
    mode = cfg.plan.plan_mode
    keep_flat = batch.keep.repeat_interleave(G)  # [B*G]
    idxs = [j for j in range(len(batch.flat_instructions)) if bool(keep_flat[j])]
    if not idxs:
        return {"rl_loss": 0.0, "n_rollouts": 0, "plan_grad_norm": 0.0, "commit_loss": 0.0}

    if mode == "hard_text":
        total, n = 0.0, 0
        for mb in microbatches(idxs, cfg.train.micro_rollouts):
            sub_instr = [batch.flat_instructions[j] for j in mb]
            adv = batch.advantages[mb].to(bundle.device)
            logp = plan_gen.plan_token_logprobs(sub_instr, [batch.plan_ids[j] for j in mb])
            loss = -(adv * logp).mean()
            loss.backward()
            total += float(loss.item()) * len(mb)
            n += len(mb)
        return {"rl_loss": total / max(1, n), "n_rollouts": n, "plan_grad_norm": 0.0, "commit_loss": 0.0}

    instructions = [batch.flat_instructions[j] for j in idxs]
    targets = [batch.responses[j] for j in idxs]
    weights = [float(batch.advantages[j]) for j in idxs]
    noises = [batch.noise[j] for j in idxs] if mode == "gumbel_codebook" else None
    out = _weighted_tf_loss_and_backward(bundle, plan_gen, instructions, targets, weights, noises, cfg, tau)
    return {"rl_loss": out["loss"], "n_rollouts": out["n"],
            "plan_grad_norm": out["plan_grad_norm"], "commit_loss": out["commit_loss"]}


def ce_loss_and_backward(bundle, plan_gen, instructions, targets, cfg, tau):
    """CE / distillation objective: teacher-force GOLD targets y* with weight 1 (no judge,
    no rollout, no advantage). Same frozen-executor backward + Gumbel codebook as RL.

    instructions: list[str], targets: list[str] (same length).
    """
    tok = bundle.tokenizer
    eos = tok.eos_token_id
    tgt_ids = []
    for t in targets:  # truncate; append EOS so the model learns to stop
        ids = tok(t, add_special_tokens=False).input_ids[: cfg.train.max_new_tokens]
        tgt_ids.append(torch.tensor((ids or [eos]) + [eos], dtype=torch.long))
    weights = [1.0] * len(instructions)
    out = _weighted_tf_loss_and_backward(bundle, plan_gen, instructions, tgt_ids, weights, None, cfg, tau)
    return {"ce_loss": out["loss"], "commit_loss": out["commit_loss"],
            "plan_grad_norm": out["plan_grad_norm"], "n": out["n"], "aux": out["aux"]}


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
