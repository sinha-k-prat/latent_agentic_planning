"""Batched rollout: sample plans -> executor generates -> judge scores -> group advantages.

The plan is sampled once (with fixed Gumbel noise for gumbel mode). Generation uses the
DETACHED plan under no_grad; the noise is returned so the loss pass can reconstruct the
exact same plan WITH gradients (the stochastic-computation-graph estimator).
"""
from dataclasses import dataclass, field
from typing import List

import torch

from .injection import build_batch_prefix
from .utils import microbatches


@dataclass
class RolloutBatch:
    instructions: List[str]                 # B unique instructions
    flat_instructions: List[str]            # B*G (each repeated G times)
    responses: List[torch.Tensor]           # B*G response token-id tensors
    response_texts: List[str]               # B*G decoded responses
    rewards: torch.Tensor                   # [B, G]
    advantages: torch.Tensor                # [B*G]
    noise: torch.Tensor                     # gumbel noise [B*G, N, K] or None
    aux: dict                               # plan aux from the (detached) sample
    keep: torch.Tensor                      # [B] bool, False for zero-variance groups
    # hard_text only:
    plan_ids: list = field(default=None)


def _trim_response(row, eos_id, pad_id):
    ids = row.tolist()
    if eos_id in ids:
        ids = ids[: ids.index(eos_id)]
    ids = [t for t in ids if t != pad_id]
    if not ids:
        ids = [eos_id]
    return torch.tensor(ids, dtype=torch.long)


@torch.no_grad()
def run_rollout(bundle, plan_gen, judge, instructions, cfg, tau):
    B, G = len(instructions), cfg.train.group_size
    flat = [instructions[i] for i in range(B) for _ in range(G)]
    sysp = cfg.plan.system_prompt
    mode = cfg.plan.plan_mode

    noise, aux, plan_ids = None, {}, None
    if mode == "gumbel_codebook":
        noise = plan_gen.sample_gumbel(B * G)
        p, aux = plan_gen.compute_plans(flat, noise=noise, tau=tau)
        plan_vectors = p.detach()
    elif mode == "vq_codebook":
        p, aux = plan_gen.compute_plans(flat, tau=tau)
        plan_vectors = p.detach()
    else:  # hard_text
        plan_ids = plan_gen.generate_text_plan(flat, cfg.plan.hard_text.max_plan_tokens)
        plan_vectors = plan_gen.embed_text_plan(plan_ids)

    # Generate responses (frozen executor), microbatched over rollouts.
    eos, pad = bundle.tokenizer.eos_token_id, bundle.tokenizer.pad_token_id
    responses = []
    for mb in microbatches(range(B * G), cfg.train.gen_microbatch):
        sub_instr = [flat[j] for j in mb]
        sub_plan = plan_vectors[list(mb)]
        emb, mask = build_batch_prefix(bundle, sub_instr, sub_plan, sysp, pad_side="left")
        gen = bundle.executor_generate(
            emb, mask, cfg.train.max_new_tokens, cfg.train.exec_temperature, cfg.train.exec_top_p,
        )
        for row in gen:
            responses.append(_trim_response(row, eos, pad))

    texts = [bundle.tokenizer.decode(r, skip_special_tokens=True) for r in responses]
    rewards = judge.score_batch(list(zip(flat, texts)))
    R = torch.tensor(rewards, dtype=torch.float32).view(B, G)

    mean = R.mean(1, keepdim=True)
    std = R.std(1, keepdim=True)
    adv = (R - mean) / (std + cfg.train.adv_eps)
    keep = torch.ones(B, dtype=torch.bool)
    if cfg.train.zero_var_skip:
        keep = std.squeeze(1) > cfg.train.adv_eps

    return RolloutBatch(
        instructions=instructions, flat_instructions=flat, responses=responses,
        response_texts=texts, rewards=R, advantages=adv.view(-1), noise=noise,
        aux=aux, keep=keep, plan_ids=plan_ids,
    )
