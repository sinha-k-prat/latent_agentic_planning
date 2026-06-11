"""Assemble the executor's input as prefix `inputs_embeds`.

Layout (matches the spec):
    [ embeds(system block) , p_1..p_N , embeds(user turn + assistant start) ]

The plan vectors are inserted right after the chat-template system tokens. With
N == 0 (no plan), the assembled embeddings equal `embed(full prompt)` exactly, so
the executor reproduces vanilla generation (unit-tested in tests/test_injection.py).
"""
import torch


def format_segments(tokenizer, instruction, system_prompt):
    """Return (system_ids, rest_ids) for the Qwen chat template, split at the system block."""
    sys_text = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
    rest_text = (
        f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
    )
    sys_ids = tokenizer(sys_text, add_special_tokens=False).input_ids
    rest_ids = tokenizer(rest_text, add_special_tokens=False).input_ids
    return sys_ids, rest_ids


def prompt_ids(tokenizer, instruction, system_prompt):
    """Full prompt token ids with NO plan (used for the N=0 equivalence test and KL anchor)."""
    sys_ids, rest_ids = format_segments(tokenizer, instruction, system_prompt)
    return sys_ids + rest_ids


def build_one_prefix(bundle, instruction, plan_vectors, system_prompt):
    """Embeddings for a single example: [sys, plan?, rest].

    plan_vectors: [N, d] (may carry grad) or None.
    Returns (prefix_emb [L, d], prefix_len).
    """
    sys_ids, rest_ids = format_segments(bundle.tokenizer, instruction, system_prompt)
    sys_emb = bundle.embed_tokens(torch.tensor(sys_ids, dtype=torch.long))
    rest_emb = bundle.embed_tokens(torch.tensor(rest_ids, dtype=torch.long))
    parts = [sys_emb]
    if plan_vectors is not None and plan_vectors.shape[0] > 0:
        parts.append(plan_vectors.to(sys_emb.dtype))
    parts.append(rest_emb)
    prefix = torch.cat(parts, dim=0)
    return prefix, prefix.shape[0]


def build_batch_prefix(bundle, instructions, plan_vectors, system_prompt, pad_side="left"):
    """Left/right padded batch of prefix embeddings.

    instructions: list[str] of length M.
    plan_vectors: [M, N, d] (grad ok) or None.
    Returns (inputs_embeds [M, Lmax, d], attention_mask [M, Lmax]).
    """
    seqs = []
    for i, instr in enumerate(instructions):
        pv = None if plan_vectors is None else plan_vectors[i]
        prefix, _ = build_one_prefix(bundle, instr, pv, system_prompt)
        seqs.append(prefix)

    d = seqs[0].shape[1]
    lmax = max(s.shape[0] for s in seqs)
    padded, masks = [], []
    for s in seqs:
        li = s.shape[0]
        pad = torch.zeros(lmax - li, d, dtype=s.dtype, device=s.device)
        m = torch.zeros(lmax, dtype=torch.long, device=s.device)
        if pad_side == "left":
            padded.append(torch.cat([pad, s], dim=0))
            m[lmax - li:] = 1
        else:
            padded.append(torch.cat([s, pad], dim=0))
            m[:li] = 1
        masks.append(m)
    return torch.stack(padded, 0), torch.stack(masks, 0)
