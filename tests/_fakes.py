"""Tiny offline fakes so the estimator / VQ / injection logic can be tested on CPU
without downloading any model. They implement only the slice of the ModelBundle API
that the unit under test touches."""
from types import SimpleNamespace

import torch
import torch.nn as nn
from omegaconf import OmegaConf


class FakeTok:
    pad_token_id = 0
    eos_token_id = 1
    pad_token = "<pad>"
    eos_token = "<eos>"
    padding_side = "left"

    def __call__(self, text, return_tensors=None, padding=False, add_special_tokens=False):
        def enc(t):
            ids = [2 + (ord(c) % 250) for c in t][:64]
            return ids or [2]
        if isinstance(text, str):
            return SimpleNamespace(input_ids=enc(text))
        seqs = [enc(t) for t in text]
        lmax = max(len(s) for s in seqs)
        ids, mask = [], []
        for s in seqs:  # left pad
            pad = [self.pad_token_id] * (lmax - len(s))
            ids.append(pad + s)
            mask.append([0] * len(pad) + [1] * len(s))
        return SimpleNamespace(input_ids=torch.tensor(ids), attention_mask=torch.tensor(mask))


class FakeBundle:
    """Embedding + a FROZEN tiny 'executor head' (Linear) standing in for the frozen LM."""

    def __init__(self, d=16, vocab=260):
        self.hidden_size = d
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.tokenizer = FakeTok()
        self.emb = nn.Embedding(vocab, d)
        self.exec_head = nn.Linear(d, vocab)
        for p in self.exec_head.parameters():
            p.requires_grad_(False)  # frozen executor

    def embed_tokens(self, ids):
        return self.emb(ids.to(self.device)).to(self.dtype)

    def executor_logits(self, inputs_embeds=None, input_ids=None, attention_mask=None):
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        # Minimal CAUSAL mixing (cumulative mean) so each position depends on all
        # preceding ones — including the plan prefix. Without this, a position-wise
        # head would make the plan->response gradient identically zero (the real LM
        # mixes via attention).
        csum = inputs_embeds.cumsum(dim=-2)
        counts = torch.arange(
            1, inputs_embeds.shape[-2] + 1, device=inputs_embeds.device, dtype=inputs_embeds.dtype
        ).unsqueeze(-1)
        pooled = csum / counts
        return self.exec_head(pooled)


def make_cfg(plan_mode="gumbel_codebook", N=4, K=8, d_code=None):
    return OmegaConf.create({
        "plan": {
            "plan_mode": plan_mode, "N": N, "K": K, "d_code": d_code,
            "system_prompt": "sys",
            "gumbel": {"tau_start": 1.0, "tau_end": 0.1, "tau_anneal_steps": 10},
            "vq": {"beta": 0.25, "ema": True, "ema_decay": 0.9,
                   "dead_code_steps": 1, "usage_log_every": 5},
            "hard_text": {"max_plan_tokens": 8},
        },
        "train": {"group_size": 2, "micro_rollouts": 2, "adv_eps": 1e-6},
    })
