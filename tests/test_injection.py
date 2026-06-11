"""Injection correctness.

Offline: with N=0 the assembled prefix embeddings equal embed(prompt_ids) exactly, and
the plan slots add exactly N rows when present; masks match real lengths.

Gated (LAP_RUN_MODEL_TESTS=1): the frozen executor produces identical logits for the
no-plan prefix-embeds path and the plain input-ids path (true vanilla equivalence).
"""
import os

import pytest
import torch

from src.injection import build_batch_prefix, build_one_prefix, prompt_ids
from ._fakes import FakeBundle


def test_n0_prefix_equals_embed_prompt_ids():
    b = FakeBundle(d=16)
    instr, sysp = "hello world", "sys"
    ids = prompt_ids(b.tokenizer, instr, sysp)
    ref = b.embed_tokens(torch.tensor(ids))
    emb, mask = build_batch_prefix(b, [instr], None, sysp, pad_side="left")
    assert emb.shape[0] == 1
    assert torch.allclose(emb[0], ref, atol=1e-6)
    assert int(mask.sum().item()) == len(ids)


def test_plan_adds_exactly_n_rows():
    b = FakeBundle(d=16)
    N = 5
    plan = torch.randn(1, N, 16)
    base_len = len(prompt_ids(b.tokenizer, "q", "sys"))
    prefix, plen = build_one_prefix(b, "q", plan[0], "sys")
    assert plen == base_len + N
    assert prefix.shape == (base_len + N, 16)


def test_batch_padding_mask_counts():
    b = FakeBundle(d=8)
    instrs = ["a", "a longer instruction here", "mid one"]
    emb, mask = build_batch_prefix(b, instrs, None, "sys", pad_side="left")
    lengths = [len(prompt_ids(b.tokenizer, x, "sys")) for x in instrs]
    assert emb.shape[0] == 3
    for i, L in enumerate(lengths):
        assert int(mask[i].sum().item()) == L


@pytest.mark.skipif(os.environ.get("LAP_RUN_MODEL_TESTS") != "1",
                    reason="set LAP_RUN_MODEL_TESTS=1 to run the real-model equivalence test")
def test_vanilla_equivalence_real_model():
    from omegaconf import OmegaConf
    from src.models import ModelBundle
    from src.utils import get_device

    cfg = OmegaConf.load("configs/default.yaml")
    cfg.model.base = cfg.smoke.model_base
    cfg.model.share_base = True
    device = get_device(cfg)
    bundle = ModelBundle(cfg, device)

    instr, sysp = "What is 2+2?", cfg.plan.system_prompt
    ids = prompt_ids(bundle.tokenizer, instr, sysp)
    ids_t = torch.tensor(ids, device=device)[None]
    emb, mask = build_batch_prefix(bundle, [instr], None, sysp, pad_side="left")

    with torch.no_grad():
        l_ids = bundle.executor_logits(input_ids=ids_t).float()
        l_emb = bundle.executor_logits(inputs_embeds=emb, attention_mask=mask).float()
    assert torch.allclose(l_ids, l_emb, atol=1e-3)
