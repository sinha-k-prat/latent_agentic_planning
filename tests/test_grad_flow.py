"""Gradient-flow / frozen-executor checks (offline).

The estimator must (a) push gradients through the frozen executor's forward into the
codebook + plan head, and (b) leave the executor's weights untouched.
"""
import torch

from src.injection import build_one_prefix
from src.losses import _response_logprob
from src.plan_head import PlanGenerator
from ._fakes import FakeBundle, make_cfg


def _plan_gen(bundle, cfg):
    pg = PlanGenerator(bundle, cfg)
    # Bypass the (real) backbone encoder with a fixed hidden state.
    pg.encode = lambda instrs: torch.zeros(len(instrs), bundle.hidden_size) + 0.1
    return pg


def test_grad_reaches_codebook_and_head():
    b = FakeBundle(d=16)
    cfg = make_cfg("gumbel_codebook", N=4, K=8)
    pg = _plan_gen(b, cfg)

    noise = pg.sample_gumbel(1)
    p, aux = pg.compute_plans(["q"], noise=noise, tau=0.5)
    prefix, plen = build_one_prefix(b, "q", p[0], "sys")
    y = torch.tensor([5, 6, 7])
    adv = torch.tensor(1.0)
    loss = -adv * _response_logprob(b, prefix, plen, y)
    loss.backward()

    assert pg.codebook.grad is not None and pg.codebook.grad.abs().sum() > 0
    assert pg.proj.weight.grad is not None and pg.proj.weight.grad.abs().sum() > 0


def test_executor_weights_frozen_after_step():
    b = FakeBundle(d=16)
    cfg = make_cfg("gumbel_codebook", N=4, K=8)
    pg = _plan_gen(b, cfg)
    opt = torch.optim.SGD(list(pg.parameters()), lr=0.1)

    before = b.exec_head.weight.detach().clone()
    noise = pg.sample_gumbel(1)
    p, _ = pg.compute_plans(["q"], noise=noise, tau=0.5)
    prefix, plen = build_one_prefix(b, "q", p[0], "sys")
    loss = -_response_logprob(b, prefix, plen, torch.tensor([3, 4]))
    opt.zero_grad()
    loss.backward()
    opt.step()

    # Frozen executor head must not have received a gradient or changed.
    assert b.exec_head.weight.grad is None
    assert torch.allclose(before, b.exec_head.weight, atol=0)
