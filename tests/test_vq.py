"""VQ mode: straight-through values, gradient to the encoder, EMA updates, dead-code reinit."""
import torch

from src.plan_head import PlanGenerator
from ._fakes import FakeBundle, make_cfg


def _vq(bundle):
    cfg = make_cfg("vq_codebook", N=3, K=6)
    pg = PlanGenerator(bundle, cfg)
    pg.encode = lambda instrs: torch.randn(len(instrs), bundle.hidden_size)
    return pg


def test_straight_through_forward_equals_codes_grad_to_encoder():
    b = FakeBundle(d=8)
    pg = _vq(b)
    p, aux = pg.compute_plans(["a", "b"], tau=1.0)
    # forward value of p equals the quantized code e (straight-through)
    assert torch.allclose(p.float(), aux["e"].float(), atol=1e-6)
    # gradient flows to the encoder projection through the ST estimator
    p.sum().backward()
    assert pg.proj.weight.grad is not None and pg.proj.weight.grad.abs().sum() > 0


def test_ema_update_changes_codebook():
    b = FakeBundle(d=8)
    pg = _vq(b)
    p, aux = pg.compute_plans(["a", "b"], tau=1.0)
    before = pg.codebook.clone()
    pg.vq_update(aux)
    assert not torch.allclose(before, pg.codebook)


def test_dead_code_reinit():
    b = FakeBundle(d=8)
    pg = _vq(b)
    # Force all assignments onto code 0 so codes 1..K-1 are unused (dead, threshold=1).
    z = torch.randn(2, 3, 8)
    aux = {"z": z, "indices": torch.zeros(2, 3, dtype=torch.long)}
    dead_before = pg.codebook[1].clone()
    pg.vq_update(aux)
    assert pg.steps_since_used[1].item() == 0          # reseeded
    assert not torch.allclose(dead_before, pg.codebook[1])  # code 1 was reinitialised
