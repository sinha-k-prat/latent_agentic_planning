"""Toy check that the advantage sign moves code logits the right way.

A self-contained mini version of the gradient estimator: logits over K codes ->
alpha = softmax(logits/tau) -> plan p = alpha @ codebook -> a frozen linear 'executor'
maps p to vocab logits -> log-prob of a fixed target token. The SCG loss is
L = -A * logp_target. With A>0 the update must INCREASE logp_target; with A<0 decrease;
and because the gradient is linear in A, the two logit updates are exact negatives.
"""
import torch
import torch.nn.functional as F


def _step(adv, K=6, d=8, V=5, tau=0.7, seed=0):
    g = torch.Generator().manual_seed(seed)
    logits = torch.zeros(K, requires_grad=True)
    codebook = torch.randn(K, d, generator=g)
    W = torch.randn(V, d, generator=g)  # frozen "executor"
    target = 2

    alpha = F.softmax(logits / tau, dim=-1)
    p = alpha @ codebook                       # [d]
    exec_logits = W @ p                        # [V]
    logp = F.log_softmax(exec_logits, dim=-1)[target]
    loss = -torch.tensor(float(adv)) * logp
    loss.backward()
    # one gradient-descent step on the logits
    new_logits = (logits - 0.5 * logits.grad).detach()
    return logits.detach(), new_logits, float(logp.detach())


def test_positive_advantage_increases_target_logprob():
    logits0, logits_pos, logp0 = _step(adv=+1.0)
    # recompute logp under updated logits
    K, d, V, tau, target = 6, 8, 5, 0.7, 2
    g = torch.Generator().manual_seed(0)
    codebook = torch.randn(K, d, generator=g)
    W = torch.randn(V, d, generator=g)
    alpha = torch.softmax(logits_pos / tau, dim=-1)
    logp_new = torch.log_softmax(W @ (alpha @ codebook), dim=-1)[target].item()
    assert logp_new > logp0


def test_sign_symmetry():
    _, up, _ = _step(adv=+1.0)
    _, down, _ = _step(adv=-1.0)
    base = torch.zeros(6)
    # gradient is linear in advantage -> updates are exact negatives of each other
    assert torch.allclose(up - base, -(down - base), atol=1e-6)
