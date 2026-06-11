"""CPU gradient-flow demo for the latent Planner-Executor estimator.

This runs the project's REAL estimator code path -- src.plan_head.PlanGenerator,
src.injection.build_one_prefix, src.losses._response_logprob -- against a tiny
FROZEN toy executor (no GPU / no Qwen download required). It shows the core claim:
gradients backprop THROUGH the frozen executor's forward into the plan vectors and
the codebook. `|p.grad|` is the signal arriving at the plan vectors; if it were ~0
the estimator would be broken.

Toy task: reward = fraction of sampled response tokens equal to a fixed TARGET token.
The planner can raise reward by selecting codes whose plan vectors bias the frozen
executor toward TARGET. Run:  python examples/grad_demo.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tests._fakes import FakeBundle, make_cfg
from src.plan_head import PlanGenerator
from src.injection import build_one_prefix
from src.losses import _response_logprob

torch.manual_seed(0)

# V must cover the toy tokenizer's id range (~252); response tokens are sampled over it.
D, V, N, K, B, G, T, STEPS, TAU, TARGET = 24, 260, 6, 16, 4, 6, 8, 40, 0.7, 7

bundle = FakeBundle(d=D, vocab=V)          # frozen toy executor (exec_head requires_grad=False)
cfg = make_cfg("gumbel_codebook", N=N, K=K)
plan_gen = PlanGenerator(bundle, cfg)

# Dense toy "judge": reward = alignment of the response's mean embedding with a fixed
# target direction, mapped to [0,1]. Dense + per-rollout variance, so the group baseline
# is informative every step (unlike sparse exact-match). The planner can raise it by
# selecting plans that steer the frozen executor's sampling toward aligned tokens.
target_dir = F.normalize(torch.randn(D), dim=0)


def toy_reward(y_ids):
    e = bundle.embed_tokens(y_ids).mean(0)
    return ((F.cosine_similarity(e, target_dir, dim=0) + 1) / 2).item()

# Fixed, instruction-dependent encoding (stands in for the planner backbone).
H = torch.randn(B, D)
plan_gen.encode = lambda instrs: H[torch.tensor([int(x) for x in instrs])]

opt = torch.optim.Adam(list(plan_gen.parameters()), lr=0.05)


def sample_response(prefix_emb, length):
    """Autoregressively sample a response from the frozen executor given the plan prefix."""
    emb = prefix_emb.unsqueeze(0)
    ids = []
    for _ in range(length):
        logits = bundle.executor_logits(inputs_embeds=emb)[0, -1]
        tok = torch.multinomial(F.softmax(logits, -1), 1).item()
        ids.append(tok)
        emb = torch.cat([emb, bundle.embed_tokens(torch.tensor([tok])).unsqueeze(0)], dim=1)
    return torch.tensor(ids)


hist = {k: [] for k in ("step", "plan_grad", "codebook_grad", "head_grad", "reward", "loss")}

for step in range(1, STEPS + 1):
    flat = [str(i) for i in range(B) for _ in range(G)]
    noise = plan_gen.sample_gumbel(B * G)

    # 1) sample responses with the DETACHED plan (the stochastic node)
    with torch.no_grad():
        p_det, _ = plan_gen.compute_plans(flat, noise=noise, tau=TAU)
    responses, rewards = [], []
    for j in range(B * G):
        prefix, _ = build_one_prefix(bundle, flat[j], p_det[j], "sys")
        with torch.no_grad():
            y = sample_response(prefix, T)
        responses.append(y)
        rewards.append(toy_reward(y))

    # 2) group-normalized advantages
    R = torch.tensor(rewards).view(B, G)
    adv = ((R - R.mean(1, keepdim=True)) / (R.std(1, keepdim=True) + 1e-6)).view(-1)

    # 3) recompute plan WITH grad from the same noise; teacher-force y back through
    #    the frozen executor; backprop into plan vectors + codebook.
    p, _ = plan_gen.compute_plans(flat, noise=noise, tau=TAU)
    p.retain_grad()
    loss = torch.zeros(())
    for j in range(B * G):
        prefix, plen = build_one_prefix(bundle, flat[j], p[j], "sys")
        loss = loss - adv[j] * _response_logprob(bundle, prefix, plen, responses[j])
    loss = loss / (B * G)

    opt.zero_grad()
    loss.backward()
    plan_gn = p.grad.norm().item()
    cb_gn = plan_gen.codebook.grad.norm().item()
    head_gn = plan_gen.proj.weight.grad.norm().item()
    opt.step()

    for k, v in zip(hist, (step, plan_gn, cb_gn, head_gn, R.mean().item(), loss.item())):
        hist[k].append(v)
    print(f"step {step:2d}  reward {R.mean():.3f}  loss {loss.item():+.4f}  "
          f"|p.grad| {plan_gn:.4f}  |codebook.grad| {cb_gn:.4f}")

# ---- plot ----
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(hist["step"], hist["plan_grad"], label="|p.grad|  (signal into plan vectors)")
ax[0].plot(hist["step"], hist["codebook_grad"], label="|codebook.grad|")
ax[0].plot(hist["step"], hist["head_grad"], label="|plan_head.grad|")
ax[0].set_xlabel("step"); ax[0].set_ylabel("grad L2 norm")
ax[0].set_title("Gradient signal through the FROZEN executor")
ax[0].legend(); ax[0].grid(alpha=.3)

ax[1].plot(hist["step"], hist["reward"], color="green", marker=".")
ax[1].set_xlabel("step"); ax[1].set_ylabel("mean reward")
ax[1].set_title("Mean reward (toy: response aligned to target direction)")
ax[1].grid(alpha=.3)

plt.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "grad_demo.png")
plt.savefig(out, dpi=120)
print("saved", out)
