# latent_agentic_planning

A latent **Planner–Executor** RL pipeline on a small Qwen model. A trainable **Planner**
emits a short sequence of latent *plan vectors* drawn from a learned codebook (not natural
language). A **frozen Executor** (the same base model, LoRA disabled) consumes the plan
vectors as prefix `inputs_embeds` and generates the response. A **Judge** scores the
response; the reward trains the planner via a stochastic-computation-graph estimator that
backprops **through the frozen executor's forward** into the plan vectors.

```
instruction x ──▶ Planner P (LoRA + plan head + codebook) ──▶ plan vectors p_1..p_N
                                                                     │ (prefix inputs_embeds)
                                                                     ▼
                                       Executor E (frozen) ──▶ response y ~ P_E(·|x,p)
                                                                     │
                                                      Judge J ──▶ reward R ∈ [0,1]
```

## The gradient estimator (core of the project)

We do **not** use plain REINFORCE on planner tokens. Per rollout:

1. Planner produces `p` (differentiable in the planner params).
2. Executor **samples** `y ~ P_E(·|x,p)` — the stochastic node (no-grad generation).
3. Judge scores `y` → advantage `A = (R − group_mean)/(group_std + ε)`.
4. `L_RL = − A · (1/|y|) Σ_t log P_E(y_t | x, p, y_<t)`, computed by **teacher-forcing the
   sampled `y` back through the frozen executor with gradients enabled**.
5. Gradients flow through the frozen executor's forward (weights have `requires_grad=False`,
   so the backward is pure chain rule into the inputs) → into `p` → into the planner.

The executor optimizer does not exist; `tests/test_grad_flow.py` asserts no executor weight
changes, and training asserts the frozen-weight signature is constant every 50 steps.

To make the estimator exact under microbatching, the plan is **recomputed with gradients**
in the loss pass from the *same* Gumbel noise used at generation time (`src/rollout.py` →
`src/losses.py`).

## Plan modes (`plan.plan_mode`)

- **`gumbel_codebook`** (default): codebook `C∈R^{K×d}` (trainable); plan head emits N×K
  logits; `p_i = Σ_k α_ik C_k`, `α_i = gumbel_softmax(logits_i, τ)`. `τ` anneals 1.0→0.1
  (exponential). fp32 at low τ. Logs τ and mean max-α (peakedness).
- **`vq_codebook`**: encoder emits `z_i`; nearest-code `e_i`; straight-through
  `p_i = z_i + sg(e_i − z_i)`; commitment loss `β‖z_i − sg(e_i)‖²` (β=0.25); EMA codebook
  (decay 0.99); codebook perplexity/usage logging and **dead-code reinit** (collapse made
  visible).
- **`hard_text`** (baseline): planner generates a NL plan; REINFORCE on plan-token log-probs
  with the group baseline.

## Calibration KL anchor (anti-drift)

On ~1k plain calibration instructions, responses are sampled once from the frozen executor
at startup. Each step a calibration minibatch is teacher-forced through both models and
`L_KL = KL(P_E ‖ P_P)` (token-level, fp32) is added: `L = L_RL + λ_kl·L_KL (+ β·L_commit)`,
`λ_kl = 0.05`.

## Install

```bash
pip install -r requirements.txt   # needs a CUDA GPU (24–48 GB) for the real run
```

## Run

```bash
# Smoke test: 0.5B base, stub judge, 32 instructions, 10 steps — runs end-to-end in minutes.
python -m src.train --smoke-test

# Full run (defaults: Qwen2.5-1.5B, gumbel, N=16, K=512, G=8, ~1000 steps).
python -m src.train

# Override anything via dotlist:
python -m src.train plan.plan_mode=vq_codebook plan.K=256 train.steps=500
python -m src.train judge.kind=api judge.api.model=claude-opus-4-8   # Anthropic judge

# Evaluate a checkpoint is done automatically every eval_every steps; standalone eval:
python -m src.inference --ckpt runs/default/checkpoints/best --question "Explain backprop." --judge
python -m src.inference --ckpt runs/default/checkpoints/best --question "Explain backprop." --no-plan
```

## Project layout

```
configs/default.yaml   model names, plan_mode, N, K, τ schedule, G, lrs, λ_kl, β, judge, data
src/models.py          backbone + LoRA planner + frozen executor (adapter sharing)
src/plan_head.py       plan head + codebook; gumbel / vq / hard_text behind one interface
src/injection.py       prefix inputs_embeds assembly, masks (N=0 == vanilla)
src/judge.py           judge interface; local 4-bit / stub / Anthropic API; caching + parsing
src/data.py            alpaca-cleaned splits (train/eval/calib, disjoint, persisted)
src/rollout.py         batched plan -> generate -> judge -> group advantages
src/losses.py          RL loss (teacher-forced, grad-through-executor), KL anchor, commitment
src/train.py           main loop, logging (wandb/CSV/JSONL), checkpointing
src/evaluate.py        three-way eval + win rates
src/inference.py       CLI
tests/                 injection, grad-flow, estimator, judge-parsing, vq
```

## Logging

Per step (CSV at `runs/<name>/metrics.csv`, optional wandb): mean/p10/p90 reward, advantage
magnitude, KL, `τ` + peakedness (gumbel), codebook perplexity/usage + commitment (vq),
**`plan_grad_norm`** (the gradient arriving at the plan vectors — if ~0 the estimator is
broken), grad norm, kept groups, judge-unparsable counter. Every `jsonl_every` steps a JSONL
dump of `(x, selected code indices, y, R)`.

## Build order (matches the spec)

1. Models + injection + one end-to-end rollout. 2. Judge + caching + parsing. 3. Gumbel RL
step with grad-flow tests. 4. Calibration KL anchor. 5. Full training + eval. 6. VQ + its
instrumentation. 7. hard_text baseline.

## Memory notes

- Single GPU, no distributed, no vLLM (`transformers.generate` with batching). 24–48 GB.
- Executor uses gradient checkpointing (we backprop through its forward). The with-grad
  teacher-forcing pass microbatches over rollouts (`train.micro_rollouts`) with gradient
  accumulation; generation microbatches over `train.gen_microbatch`.
- The local 7B judge loads in 4-bit (bitsandbytes, Linux). If memory is tight, run the judge
  as a separate process/server or use `judge.kind=api`.

## Known failure modes

- **VQ codebook collapse** is expected — watch `vq_perplexity`/`vq_usage`; dead-code reinit
  fights it but does not guarantee recovery.
- **`plan_grad_norm ≈ 0`** means the executor isn't passing signal to the plan (check that
  the executor really runs with grad enabled and that `y` is teacher-forced exactly as
  sampled — no re-tokenization).
- **Same-judge eval favours reward hacking.** Held-out eval uses the *same* judge as
  training; treat the numbers with suspicion and report a second judge (API) when available.
- Gumbel-softmax in bf16 underflows at low `τ`; we keep it fp32 (KL too).

## Correctness pitfalls covered by tests

`L_RL` over response tokens only with the exact sampled `y`; gradients reach codebook/plan
head; executor weights never change; KL/gumbel in fp32; judge parsing never crashes
(unparseable → reward 0 + counter); N=0 injection reproduces vanilla generation.

## A note on the Anthropic API judge

`judge.kind=api` calls `client.messages.create(model=..., max_tokens=8, messages=[...])`
(SDK default model `claude-opus-4-8`; set `judge.api.model` to a cheaper model for
high-volume judging) and parses the integer rating. Requires `ANTHROPIC_API_KEY`.
```
