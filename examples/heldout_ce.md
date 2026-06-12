# Held-out generalization: NLL of unseen-topic targets (lower = better)

Planner trained on 8 *different* topics; the 4 targets below were never trained on. `base` = frozen Qwen, no plan. `planner` = same frozen Qwen + trained latent plan.

| held-out prompt | base NLL | planner NLL | Δ |
|---|---|---|---|
| List the parts of the human body, then write two sentences a… | 2.060 | 1.193 | -0.867 |
| List the oceans of the Earth, then write two sentences about… | 1.855 | 0.804 | -1.051 |
| List the basic arithmetic operations, then write two sentenc… | 1.924 | 1.062 | -0.861 |
| List three programming languages, then write two sentences a… | 2.324 | 1.414 | -0.910 |

**Mean held-out NLL — base 2.041 · planner 1.118 · Δ -0.922**

Planner LOWERS held-out NLL → the latent plan transfers (generalization).
