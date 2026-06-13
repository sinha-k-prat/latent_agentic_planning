# Plan-supervision corpus (path B)

A dataset for **supervising the latent planner** with explicit, interpretable plan-token
sequences — the fix for the *input-independence collapse* the latent (path-A) planner showed.
Instead of discovering codes from reward alone, each instruction is labeled with a sequence of
operations drawn from a fixed **64-op "cognitive instruction set"**, so the planner is forced to
emit *different* plans for *different* instructions (no collapse) and the codes stay interpretable.

## Schema — `plan_dataset.jsonl` (800 unique examples)

Each line:
```json
{
  "instruction":    "From 36, 23, 29, 34, 32, 16, 11, keep only the even numbers.",
  "plan":           "Plan: 1) extract numbers; 2) filter by condition; 3) format list.",
  "plan_tokens":    ["EXTRACT_NUMBERS", "FILTER_BY_CONDITION", "FORMAT_LIST", "EOP"],
  "plan_token_ids": [3, 17, 58, 64],
  "response":       "Kept: 36, 34, 32, 16."
}
```
- **instruction** — a general NL task (35+ families: arithmetic, sort/median, filter, top-k,
  unit conversion, counting, extraction, classification, comparison, formatting (list/table/JSON),
  summarize, explain, analogy, pros/cons, steps, code, critique/repair, fact-check, cause,
  language ID, keywords, dedup, estimate, syllogistic deduction, brainstorm, question-gen,
  tone rewrite, recommendation, multi-step compositions, …).
- **plan** — natural-language decomposition.
- **plan_tokens / plan_token_ids** — the symbolic plan: a sequence of the 64 ops, terminated by `EOP`.
- **response** — gold worked answer. **Computable tasks are exact** (generated deterministically:
  arithmetic, sort, filter, convert, count, extract, format, prime-check, …); knowledge/writing
  tasks use small curated tables. No LLM was used — it's fully reproducible.

## The 64 operations — `operations.json`

A small ISA over task operations, 8 families × 8 (ids 0–63), plus `EOP` (id 64) as the plan
terminator:

| family | ops |
|---|---|
| **GROUND** | IDENTIFY_TASK, PARSE_INPUT, EXTRACT_ENTITIES, EXTRACT_NUMBERS, EXTRACT_CONSTRAINTS, IDENTIFY_KEYWORDS, SEGMENT_TEXT, DETECT_LANGUAGE |
| **RECALL** | ENUMERATE, RECALL_FACTS, RETRIEVE_DEFINITION, LIST_EXAMPLES, LIST_STEPS, LIST_PROS_CONS, GENERATE_CANDIDATES, BRAINSTORM |
| **SELECT** | SELECT_SUPERLATIVE, FILTER_BY_CONDITION, RANK, TOP_K, PICK_BEST, SELECT_RELEVANT, DEDUP, CHOOSE_BY_CRITERIA |
| **COMPUTE** | COMPUTE_ARITHMETIC, SORT, AGGREGATE, CONVERT_UNITS, MAP_TRANSFORM, NORMALIZE, COUNT, ROUND_ESTIMATE |
| **REASON** | COMPARE, CLASSIFY, EVALUATE_CONDITION, INFER_CAUSE, DECOMPOSE, DEDUCE, CHECK_LOGIC, ESTIMATE |
| **GENERATE** | WRITE_SENTENCE, WRITE_PARAGRAPH, WRITE_CODE, GIVE_EXAMPLE, CONSTRUCT_ANALOGY, DRAFT_OUTLINE, COMPOSE_MESSAGE, GENERATE_QUESTION |
| **VERIFY** | VERIFY_FORMAT, CHECK_CONSTRAINT, FACT_CHECK, CRITIQUE, REVISE, VALIDATE_NUMBER, SELF_CORRECT, SUMMARIZE_CHECK |
| **COMMUNICATE** | SUMMARIZE, EXPLAIN_SIMPLE, FORMAT_LIST, FORMAT_TABLE, FORMAT_JSON, ADAPT_TONE, ADD_CAVEAT, CONCLUDE |

## Stats
- 800 unique instructions · **all 64 ops used** · **45 distinct plan-token sequences** ·
  plan length 2–5 ops (`EOP` excluded).

## Reproduce
```bash
python dataset/build_plan_dataset.py   # -> plan_dataset.jsonl, operations.json (deterministic, SEED=7)
```

## How this plugs into the planner (next step)
Teacher-force the (autoregressive) plan head on `plan_token_ids` + CE on `response`:
```
L = CE(response | x, plan)  +  CE(plan_tokens | x)
```
The plan-token supervision **forces input-dependence** (kills the collapse), makes the codes
**interpretable** (each id = a named operation), and — because we now have plan targets — lets the
autoregressive plan head be **teacher-forced** (fast, no rollout). `EOP` supports dynamic plan
length (halt at the first `EOP`). The 64 ops are the bridge to a grounded skill inventory
(à la an O*NET/skill-DAG ontology): codes = reusable skills.
