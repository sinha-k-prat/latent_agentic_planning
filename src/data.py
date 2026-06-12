"""Instruction (+ optional target) data loading.

Each split item is a dict {"instruction": str, "target": str}. The instruction is the
prompt distribution; the target is a gold/distilled response used ONLY in CE / distillation
mode (loss_mode=ce). In RL mode the target is ignored (responses are sampled + judged).

Schema auto-detection covers common instruction datasets:
  alpaca-cleaned : instruction (+ input) / output
  OpenOrca/SlimOrca/Hermes : question (or system+question) / response
  multi-hop QA   : question / answer

Big datasets (e.g. Open-Orca/OpenOrca, ~4M rows) should set data.streaming=true so we
stop after collecting enough rows instead of downloading everything.
"""
import json
import os
import random

_INSTR_KEYS = ("instruction", "question", "prompt", "query")
_TARGET_KEYS = ("output", "response", "answer", "completion")


def _instruction_text(ex):
    if ex.get("instruction"):
        instr = ex["instruction"].strip()
        inp = (ex.get("input") or "").strip()
        return f"{instr}\n\n{inp}" if inp else instr
    for k in _INSTR_KEYS[1:]:
        v = ex.get(k)
        if v:
            return str(v).strip()
    return ""


def _target_text(ex):
    for k in _TARGET_KEYS:
        v = ex.get(k)
        if v:
            return str(v).strip()
    return ""


def _as_pair(x):
    """Tolerate both the new dict form and the legacy plain-string form."""
    if isinstance(x, dict):
        return {"instruction": x.get("instruction", ""), "target": x.get("target", "")}
    return {"instruction": x, "target": ""}


def split_instructions(split):
    return [_as_pair(x)["instruction"] for x in split]


def split_pairs(split):
    return [_as_pair(x) for x in split]


def build_or_load_splits(cfg, tokenizer):
    """Return {'train','eval','calib'} -> list[{'instruction','target'}].

    Persisted to cfg.data.split_path and reused (fixed seed) on later runs. Backward
    compatible with older split files that stored plain instruction strings.
    """
    path = cfg.data.split_path
    if os.path.exists(path):
        with open(path) as f:
            raw = json.load(f)
        return {k: [_as_pair(x) for x in v] for k, v in raw.items()}

    from datasets import load_dataset

    streaming = bool(cfg.data.get("streaming", False))
    ds = load_dataset(cfg.data.dataset, split="train", streaming=streaming)

    n_tr, n_ev, n_ca = cfg.data.n_train, cfg.data.n_eval, cfg.data.n_calib
    need = n_tr + n_ev + n_ca
    collect = need * 2  # oversample so the seeded shuffle has variety
    max_tok = cfg.data.max_prompt_tokens

    seen, items = set(), []
    for ex in ds:
        text = _instruction_text(ex)
        if not text or text in seen:
            continue
        if len(tokenizer(text, add_special_tokens=False).input_ids) > max_tok:
            continue
        seen.add(text)
        items.append({"instruction": text, "target": _target_text(ex)})
        if len(items) >= collect:
            break

    if len(items) < need:
        raise ValueError(f"dataset gave {len(items)} usable items, need {need}")

    random.Random(cfg.seed).shuffle(items)
    splits = {
        "train": items[:n_tr],
        "eval": items[n_tr:n_tr + n_ev],
        "calib": items[n_tr + n_ev:n_tr + n_ev + n_ca],
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(splits, f)
    return splits
