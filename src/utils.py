"""Config loading, seeding, device selection, and lightweight loggers."""
import csv
import json
import os
import random

import numpy as np
import torch
from omegaconf import OmegaConf


def load_config(path, overrides=None, smoke=False):
    """Load YAML config, apply optional --smoke-test overrides, then CLI dotlist overrides."""
    cfg = OmegaConf.load(path)
    if smoke:
        s = cfg.smoke
        cfg.model.base = s.model_base
        cfg.model.judge = s.model_judge
        cfg.train.steps = s.steps
        cfg.train.group_size = s.G
        cfg.train.batch_size = min(cfg.train.batch_size, 4)
        cfg.train.gen_microbatch = min(cfg.train.gen_microbatch, 8)
        cfg.plan.N = s.N
        cfg.plan.K = s.K
        cfg.judge.kind = s.judge_kind
        cfg.data.n_train = s.n_instructions
        cfg.data.n_eval = min(cfg.data.n_eval, s.n_instructions)
        cfg.data.n_calib = min(cfg.data.n_calib, s.n_instructions)
        cfg.plan.gumbel.tau_anneal_steps = max(2, s.steps)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    return cfg


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(cfg):
    if cfg.device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if cfg.device == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def gumbel_tau(step, cfg):
    """Exponential anneal from tau_start -> tau_end over tau_anneal_steps."""
    g = cfg.plan.gumbel
    t = min(1.0, step / max(1, g.tau_anneal_steps))
    return float(g.tau_start * (g.tau_end / g.tau_start) ** t)


def microbatches(seq, size):
    seq = list(seq)
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


class CSVLogger:
    """Append-only CSV logger that lazily writes the header from the first row's keys."""

    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._fields = None
        self._fh = None
        self._writer = None

    def log(self, row):
        if self._writer is None:
            self._fields = list(row.keys())
            self._fh = open(self.path, "w", newline="")
            self._writer = csv.DictWriter(self._fh, fieldnames=self._fields)
            self._writer.writeheader()
        # Only write known fields; ignore extras to keep the CSV rectangular.
        self._writer.writerow({k: row.get(k, "") for k in self._fields})
        self._fh.flush()

    def close(self):
        if self._fh:
            self._fh.close()


class JsonlLogger:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._fh = open(path, "a")

    def log(self, obj):
        self._fh.write(json.dumps(obj) + "\n")
        self._fh.flush()

    def close(self):
        self._fh.close()
