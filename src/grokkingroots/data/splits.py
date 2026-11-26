"""Leakage-controlled splits for root extraction.

Three families:
  - root_disjoint: test roots never appear in train (tests rule generalization)
  - seen_unseen:   all roots in train, held-out surface forms in test
  - iid:           random row split (upper-bound reference)

Plus dcrit_splits: root-disjoint with a FIXED test/val across training fractions
p, to locate the critical dataset size. Test sets are SHA256-frozen so a sweep
cell always evaluates on identical data.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _keys(df: pd.DataFrame):
    return [[s, r] for s, r in zip(df["surface_clean"], df["root_clean"])]


def sha256_of_keys(keys) -> str:
    blob = json.dumps(sorted("\t".join(k) for k in keys), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def root_disjoint_split(df, test_frac=0.2, val_frac=0.1, seed=42,
                        stratify_by="difficulty_level"):
    rng = np.random.default_rng(seed)
    train_roots, val_roots, test_roots = set(), set(), set()
    roots_df = df.drop_duplicates("root_clean")[["root_clean", stratify_by]]
    for _, group in roots_df.groupby(stratify_by):
        roots = group["root_clean"].to_numpy().copy()
        rng.shuffle(roots)
        n, n_test, n_val = len(roots), int(len(roots) * test_frac), int(len(roots) * val_frac)
        test_roots.update(roots[:n_test])
        val_roots.update(roots[n_test:n_test + n_val])
        train_roots.update(roots[n_test + n_val:])
    assert not (train_roots & test_roots) and not (train_roots & val_roots)
    split = {
        "train": _keys(df[df["root_clean"].isin(train_roots)]),
        "val": _keys(df[df["root_clean"].isin(val_roots)]),
        "test": _keys(df[df["root_clean"].isin(test_roots)]),
    }
    return split, {"family": "root_disjoint", "seed": seed, "test_frac": test_frac,
                   "val_frac": val_frac, "n_train_roots": len(train_roots),
                   "n_val_roots": len(val_roots), "n_test_roots": len(test_roots)}


def seen_unseen_split(df, test_frac=0.2, val_frac=0.1, seed=42):
    rng = np.random.default_rng(seed)
    train_idx, val_idx, test_idx = [], [], []
    for _, group in df.groupby("root_clean"):
        idx = group.index.to_numpy().copy()
        rng.shuffle(idx)
        n = len(idx)
        if n == 1:
            train_idx.append(idx[0])
            continue
        n_test = min(int(n * test_frac), n - 1)
        n_val = min(int(n * val_frac), n - 1 - n_test)
        test_idx += list(idx[:n_test])
        val_idx += list(idx[n_test:n_test + n_val])
        train_idx += list(idx[n_test + n_val:])
    split = {"train": _keys(df.loc[train_idx]), "val": _keys(df.loc[val_idx]),
             "test": _keys(df.loc[test_idx])}
    return split, {"family": "seen_unseen", "seed": seed, "test_frac": test_frac,
                   "val_frac": val_frac}


def iid_split(df, test_frac=0.2, val_frac=0.1, seed=42):
    rng = np.random.default_rng(seed)
    idx = df.index.to_numpy().copy()
    rng.shuffle(idx)
    n, n_test, n_val = len(idx), int(len(idx) * test_frac), int(len(idx) * val_frac)
    split = {"test": _keys(df.loc[idx[:n_test]]), "val": _keys(df.loc[idx[n_test:n_test + n_val]]),
             "train": _keys(df.loc[idx[n_test + n_val:]])}
    return split, {"family": "iid", "seed": seed, "test_frac": test_frac, "val_frac": val_frac}


def dcrit_splits(df, fractions, test_frac=0.2, val_frac=0.1, seed=42):
    """Root-disjoint with a FIXED test/val; train = nested fraction p of the train pool."""
    rng = np.random.default_rng(seed)
    roots = df.drop_duplicates("root_clean")["root_clean"].to_numpy().copy()
    rng.shuffle(roots)
    n, n_test, n_val = len(roots), int(len(roots) * test_frac), int(len(roots) * val_frac)
    test_roots = set(roots[:n_test])
    val_roots = set(roots[n_test:n_test + n_val])
    pool = roots[n_test + n_val:]
    val_df = df[df["root_clean"].isin(val_roots)]
    test_df = df[df["root_clean"].isin(test_roots)]
    out = {}
    for p in fractions:
        train_roots = set(pool[:max(1, int(len(pool) * p))])
        split = {"train": _keys(df[df["root_clean"].isin(train_roots)]),
                 "val": _keys(val_df), "test": _keys(test_df)}
        out[p] = (split, {"family": "dcrit", "p": p, "seed": seed,
                          "n_train_roots": len(train_roots), "n_test_roots": len(test_roots)})
    return out


def save_split(path, split, meta) -> str:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sha = sha256_of_keys(split["test"])
    payload = {"meta": meta, "test_sha256": sha, **split}
    Path(path).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return sha
