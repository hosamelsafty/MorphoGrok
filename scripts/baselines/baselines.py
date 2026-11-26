"""Evaluate non-neural baselines on a split's test set (overall + by difficulty).

Usage:
    python scripts/baselines/baselines.py --split SPLIT.json [--dataset DATA.parquet] [--out OUT.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from grokkingroots.baselines.skeleton import skeleton_root
from grokkingroots.data.encoding import load_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="artefacts/phase0/roots_dataset.parquet")
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    df = pd.read_parquet(args.dataset)
    difficulty = dict(zip(df["surface_clean"], df["difficulty_level"]))
    _, _, test, _ = load_split(args.split)

    overall, by = [], {}
    for surf, root in test:
        ok = int(skeleton_root(surf, n=len(root)) == root)
        overall.append(ok)
        by.setdefault(difficulty.get(surf, "?"), []).append(ok)

    res = {
        "split": args.split,
        "n_test": len(test),
        "skeleton_exact_overall": round(sum(overall) / len(overall), 4) if overall else 0.0,
        "skeleton_exact_by_difficulty": {k: round(sum(v) / len(v), 4) for k, v in sorted(by.items())},
        "counts_by_difficulty": {k: len(v) for k, v in sorted(by.items())},
    }
    print(json.dumps(res, indent=2, ensure_ascii=False))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
