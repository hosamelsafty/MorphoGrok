"""Build leakage-controlled splits into artefacts/phase0/splits/ and verify G0.

Usage:
    python scripts/data/make_splits.py [--dataset PATH] [--out-dir DIR] [--seed 42]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from grokkingroots.data.splits import (
    dcrit_splits, iid_split, root_disjoint_split, save_split, seen_unseen_split,
)


def surface_overlap(split) -> int:
    tr = {tuple(k) for k in split["train"]}
    te = {tuple(k) for k in split["test"]}
    return len(tr & te)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="artefacts/phase0/roots_dataset.parquet")
    ap.add_argument("--out-dir", default="artefacts/phase0/splits")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--levels", default="L1,L2",
                    help="comma-separated difficulty levels to build. The default "
                         "reproduces the frozen family byte-for-byte; anything else "
                         "MUST be written to a different --out-dir, since the frozen "
                         "_summary.json carries the SHA256 freeze that every existing "
                         "run is validated against.")
    args = ap.parse_args()
    levels = [x.strip() for x in args.levels.split(",") if x.strip()]

    df = pd.read_parquet(args.dataset)
    outdir = Path(args.out_dir)
    summary = {}

    def record(name, sp, meta):
        sha = save_split(outdir / f"{name}.json", sp, meta)
        summary[name] = {**meta, "sizes": {k: len(v) for k, v in sp.items()},
                         "test_sha256": sha[:12], "surface_overlap": surface_overlap(sp)}

    # full dataset, root-disjoint, stratified by difficulty. Only emitted on the
    # default run, so a per-level build cannot perturb the frozen family.
    if levels == ["L1", "L2"]:
        record("root_disjoint_all", *root_disjoint_split(df, seed=args.seed))
    # per-difficulty families + D_crit (fixed test/val across p)
    for level in levels:
        sub = df[df["difficulty_level"] == level].reset_index(drop=True)
        record(f"{level}_root_disjoint", *root_disjoint_split(sub, seed=args.seed))
        record(f"{level}_seen_unseen", *seen_unseen_split(sub, seed=args.seed))
        record(f"{level}_iid", *iid_split(sub, seed=args.seed))
        for p, (sp, meta) in dcrit_splits(sub, [0.05, 0.10, 0.20, 0.30, 0.50, 0.80], seed=args.seed).items():
            record(f"{level}_dcrit_p{int(p * 100):02d}", sp, meta)

    (outdir / "_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                          encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    leaks = {k: s["surface_overlap"] for k, s in summary.items() if s["surface_overlap"]}
    assert not leaks, f"LEAKAGE (train∩test surface overlap): {leaks}"
    print("\nG0 leakage check: PASS — 0 surface overlap across all splits")


if __name__ == "__main__":
    main()
