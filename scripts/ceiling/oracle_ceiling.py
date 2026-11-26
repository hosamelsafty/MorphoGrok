"""P2 (no GPU): oracle ambiguity ceiling for root-disjoint test sets.

For each test surface, the best any DETERMINISTIC model could do is predict the
majority gold root for that undiacritized surface across the full lexicon
(main + ambiguous bin). If that ceiling is 1.0, surface-collision ambiguity does
NOT explain the model's accuracy gap.

Usage: python scripts/ceiling/oracle_ceiling.py
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd

ART = Path("artefacts/phase0")


def main():
    m = pd.read_parquet(ART / "roots_dataset.parquet")
    a = pd.read_parquet(ART / "roots_dataset_ambiguous.parquet")
    full = pd.concat([m, a], ignore_index=True)

    sr = {}
    for s, r in zip(full["surface_clean"], full["root_clean"]):
        sr.setdefault(s, Counter())[r] += 1

    for lvl in ["L1", "L2"]:
        split = json.loads((ART / "splits" / f"{lvl}_root_disjoint.json").read_text())
        test = split["test"]  # list of [surface, root]
        n = len(test)
        amb = oracle = unseen = 0
        for s, gold in test:
            c = sr.get(s)
            if c is None:
                unseen += 1
                continue
            if len(c) > 1:
                amb += 1
            if c.most_common(1)[0][0] == gold:
                oracle += 1
        print(f"{lvl} root-disjoint test: n={n}")
        print(f"  ambiguous surfaces (>1 root in lexicon): {amb} ({100*amb/n:.1f}%)")
        print(f"  oracle ceiling (majority-root):          {oracle/n:.3f}")
        print(f"  surfaces absent from lexicon:            {unseen}")


if __name__ == "__main__":
    main()
