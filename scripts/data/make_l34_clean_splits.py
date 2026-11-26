"""Junk-free test sets for the L3 and L4 strata, with the root partition kept.

The L3/L4 models were trained on artefacts/phase0/splits_l34/ (built by
make_splits.py from the uncleaned dataset). Rebuilding those splits from the
cleaned dataset would reshuffle the root partition and put training roots into
the test set. This script therefore keeps train, val and the root partition
exactly as they are and only drops lexicon-junk items from the test list:
surfaces that carry a homograph sense index (a digit) or are pipe-joined
variants. Train and val are copied unchanged; the test SHA-256 is recomputed.

Usage:
    python scripts/data/make_l34_clean_splits.py \
        [--src artefacts/phase0/splits_l34] [--out artefacts/phase0/splits_l34_clean]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from grokkingroots.data.splits import save_split

JUNK = re.compile(r"[0-9|]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="artefacts/phase0/splits_l34")
    ap.add_argument("--out", default="artefacts/phase0/splits_l34_clean")
    args = ap.parse_args()
    for level in ("L3", "L4"):
        name = f"{level}_dcrit_p30.json"
        sp = json.loads((Path(args.src) / name).read_text(encoding="utf-8"))
        test = [t for t in sp["test"] if not JUNK.search(t[0])]
        split = {"train": sp["train"], "val": sp["val"], "test": test}
        sha = save_split(Path(args.out) / name, split, sp["meta"])
        print(f"{level}: test {len(sp['test'])} -> {len(test)} items, sha256 {sha}")


if __name__ == "__main__":
    main()
