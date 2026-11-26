"""Quantify the achievable (Bayes-style) ceiling for L2 root extraction.

Decomposes the L2 test set of a dcrit split into:
  - VISIBLE:  the weak/hamza radical appears in the surface -> recoverable in
              principle (ceiling contribution 1.0)
  - HIDDEN:   the weak radical is absent from the surface -> the best any
              learner can do from a single form is the prior argmax over the
              weak slot given the visible radicals, estimated from TRAIN roots
  - JUNK:     surface contains non-Arabic characters (sense indices, pipes) —
              reported separately since these rows are noise, not linguistics

Outputs per dataset variant: visible/hidden/junk fractions, the prior-argmax
accuracy on hidden items, and the composite ceiling estimate
    ceiling = P(visible) * 1 + P(hidden) * acc(prior-argmax)
(assuming non-weak radicals are read perfectly — so this is an UPPER bound).

Usage:
  python scripts/ceiling/ambiguity_audit.py --dataset artefacts/phase0/roots_dataset.parquet \
      --split artefacts/phase0/splits/L2_dcrit_p30.json
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict

import pandas as pd

ARABIC_ONLY = re.compile(r"^[ء-يٱ-ۓ]+$")
WEAK = set("ويىا")
HAMZA_FORMS = {"ء": "ءأإؤئآ", "أ": "ءأإؤئآ", "إ": "ءأإؤئآ", "ؤ": "ءأإؤئآ", "ئ": "ءأإؤئآ", "آ": "ءأإؤئآ"}


def weak_slot(root_radicals, root_type):
    """Return (index, radical) of the weak/hamza radical for L2 root types."""
    rads = root_radicals.split() if " " in root_radicals else list(root_radicals)
    if root_type == "hollow":
        return 1, rads[1]
    if root_type == "defective":
        return 2, rads[2]
    if root_type == "assimilated":
        return 0, rads[0]
    if root_type == "hamzated":
        for i, r in enumerate(rads):
            if r in HAMZA_FORMS:
                return i, r
    return None, None


def surface_shows(radical, surface):
    """Is the weak radical (or an equivalent carrier) present in the surface?"""
    if radical in HAMZA_FORMS:
        return any(c in surface for c in HAMZA_FORMS[radical])
    if radical == "و":
        return "و" in surface
    if radical == "ي":
        return "ي" in surface or "ى" in surface
    return radical in surface


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="artefacts/phase0/roots_dataset.parquet")
    ap.add_argument("--split", default="artefacts/phase0/splits/L2_dcrit_p30.json")
    args = ap.parse_args()

    df = pd.read_parquet(args.dataset)
    info = {(s, rc): (rt, r) for s, rc, rt, r in
            zip(df["surface_clean"], df["root_clean"], df["root_type"], df["root"])}

    sp = json.loads(open(args.split, encoding="utf-8").read())
    train_roots = {}
    for s, rc in sp["train"]:
        rec = info.get((s, rc))
        if rec:
            train_roots[rc] = rec
    # weak-slot priors from TRAIN root TYPES (root-level, not pair-level)
    prior_ctx = defaultdict(Counter)   # (root_type, visible radicals) -> weak radical
    prior_glob = defaultdict(Counter)  # root_type -> weak radical
    for rc, (rt, root) in train_roots.items():
        idx, w = weak_slot(root, rt)
        if idx is None:
            continue
        rads = root.split()
        ctx = tuple(r for i, r in enumerate(rads) if i != idx)
        prior_ctx[(rt, ctx)][w] += 1
        prior_glob[rt][w] += 1

    n = vis = hid = hid_correct = junk = other = 0
    per_type = defaultdict(lambda: Counter())
    for s, rc in sp["test"]:
        rec = info.get((s, rc))
        if rec is None:
            continue
        rt, root = rec
        n += 1
        if not ARABIC_ONLY.match(s):
            junk += 1
            per_type[rt]["junk"] += 1
            continue
        idx, w = weak_slot(root, rt)
        if idx is None:  # e.g. doubled classed oddly — treat as visible
            other += 1
            per_type[rt]["visible"] += 1
            continue
        if surface_shows(w, s):
            vis += 1
            per_type[rt]["visible"] += 1
        else:
            hid += 1
            per_type[rt]["hidden"] += 1
            rads = root.split()
            ctx = tuple(r for i, r in enumerate(rads) if i != idx)
            c = prior_ctx.get((rt, ctx)) or prior_glob[rt]
            guess = c.most_common(1)[0][0] if c else None
            if guess == w:
                hid_correct += 1
                per_type[rt]["hidden_prior_ok"] += 1

    vis_frac = (vis + other) / n
    hid_frac = hid / n
    junk_frac = junk / n
    hid_acc = hid_correct / hid if hid else 0.0
    ceiling_junk_zero = vis_frac + hid_frac * hid_acc            # junk scored wrong
    ceiling_junk_free = (vis + other + hid_correct) / max(1, n - junk)  # junk excluded

    print(f"dataset={args.dataset}  split={args.split}")
    print(f"test items analysed: {n}")
    print(f"  weak radical VISIBLE : {vis_frac:6.1%}  (recoverable in principle)")
    print(f"  weak radical HIDDEN  : {hid_frac:6.1%}  -> prior-argmax accuracy {hid_acc:.1%}")
    print(f"  JUNK surfaces        : {junk_frac:6.1%}")
    print(f"  ceiling (junk counted as errors): {ceiling_junk_zero:.4f}")
    print(f"  ceiling (junk-free test)       : {ceiling_junk_free:.4f}")
    print("\nper root_type breakdown:")
    for rt, c in sorted(per_type.items()):
        tot = sum(v for k, v in c.items() if k in ("visible", "hidden", "junk"))
        print(f"  {rt:14s} n={tot:5d}  visible={c['visible']:5d}  hidden={c['hidden']:5d} "
              f"(prior-ok {c['hidden_prior_ok']})  junk={c['junk']}")


if __name__ == "__main__":
    main()
