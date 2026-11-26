"""Paired self-validation check for the computable ceiling (paper §4.2).

The ceiling estimate is the mean over test items of the posterior of the argmax
candidate; the attained accuracy is the mean of 1[argmax == gold]. Both come
from the same items, so the check is paired: per item, d_i = max_post_i -
correct_i, and the check passes when the 95% percentile bootstrap interval of
mean(d) contains 0 (10,000 resamples over items, fixed seed 0).

Items with no prediction (gold outside the candidate set) are credited 1.0 in
the estimate by scripts/ceiling/ceiling_method.py; here they are also counted correct,
so they contribute d_i = 0. (ceiling_method.py instead drops them from the
attained-accuracy denominator, which moves attained by <= 0.0004; no verdict
depends on it.) Under --lenient files, correctness uses lenient_key, as in
ceiling_method.py.

Reads the *.items.json files written by ceiling_method.py --items-out together
with their summary JSON (same name without .items).

    python3 scripts/ceiling/ceiling_selfcheck.py \
        artefacts/camera_ready/ceiling/*.items.json \
        artefacts/camera_ready/ceiling_l34/*.items.json \
        --out artefacts/camera_ready/ceiling_selfcheck.json
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from grokkingroots.data.arabic_text import lenient_key


def check(items_path: Path, n_boot: int = 10_000, seed: int = 0) -> dict:
    summary = json.loads(Path(str(items_path).replace(".items.json", ".json")).read_text())
    items = json.loads(items_path.read_text())
    lenient = summary.get("lenient", False)
    key = lenient_key if lenient else (lambda r: r)

    est = [it["max_post"] for it in items]
    cor = [1.0 if it["pred"] is None else float(it["pred"] == key(it["gold"])) for it in items]
    d = [e - c for e, c in zip(est, cor)]
    n = len(d)
    rng = random.Random(seed)
    boots = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    lo, hi = boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot) - 1]
    return {
        "file": str(items_path),
        "lenient": lenient,
        "n": n,
        "estimate": round(sum(est) / n, 4),
        "estimate_file": summary["ceiling"],
        "estimate_ci95_file": summary["ceiling_ci95"],
        "attained_all_items": round(sum(cor) / n, 4),
        "attained_file": summary["realized_argmax_acc"],
        "mean_diff": round(sum(d) / n, 4),
        "paired_ci95": [round(lo, 4), round(hi, 4)],
        "passes": lo <= 0 <= hi,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("items", nargs="+", type=Path)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = [check(p) for p in sorted(args.items)]
    for r in rows:
        print(f"{Path(r['file']).name:52s} est {r['estimate']:.4f} att {r['attained_all_items']:.4f} "
              f"diff {r['mean_diff']:+.4f} [{r['paired_ci95'][0]:+.4f}, {r['paired_ci95'][1]:+.4f}] "
              f"{'pass' if r['passes'] else 'FAIL'}")
    Path(args.out).write_text(json.dumps(rows, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
