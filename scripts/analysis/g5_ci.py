"""Gen-5 val-selected results with seed-level bootstrap CIs.

Reuses aggregate_ci.py's exact method (deterministic LCG, fixed seed) on the
gen-5 v2clean intervention runs in artefacts/runs/. Metric = test exact-match
at the best-val checkpoint (val-selected). CIs are SEED-LEVEL (n=5 seeds each);
the example-paired bootstrap needs GPU checkpoint re-eval (P3/P7).

Usage: python scripts/analysis/g5_ci.py
"""
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from aggregate_ci import test_at_best_val, bootstrap_ci, paired_diff_ci

ROOT = "artefacts/runs"
CONDS = [
    ("k1", "k1 single-form (baseline)"),
    ("k2_para", "k2 paradigm"),
    ("k3_para", "k3 paradigm"),
    ("k3_self", "k3 self (length control)"),
    ("k3_shuf", "k3 shuf (specificity control)"),
    ("k3_reveal", "reveal (witness; changed interface)"),
    ("kmix", "kmix (same interface)"),
]


def seed_vals(cond):
    """Val-selected test acc per seed, ordered by seed index."""
    out = {}
    for d in glob.glob(f"{ROOT}/g5_L2_{cond}_v2c_wd005_500k_s*/"):
        if not os.path.exists(d + "DONE"):
            continue
        s = int(re.search(r"_s(\d+)/$", d).group(1))
        out[s] = test_at_best_val(json.load(open(d + "history.json")))
    return [out[s] for s in sorted(out)]


def main():
    vals = {c: seed_vals(c) for c, _ in CONDS}
    base = vals["k1"]
    print(f"{'condition':36s} n  mean   95% CI            Δ vs k1 (95% CI)")
    for c, lab in CONDS:
        v = vals[c]
        m, lo, hi = bootstrap_ci(v)
        if c == "k1":
            delta = ""
        else:
            dm, dlo, dhi = paired_diff_ci(v, base)
            delta = f"{dm:+.3f} [{dlo:+.3f}, {dhi:+.3f}]"
        print(f"{lab:36s} {len(v)}  {m:.3f}  [{lo:.3f}, {hi:.3f}]   {delta}")


if __name__ == "__main__":
    main()
