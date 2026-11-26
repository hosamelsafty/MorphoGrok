"""How robust is the grok / no-grok label?

The paper's criterion (section 5, "When a run groks"): a run groks iff
held-out exact match is still below HALF its eventual peak when train exact
match first passes 0.99; the grok step is then the first later checkpoint at
half-peak. This script recomputes, from the logged trajectories only:

  1. per run: memorization step, held-out accuracy there, eventual peak, their
     ratio, and the absolute gain after memorization;
  2. the grok / no-grok label at cutoffs 0.4, 0.5, 0.6, 0.7 of peak;
  3. a cutoff-free description of the second rise: its size (peak minus
     accuracy at memorization) and the step at which half of it is complete
     (first checkpoint after memorization at the midpoint between the two).

Metric = history.json `test_exact` / `train_exact`, i.e. greedy exact match on
the training loop's 2,000-item test prefix, the same series the published
criterion reads. --max-step restricts every run to a common budget.

No GPU, no checkpoints. Usage:
  python scripts/analysis/grok_robustness.py --out artefacts/camera_ready/grok_robustness.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from pathlib import Path

ROOT = "artefacts/runs"
FAMILIES = [  # (label, run-dir glob)
    ("Single-Form", "g5_L2_k1_v2c_wd005_500k_s*"),
    ("Support-1", "g5_L2_k2_para_v2c_wd005_500k_s*"),
    ("Support-2", "g5_L2_k3_para_v2c_wd005_500k_s*"),
    ("Witness", "g5_L2_k3_reveal_v2c_wd005_500k_s*"),
    ("Repeat", "g5_L2_k3_self_v2c_wd005_500k_s*"),
    ("Cross-Root", "g5_L2_k3_shuf_v2c_wd005_500k_s*"),
    ("Mixed-k", "g5_L2_kmix_v2c_wd005_500k_s*"),
    ("L1 (wd 0.1, init 3)", "g1b_L1_p30_wd01_is3_1M_s*"),
    ("Single-Form legacy 1M", "g1_L2nondiac_wd005_1M_s*"),
    ("Single-Form init 3 (wd 0.05)", "cr_L2_k1_is3_v2c_wd005_500k_s*"),
    ("Single-Form init 3 (wd 0.1)", "cr_L2_k1_is3_v2c_wd01_500k_s*"),
]
CUTOFFS = (0.4, 0.5, 0.6, 0.7)


def analyse(hist, max_step=0):
    h = [r for r in hist if not max_step or r["step"] <= max_step]
    test = [r["test_exact"] for r in h]
    mem = next((i for i, r in enumerate(h) if r["train_exact"] > 0.99), None)
    if mem is None:
        return None
    peak, at_mem = max(test), test[mem]
    out = {"last_step": h[-1]["step"], "mem_step": h[mem]["step"],
           "test_at_mem": at_mem, "peak": peak, "ratio": at_mem / peak,
           "gain_after_mem": peak - at_mem,
           "val_selected": max(h, key=lambda r: r["val_exact"])["test_exact"]}
    for c in CUTOFFS:   # the published rule at cutoff c (c = 0.5 is the paper's)
        step = None
        if at_mem < c * peak:
            step = next(r["step"] for r in h[mem:] if r["test_exact"] >= c * peak)
        out[f"grok_step@{c}"] = step
    mid = (at_mem + peak) / 2
    out["midrise_step"] = next(r["step"] for r in h[mem:] if r["test_exact"] >= mid)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-step", type=int, default=0)
    args = ap.parse_args()

    result = {"metric": "history test_exact (greedy, 2000-item prefix)",
              "max_step": args.max_step or None, "families": {}}
    for label, pat in FAMILIES:
        runs = {}
        for d in sorted(glob.glob(f"{ROOT}/{pat}")):
            hp = os.path.join(d, "history.json")
            if os.path.exists(os.path.join(d, "DONE")) and os.path.exists(hp):
                a = analyse(json.load(open(hp)), args.max_step)
                if a:
                    runs[int(re.search(r"_s(\d+)$", d).group(1))] = a
        if runs:
            result["families"][label] = runs

    print(f"{'family':30s} n  mem step     test@mem   peak     ratio        gain     "
          f"groks@.4/.5/.6/.7   mid-rise step")
    for label, runs in result["families"].items():
        v = list(runs.values())

        def rng(k, fmt="{:.3f}"):
            xs = [r[k] for r in v]
            lo, hi = fmt.format(min(xs)), fmt.format(max(xs))
            return lo if lo == hi else f"{lo}-{hi}"

        grok = "/".join(str(sum(r[f"grok_step@{c}"] is not None for r in v)) for c in CUTOFFS)
        print(f"{label:30s} {len(v)}  {rng('mem_step', '{}'):11s}  {rng('test_at_mem'):10s} "
              f"{rng('peak'):11s} {rng('ratio', '{:.2f}'):9s}  {rng('gain_after_mem'):11s} "
              f"{grok:10s} of {len(v)}   {rng('midrise_step', '{}')}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
