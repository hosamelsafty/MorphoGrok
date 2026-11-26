"""Assemble the C_phi sensitivity block into one table.

Reads the artefacts written by ceiling_method.py under the two sensitivity
directories and prints the predicted ceiling, the accuracy the estimator
realizes when run as a predictor, and the gap between them -- the estimator's
own self-validation statistic (App. "scope condition on C_phi").

  uv run --with pandas --with pyarrow python scripts/ceiling/ceiling_sensitivity_table.py
"""
from __future__ import annotations

import json
from pathlib import Path

ART = Path(__file__).resolve().parents[2] / "artefacts" / "phase0"

ROWS = [
    ("published (max, freq prior, LOO)", "ceiling_k1v2c_template.json"),
    ("scoring: sum over D(x,y)",         "ceiling_sensitivity/score_sum.json"),
    ("LOO disabled",                     "ceiling_sensitivity/no_loo.json"),
] + [(f"smoothing alpha={a}", f"phi_sensitivity/ceiling_smooth_{a}.json")
     for a in ("0.1", "0.5", "1.0", "5.0", "20.0")]


def main():
    print(f"{'configuration':36s} {'C_phi':>7s} {'95% CI':>18s} "
          f"{'realized':>9s} {'gap':>8s}  flag")
    for label, rel in ROWS:
        p = ART / rel
        if not p.exists():
            print(f"{label:36s} {'--':>7s} {'(missing)':>18s}")
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        c, rz = d["ceiling"], d.get("realized_argmax_acc")
        lo, hi = d["ceiling_ci95"]
        gap = c - rz if rz is not None else None
        # the bound is violated whenever the estimator, run as a predictor,
        # beats the ceiling it predicted for itself
        flag = "BOUND VIOLATED" if gap is not None and gap < 0 else ""
        print(f"{label:36s} {c:7.4f} [{lo:.4f}, {hi:.4f}] "
              f"{rz if rz is None else f'{rz:9.4f}'} "
              f"{'' if gap is None else f'{gap:+8.4f}'}  {flag}")
    print("\nself-validation gap = predicted C_phi - accuracy realized as a "
          "predictor;\nnear zero is the well-specified case.")


if __name__ == "__main__":
    main()
