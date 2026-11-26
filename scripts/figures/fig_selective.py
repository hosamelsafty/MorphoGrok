"""Figure: calibrated selective extraction (risk-coverage).

REPLACES the TikZ `fig:selective`. The port fixes one real defect: the TikZ
version hand-placed six tau labels with per-node anchors, and the tau=0.3 and
tau=0.5 labels sat close enough to collide at column width. Here the labels
are offset programmatically along the curve.

The y-axis deliberately starts at 0.60 -- the whole figure lives in a narrow
accuracy band and a 0-1 axis would flatten it to a line. That is a stated
exception to the full-axis rule, and the caption says so.
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import matplotlib.pyplot as plt
import style, data

# label offsets in points, keyed by tau -- tuned so no two collide at 3.03in
OFFSETS = {0.3: (0, -11), 0.5: (0, -11), 0.7: (-2, 7), 0.8: (-2, 7), 0.9: (8, 0), 0.95: (8, -8)}


def build():
    style.apply()
    rc = data.risk_coverage("Single-Form")
    pts = sorted(rc["points"], key=lambda p: -p["coverage"])

    # full-coverage endpoint: no abstention, at the observed top-1
    xs = [1.0] + [p["coverage"] for p in pts]
    ys = [rc["top1"]] + [p["selective_acc"] for p in pts]

    fig, ax = plt.subplots(figsize=(style.COL_IN, 2.1))

    ax.axhline(rc["expected_acc"], color=style.CTRLGRAY, ls="--", lw=1.0, zorder=1)
    ax.text(0.02, rc["expected_acc"] + 0.008,
            f"expected full-coverage acc.: {rc['expected_acc']:.3f}",
            fontsize=style.ANNOT, color="#5d6568")

    ax.plot(xs, ys, color=style.TEAL, ls="-", marker="o", ms=3.2, zorder=3)
    ax.plot([1.0], [rc["top1"]], marker="o", ms=4.5, color=style.TEAL,
            markeredgecolor="white", markeredgewidth=0.5, zorder=4)
    ax.annotate(f"no abstention: {rc['top1']:.3f}", (1.0, rc["top1"]),
                textcoords="offset points", xytext=(-2, -10), ha="right",
                fontsize=style.ANNOT, color="#1d4a58")

    for p in pts:
        dx, dy = OFFSETS.get(p["tau"], (0, 7))
        ax.annotate(f"$\\tau{{=}}{p['tau']:g}$", (p["coverage"], p["selective_acc"]),
                    textcoords="offset points", xytext=(dx, dy), ha="center",
                    fontsize=style.ANNOT, color="#1d4a58")

    ax.set_xlim(0, 1.04)
    ax.set_ylim(0.60, 0.975)     # stated exception to the full-axis rule
    ax.set_xlabel("coverage (fraction of queries answered)")
    ax.set_ylabel("selective accuracy")
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticks([0.6, 0.7, 0.8, 0.9])

    style.save(fig, "fig_selective")
    print(f"  ({rc['n_seeds']} seeds, T={rc['temperature']:.2f}, "
          f"ECE {rc['ece_raw']:.3f}->{rc['ece_cal']:.3f})")


if __name__ == "__main__":
    build()
