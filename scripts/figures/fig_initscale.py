"""Figure: L2 Single-Form at initialization scale 1 vs 3.

Backs the claim that the initialization scale decides whether generalization
is delayed, and not where training ends: same cell, same weight decay (0.05),
5 seeds each, only the scale differs. Colour is the condition (both are
Single-Form), so the two scales are told apart by linestyle and marker.

Held-out accuracy is the logged 2,000-item prefix, as in fig_grokking_curves.
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import matplotlib.pyplot as plt
import style, data

RUNS = [  # (arm key in data.ARMS, legend label, linestyle, marker)
    ("Single-Form",          "scale 1", "-",  "o"),
    ("Single-Form, scale 3", "scale 3", "--", "s"),
]


def build():
    style.apply()
    fig, ax = plt.subplots(figsize=(style.COL_IN, 2.2), constrained_layout=True)

    for arm, label, ls, mk in RUNS:
        s, m, lo, hi = data.curves(arm, "test_exact")
        ax.fill_between(s, lo, hi, color=style.TEAL, alpha=0.15, lw=0)
        ax.plot(s, m, color=style.TEAL, ls=ls, marker=mk, ms=2.2, markevery=5,
                label=f"held-out, {label}")
        s, m, _, _ = data.curves(arm, "train_exact")
        ax.plot(s, m, color=style.INK, ls=ls, lw=0.9, alpha=0.55,
                label=f"train, {label}")

    ax.set_xscale("log"); ax.set_xlim(1, 5.5e5); ax.set_ylim(0, 1.02)
    ax.set_xlabel("training step"); ax.set_ylabel("exact-root accuracy")
    # above the axes: inside, every free region is crossed by a curve
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
              handlelength=2.2, columnspacing=1.2)

    style.save(fig, "fig_initscale")


if __name__ == "__main__":
    build()
