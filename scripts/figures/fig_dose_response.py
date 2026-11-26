"""Figure: the information-supply law -- dose-response with matched placebos.

REPLACES the TikZ `fig:dose`. Second redesign (2026-08-03): one condition per
ROW, same structural fix as fig_gap_anatomy. The vertical dot-plot version
fought its own content -- rotated tick labels, a half-empty 0-1 y-axis, and
the two Witness budgets dodged sideways because a categorical x-axis has
nowhere to put a second point. Rows solve all three by construction: labels
sit horizontal at full size, the 0-1 axis lies along the reading direction so
stems fill it, and the second Witness budget is simply its own row.

Design choices carried over from review, enforced in code rather than by hand:
  * full 0-1 accuracy axis (the original truncated at 0.45, exaggerating
    every difference);
  * one marker shape per group, so colour is never load-bearing; groups are
    also named in the row labels, so no legend is needed;
  * both Witness budgets shown -- open square 500k, filled square 10^6;
  * the learning-gap band (baseline..ceiling) shaded as in fig_gap_anatomy,
    so "the dose progressively fills the gap" is the visible story;
  * placebos in their own row group under a "placebos" header, so the eye
    separates "more evidence" from "more text".
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import matplotlib.pyplot as plt
import style, data

# headline scalars come from data.PAPER_ACC (full 2,109-item test set), not
# from the history prefix -- see the data module docstring


def build():
    style.apply()
    d = data
    base = d.PAPER_ACC["Single-Form"][0]
    ceil = d.CEILING_TEMPLATE

    # (label, value, ci, colour, marker, open_marker, y). Top-to-bottom:
    # ascending dose, then both Witness budgets, then the placebo group.
    w5, w5ci = d.WITNESS_500K
    rows = [
        ("Single-Form ($k{=}1$)", *d.PAPER_ACC["Single-Form"], style.TEAL, "o", False, 6.4),
        ("Support-1 ($k{=}2$)",   *d.PAPER_ACC["Support-1"],   style.TEAL, "o", False, 5.4),
        ("Support-2 ($k{=}3$)",   *d.PAPER_ACC["Support-2"],   style.TEAL, "o", False, 4.4),
        ("Witness ($k{=}3$), 500k",   w5, w5ci,                style.AMBER, "s", True, 3.4),
        ("Witness ($k{=}3$), $10^6$", *d.PAPER_ACC["Witness"], style.AMBER, "s", False, 2.4),
        ("Repeat ($k{=}3$)",      *d.PAPER_ACC["Repeat"],      style.CTRLGRAY, "^", False, 1.0),
        ("Cross-Root ($k{=}3$)",  *d.PAPER_ACC["Cross-Root"],  style.CTRLGRAY, "v", False, 0.0),
    ]

    fig, ax = plt.subplots(figsize=(style.COL_IN, 2.2))

    # the learning gap as a band, same treatment as fig_gap_anatomy: the dose
    # markers land progressively deeper inside it, the witness at its edge
    ax.axvspan(base, ceil, color=style.AMBER, alpha=0.10, lw=0, zorder=0)
    ax.axvline(base, color=style.CTRLGRAY, ls=":", lw=0.9, zorder=1)
    style.ceiling_line(ax, ceil, horizontal=False)
    # two lines so the label starts right of the baseline rule at 0.563 --
    # on one line it is wide enough that the dotted rule strikes through it
    ax.text(0.878, 7.0, "computable ceiling\n$C_\\varphi{=}0.890$", ha="right",
            va="center", fontsize=style.ANNOT, color=style.CEILRED, linespacing=1.15)
    # both labels live in the gap row between the arms and the placebos
    ax.text(0.005, 1.7, "placebos", fontsize=style.ANNOT, color="#666666", va="center")
    ax.text(0.553, 1.7, "baseline", fontsize=style.ANNOT, color="#6b7275",
            ha="right", va="center")

    for label, v, ci, c, m, openm, y in rows:
        ax.plot([0, v], [y, y], color=c, lw=0.8, alpha=0.45, zorder=2)
        ax.errorbar([v], [y], xerr=[ci], fmt=m, ms=4.2, color=c,
                    mfc="white" if openm else c,
                    markeredgecolor=c if openm else "white",
                    mew=1.1 if openm else 0.4,
                    ecolor=c, capsize=1.5, lw=1.0, zorder=3)
        # values as a right-hand column, past the ceiling rule, so no value
        # ever collides with a whisker, the band edge, or the rule itself
        ax.text(1.025, y, f"{v:.3f}", ha="right", va="center",
                fontsize=style.ANNOT, color=c)

    ax.set_yticks([r[-1] for r in rows])
    ax.set_yticklabels([r[0] for r in rows], fontsize=style.TICK)
    ax.set_ylim(-0.55, 7.45)
    ax.set_xlim(0, 1.03)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("exact-root accuracy")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", alpha=0.25)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    style.save(fig, "fig_dose_response")


if __name__ == "__main__":
    build()
