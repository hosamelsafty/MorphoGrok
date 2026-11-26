"""Figure: the stall IS the deleted-radical bucket, and evidence erases it.

REPLACES the TikZ `fig:buckets`. Two changes beyond a straight port:

1. The gap is *drawn as a filled region* between the visible and hidden
   curves, not left for the reader to subtract. The paper's claim is about
   the gap, so the gap should be the ink. Panel (c) then plots that gap
   directly for both arms -- the punchline in one line each.
2. All five seeds, not seed s0. The TikZ version plotted s0 while the text
   quoted five-seed means, a discrepancy that cost a review round. Reading
   from circuit_actv.json across seeds removes the possibility.
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import matplotlib.pyplot as plt
import style, data

def _panel(ax, arm, title, show_grok=False):
    b = data.bucket_curves(arm)
    s = b["steps"]
    vis, hid = b["visible"], b["hidden"]

    # the gap, as ink
    ax.fill_between(s, hid["mean"], vis["mean"], color=style.AMBER,
                    alpha=0.16, lw=0, zorder=1, label="gap")
    for key in ("visible", "hidden"):
        cfg = style.BUCKET[key]
        ax.fill_between(s, b[key]["lo"], b[key]["hi"], color=cfg["color"],
                        alpha=0.15, lw=0, zorder=2)
        ax.plot(s, b[key]["mean"], color=cfg["color"], ls=cfg["ls"],
                marker=cfg["marker"], ms=2.2, markevery=6, zorder=3,
                label=cfg["label"])
    if show_grok:
        style.grok_band(ax, data.grok_steps(arm), label="grok\n(5 seeds)", y=0.04)

    ax.set_xscale("log")
    ax.set_xlim(1, 1.1e6)
    ax.set_ylim(0, 0.95)
    ax.set_title(title, loc="left", fontsize=style.TITLE)
    ax.set_xlabel("training step")
    return b


def build():
    style.apply()
    fig, axes = plt.subplots(1, 3, figsize=(style.TEXT_IN, 1.95), constrained_layout=True)

    b1 = _panel(axes[0], "Single-Form", "(a) Single-Form: gap never closes")
    b2 = _panel(axes[1], "Witness", "(b) Witness: gap collapses", show_grok=True)
    axes[0].set_ylabel("exact-root accuracy")
    axes[0].legend(loc="upper left", handlelength=1.6)

    # (c) the gap itself -- the claim, stated directly
    ax = axes[2]
    ax.axhline(0, color=style.INK, lw=0.6, alpha=0.5)
    ax.plot(b1["steps"], b1["gap"], color=style.CTRLGRAY, ls="-", marker="o",
            ms=2.2, markevery=6)
    ax.plot(b2["steps"], b2["gap"], color=style.AMBER, ls="--", marker="s",
            ms=2.2, markevery=6)
    style.grok_band(ax, data.grok_steps("Witness"), label=False)
    # Direct labels instead of a legend: any legend box in this panel gets
    # struck through by one of the curves. Both go in the right-hand clear
    # space -- above the flat Single-Form gap, and in the wedge opened up by
    # the Witness collapse. Colour ties each to its curve.
    ax.text(1.0e6, 0.50, f"Single-Form\nends {b1['gap'][-1]:+.2f}", va="top",
            ha="right", fontsize=style.ANNOT, color="#6b7275")
    # below the Witness plateau, not right of it: right-aligned at 1e6 the
    # second line ran straight through the curve's collapse
    ax.text(200, 0.115, f"Witness\nends {b2['gap'][-1]:+.2f}", va="top",
            ha="left", fontsize=style.ANNOT, color="#8a5220")
    ax.set_xscale("log")
    ax.set_xlim(1, 1.1e6)
    ax.set_xlabel("training step")
    ax.set_ylabel("visible $-$ hidden")
    ax.set_title("(c) the gap, directly", loc="left", fontsize=style.TITLE)
    ax.set_ylim(-0.03, 0.52)     # headroom for the Single-Form label

    n = b1["n_seeds"]
    fig.suptitle("", y=0)   # keep constrained_layout from clipping
    style.save(fig, "fig_bucket_gap")
    print(f"  (built from {n} seeds per arm, {len(b1['steps'])}/{len(b2['steps'])} checkpoints)")


if __name__ == "__main__":
    build()
