"""Figure: anatomy of the learning gap -- every system on one accuracy scale.

REPLACES the TikZ `fig:ladder`, which put ~11 labels on a single horizontal
axis and collided at every width it was ever tried at (main body, appendix
column, full width). The fix is structural rather than cosmetic: rotate the
axis. One system per ROW means labels cannot collide by construction, no
matter how many systems are added or how the column width changes.

Reading: the shaded band is the greedy learning gap. Everything left of it is
what the field's tooling and the model actually achieve; the right edge is
what the input provably permits.
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import matplotlib.pyplot as plt
import style, data

GREEDY = 0.563


def build():
    style.apply()
    d = data
    # (label, value, group). Order is bottom-to-top in the plot.
    rows = [
        ("skeleton (copy consonants)", 0.000, "classical"),
        ("CAMeL Tools",                0.000, "classical"),
        ("compat $C_\\varphi$ (evidence-poor)", 0.151, "ceiling"),
        ("ISRI",                       0.333, "classical"),
        ("Tashaphyne",                 0.338, "classical"),
        ("supervised, encoder-decoder", d.BASELINE_ACC["encoder-decoder"], "baseline"),
        ("Qalsadi",                    0.421, "classical"),
        ("supervised, MorphoGrok arch.", d.BASELINE_ACC["MorphoGrok architecture"], "baseline"),
        ("MorphoGrok, greedy",         GREEDY, "model"),
        ("MorphoGrok, constrained",    d.MODEL_CONSTRAINED_TOP1, "model"),
        ("+ Support-2 (2 same-root forms)", d.PAPER_ACC["Support-2"][0], "evidence"),
        ("+ Witness (radical shown)",   d.PAPER_ACC["Witness"][0], "evidence"),
        ("template $C_\\varphi$ (closed form)", d.CEILING_TEMPLATE, "ceiling"),
        ("audit bound (oracle radical)", d.CEILING_AUDIT, "ceiling"),
    ]
    # no reversal: the list above is already bottom-to-top, so the ladder
    # ascends to the ceilings at the top (the [::-1] here inverted it)

    colour = {"classical": style.CTRLGRAY, "baseline": style.TEAL, "model": style.TEAL,
              "evidence": style.AMBER, "ceiling": style.CEILRED}
    marker = {"classical": "s", "baseline": "^", "model": "o", "evidence": "D", "ceiling": "*"}

    # 3.15in carried 12 rows; same row pitch for 14
    fig, ax = plt.subplots(figsize=(style.COL_IN, 3.15 * len(rows) / 12))

    # the gap itself, drawn first so everything sits on top of it
    ax.axvspan(GREEDY, d.CEILING_TEMPLATE, color=style.AMBER, alpha=0.13, lw=0, zorder=0)
    ax.annotate("", xy=(d.CEILING_TEMPLATE, len(rows) - 0.35), xytext=(GREEDY, len(rows) - 0.35),
                arrowprops=dict(arrowstyle="<->", color=style.AMBER, lw=0.9))
    ax.text((GREEDY + d.CEILING_TEMPLATE) / 2, len(rows) - 0.05,
            "learning gap 0.33", ha="center", va="bottom",
            fontsize=style.TICK, color="#8a5220")

    for i, (label, val, grp) in enumerate(rows):
        # lollipop stem: makes magnitude readable at a glance
        ax.plot([0, val], [i, i], color=colour[grp], lw=0.8, alpha=0.45, zorder=2)
        ax.plot([val], [i], marker=marker[grp], color=colour[grp],
                ms=6 if grp == "ceiling" else 4.2, zorder=3, clip_on=False,
                markeredgecolor="white", markeredgewidth=0.4)
        ax.text(val + 0.022, i, f"{val:.3f}", va="center", ha="left",
                fontsize=style.ANNOT, color=colour[grp])

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=style.TICK)
    ax.set_xlim(0, 1.06)
    ax.set_ylim(-0.7, len(rows) - 0.1)
    ax.set_xlabel("exact-root accuracy (frozen L2 test)")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", alpha=0.25)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    style.save(fig, "fig_gap_anatomy")


if __name__ == "__main__":
    build()
