"""Figure: the canonical grokking curves the paper never showed.

NEW -- there is no equivalent TikZ figure. The paper's entire framing is
grokking, and the compiled draft contains no train-vs-test-over-log-steps
plot: the one figure every paper in this lineage (Power, Nanda, Omnigrok,
Varma) opens with. `fig:dynamics` shows the constrained-posterior ladder
metric on one seed; this shows the *primary* metric (greedy exact match)
across all five, which is what the tables report.

Panel (b) also discharges a standing debt: the paper invokes Omnigrok's
weight-norm account but never plots a weight norm. What the data actually
shows is compress-then-expand, NOT the textbook rise-then-cleanup-drop --
and the compression minimum lands on the Witness grok step. Report the shape
observed, not the shape expected.
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import matplotlib.pyplot as plt
import style, data


def build():
    style.apply()
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(style.COL_IN, 3.5),
                                 sharex=True, constrained_layout=True)

    # ---------------- (a) train vs test ---------------------------------
    ceil_lbl = style.ceiling_line(a1, data.CEILING_TEMPLATE)
    for arm in ("Single-Form", "Witness"):
        cfg = style.ARM[arm]
        s, m, lo, hi = data.curves(arm, "test_exact")
        a1.fill_between(s, lo, hi, color=cfg["color"], alpha=0.15, lw=0)
        a1.plot(s, m, color=cfg["color"], ls=cfg["ls"], marker=cfg["marker"],
                ms=2.2, markevery=5, label=f"{arm} (test)")
    # both train curves, drawn separately: they are near-identical (max
    # divergence 0.129, a transient at step 35) but reach 1.0 at different
    # steps (208 vs 106), so one curve cannot honestly stand for both
    mems = []
    for arm in ("Single-Form", "Witness"):
        s, m, _, _ = data.curves(arm, "train_exact")
        a1.plot(s, m, color=style.INK, ls=(0, (1, 1.5)), lw=0.9, alpha=0.55,
                label="train (both arms)" if arm == "Single-Form" else None)
        mems.append(next(st for st, v in zip(s, m) if v >= 0.999))
    # sits in the empty band between the test curves and the ceiling rule;
    # at y=1.0 it collided with the ceiling label
    a1.annotate(f"train$=$1.0 by step {max(mems)}", (max(mems), 1.0),
                textcoords="offset points", xytext=(4, -40), fontsize=style.ANNOT, color="#555")

    a1.set_xscale("log"); a1.set_ylim(0, 1.02)
    a1.set_ylabel("exact-root accuracy")
    a1.legend(loc="lower right", handlelength=1.8)
    a1.text(1.05e6, data.CEILING_TEMPLATE + 0.02, ceil_lbl, ha="right",
            fontsize=style.ANNOT, color=style.CEILRED)
    a1.set_title("(a) same memorisation, different second rise", loc="left", fontsize=style.TITLE)

    # ---------------- (b) weight norm -----------------------------------
    for arm in ("Single-Form", "Witness"):
        cfg = style.ARM[arm]
        s, m, lo, hi = data.curves(arm, "weight_norm")
        a2.fill_between(s, lo, hi, color=cfg["color"], alpha=0.15, lw=0)
        a2.plot(s, m, color=cfg["color"], ls=cfg["ls"], marker=cfg["marker"],
                ms=2.2, markevery=5, label=arm)
        imin = min(range(len(m)), key=lambda i: m[i])
        a2.plot([s[imin]], [m[imin]], marker="v", color=cfg["color"], ms=5,
                markeredgecolor="white", markeredgewidth=0.4)
        # Witness above-left, Single-Form below-right: the old stacked offsets
        # put the teal label on the Witness curve.
        dxy = (4, -13) if arm == "Single-Form" else (-34, 8)
        a2.annotate(f"min @{s[imin]:,}", (s[imin], m[imin]), textcoords="offset points",
                    xytext=dxy, fontsize=style.ANNOT, color=cfg["color"])
    # Both minima fall inside this band, which is the point of drawing it:
    # the witness minimum sits at a grok step, but so does Single-Form's, and
    # Single-Form never groks. The coincidence is not evidence of alignment.
    # y=0.50: the band's mid-height is empty (curves sit near 0.1 there, the
    # legend ends around 0.72); at 0.90 the label touched "Single-Form"
    style.grok_band(a2, data.grok_steps("Witness"), label="grok (5 seeds)", y=0.50)
    a2.margins(y=0.22)          # headroom for the two min annotations
    a2.set_xscale("log"); a2.set_xlim(1, 1.1e6)
    a2.set_xlabel("training step"); a2.set_ylabel("parameter $L_2$ norm")
    a2.legend(loc="upper left", handlelength=1.8)
    a2.set_title("(b) compress, then expand", loc="left", fontsize=style.TITLE)

    style.save(fig, "fig_grokking_curves")


if __name__ == "__main__":
    build()
