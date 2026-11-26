"""Figure: fast-fit-and-stall vs memorise-grok-converge.

REPLACES the TikZ `fig:dynamics`. Beyond a port, this adds *phase shading* --
memorisation / stall / second rise -- so the qualitative claim ("the failure
mode is not 'groks slowly', it is 'never groks'") is legible from the shape
alone rather than only from the caption.

Single seed (s0) by construction: ladder_posterior.json is only computed for
Single-Form s0 and Witness s0. The panel titles say so; do not let a reader
mistake this for a seed-averaged result.
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import matplotlib.pyplot as plt
import style, data

# Seed s0's grok step. A single vline is correct HERE and only here: this
# figure is s0 by construction (see the docstring). The five-seed figures draw
# style.grok_band() instead, because grok steps span 133--7,306 across seeds.
GROK_STEP = 3001


def build():
    style.apply()
    d = data
    sf = d.ladder_posterior("Single-Form")
    wt = d.ladder_posterior("Witness")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(style.TEXT_IN, 2.15), constrained_layout=True)

    # ---------------- (a) constrained top-1 -----------------------------
    a1.axvspan(1, 130, color="#000000", alpha=0.045, lw=0)
    a1.axvspan(130, GROK_STEP, color="#000000", alpha=0.02, lw=0)
    a1.text(11, 0.03, "memorise", fontsize=style.ANNOT, color="#666", ha="center")
    a1.text(620, 0.03, "plateau", fontsize=style.ANNOT, color="#666", ha="center")
    a1.text(1.2e5, 0.03, "second rise (Witness only)", fontsize=style.ANNOT, color="#8a5220", ha="center")

    ceil_lbl = style.ceiling_line(a1, d.CEILING_TEMPLATE)
    for traj, name in ((sf, "Single-Form"), (wt, "Witness")):
        cfg = style.ARM[name]
        a1.plot([p["step"] for p in traj], [p["top1"] for p in traj],
                color=cfg["color"], ls=cfg["ls"], marker=cfg["marker"],
                ms=2.6, label=name)
    a1.axvline(GROK_STEP, color=style.TEAL, ls=(0, (1, 2)), lw=0.9)
    a1.annotate("grok", (GROK_STEP, 0.13), textcoords="offset points", xytext=(3, 0),
                fontsize=style.ANNOT, color=style.TEAL)
    a1.set_xscale("log"); a1.set_xlim(1, 1.1e6); a1.set_ylim(0, 1.0)
    a1.set_xlabel("training step"); a1.set_ylabel("constrained top-1")
    a1.set_title("(a) stall vs. grok-and-converge  (seed s0)", loc="left", fontsize=style.TITLE)
    # upper left is the only region with no curve, phase label, or ceiling rule;
    # lower right collided with the "second rise" phase annotation
    a1.legend(loc="upper left", bbox_to_anchor=(0.01, 0.86), handlelength=1.8)
    a1.text(1.05e6, d.CEILING_TEMPLATE + 0.015, ceil_lbl, ha="right",
            fontsize=style.ANNOT, color=style.CEILRED)

    # ---------------- (b) calibration temperature ------------------------
    a2.axhline(1.0, color=style.INK, lw=0.9, alpha=0.65)
    a2.text(1.4, 1.06, "calibrated ($T{=}1$)", fontsize=style.ANNOT, color=style.INK)
    for traj, name in ((sf, "Single-Form"), (wt, "Witness")):
        cfg = style.ARM[name]
        a2.plot([p["step"] for p in traj], [p["T"] for p in traj],
                color=cfg["color"], ls=cfg["ls"], marker=cfg["marker"],
                ms=2.6, label=name)
    a2.axvline(GROK_STEP, color=style.TEAL, ls=(0, (1, 2)), lw=0.9)
    a2.set_xscale("log"); a2.set_xlim(1, 1.1e6); a2.set_ylim(0.5, 6.0)
    a2.set_xlabel("training step"); a2.set_ylabel("fitted temperature $T_t$")
    a2.set_title("(b) both arms drift from calibrated; Witness less", loc="left", fontsize=style.TITLE)
    a2.legend(loc="upper left", handlelength=1.8)

    style.save(fig, "fig_dynamics")


if __name__ == "__main__":
    build()
