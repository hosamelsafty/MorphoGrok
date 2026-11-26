"""Shared house style for the paper figures.

One place for every visual decision, so a change is made once rather than
argued per-figure. Mirrors the \\definecolor block of the paper's LaTeX source.

Grayscale rule: no series may be distinguished by colour alone. Every series
carries a marker shape AND a linestyle in addition to its colour -- three
review rounds flagged colour-only encoding.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")            # headless; must precede pyplot
import matplotlib.pyplot as plt  # noqa: E402

# acl.sty, 11pt, A4, two-column: \columnwidth 7.7cm, \textwidth 16cm
COL_IN = 3.03       # \columnwidth  -> figure
TEXT_IN = 6.30      # \textwidth    -> figure*

# Type sizes in points, and they mean points ON THE PAGE: save() normalises the
# saved width to the include width, so \includegraphics does not rescale them.
# ANNOT is the floor. The paper's own smallest type is 7pt (math subscripts);
# 8pt is \scriptsize, used for the CI brackets in tables. Nothing in a figure
# should be smaller than the smallest thing in the body text.
ANNOT = 7.0         # in-plot annotations and direct labels
TICK = 7.5          # axis tick labels
LABEL = 8.0         # axis labels
LEGEND = 7.5
TITLE = 8.5         # panel titles ((a), (b), ...)

# identical RGB to main.tex's \definecolor
TEAL = "#2c6072"
AMBER = "#c9772f"
CEILRED = "#9a2a2a"
CTRLGRAY = "#9aa3a7"
INK = "#222222"

ARM = {
    "Single-Form": dict(color=TEAL,     marker="o", ls="-"),
    "Support-1":   dict(color=TEAL,     marker="o", ls="-"),
    "Support-2":   dict(color=TEAL,     marker="o", ls="-"),
    "Witness":     dict(color=AMBER,    marker="s", ls="--"),
    "Repeat":      dict(color=CTRLGRAY, marker="^", ls=":"),
    "Cross-Root":  dict(color=CTRLGRAY, marker="v", ls=":"),
    "Mixed-k":     dict(color=CTRLGRAY, marker="D", ls="-."),
}
BUCKET = {
    "visible": dict(color=TEAL,  marker="o", ls="-",  label="radical visible"),
    "hidden":  dict(color=AMBER, marker="^", ls="--", label="radical hidden"),
}


def apply():
    plt.rcParams.update({
        "pdf.fonttype": 42,          # real embedded fonts, not Type-3 bitmaps
        "ps.fonttype": 42,
        "font.family": "serif",      # matches texgyretermes in the paper
        "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
        "font.size": LABEL,
        "axes.labelsize": LABEL,
        "axes.titlesize": TITLE,
        "xtick.labelsize": TICK,
        "ytick.labelsize": TICK,
        "legend.fontsize": LEGEND,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "grid.linestyle": "--",
        "legend.frameon": False,
        "lines.linewidth": 1.3,
        "lines.markersize": 3.0,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.01,
    })


def ceiling_line(ax, y=0.890, label="computable ceiling $C_\\varphi$=0.890", horizontal=True):
    """The 0.890 reference. Always dashed CEILRED, always labelled the same."""
    f = ax.axhline if horizontal else ax.axvline
    f(y, color=CEILRED, ls=(0, (5, 2)), lw=1.1, zorder=1)
    return label


def grok_band(ax, steps, label=None, y=0.03):
    """Shade the min--max range of per-seed grok steps. `y` is axes fraction.

    A single vline is only honest on a single-seed figure (fig_dynamics). On
    the five-seed figures the seeds span two orders of magnitude, so drawing
    one seed's step as *the* grok step overstates how sharp the landmark is.
    """
    if not steps:
        return
    lo, hi = min(steps), max(steps)
    ax.axvspan(lo, hi, color=TEAL, alpha=0.10, lw=0, zorder=1)
    for x in (lo, hi):
        ax.axvline(x, color=TEAL, ls=(0, (1, 2)), lw=0.7, alpha=0.55, zorder=1)
    if label:
        # x in data coords (geometric mid-band, since the axis is log),
        # y in axes fraction so the caller need not know the y units
        ax.text((lo * hi) ** 0.5, y, label, fontsize=ANNOT, color=TEAL,
                ha="center", va="bottom", linespacing=0.95,
                transform=ax.get_xaxis_transform())


def _fit_width(fig, target_in):
    """Rescale figsize until the tight-bbox output is exactly `target_in` wide.

    savefig(bbox_inches="tight") trims to content, so the saved width is NOT
    figsize -- the two dot plots, whose long y labels extend past the axes,
    came out 15-21% wider than the column. \\includegraphics[width=\\columnwidth]
    then shrank the whole figure, fonts included, so a nominal 7pt tick reached
    the page anywhere between 5.8pt and 7.6pt depending on which figure it was
    in. Fitting the width here makes the include a no-op scale, which is what
    lets the sizes above be stated in real points.
    """
    pad = plt.rcParams["savefig.pad_inches"]
    for _ in range(6):
        fig.canvas.draw()
        w = fig.get_tightbbox(fig.canvas.get_renderer()).width + 2 * pad
        if abs(w - target_in) < 0.004:
            break
        s = target_in / w
        sw, sh = fig.get_size_inches()
        fig.set_size_inches(sw * s, sh * s)


def save(fig, name, outdir="figures", target_in=None):
    """Write the figure as PDF and PNG into `outdir`.

    `target_in` is the width the figure is included at; it defaults to whichever
    of COL_IN/TEXT_IN the figure was built for.
    """
    import os
    if target_in is None:
        target_in = TEXT_IN if fig.get_size_inches()[0] > (COL_IN + TEXT_IN) / 2 else COL_IN
    _fit_width(fig, target_in)
    os.makedirs(outdir, exist_ok=True)
    pdf = os.path.join(outdir, name + ".pdf")
    png = os.path.join(outdir, name + ".png")
    fig.savefig(pdf)                 # vector, for \includegraphics
    fig.savefig(png, dpi=200)        # raster, for quick eyeballing / review
    w, h = fig.get_size_inches()
    plt.close(fig)
    print(f"  wrote {pdf} and {png}  (fitted to {target_in:.2f}in, canvas {w:.2f}x{h:.2f}in)")
