"""Plot grokking curves (exact-root train/val/test + weight norm) vs log-step.

Usage: python scripts/analysis/plot_run.py OUT_DIR [OUT_DIR ...] [--out FIG_DIR]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", default="artefacts/phase3/figures")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for d in args.dirs:
        hist_path = Path(d) / "history.json"
        if not hist_path.exists():
            print("skip (no history):", d)
            continue
        h = json.loads(hist_path.read_text())
        if not h:
            continue
        steps = [x["step"] for x in h]
        fig, ax1 = plt.subplots(figsize=(7, 4.5))
        ax1.plot(steps, [x["train_exact"] for x in h], label="train", color="tab:blue")
        ax1.plot(steps, [x["val_exact"] for x in h], label="val", color="tab:green")
        ax1.plot(steps, [x["test_exact"] for x in h], label="test (unseen roots)", color="tab:red")
        ax1.set_xscale("log")
        ax1.set_xlabel("optimizer step (log scale)")
        ax1.set_ylabel("exact-root accuracy")
        ax1.set_ylim(-0.02, 1.02)
        ax1.grid(True, which="both", alpha=0.3)
        ax1.legend(loc="center left")
        ax2 = ax1.twinx()
        ax2.plot(steps, [x["weight_norm"] for x in h], color="gray", ls="--", alpha=0.7)
        ax2.set_ylabel("weight norm (dashed)")
        ax1.set_title(Path(d).name)
        fig.tight_layout()
        p = out / f"{Path(d).name}.png"
        fig.savefig(p, dpi=110)
        plt.close(fig)
        print("wrote", p)


if __name__ == "__main__":
    main()
