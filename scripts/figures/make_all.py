"""Build every paper figure. Run from the repo root, in an env with matplotlib.

    python scripts/figures/make_all.py [--only fig_dose_response]
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys

FIGURES = [
    "fig_gap_anatomy",
    "fig_bucket_gap",
    "fig_dose_response",
    "fig_dynamics",
    "fig_grokking_curves",
    "fig_initscale",
    "fig_selective",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None, help="subset of module names")
    args = ap.parse_args()

    if not os.path.isdir("artefacts/runs"):
        sys.exit("run from the repo root (artefacts/runs not found)")

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    targets = args.only or FIGURES
    failed = []
    for name in targets:
        print(f"[{name}]")
        try:
            importlib.import_module(name).build()
        except Exception as e:                       # keep going; report at the end
            print(f"  FAILED: {type(e).__name__}: {e}")
            failed.append(name)
    print()
    print(f"{len(targets) - len(failed)}/{len(targets)} built"
          + (f" -- failed: {failed}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
