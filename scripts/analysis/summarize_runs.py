"""Summarize grokking runs from their history.json (the grokking gap + finals).

Usage: python scripts/analysis/summarize_runs.py OUT_DIR [OUT_DIR ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def step_to(hist, key, thr):
    for h in hist:
        if h[key] >= thr:
            return h["step"]
    return None


def summarize(d):
    h = json.loads((Path(d) / "history.json").read_text())
    if not h:
        return {"dir": Path(d).name, "empty": True}
    final = h[-1]
    peak_test = max(x["test_exact"] for x in h)
    tr99 = step_to(h, "train_exact", 0.99)
    te_peak80 = step_to(h, "test_exact", 0.8 * peak_test) if peak_test > 0 else None
    gap = (te_peak80 / tr99) if (tr99 and te_peak80) else None
    return {
        "dir": Path(d).name,
        "steps_logged": final["step"],
        "train@99_step": tr99,
        "test@80%peak_step": te_peak80,
        "grok_gap_ratio": round(gap, 1) if gap else None,
        "final_train": round(final["train_exact"], 3),
        "final_val": round(final["val_exact"], 3),
        "final_test": round(final["test_exact"], 3),
        "peak_test": round(peak_test, 3),
        "final_wnorm": round(final["weight_norm"], 1),
    }


if __name__ == "__main__":
    for d in sys.argv[1:]:
        try:
            print(json.dumps(summarize(d), ensure_ascii=False))
        except Exception as e:  # noqa: BLE001
            print(json.dumps({"dir": d, "error": str(e)}, ensure_ascii=False))
