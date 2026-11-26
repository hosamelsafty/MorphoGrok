"""Re-score val-selected checkpoints on the FULL frozen test set.

Training-time `test_exact` in history.json is computed on a PREFIX of the test
set (`--gen-subsample`, default 2000), while the ceiling/constrained/calibration
artefacts are computed on the full split. Every headline comparison in the paper
therefore mixes two denominators (e.g. greedy 0.564 on 2000 items vs ceiling
0.890 and constrained 0.639 on 2109).

This script removes that mismatch: for each run it finds the val-selected step
(argmax val_exact over history -- the project's reporting convention), snaps to
the nearest ladder checkpoint, and re-runs greedy decoding over ALL test items.
It reports the full-set number alongside the prefix number so the shift induced
by truncation is measurable rather than assumed.

Usage:
  python scripts/evaluation/fulltest_eval.py --runs 'artefacts/runs/g5_*' \
      --out artefacts/runs/fulltest_eval.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

import torch

from grokkingroots.data.encoding import build_vocab, greedy_decode, load_split, max_input_len
from grokkingroots.model.char_transformer import CharTransformer


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="glob over run dirs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prefix", type=int, default=2000,
                    help="the training-time gen_subsample, for the comparison column")
    ap.add_argument("--max-step", type=int, default=0,
                    help="0 = select over the whole run. Otherwise restrict "
                         "val-selection to steps <= this, i.e. report what the "
                         "run would have yielded under a smaller training "
                         "budget -- needed for the 500k-vs-1M convergence "
                         "comparison on runs that were later extended.")
    ap.add_argument("--device", default=None)
    return ap.parse_args()


def val_selected_step(hist, max_step=0):
    rows = [r for r in hist if not max_step or r["step"] <= max_step]
    if not rows:
        raise ValueError(f"no logged step <= {max_step}")
    return max(rows, key=lambda r: r["val_exact"])["step"]


def main():
    args = parse()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cache = {}                      # split path -> (vocab, mlen, test pairs)
    rows = []

    for run_dir in sorted(glob.glob(args.runs)):
        mp = os.path.join(run_dir, "meta.json")
        hp = os.path.join(run_dir, "history.json")
        if not (os.path.exists(mp) and os.path.exists(hp)):
            continue
        meta = json.load(open(mp))
        split = meta["split"]
        margs = meta["args"]
        hist = json.load(open(hp))

        if split not in cache:
            tr, va, te, _ = load_split(split)
            allp = tr + va + te
            vocab = build_vocab(allp)
            max_s = max(len(s) for s, _ in allp)
            mlen = max(max_input_len(allp), max_s + 5 + 3)
            cache[split] = (vocab, mlen, te)
        vocab, mlen, te = cache[split]

        want = val_selected_step(hist, args.max_step)
        ladder = glob.glob(os.path.join(run_dir, "ckpts", "ckpt_*.pt"))
        if not ladder:
            print(f"SKIP {os.path.basename(run_dir)}: no checkpoints", flush=True)
            continue
        steps = {int(re.search(r"ckpt_(\d+)\.pt", p).group(1)): p for p in ladder}
        got = min(steps, key=lambda s: abs(s - want))

        model = CharTransformer(len(vocab), margs["d_model"], margs["n_heads"],
                                margs["n_layers"], margs["d_ff"], 0.0,
                                max_len=mlen, pad_id=vocab.pad).to(device)
        sd = torch.load(steps[got], map_location=device, weights_only=False)
        state = sd["model"] if "model" in sd else sd
        # aux-roottype runs carry an auxiliary classification head used only in
        # the training loss; it takes no part in decoding, so drop it rather
        # than rebuild the module (any OTHER key mismatch still errors).
        state = {k: v for k, v in state.items() if not k.startswith("aux_head.")}
        model.load_state_dict(state)

        surfaces = [s for s, _ in te]
        golds = [r for _, r in te]
        preds = greedy_decode(model, vocab, surfaces, device)
        hit = [p == g for p, g in zip(preds, golds)]

        full = sum(hit) / len(hit)
        pref = sum(hit[: args.prefix]) / min(args.prefix, len(hit))
        reported = next(r["test_exact"] for r in hist if r["step"] == got) \
            if any(r["step"] == got for r in hist) else None

        rows.append({
            "run": os.path.basename(run_dir), "split": split,
            "val_selected_step": want, "scored_step": got,
            "max_step": args.max_step or None,
            "n_test_full": len(hit), "n_test_prefix": min(args.prefix, len(hit)),
            "test_exact_full": round(full, 4),
            "test_exact_prefix": round(pref, 4),
            "history_test_exact": reported,
            "full_minus_prefix": round(full - pref, 4),
        })
        print(f"{os.path.basename(run_dir):46s} step={got:>7} "
              f"full={full:.4f} prefix={pref:.4f} delta={full - pref:+.4f}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump({"n_runs": len(rows), "prefix": args.prefix, "runs": rows},
              open(args.out, "w"), ensure_ascii=False, indent=1)
    print(f"\nwrote {args.out}  ({len(rows)} runs)")
    print("FULLTEST_EVAL_DONE", flush=True)


if __name__ == "__main__":
    main()
