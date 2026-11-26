"""Circuit analysis over a run's checkpoint ladder.

Two products, both keyed to the grokking transition:
1. CURVES — per-bucket test accuracy (weak-radical HIDDEN vs VISIBLE) at every
   ladder checkpoint: does the hidden bucket's rise coincide with the grok
   step while the visible bucket rises earlier/smoothly?
2. ABLATIONS — at pre-grok / grok / post-grok / final checkpoints, zero each
   attention head (out_proj column surgery) and each FFN, and measure the
   per-bucket damage. The story's prediction: some component's ablation
   selectively destroys HIDDEN-bucket accuracy, and that selectivity is absent
   pre-grok — the restoration mechanism "comes online" at the transition.

Usage (GPU strongly recommended):
  python scripts/analysis/circuit_analysis.py --run-dir <dir with ckpts/ + history.json>
      --split <split.json> --out analysis.json [--subsample 800]
"""
from __future__ import annotations

import argparse
import copy
import glob
import json
import os
import re

import torch

from grokkingroots.data.arabic_text import strip_diacritics
from grokkingroots.data.encoding import build_vocab, greedy_decode, load_split, max_input_len
from grokkingroots.model.char_transformer import CharTransformer

DELIM = "؞"
HAMZA = {"ء": "ءأإؤئآ", "أ": "ءأإؤئآ", "إ": "ءأإؤئآ", "ؤ": "ءأإؤئآ",
         "ئ": "ءأإؤئآ", "آ": "ءأإؤئآ"}


def weak_of(root_spaced, rtype):
    r = root_spaced.split()
    if rtype == "hollow":
        return r[1]
    if rtype == "defective":
        return r[2]
    if rtype == "assimilated":
        return r[0]
    if rtype == "hamzated":
        for x in r:
            if x in HAMZA:
                return x
    return None


def shows(w, s):
    if w is None:
        return True
    if w in HAMZA:
        return any(c in s for c in HAMZA[w])
    if w == "ي":
        return "ي" in s or "ى" in s
    return w in s


def bucket_tests(test_pairs, dataset_parquet):
    """Map each test item -> 'hidden' | 'visible' | 'all' (no parquet info)."""
    try:
        import pandas as pd
        df = pd.read_parquet(dataset_parquet)
        by_root = {}
        for rc, rt, root in zip(df["root_clean"], df["root_type"], df["root"]):
            by_root.setdefault(rc, (rt, root))
    except Exception:
        by_root = {}
    buckets = []
    for s, r in test_pairs:
        info = by_root.get(r)
        if not info:
            buckets.append("all")
            continue
        rt, root = info
        w = weak_of(root, rt)
        if w is None:
            buckets.append("visible")
            continue
        query = strip_diacritics(s.split(DELIM)[0])
        buckets.append("visible" if shows(w, query) else "hidden")
    return buckets


def bucket_acc(preds, golds, buckets):
    out = {}
    for b in sorted(set(buckets)):
        idx = [i for i, x in enumerate(buckets) if x == b]
        out[b] = round(sum(preds[i] == golds[i] for i in idx) / len(idx), 4)
    out["overall"] = round(sum(p == g for p, g in zip(preds, golds)) / len(golds), 4)
    return out


def components(n_layers, n_heads):
    for l in range(n_layers):
        for h in range(n_heads):
            yield f"L{l}H{h}"
        yield f"L{l}FFN"


def ablate(model, name, d_model, n_heads):
    """Return a copy of model with the named component zeroed."""
    m = copy.deepcopy(model)
    l = int(name[1])
    dh = d_model // n_heads
    layer = m.blocks.layers[l]
    with torch.no_grad():
        if name.endswith("FFN"):
            layer.linear2.weight.zero_()
            if layer.linear2.bias is not None:
                layer.linear2.bias.zero_()
        else:
            h = int(name.split("H")[1])
            layer.self_attn.out_proj.weight[:, h * dh:(h + 1) * dh] = 0
    return m


def grok_step_of(history):
    peak = max(r["test_exact"] for r in history)
    if peak <= 0.05:
        return None
    mem = next((i for i, r in enumerate(history) if r.get("train_exact", 0) > 0.99), None)
    if mem is None or history[mem]["test_exact"] >= 0.5 * peak:
        return None
    for r in history[mem + 1:]:
        if r["test_exact"] >= 0.5 * peak:
            return r["step"]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--dataset", default="artefacts/phase0/roots_dataset_enrclean.parquet")
    ap.add_argument("--out", required=True)
    ap.add_argument("--subsample", type=int, default=800)
    ap.add_argument("--device", default=None)
    ap.add_argument("--ablate-steps", default=None,
                    help="comma-separated wanted steps; each snaps to the nearest "
                         "ladder ckpt (final always added). Overrides the "
                         "grok-derived epochs — needed to give a never-grokking "
                         "run (k1) the SAME epochs as its grokking contrast (reveal).")
    ap.add_argument("--curve-stride", type=int, default=1,
                    help="evaluate curves on every Nth ladder ckpt (smoke: 20)")
    args = ap.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    tr, va, te, _ = load_split(args.split)
    allp = tr + va + te
    vocab = build_vocab(allp)
    max_s = max(len(s) for s, _ in allp)
    mlen = max(max_input_len(allp), max_s + 5 + 3)
    meta = json.load(open(os.path.join(args.run_dir, "meta.json")))["args"]
    hist = json.load(open(os.path.join(args.run_dir, "history.json")))
    gs = grok_step_of(hist)
    print(f"grok step: {gs}")

    # fixed evaluation subsample (deterministic: first N)
    te_s = te[:args.subsample]
    surfaces = [s for s, _ in te_s]
    golds = [r for _, r in te_s]
    buckets = bucket_tests(te_s, args.dataset)
    from collections import Counter
    print("buckets:", dict(Counter(buckets)))

    def make_model():
        return CharTransformer(len(vocab), meta["d_model"], meta["n_heads"],
                               meta["n_layers"], meta["d_ff"], 0.0,
                               max_len=mlen, pad_id=vocab.pad).to(device)

    ladder = sorted(glob.glob(os.path.join(args.run_dir, "ckpts", "ckpt_*.pt")),
                    key=lambda p: int(re.search(r"ckpt_(\d+)\.pt", p).group(1)))
    print(f"ladder: {len(ladder)} checkpoints")

    # ---- 1. curves ----
    curves = []
    model = make_model()
    curve_ladder = ladder[::args.curve_stride]
    if ladder and ladder[-1] not in curve_ladder:
        curve_ladder.append(ladder[-1])
    for p in curve_ladder:
        step = int(re.search(r"ckpt_(\d+)\.pt", p).group(1))
        sd = torch.load(p, map_location=device, weights_only=False)
        model.load_state_dict(sd["model"] if "model" in sd else sd)
        preds = greedy_decode(model, vocab, surfaces, device)
        curves.append({"step": step, **bucket_acc(preds, golds, buckets)})
        print(f"  step {step:>8}: " + " ".join(f"{k}={v}" for k, v in curves[-1].items() if k != "step"))

    # ---- 2. ablations at pre-grok / grok / post-grok / final ----
    all_steps = [int(re.search(r"ckpt_(\d+)\.pt", p).group(1)) for p in ladder]
    targets = []
    if args.ablate_steps:
        for want in [int(x) for x in args.ablate_steps.split(",")]:
            targets.append(min(all_steps, key=lambda s: abs(s - want)))
    elif gs:
        for want in [max(1, gs // 4), gs, gs * 3]:
            targets.append(min(all_steps, key=lambda s: abs(s - want)))
    targets.append(all_steps[-1])
    targets = sorted(set(targets))
    print(f"ablation checkpoints: {targets}")

    ablations, selectivity = {}, {}
    for tstep in targets:
        p = ladder[all_steps.index(tstep)]
        sd = torch.load(p, map_location=device, weights_only=False)
        model.load_state_dict(sd["model"] if "model" in sd else sd)
        base = bucket_acc(greedy_decode(model, vocab, surfaces, device), golds, buckets)
        rows = {"baseline": base}
        for comp in components(meta["n_layers"], meta["n_heads"]):
            am = ablate(model, comp, meta["d_model"], meta["n_heads"])
            rows[comp] = bucket_acc(greedy_decode(am, vocab, surfaces, device), golds, buckets)
            del am
        ablations[str(tstep)] = rows
        sel = {c: round(base.get("hidden", 0) - r.get("hidden", 0)
                        - (base.get("visible", 0) - r.get("visible", 0)), 4)
               for c, r in rows.items() if c != "baseline"}
        selectivity[str(tstep)] = sel
        top = sorted(sel.items(), key=lambda kv: -kv[1])[:3]
        print(f"  step {tstep}: most hidden-selective ablations: {top}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump({"run_dir": args.run_dir, "split": args.split, "grok_step": gs,
               "subsample": len(te_s), "buckets": dict(Counter(buckets)),
               "curves": curves, "ablations": ablations, "selectivity": selectivity},
              open(args.out, "w"), ensure_ascii=False, indent=1)
    print(f"wrote {args.out}")
    print("CIRCUIT_ACTV_DONE", flush=True)


if __name__ == "__main__":
    main()
