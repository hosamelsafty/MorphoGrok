"""Input-conditional calibration (extends calibrated_decode's single global T).

calibrated_decode.py fits ONE temperature T* on the whole val set. But the
model's overconfidence is not uniform: it is concentrated on the restoration
bucket (deleted radical), where the task is harder and the posterior is
flatter to begin with. This script fits a SEPARATE temperature per bucket
(substring / restoration) on that bucket's own val items, and checks whether
bucket-conditional calibration tightens ECE and the risk-coverage curve
beyond the single global scalar — a strict generalization, never worse by
construction (global T is the special case where both buckets share one fit).

Reuses the exact same scorer as constrained_topk_eval / calibrated_decode so
numbers cannot drift between scripts.

Run (CPU fine for this model size):
  python scripts/evaluation/conditional_calibration.py \
      --ckpt CKPT.pt --meta meta.json --split SPLIT.json \
      --inventory artefacts/phase0/root_type_map.json --out OUT.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from constrained_topk_eval import load_model, build_candidates, make_scorer
from calibrated_decode import fit_temperature, ece_15, score_matrix


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--meta", required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--inventory", default="artefacts/phase0/root_type_map.json")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default=None)
    p.add_argument("--max-items", type=int, default=0,
                   help="0 = all (recommended for reported numbers). NOTE: takes "
                        "a PREFIX of val/test, not a random sample -- val/test "
                        "rows are grouped contiguously by root (not i.i.d. "
                        "shuffled), so any nonzero value here is a biased "
                        "subsample of the frozen split. Smoke-testing only.")
    p.add_argument("--max-cands", type=int, default=0)
    p.add_argument("--batch-rows", type=int, default=2048)
    return p.parse_args()


def bucket_of(surface, gold):
    # match calibrated_decode.py's convention: bucket on the TARGET form only
    # (the one actually scored), not on a multi-form delimiter-joined surface.
    return "substring" if gold in surface.split("؞")[0] else "restoration"


def main():
    args = parse()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, vocab, (train_pairs, val_pairs, test_pairs) = load_model(
        args.meta, args.split, args.ckpt, device)
    cands, skipped = build_candidates(args.inventory, test_pairs, vocab, args.max_cands)
    c2i = {r: i for i, r in enumerate(cands)}
    score_item = make_scorer(model, vocab, cands, device, args.batch_rows)

    val = val_pairs[: args.max_items] if args.max_items else val_pairs
    test = test_pairs[: args.max_items] if args.max_items else test_pairs
    val = [(s, g) for s, g in val if g in c2i]

    print(f"scoring: {len(val)} val + {len(test)} test x {len(cands)} cands on {device}", flush=True)
    lp_val, gold_val = score_matrix(score_item, val, "val")
    gi_val = torch.tensor([c2i[g] for g in gold_val], device=lp_val.device)
    lp_te, gold_te = score_matrix(score_item, test, "test")
    order = lp_te.argsort(dim=1, descending=True)
    top1_pred = [cands[int(i)] for i in order[:, 0]]
    hit1 = [p == g for p, g in zip(top1_pred, gold_te)]

    val_bucket = [bucket_of(s, g) for s, g in val]
    te_bucket = [bucket_of(s, g) for s, g in test]

    def confidences(lp_rows, T):
        return F.softmax(lp_rows / T, dim=1).max(dim=1).values.tolist()

    # --- global T (reproduces calibrated_decode's number exactly when
    # --max-items is left at its default 0 -- NOT when a prefix is taken) ---
    T_glob, _ = fit_temperature(lp_val, gi_val)
    conf_glob = confidences(lp_te, T_glob)
    ece_glob = ece_15(conf_glob, hit1)
    exp_acc_glob = sum(conf_glob) / len(conf_glob)

    # --- per-bucket T (fit on that bucket's val items only) ---
    T_bucket = {}
    for b in ("substring", "restoration"):
        idx = [i for i, bb in enumerate(val_bucket) if bb == b]
        if len(idx) < 20:       # too few val items to fit a stable scalar
            T_bucket[b] = T_glob
            continue
        sub_lp = lp_val[idx]
        sub_gi = gi_val[idx]
        T_b, _ = fit_temperature(sub_lp, sub_gi)
        T_bucket[b] = T_b

    conf_cond = []
    for r in range(len(test)):
        T = T_bucket[te_bucket[r]]
        post = F.softmax(lp_te[r] / T, dim=0)
        conf_cond.append(float(post.max()))
    ece_cond = ece_15(conf_cond, hit1)
    exp_acc_cond = sum(conf_cond) / len(conf_cond)
    observed_acc = sum(hit1) / len(hit1)

    # risk-coverage under both calibration schemes
    def risk_coverage(conf):
        out = []
        n = len(conf)
        for tau in (0.3, 0.5, 0.7, 0.8, 0.9, 0.95):
            keep = [r for r in range(n) if conf[r] >= tau]
            if keep:
                out.append({"tau": tau, "coverage": round(len(keep) / n, 4),
                           "selective_acc": round(sum(hit1[r] for r in keep) / len(keep), 4)})
        return out

    result = {
        "ckpt": args.ckpt, "split": args.split, "n_val": len(val), "n_test": len(test),
        "n_cands": len(cands), "n_cands_skipped_oov": skipped,
        "observed_top1_acc": round(observed_acc, 4),
        "global": {
            "T": round(T_glob, 4), "ece": round(ece_glob, 4),
            "expected_acc": round(exp_acc_glob, 4),
            "expected_minus_observed": round(exp_acc_glob - observed_acc, 4),
            "risk_coverage": risk_coverage(conf_glob),
        },
        "per_bucket": {
            "T": {k: round(v, 4) for k, v in T_bucket.items()},
            "ece": round(ece_cond, 4),
            "expected_acc": round(exp_acc_cond, 4),
            "expected_minus_observed": round(exp_acc_cond - observed_acc, 4),
            "risk_coverage": risk_coverage(conf_cond),
        },
        "improvement": {
            "ece_reduction": round(ece_glob - ece_cond, 4),
            "ece_reduction_pct": round(100 * (ece_glob - ece_cond) / ece_glob, 1) if ece_glob else None,
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
