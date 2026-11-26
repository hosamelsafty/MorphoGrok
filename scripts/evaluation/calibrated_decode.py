"""Calibration-aware constrained decoding.

Extends constrained_topk_eval: same shared scorer (one implementation, numbers
cannot drift), plus
  1. TEMPERATURE FIT on the VALIDATION split: T* = argmin_T NLL of the gold
     root under softmax(loglik / T) over the valid-root inventory. One scalar,
     fitted on val only — test is never touched during calibration.
  2. Calibrated TEST report: ECE(T=1) vs ECE(T*), mean p(gold), expected
     accuracy E[max posterior] vs observed top-1 (equal iff calibrated), and
     constrained top-1/3/5 (ranking is T-invariant — calibration changes
     confidence, never the argmax).
  3. SELECTIVE PREDICTION: with a calibrated posterior the decoder can abstain
     below a confidence floor; reports the risk-coverage curve (coverage +
     selective accuracy at thresholds) — the deployable face of the method.
  4. Per-item dump (conf, p_gold, hit, bucket) for figures.

Usage (GPU fast, CPU fine):
  python scripts/evaluation/calibrated_decode.py \
      --ckpt RUNS/<id>/ckpts/ckpt_<step>.pt --meta RUNS/<id>/meta.json \
      --split artefacts/phase0/splits_gen2_v2clean/L2_p30_k1.json \
      --out OUT.json [--ceiling-json artefacts/phase0/ceiling_k1v2c_compat.json]
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


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--meta", required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--inventory", default="artefacts/phase0/root_type_map.json")
    p.add_argument("--out", required=True)
    p.add_argument("--ceiling-json", default=None,
                   help="ceiling_method output to report saturation against")
    p.add_argument("--device", default=None)
    p.add_argument("--max-items", type=int, default=0, help="0 = all (smoke only)")
    p.add_argument("--max-cands", type=int, default=0, help="0 = full inventory (smoke only)")
    p.add_argument("--batch-rows", type=int, default=2048)
    return p.parse_args()


def score_matrix(score_item, pairs, tag):
    L, golds = [], []
    for i, (s, g) in enumerate(pairs):
        L.append(score_item(s.split("؞")[0]))
        golds.append(g)
        if (i + 1) % 200 == 0:
            print(f"  {tag}: {i + 1}/{len(pairs)}", flush=True)
    return torch.stack(L), golds


def fit_temperature(lp, gold_idx):
    """1-D NLL minimization over T (log-grid + golden refine). lp: [N, C]."""
    def nll(T):
        return float(F.cross_entropy(lp / T, gold_idx))
    Ts = torch.logspace(-1.3, 1.3, 53).tolist()
    T = min(Ts, key=nll)
    lo, hi = T / 1.35, T * 1.35
    for _ in range(40):                       # golden-section refine
        a = lo + 0.382 * (hi - lo); b = lo + 0.618 * (hi - lo)
        if nll(a) < nll(b):
            hi = b
        else:
            lo = a
    return (lo + hi) / 2, nll((lo + hi) / 2)


def ece_15(conf, hit):
    conf = torch.tensor(conf); hit = torch.tensor(hit, dtype=torch.float32)
    e = 0.0
    for b in range(15):
        m = (conf > b / 15) & (conf <= (b + 1) / 15)
        if int(m.sum()):
            e += float(m.float().mean() * abs(hit[m].mean() - conf[m].mean()))
    return e


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
    # val golds outside the candidate set can't be used for NLL fitting
    val = [(s, g) for s, g in val if g in c2i]

    print(f"scoring: {len(val)} val + {len(test)} test items x {len(cands)} cands "
          f"on {device}", flush=True)
    lp_val, gold_val = score_matrix(score_item, val, "val")
    gi_val = torch.tensor([c2i[g] for g in gold_val], device=lp_val.device)
    T, val_nll = fit_temperature(lp_val, gi_val)
    ece_val_raw = ece_15(*_conf_hit(lp_val, gi_val, 1.0))
    ece_val_cal = ece_15(*_conf_hit(lp_val, gi_val, T))
    print(f"T* = {T:.3f}  (val NLL {val_nll:.4f}; val ECE {ece_val_raw:.3f} -> {ece_val_cal:.3f})",
          flush=True)

    lp_te, gold_te = score_matrix(score_item, test, "test")
    order = lp_te.argsort(dim=1, descending=True)
    top1 = [cands[int(i)] for i in order[:, 0]]
    hit1 = [p == g for p, g in zip(top1, gold_te)]
    hit3 = [g in {cands[int(i)] for i in order[r, :3]} for r, g in enumerate(gold_te)]
    hit5 = [g in {cands[int(i)] for i in order[r, :5]} for r, g in enumerate(gold_te)]

    post_raw = F.softmax(lp_te, dim=1)
    post_cal = F.softmax(lp_te / T, dim=1)
    conf_raw = post_raw.max(dim=1).values.tolist()
    conf_cal = post_cal.max(dim=1).values.tolist()
    gi_te = [c2i.get(g, -1) for g in gold_te]
    pg_raw = [float(post_raw[r, i]) if i >= 0 else 0.0 for r, i in enumerate(gi_te)]
    pg_cal = [float(post_cal[r, i]) if i >= 0 else 0.0 for r, i in enumerate(gi_te)]

    n = len(test)
    buckets = {}
    items = []
    for r, (s, g) in enumerate(test):
        b = "substring" if g in s.split("؞")[0] else "restoration"
        bb = buckets.setdefault(b, {"n": 0, "top1": 0, "conf_cal": 0.0, "p_gold_cal": 0.0})
        bb["n"] += 1; bb["top1"] += hit1[r]
        bb["conf_cal"] += conf_cal[r]; bb["p_gold_cal"] += pg_cal[r]
        items.append({"surface": s.split("؞")[0], "gold": g, "pred": top1[r],
                      "hit": bool(hit1[r]), "conf_cal": round(conf_cal[r], 4),
                      "p_gold_cal": round(pg_cal[r], 4), "bucket": b})
    for bb in buckets.values():
        for k in ("top1", "conf_cal", "p_gold_cal"):
            bb[k] = round(bb[k] / bb["n"], 4)

    # selective prediction (calibrated confidence thresholds)
    risk_cov = []
    for tau in (0.3, 0.5, 0.7, 0.8, 0.9, 0.95):
        keep = [r for r in range(n) if conf_cal[r] >= tau]
        if keep:
            risk_cov.append({"tau": tau, "coverage": round(len(keep) / n, 4),
                             "selective_acc": round(sum(hit1[r] for r in keep) / len(keep), 4)})

    result = {
        "ckpt": args.ckpt, "split": args.split, "n_val_fit": len(val), "n_test": n,
        "n_cands": len(cands), "n_cands_skipped_oov": skipped,
        "temperature": round(T, 4),
        "constrained_top1": round(sum(hit1) / n, 4),
        "top3": round(sum(hit3) / n, 4), "top5": round(sum(hit5) / n, 4),
        "ece_raw": round(ece_15(conf_raw, hit1), 4),
        "ece_calibrated": round(ece_15(conf_cal, hit1), 4),
        "mean_p_gold_raw": round(sum(pg_raw) / n, 4),
        "mean_p_gold_calibrated": round(sum(pg_cal) / n, 4),
        "expected_acc_calibrated": round(sum(conf_cal) / n, 4),
        "buckets": buckets, "risk_coverage": risk_cov,
        "val_ece_raw": round(ece_val_raw, 4), "val_ece_calibrated": round(ece_val_cal, 4),
    }
    if args.ceiling_json:
        c = json.loads(Path(args.ceiling_json).read_text(encoding="utf-8"))
        result["ceiling_ref"] = {"phi": c["phi"], "ceiling": c["ceiling"]}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    out.with_suffix(".items.json").write_text(json.dumps(items, ensure_ascii=False),
                                              encoding="utf-8")
    print(json.dumps(result, indent=1, ensure_ascii=False))


def _conf_hit(lp, gold_idx, T):
    post = F.softmax(lp / T, dim=1)
    conf = post.max(dim=1).values.tolist()
    hit = (post.argmax(dim=1) == gold_idx).tolist()
    return conf, hit


if __name__ == "__main__":
    main()
