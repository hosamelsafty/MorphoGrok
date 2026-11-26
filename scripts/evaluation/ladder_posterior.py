"""Posterior & calibration dynamics ACROSS TRAINING (method d — the
grokking-native analysis): does the constrained posterior concentrate, and does
calibration improve, at the grok transition?

For a run's checkpoint ladder, at each (log-spaced subset of) checkpoint:
  - score the valid-root inventory on fixed val/test item subsets (shared
    scorer with constrained_topk_eval/calibrated_decode);
  - fit the temperature on the val subset (per-checkpoint T_t);
  - report constrained top-1/top-3, mean p(gold), ECE raw vs calibrated, mean
    calibrated confidence, and T_t itself.
Read against the run's grok step: a posterior-concentration jump at the
transition is a progress measure in the Nanda/Varma sense; T_t's trajectory
says when the model becomes (over)confident.

Usage (GPU ~2 min/ckpt at default sizes):
  python scripts/evaluation/ladder_posterior.py \
      --run-dir artefacts/runs/<id> \
      --split artefacts/phase0/splits_gen2_v2clean/L2_p30_k1.json --out OUT.json
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from constrained_topk_eval import load_model, build_candidates, make_scorer
from calibrated_decode import fit_temperature, ece_15

import torch
import torch.nn.functional as F


def score_full(score_item, pairs, tag):
    """Score pairs on their FULL input string. calibrated_decode.score_matrix
    strips to the first delimiter field (right for k1, wrong for multi-form
    splits like reveal — the model must be evaluated on its own interface)."""
    L, golds = [], []
    for i, (s, g) in enumerate(pairs):
        L.append(score_item(s))
        golds.append(g)
        if (i + 1) % 200 == 0:
            print(f"  {tag}: {i + 1}/{len(pairs)}", flush=True)
    return torch.stack(L), golds


def grok_step_of(history):
    """The paper's grokking criterion: delayed generalization only."""
    acc = [r["test_exact"] for r in history]
    peak = max(acc) if acc else 0.0
    if peak <= 0.05:
        return None
    mem = next((i for i, r in enumerate(history) if r.get("train_exact", 0) > 0.99), None)
    if mem is None or history[mem]["test_exact"] >= 0.5 * peak:
        return None
    for r in history[mem + 1:]:
        if r["test_exact"] >= 0.5 * peak:
            return r["step"]
    return None


def pick_steps(steps, k):
    """Log-spaced subset of available ladder steps, always keeping ends."""
    if len(steps) <= k:
        return steps
    import math
    lo, hi = math.log(steps[0]), math.log(steps[-1])
    want = [math.exp(lo + (hi - lo) * i / (k - 1)) for i in range(k)]
    out, used = [], set()
    for w in want:
        s = min(steps, key=lambda x: abs(math.log(x) - math.log(w)))
        if s not in used:
            used.add(s); out.append(s)
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--inventory", default="artefacts/phase0/root_type_map.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-ckpts", type=int, default=15)
    ap.add_argument("--test-items", type=int, default=400)
    ap.add_argument("--val-items", type=int, default=250)
    ap.add_argument("--max-cands", type=int, default=0, help="0 = full inventory (smoke only)")
    ap.add_argument("--batch-rows", type=int, default=2048)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    rd = Path(args.run_dir)

    hist = json.loads((rd / "history.json").read_text(encoding="utf-8"))
    gstep = grok_step_of(hist)
    ladder = sorted(int(re.search(r"ckpt_(\d+)\.pt$", p).group(1))
                    for p in glob.glob(str(rd / "ckpts" / "ckpt_*.pt")))
    steps = pick_steps(ladder, args.max_ckpts)
    print(f"ladder: {len(ladder)} ckpts -> analysing {len(steps)}; grok_step={gstep}", flush=True)

    traj = []
    for st in steps:
        model, vocab, (train_p, val_p, test_p) = load_model(
            rd / "meta.json", args.split, rd / "ckpts" / f"ckpt_{st}.pt", device)
        cands, _ = build_candidates(args.inventory, test_p, vocab, args.max_cands)
        c2i = {r: i for i, r in enumerate(cands)}
        # deterministic stride subsample (identical at every checkpoint)
        def sub(pairs, n):
            pairs = [(s, g) for s, g in pairs if g in c2i]
            if len(pairs) <= n:
                return pairs
            stride = len(pairs) // n
            return pairs[::stride][:n]
        val = sub(val_p, args.val_items)
        test = sub(test_p, args.test_items)
        scorer = make_scorer(model, vocab, cands, device, args.batch_rows)
        lp_v, gold_v = score_full(scorer, val, f"val@{st}")
        gi_v = torch.tensor([c2i[g] for g in gold_v], device=lp_v.device)
        T, _ = fit_temperature(lp_v, gi_v)
        lp_t, gold_t = score_full(scorer, test, f"test@{st}")
        gi_t = torch.tensor([c2i[g] for g in gold_t], device=lp_t.device)
        top = lp_t.argsort(dim=1, descending=True)
        hit1 = (top[:, 0] == gi_t).tolist()
        hit3 = (top[:, :3] == gi_t.unsqueeze(1)).any(1).tolist()
        post_r = F.softmax(lp_t, dim=1)
        post_c = F.softmax(lp_t / T, dim=1)
        conf_r = post_r.max(1).values.tolist()
        conf_c = post_c.max(1).values.tolist()
        pg_c = post_c.gather(1, gi_t.unsqueeze(1)).squeeze(1)
        n = len(test)
        traj.append({
            "step": st, "T": round(T, 3),
            "top1": round(sum(hit1) / n, 4), "top3": round(sum(hit3) / n, 4),
            "p_gold_cal": round(float(pg_c.mean()), 4),
            "ece_raw": round(ece_15(conf_r, hit1), 4),
            "ece_cal": round(ece_15(conf_c, hit1), 4),
            "conf_cal_mean": round(sum(conf_c) / n, 4),
        })
        print(json.dumps(traj[-1]), flush=True)

    out = {"run_dir": str(rd), "split": args.split, "grok_step": gstep,
           "n_test_sub": len(test), "n_val_sub": len(val),
           "n_cands": len(cands), "trajectory": traj}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print("LADDER_POSTERIOR_DONE", flush=True)


if __name__ == "__main__":
    main()
