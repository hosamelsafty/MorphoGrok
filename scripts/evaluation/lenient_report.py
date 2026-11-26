"""Aggregation of the reported accuracies: strict vs و/ي-lenient accuracy, with the
paper's seed-level bootstrap, from the per-item dumps of dump_predictions.py.

Reads   artefacts/camera_ready/preds/<run>[.<tag>].json   (dump_predictions.py)
        artefacts/camera_ready/runs/sup_*/history.json    (baseline val curves)
        artefacts/camera_ready/ceiling/*.json             (ceiling_method.py)
Writes  one JSON with every number and prints the tables.

Sections:
  conditions   Table 3 conditions, greedy, strict and lenient, mean [95% CI]
               over seeds and the paired delta vs Single-Form. The strict
               column must reproduce the published table.
  constrained  inventory-constrained top-1 / top-3, strict and lenient. Lenient
               is reported two ways: the strict ranking scored leniently
               ("rescored"), and the ranking after merging و/ي-equivalent
               candidates ("merged", the criterion the lenient ceiling uses).
  baseline     the supervised baseline, per (architecture, dropout) config.
  errors       A3: what the remaining Single-Form errors are.

No GPU. Usage: python scripts/evaluation/lenient_report.py
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))  # aggregate_ci
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from aggregate_ci import bootstrap_ci, paired_diff_ci  # noqa: E402
from grokkingroots.data.arabic_text import lenient_key, lenient_match  # noqa: E402

CR = "artefacts/camera_ready"
CONDS = [  # (paper name, run stem, tag)
    ("Single-Form", "g5_L2_k1_v2c_wd005_500k", ""),
    ("Support-1", "g5_L2_k2_para_v2c_wd005_500k", ""),
    ("Support-2", "g5_L2_k3_para_v2c_wd005_500k", ""),
    ("Support-2 @500k", "g5_L2_k3_para_v2c_wd005_500k", ".500k"),
    ("Witness", "g5_L2_k3_reveal_v2c_wd005_500k", ""),
    ("Witness @500k", "g5_L2_k3_reveal_v2c_wd005_500k", ".500k"),
    ("Repeat", "g5_L2_k3_self_v2c_wd005_500k", ""),
    ("Cross-Root", "g5_L2_k3_shuf_v2c_wd005_500k", ""),
    ("Mixed-k", "g5_L2_kmix_v2c_wd005_500k", ""),
    ("Single-Form init 3 (wd 0.05)", "cr_L2_k1_is3_v2c_wd005_500k", ""),
    ("Single-Form init 3 (wd 0.1)", "cr_L2_k1_is3_v2c_wd01_500k", ""),
]
BASELINES = [("enc-dec, dropout 0.3", "sup_encdec_do03"), ("enc-dec, dropout 0.1", "sup_encdec_do01"),
             ("dec-only, dropout 0.3", "sup_dec_do03"), ("dec-only, dropout 0.1", "sup_dec_do01"),
             ("dec-only, dropout 0.1, 500k", "sup_dec_do01_500k"),
             ("dec-only, dropout 0.0, 500k", "sup_dec_do00_500k")]

_splits = {}


def golds_of(split_path):
    if split_path not in _splits:
        test = json.load(open(split_path, encoding="utf-8"))["test"]
        _splits[split_path] = ([x.split("؞")[0] for x, _ in test], [y for _, y in test])
    return _splits[split_path]


def load(stem, tag=""):
    """{seed: dump} for one condition."""
    out = {}
    for f in glob.glob(f"{CR}/preds/{stem}_s*{tag}.json"):
        m = re.fullmatch(rf"{re.escape(stem)}_s(\d+){re.escape(tag)}\.json", os.path.basename(f))
        if m:
            out[int(m.group(1))] = json.load(open(f, encoding="utf-8"))
    return dict(sorted(out.items()))


def acc(hits):
    # Per-run accuracies are rounded to 4 dp before aggregation, as in
    # fulltest_eval.json / calibrated_decode.json: the published means, deltas
    # and intervals were computed from those rounded values, and two deltas
    # (Support-2, Mixed-k) differ in the third decimal otherwise.
    return round(sum(hits) / len(hits), 4)


def ci(vals):
    m, lo, hi = bootstrap_ci(vals)
    return {"mean": m, "ci95": [lo, hi], "n": len(vals), "seeds": vals}


def fmt(c):
    return f"{c['mean']:.3f} [{c['ci95'][0]:.3f}, {c['ci95'][1]:.3f}]"


def greedy_scores(dumps):
    strict, lenient = [], []
    for d in dumps.values():
        _, gold = golds_of(d["split"])
        strict.append(acc([p == g for p, g in zip(d["greedy"], gold)]))
        lenient.append(acc([lenient_match(p, g) for p, g in zip(d["greedy"], gold)]))
    return strict, lenient


def constrained_scores(dumps):
    keys = ["top1_strict", "top3_strict", "top1_lenient_rescored", "top3_lenient_rescored",
            "top1_lenient_merged", "top3_lenient_merged"]
    out = {k: [] for k in keys}
    for d in dumps.values():
        _, gold = golds_of(d["split"])
        gk = [lenient_key(g) for g in gold]
        t5, m3 = d["top5"], d["merged_top3"]
        out["top1_strict"].append(acc([t[0] == g for t, g in zip(t5, gold)]))
        out["top3_strict"].append(acc([g in t[:3] for t, g in zip(t5, gold)]))
        out["top1_lenient_rescored"].append(acc([lenient_key(t[0]) == k for t, k in zip(t5, gk)]))
        out["top3_lenient_rescored"].append(
            acc([k in {lenient_key(c) for c in t[:3]} for t, k in zip(t5, gk)]))
        out["top1_lenient_merged"].append(acc([m[0] == k for m, k in zip(m3, gk)]))
        out["top3_lenient_merged"].append(acc([k in m for m, k in zip(m3, gk)]))
    return out


def error_taxonomy(dumps, inventory, ceiling_items):
    """A3. Shares are of that decoding's own errors, averaged over seeds."""
    rows = {"greedy": [], "constrained": []}
    for d in dumps.values():
        surf, gold = golds_of(d["split"])
        for name, preds in (("greedy", d["greedy"]), ("constrained", [t[0] for t in d["top5"]])):
            err = [i for i, (p, g) in enumerate(zip(preds, gold)) if p != g]
            swap = [i for i in err if lenient_match(preds[i], gold[i])]
            real = [i for i in swap if preds[i] in inventory]
            r = {"error_rate": len(err) / len(gold),
                 "wawya_swap_share": len(swap) / len(err),
                 "wawya_swap_between_real_roots_share": len(real) / len(err),
                 "pred_not_in_inventory_share":
                     sum(preds[i] not in inventory for i in err) / len(err),
                 "gold_in_constrained_top3_share":
                     sum(gold[i] in d["top5"][i][:3] for i in err) / len(err)}
            if ceiling_items:   # does the ceiling's closed-form predictor get these right?
                assert [c["gold"] for c in ceiling_items] == gold
                r["ceiling_predictor_acc_on_swap_errors"] = (
                    sum(ceiling_items[i]["pred"] == gold[i] for i in swap) / len(swap))
                r["ceiling_predictor_acc_on_all_errors"] = (
                    sum(ceiling_items[i]["pred"] == gold[i] for i in err) / len(err))
            rows[name].append(r)
    return {name: {k: sum(r[k] for r in rs) / len(rs) for k in rs[0]}
            for name, rs in rows.items() if rs}


def main():
    report = {"conditions": {}, "constrained": {}, "baseline": {}, "ceiling": {}}

    print("== greedy, val-selected, full frozen test (2,109 items)")
    print(f"{'condition':30s} n  strict                 lenient                "
          f"d vs Single-Form strict       lenient")
    base = None
    for name, stem, tag in CONDS:
        dumps = load(stem, tag)
        if not dumps:
            continue
        s, l = greedy_scores(dumps)
        row = {"strict": ci(s), "lenient": ci(l), "seeds": list(dumps),
               "scored_steps": [d["scored_step"] for d in dumps.values()]}
        delta = ""
        if name == "Single-Form":
            base = (s, l)
        elif base and len(s) == len(base[0]):
            ds, dl = paired_diff_ci(s, base[0]), paired_diff_ci(l, base[1])
            row["delta_strict"], row["delta_lenient"] = ds, dl
            delta = (f"{ds[0]:+.3f} [{ds[1]:+.3f}, {ds[2]:+.3f}]   "
                     f"{dl[0]:+.3f} [{dl[1]:+.3f}, {dl[2]:+.3f}]")
        report["conditions"][name] = row
        print(f"{name:30s} {len(s)}  {fmt(row['strict'])}   {fmt(row['lenient'])}   {delta}")

    print("\n== supervised baseline (A1), val-selected best-of-50 checkpoint")
    print(f"{'config':24s} n  best val   greedy strict          greedy lenient         "
          f"constr top-1           top-3")
    for name, stem in BASELINES:
        dumps = load(stem)
        if not dumps:
            continue
        s, l = greedy_scores(dumps)
        c = {k: ci(v) for k, v in constrained_scores(dumps).items()}
        val = []
        for seed in dumps:
            h = json.load(open(f"{CR}/runs/{stem}_s{seed}/history.json"))
            val.append(max(r["val_exact"] for r in h))
        row = {"best_val": ci(val), "greedy_strict": ci(s), "greedy_lenient": ci(l),
               "constrained": c, "selected_steps": [d["scored_step"] for d in dumps.values()]}
        if base and len(s) == len(base[0]):
            row["delta_strict_vs_single_form"] = paired_diff_ci(s, base[0])
        report["baseline"][name] = row
        print(f"{name:24s} {len(s)}  {row['best_val']['mean']:.3f}      {fmt(row['greedy_strict'])}   "
              f"{fmt(row['greedy_lenient'])}   {fmt(c['top1_strict'])}   {fmt(c['top3_strict'])}")

    print("\n== Single-Form, inventory-constrained decoding")
    sf = load("g5_L2_k1_v2c_wd005_500k")
    if sf and "top5" in next(iter(sf.values())):
        c = {k: ci(v) for k, v in constrained_scores(sf).items()}
        report["constrained"]["Single-Form"] = c
        for k, v in c.items():
            print(f"  {k:24s} {fmt(v)}")

    print("\n== ceiling (template statistics), strict vs lenient")
    for f in sorted(glob.glob(f"{CR}/ceiling/template_*.json")):
        if f.endswith(".items.json"):
            continue
        a = json.load(open(f))
        key = f"prior={a['prior']}, templates={a['template_source']}, " \
              f"{'lenient' if a['lenient'] else 'strict'}"
        report["ceiling"][key] = {"ceiling": a["ceiling"], "ci95": a["ceiling_ci95"],
                                  "realized": a["realized_argmax_acc"]}
        print(f"  {key:50s} C={a['ceiling']:.4f} {a['ceiling_ci95']}  realized={a['realized_argmax_acc']:.4f}")

    if sf and "top5" in next(iter(sf.values())):
        inv = set(json.load(open("artefacts/phase0/root_type_map.json", encoding="utf-8"))["map"])
        items_path = f"{CR}/ceiling/template_prior-freq_tpl-lexicon_strict.items.json"
        items = json.load(open(items_path, encoding="utf-8")) if os.path.exists(items_path) else None
        report["errors"] = error_taxonomy(sf, inv, items)
        print("\n== A3: what the remaining Single-Form errors are (mean over seeds)")
        for name, r in report["errors"].items():
            print(f"  {name}")
            for k, v in r.items():
                print(f"    {k:42s} {v:.3f}")

    Path(f"{CR}/lenient_report.json").write_text(
        json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {CR}/lenient_report.json")


if __name__ == "__main__":
    main()
