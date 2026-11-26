"""Artefact loaders for the paper figures.

Pure stdlib on purpose: no numpy, no matplotlib. This half can therefore be
run and verified anywhere, without the plotting stack. Every figure script imports its numbers from here and
never hardcodes one -- that is the whole point.

Two tiers, and the distinction matters:
  * TRAJECTORIES (curves, buckets, ladders, risk-coverage) come from the run
    artefacts. They exist nowhere else.
  * HEADLINE SCALARS come from PAPER_ACC, re-scored on the full 2,109-item
    test set. history.json logs only a 2,000-item prefix, so the artefact
    figure is 0.001-0.003 low; a figure built from it would contradict
    Table 1. drift_report() checks the two stay consistent.
"""
from __future__ import annotations

import glob
import json
import os
import random
import statistics as st

AL = "artefacts/runs"
TOOLS_JSON = "artefacts/phase0/arabic_tools.json"

# Paper condition name -> run-directory prefix (see the reproducibility appendix).
ARMS = {
    "Single-Form": "g5_L2_k1_v2c_wd005_500k",
    "Support-1":   "g5_L2_k2_para_v2c_wd005_500k",
    "Support-2":   "g5_L2_k3_para_v2c_wd005_500k",
    "Witness":     "g5_L2_k3_reveal_v2c_wd005_500k",
    "Repeat":      "g5_L2_k3_self_v2c_wd005_500k",
    "Cross-Root":  "g5_L2_k3_shuf_v2c_wd005_500k",
    "Mixed-k":     "g5_L2_kmix_v2c_wd005_500k",
    # Single-Form re-trained at initialization scale 3 
    "Single-Form, scale 3": "cr_L2_k1_is3_v2c_wd005_500k",
}

CEILING_TEMPLATE = 0.890      # C_phi, template evidence model (LOO)
CEILING_AUDIT = 0.92          # oracle-radical upper bracket
CEILING_COMPAT = 0.151        # bare-compatibility bracket


def _seed_dirs(prefix: str) -> list[str]:
    return sorted(glob.glob(os.path.join(AL, prefix + "_s*")))


def _history(run_dir: str) -> list[dict]:
    h = json.load(open(os.path.join(run_dir, "history.json")))
    return h["history"] if isinstance(h, dict) and "history" in h else h


def val_selected(run_dir: str) -> float:
    """test_exact at the argmax of val_exact -- the project's selection rule.

    Never peak-over-training, which would be oracle early stopping on test.
    NOTE: history logs a 2,000-item prefix of the frozen test set; the paper's
    headline numbers are re-scored on all 2,109. Expect ~0.001 disagreement on
    L2 and treat the paper's value as authoritative for the main tables.
    """
    h = _history(run_dir)
    i = max(range(len(h)), key=lambda j: h[j]["val_exact"])
    return h[i]["test_exact"]


def bootstrap_ci(vals: list[float], n: int = 10000, seed: int = 20260803):
    """Seed-level bootstrap CI. At n<=3 seeds this is not a meaningful
    interval -- callers should render those as an observed range instead."""
    if len(vals) < 2:
        return vals[0], vals[0]
    rng = random.Random(seed)
    means = sorted(st.mean(rng.choices(vals, k=len(vals))) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n)]


def arm_summary(arm: str) -> dict:
    """{mean, lo, hi, n, vals} of val-selected test accuracy across seeds."""
    dirs = _seed_dirs(ARMS[arm])
    vals = [val_selected(d) for d in dirs]
    lo, hi = bootstrap_ci(vals)
    return {"mean": st.mean(vals), "lo": lo, "hi": hi, "n": len(vals), "vals": vals}


def curves(arm: str, key: str) -> tuple[list[int], list[float], list[float], list[float]]:
    """Per-step (steps, mean, min, max) of `key` in history.json across seeds.

    Seeds share a log-spaced grid but not always its length (arms extended to
    1e6 have more points than those stopped at 5e5), so the grid is truncated
    to the shortest seed to keep the band honest.
    """
    runs = [_history(d) for d in _seed_dirs(ARMS[arm])]
    n = min(len(r) for r in runs)
    steps = [runs[0][i]["step"] for i in range(n)]
    mean, lo, hi = [], [], []
    for i in range(n):
        col = [r[i][key] for r in runs]
        mean.append(st.mean(col)); lo.append(min(col)); hi.append(max(col))
    return steps, mean, lo, hi


def bucket_curves(arm: str) -> dict:
    """Radical-visibility bucket accuracy over training, across all seeds.

    Source: circuit_actv.json 'curves' ({step, hidden, visible, overall}).
    Using every seed here is what lets the bucket figure show a 5-seed band
    rather than a single seed -- the mismatch between a seed-s0 plot and
    5-seed means quoted in the text was a real reported defect.
    """
    files = sorted(glob.glob(os.path.join(AL, ARMS[arm] + "_s*", "circuit_actv.json")))
    runs = [json.load(open(f))["curves"] for f in files]
    runs = [c for c in runs if c]
    n = min(len(c) for c in runs)
    steps = [runs[0][i]["step"] for i in range(n)]
    out = {"steps": steps, "n_seeds": len(runs)}
    for b in ("hidden", "visible"):
        m, lo, hi = [], [], []
        for i in range(n):
            col = [c[i][b] for c in runs]
            m.append(st.mean(col)); lo.append(min(col)); hi.append(max(col))
        out[b] = {"mean": m, "lo": lo, "hi": hi}
    out["gap"] = [v - h for v, h in zip(out["visible"]["mean"], out["hidden"]["mean"])]
    return out


def grok_steps(arm: str) -> list[int]:
    """Per-seed grok step from circuit_actv.json, ascending; [] if none grok.

    Seeds disagree by two orders of magnitude (Witness: 133--7,306), so a
    figure over all five seeds must draw the range, not one seed's landmark.
    Single-Form records `null` for every seed -- it never groks, which is the
    claim -- and that arm therefore returns [].
    """
    files = sorted(glob.glob(os.path.join(AL, ARMS[arm] + "_s*", "circuit_actv.json")))
    return sorted(s for s in (json.load(open(f)).get("grok_step") for f in files)
                  if s is not None)


def ladder_posterior(arm: str) -> list[dict]:
    """Per-checkpoint constrained-posterior trajectory (step, T, top1, ...).

    Only computed for Single-Form s0 and Witness s0 -- single seed by
    construction, and any figure using it must say so.
    """
    f = glob.glob(os.path.join(AL, ARMS[arm] + "_s0", "ladder_posterior.json"))
    if not f:
        raise FileNotFoundError(f"no ladder_posterior.json for {arm}")
    return json.load(open(f[0]))["trajectory"]


def risk_coverage(arm: str = "Single-Form") -> dict:
    """Mean risk-coverage curve + calibration scalars across seeds."""
    files = sorted(glob.glob(os.path.join(AL, ARMS[arm] + "_s*", "calibrated_decode.json")))
    js = [json.load(open(f)) for f in files]
    # seeds do not all carry the same tau grid, so key by tau and keep only
    # the taus every seed reports -- indexing by position silently misaligns
    per_seed = [{p["tau"]: p for p in j["risk_coverage"]} for j in js]
    taus = sorted(set.intersection(*(set(d) for d in per_seed)))
    pts = [{
        "tau": tau,
        "coverage": st.mean(d[tau]["coverage"] for d in per_seed),
        "selective_acc": st.mean(d[tau]["selective_acc"] for d in per_seed),
    } for tau in taus]
    return {
        "points": pts,
        "n_seeds": len(js),
        "temperature": st.mean(j["temperature"] for j in js),
        "top1": st.mean(j["constrained_top1"] for j in js),
        "expected_acc": st.mean(j["expected_acc_calibrated"] for j in js),
        "ece_raw": st.mean(j["ece_raw"] for j in js),
        "ece_cal": st.mean(j["ece_calibrated"] for j in js),
    }


# Third-party / closed-form reference systems, frozen L2 test set.
# Classical rows are derivable from artefacts/phase0/arabic_tools.json (L2,
# strict, all-bucket) and are guarded by drift_report() below; ceiling rows come
# from artefacts/phase0/ceiling_k1v2c_*.json. Kept here so a figure never
# inlines them -- fig_gap_anatomy.py currently still does, with its own labels.
REFERENCE_SYSTEMS = [
    ("skeleton",       0.000, "classical"),
    ("CAMeL Tools",    0.000, "classical"),
    ("ISRI",           0.333, "classical"),
    ("Tashaphyne",     0.338, "classical"),
    ("Qalsadi",        0.421, "classical"),
    ("compat $C_\\varphi$", CEILING_COMPAT, "ceiling"),
    ("template $C_\\varphi$", CEILING_TEMPLATE, "ceiling"),
]
MODEL_CONSTRAINED_TOP1 = 0.638   # Single-Form, decoding constrained to inventory
                                 # (5-seed mean of constrained_top1, 0.6382)

# Authoritative headline accuracies -- Table 1 (tab:dose), as (value, 95% CI
# half-width). Re-scored on the FULL frozen L2 test set (2,109 items).
#
# history.json only logs a 2,000-item prefix (--gen-subsample), so
# val_selected() above disagrees with these by up to 0.003 in either direction
# -- low on the paradigm arms, high on Single-Form, Repeat and Cross-Root, as
# drift_report() prints. Any figure showing
# a headline number must use THESE, or it silently contradicts the table it
# is supposed to illustrate. Trajectories still come from the artefacts --
# they are the only place per-step data exists.
PAPER_ACC = {
    "Single-Form": (0.563, 0.014),
    "Support-1":   (0.669, 0.010),
    "Support-2":   (0.711, 0.022),
    "Witness":     (0.867, 0.006),
    "Repeat":      (0.570, 0.009),
    "Cross-Root":  (0.525, 0.031),
}
WITNESS_500K = (0.855, 0.012)    # matched-budget Witness, Table 1 footnote

# Supervised baseline (recipe of Wu et al. 2021): val-selected greedy exact
# match on the full frozen L2 test set, 5 seeds, at the validation-chosen
# dropout. Headline scalars like PAPER_ACC. Source: the paper's baseline table
# (Appendix A).
BASELINE_ACC = {"encoder-decoder": 0.397, "MorphoGrok architecture": 0.519}


def drift_report() -> list[str]:
    """Compare artefact-derived accuracy against the paper's table.

    A guard, not a fix: if a re-run ever moves an arm by more than the known
    prefix effect, this is what catches it before a figure ships.
    """
    out = []
    for arm, (paper, _) in PAPER_ACC.items():
        got = arm_summary(arm)["mean"]
        d = got - paper
        flag = "  <-- CHECK" if abs(d) > 0.006 else ""
        out.append(f"  {arm:12s} paper={paper:.3f}  artefact={got:.4f}  diff={d:+.4f}{flag}")

    # classical rows: exact match required, they are read straight off the
    # frozen tools artefact (run under PYTHONHASHSEED=0; unpinned, Tashaphyne
    # drifts by ~0.03, which is how 0.340 went stale here once).
    key = {"skeleton": "skeleton", "CAMeL Tools": "camel", "ISRI": "isri",
           "Tashaphyne": "tashaphyne", "Qalsadi": "qalsadi"}
    tools = {x["split"]: x for x in json.load(open(TOOLS_JSON))}["L2"]["systems"]
    for lab, val, grp in REFERENCE_SYSTEMS:
        if grp != "classical":
            continue
        act = round(tools[key[lab]]["all"]["strict"], 3)
        flag = "" if abs(val - act) < 1e-9 else "  <-- STALE"
        out.append(f"  {lab:12s} figure={val:.3f}  artefact={act:.3f}{flag}")
    return out


if __name__ == "__main__":
    print("=== val-selected (history prefix metric) ===")
    for a in ARMS:
        s = arm_summary(a)
        print(f"  {a:12s} n={s['n']} mean={s['mean']:.4f} [{s['lo']:.4f},{s['hi']:.4f}]")
    print("\n=== bucket curves ===")
    for a in ("Single-Form", "Witness"):
        b = bucket_curves(a)
        print(f"  {a:12s} seeds={b['n_seeds']} pts={len(b['steps'])} "
              f"final hidden={b['hidden']['mean'][-1]:.3f} visible={b['visible']['mean'][-1]:.3f} "
              f"gap={b['gap'][-1]:+.3f}")
    print("\n=== grok steps (per seed) ===")
    for a in ("Single-Form", "Witness"):
        g = grok_steps(a)
        print(f"  {a:12s} {g if g else 'no seed groks'}")
    print("\n=== ladder posterior ===")
    for a in ("Single-Form", "Witness"):
        t = ladder_posterior(a)
        print(f"  {a:12s} pts={len(t)} final top1={t[-1]['top1']:.3f} T={t[-1]['T']:.2f}")
    print("\n=== drift vs Table 1 (authoritative) ===")
    for line in drift_report():
        print(line)
    print("\n=== risk coverage ===")
    rc = risk_coverage()
    print(f"  seeds={rc['n_seeds']} T={rc['temperature']:.3f} top1={rc['top1']:.4f} "
          f"expected={rc['expected_acc']:.4f} ece {rc['ece_raw']:.3f}->{rc['ece_cal']:.3f}")
