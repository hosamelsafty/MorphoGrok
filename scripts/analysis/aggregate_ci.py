"""Phase-2.5 analysis (no GPU): seed-level aggregation + bootstrap CIs, and the
init-scale delay-law fit. All from existing history.json — no checkpoints needed.

NOTE: history.json stores only aggregate test_exact per log-point, so the CIs here
are SEED-LEVEL (each seed = one sample), not example-paired. The rigorous
example-paired bootstrap needs ckpt re-eval (P3/P7, GPU).

Usage:
  python scripts/analysis/aggregate_ci.py            # runs the built-in report
"""
from __future__ import annotations

import glob
import json
import math
from pathlib import Path

ART = Path("artefacts/phase1")
RNG_SEED = 12345  # fixed for reproducibility (no Date/random-time dependence)


def load(dirname):
    p = ART / dirname / "history.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def peak_test(h):
    return max(x["test_exact"] for x in h)


def test_at_best_val(h):
    best = max(h, key=lambda x: x["val_exact"])
    return best["test_exact"]


def grok_step(h, frac=0.5):
    """Onset step: first step where test_exact >= frac * peak_test."""
    pk = peak_test(h)
    if pk <= 0:
        return None
    thr = frac * pk
    for x in h:
        if x["test_exact"] >= thr:
            return x["step"]
    return None


def _lcg(seed):
    x = seed & 0xFFFFFFFF
    while True:
        x = (1103515245 * x + 12345) & 0x7FFFFFFF
        yield x / 0x7FFFFFFF


def bootstrap_ci(vals, n=10000, seed=RNG_SEED):
    """Seed-level bootstrap CI of the mean. Deterministic (LCG), no time deps."""
    vals = list(vals)
    k = len(vals)
    if k == 0:
        return (None, None, None)
    g = _lcg(seed)
    means = []
    for _ in range(n):
        s = sum(vals[int(next(g) * k)] for _ in range(k))
        means.append(s / k)
    means.sort()
    lo = means[int(0.025 * n)]
    hi = means[int(0.975 * n)]
    return (sum(vals) / k, lo, hi)


def paired_diff_ci(a, b, n=10000, seed=RNG_SEED):
    """Bootstrap CI of mean(a-b) over paired seeds (a,b aligned by index)."""
    d = [ai - bi for ai, bi in zip(a, b)]
    return bootstrap_ci(d, n=n, seed=seed)


def collect(prefix, seeds_dirs, metric):
    out = []
    for d in seeds_dirs:
        h = load(d)
        if h is None:
            continue
        out.append(metric(h))
    return out


def report_group(title, grok_dirs, twin_dirs):
    print(f"\n=== {title} ===")
    gp = collect("grok", grok_dirs, peak_test)
    tw = collect("twin", twin_dirs, test_at_best_val)
    gm, glo, ghi = bootstrap_ci(gp)
    tm, tlo, thi = bootstrap_ci(tw)
    print(f"  grok peak_test  n={len(gp)}: {[round(x,3) for x in gp]}")
    print(f"    mean={gm:.3f}  95% CI [{glo:.3f}, {ghi:.3f}]")
    print(f"  twin test@bestval n={len(tw)}: {[round(x,3) for x in tw]}")
    print(f"    mean={tm:.3f}  95% CI [{tlo:.3f}, {thi:.3f}]")
    if len(gp) == len(tw) and len(gp) > 0:
        dm, dlo, dhi = paired_diff_ci(gp, tw)
        excl = "EXCLUDES 0 (sig)" if (dlo > 0 or dhi < 0) else "straddles 0 (n.s.)"
        print(f"  grok - twin (paired by seed): mean={dm:+.3f}  95% CI [{dlo:+.3f}, {dhi:+.3f}]  -> {excl}")


def report_within_config(title, grok_dirs):
    """Cleanest H7 isolation: SAME run, extended (peak) vs early-stopped (test@best-val).
    Removes the hyperparameter confound — the only difference is when you stop."""
    print(f"\n=== {title}  [within-config: extended vs early-stopped, same run] ===")
    ext = collect("ext", grok_dirs, peak_test)
    es = collect("es", grok_dirs, test_at_best_val)
    if not ext:
        print("  MISSING")
        return
    em, elo, ehi = bootstrap_ci(ext)
    sm, slo, shi = bootstrap_ci(es)
    print(f"  extended (peak)      n={len(ext)}: mean={em:.3f}  95% CI [{elo:.3f}, {ehi:.3f}]")
    print(f"  early-stop (best-val) n={len(es)}: mean={sm:.3f}  95% CI [{slo:.3f}, {shi:.3f}]")
    dm, dlo, dhi = paired_diff_ci(ext, es)
    excl = "EXCLUDES 0 (grokking helps)" if (dlo > 0 or dhi < 0) else "straddles 0 (no grok benefit)"
    print(f"  extended - early-stop (paired): mean={dm:+.3f}  95% CI [{dlo:+.3f}, {dhi:+.3f}]  -> {excl}")


def report_diac(title, diac_dirs, undiac_dirs, undiac_ref=None):
    """P1 read-out: diacritized vs undiacritized peak_test at the same grok config."""
    print(f"\n=== {title}  [P1 diac vs undiac] ===")
    dv = collect("diac", diac_dirs, peak_test)
    uv = collect("undiac", undiac_dirs, peak_test)
    if not dv:
        print(f"  diac runs not finished yet ({len(dv)}/{len(diac_dirs)} present) — skip")
        return
    dm, dlo, dhi = bootstrap_ci(dv)
    print(f"  diac   peak_test n={len(dv)}: {[round(x,3) for x in dv]}  mean={dm:.3f} CI [{dlo:.3f},{dhi:.3f}]")
    if uv:
        um, ulo, uhi = bootstrap_ci(uv)
        print(f"  undiac peak_test n={len(uv)}: {[round(x,3) for x in uv]}  mean={um:.3f} CI [{ulo:.3f},{uhi:.3f}]")
        if len(dv) == len(uv):
            gm, glo, ghi = paired_diff_ci(dv, uv)
            verdict = ("UNDERDETERMINATION (diac lifts ceiling)" if glo > 0.10
                       else "no lift (sparsity/capacity)" if ghi < 0.05
                       else "partial/inconclusive")
            print(f"  diac - undiac (paired): mean={gm:+.3f} CI [{glo:+.3f},{ghi:+.3f}]  -> {verdict}")
    elif undiac_ref is not None:
        print(f"  undiac reference mean={undiac_ref:.3f} (from prior 5-seed set)  -> diac Δ ≈ {dm-undiac_ref:+.3f}")


def diac_report():
    """One-command P1 read-out for the 4h loop. Safe to run before runs finish."""
    def dg(cell):  # diac seed dirs
        return [f"{cell}_diac_wd01_is3_s{s}" for s in range(5)]
    report_diac("L2_p30 (decisive: the ~0.57 ceiling)",
                dg("L2_p30"),
                [f"L2_p30_wd01_is3_s{s}" for s in range(5)])
    report_diac("L1_p05 (easy control)",
                dg("L1_p05"),
                [f"L1_p05_wd01_is3_s{s}" for s in range(5)])
    # cells without a matched undiac wd01_is3 baseline: report diac alone
    report_diac("L1_p30 (diac only)", dg("L1_p30"), [], undiac_ref=None)
    report_diac("L2_p05 (diac only)", dg("L2_p05"), [], undiac_ref=None)


def grid_2d():
    """P6 phase-boundary map: L2_p30 peak_test over init x wd (seed-0 scouts exist)."""
    print("\n=== P6  init x wd phase map (L2_p30, peak_test, seed 0) ===")
    inits = [1, 2, 3, 5]
    wds = [("0.01", "001"), ("0.05", "005"), ("0.1", "01"), ("0.3", "03")]
    print("         " + "".join(f"wd{w:>6}" for w, _ in wds))
    for it in inits:
        row = [f"init={it:<3}"]
        for _, wt in wds:
            d = f"L2_p30_wd{wt}_s0" if it == 1 else f"L2_p30_wd{wt}_is{it}_s0"
            h = load(d)
            row.append(f"{peak_test(h):.3f}" if h else "  -  ")
        print("  " + row[0] + " " + " ".join(f"{c:>6}" for c in row[1:]))
    print("  (init=1 col from L2_p30_wd*_s0; peak >~0.60 anywhere = ceiling broken)")


def delay_law():
    print("\n=== Init-scale delay law:  T_grok vs init ===")
    cells = {
        0.5: "L1_p05_wd01_is05_s0",
        2.0: "L1_p05_wd01_is2_s0",
        3.0: "L1_p05_wd01_is3_s0",
        5.0: "L1_p05_wd01_is5_s0",
    }
    xs, ys = [], []
    for init, d in sorted(cells.items()):
        h = load(d)
        if h is None:
            print(f"  init={init}: MISSING ({d})")
            continue
        gs = grok_step(h)
        pk = peak_test(h)
        print(f"  init={init}: grok_step={gs}  peak_test={pk:.3f}")
        if gs and gs > 0:
            xs.append(init)
            ys.append(math.log(gs))
    if len(xs) >= 3:
        # linear fit of ln(T_grok) vs init  ->  exponential law T ~ e^(b*init)
        k = len(xs)
        mx = sum(xs) / k
        my = sum(ys) / k
        sxx = sum((x - mx) ** 2 for x in xs)
        sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        b = sxy / sxx
        a = my - b * mx
        ss_tot = sum((y - my) ** 2 for y in ys)
        ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        print(f"  fit ln(T) = {a:.3f} + {b:.3f}*init   ->  T ~ e^({b:.3f}*init)   R^2={r2:.4f}")
        print("  (Omnigrok / 2026 exponential delay law; peak ~const across init = delay-not-ceiling)")


if __name__ == "__main__":
    # H7 twin comparison. Two candidate 'grok' groups (config choice is a research
    # judgment): canonical (matches PLAN's grok~0.60) vs headline wd0.1/init3 cell.
    report_group(
        "L1_p05  canonical grok  vs  fixed twin",
        ["L1_p05_grok", "L1_p05_grok_s1", "L1_p05_grok_s2", "L1_p05_grok_s3", "L1_p05_grok_s4"],
        ["L1_p05_twinfix_s0", "L1_p05_twinfix_s1", "L1_p05_twinfix_s2", "L1_p05_twinfix_s3", "L1_p05_twinfix_s4"],
    )
    report_group(
        "L1_p05  headline grok (wd0.1,init3)  vs  fixed twin",
        ["L1_p05_wd01_is3_s0", "L1_p05_wd01_is3_s1", "L1_p05_wd01_is3_s2", "L1_p05_wd01_is3_s3", "L1_p05_wd01_is3_s4"],
        ["L1_p05_twinfix_s0", "L1_p05_twinfix_s1", "L1_p05_twinfix_s2", "L1_p05_twinfix_s3", "L1_p05_twinfix_s4"],
    )
    report_group(
        "L2_p30  canonical grok (wd1.0)  vs  fixed twin",
        ["L2_p30_grok_s0b", "L2_p30_grok_s1", "L2_p30_grok_s2", "L2_p30_grok_s3", "L2_p30_grok_s4"],
        ["L2_p30_twinfix_s0", "L2_p30_twinfix_s1", "L2_p30_twinfix_s2", "L2_p30_twinfix_s3", "L2_p30_twinfix_s4"],
    )
    report_group(
        "L2_p30  best grok (wd0.05, optimum)  vs  fixed twin",
        ["L2_p30_wd005_s0", "L2_p30_wd005_s1", "L2_p30_wd005_s2", "L2_p30_wd005_s3", "L2_p30_wd005_s4"],
        ["L2_p30_twinfix_s0", "L2_p30_twinfix_s1", "L2_p30_twinfix_s2", "L2_p30_twinfix_s3", "L2_p30_twinfix_s4"],
    )
    # Cleanest H7: same config, extended vs early-stopped (no hyperparameter confound).
    report_within_config(
        "L1_p05 headline grok (wd0.1,init3)",
        ["L1_p05_wd01_is3_s0", "L1_p05_wd01_is3_s1", "L1_p05_wd01_is3_s2", "L1_p05_wd01_is3_s3", "L1_p05_wd01_is3_s4"],
    )
    report_within_config(
        "L2_p30 best grok (wd0.05)",
        ["L2_p30_wd005_s0", "L2_p30_wd005_s1", "L2_p30_wd005_s2", "L2_p30_wd005_s3", "L2_p30_wd005_s4"],
    )
    diac_report()
    grid_2d()
    delay_law()
