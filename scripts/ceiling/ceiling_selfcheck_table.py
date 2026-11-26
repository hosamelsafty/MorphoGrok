"""LaTeX rows for the paper's paired validation-check table (Appendix B).

Reads artefacts/camera_ready/ceiling_selfcheck.json (scripts/ceiling/ceiling_selfcheck.py)
and prints one row per ceiling estimate. Est. and Att. are the values stored by
scripts/ceiling/ceiling_method.py (the ones printed elsewhere in the paper), rounded
half-up to 3 decimals; the interval is the paired bootstrap interval.

    python3 scripts/ceiling/ceiling_selfcheck_table.py > artefacts/camera_ready/ceiling_selfcheck_table.tex
"""
import json
from decimal import ROUND_HALF_UP, Decimal

rows = json.load(open("artefacts/camera_ready/ceiling_selfcheck.json"))
by = {r["file"].split("/")[-1].replace(".items.json", ""): r for r in rows}


def f(x: float) -> str:
    return str(Decimal(str(x)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))


def cells(r: dict) -> str:
    lo, hi = r["paired_ci95"]
    return (f"{f(r['estimate_file'])} & {f(r['attained_file'])} & "
            f"$[{lo:+.3f}, {hi:+.3f}]$ & {'pass' if r['passes'] else 'fail'}")


for name, key in [("template (main)", "template_prior-freq_tpl-lexicon"),
                  ("training-pair templates", "template_prior-freq_tpl-train"),
                  ("uniform root prior", "template_prior-uniform_tpl-lexicon"),
                  ("both", "template_prior-uniform_tpl-train")]:
    print(f"{name} & {cells(by[key + '_strict'])} & {cells(by[key + '_lenient'])} \\\\")
print(r"\addlinespace[2pt]")
for name, key in [("L3 (geminate)", "L3_strict"), ("L4 (quadriliteral)", "L4_strict")]:
    print(f"{name} & {cells(by[key])} & & & & \\\\")
print(r"\addlinespace[2pt]")
for a in ("0.1", "0.5", "1", "5", "20"):
    print(f"L2, $\\alpha{{=}}{a}$ & {cells(by[f'alpha{a}_strict'])} & & & & \\\\")
