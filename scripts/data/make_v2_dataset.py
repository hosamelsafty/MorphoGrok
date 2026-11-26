"""Build CLEANED dataset variants + their L2 dcrit_p30 splits (gen-1c).

Motivation (ambiguity_audit, 2026-07-04): 16.7% of the frozen L2 test surfaces
contain lexicon junk (homograph sense indices like شر2, pipe-joined variants);
the achievable ceiling is ~0.75 on dirty data but ~0.90 junk-free. Gen-1c
measures how much of the observed ~0.58 plateau the junk actually costs.

Variants:
  v2clean  — base parquet, cleaned      (isolates the CLEANING effect)
  enrclean — enriched parquet, cleaned  (cleaning + denser forms-per-root;
             density effect = enrclean - v2clean)

Cleaning: explode pipe-joined variants into rows, strip non-Arabic chars,
drop residues < 2 chars, dedupe (surface, root), re-quarantine surfaces with
>1 gold root. Roots are untouched. New SHA freeze in splits_v2/_summary.json
(results on the cleaned data are not comparable with the pre-cleaning runs).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from grokkingroots.data.splits import dcrit_splits, save_split

NON_ARABIC = re.compile(r"[^ء-يٱ-ۓ]")
OUT = Path("artefacts/phase0")
SPLITS_OUT = OUT / "splits_v2"


def clean(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        for part in str(r["surface_clean"]).split("|"):
            s = NON_ARABIC.sub("", part)
            if len(s) >= 2:
                rr = r.copy()
                rr["surface_clean"] = s
                rows.append(rr)
    out = pd.DataFrame(rows).drop_duplicates(subset=["surface_clean", "root_clean"])
    # re-quarantine ambiguity introduced by cleaning (e.g. شر1/شر2 collapsing)
    amb = out.groupby("surface_clean")["root_clean"].nunique()
    ambiguous = set(amb[amb > 1].index)
    return out[~out["surface_clean"].isin(ambiguous)].reset_index(drop=True)


def build(name: str, src: str, summary: dict):
    df = pd.read_parquet(src)
    n0 = len(df)
    dfc = clean(df)
    dfc.to_parquet(OUT / f"roots_dataset_{name}.parquet", index=False)
    l2 = dfc[dfc["difficulty_level"] == "L2"].reset_index(drop=True)
    print(f"{name}: {n0} -> {len(dfc)} rows after cleaning "
          f"({len(dfc.surface_clean.unique())} surfaces); "
          f"L2 {len(l2)} pairs / {l2.root_id.nunique()} roots")
    for p, (sp, meta) in dcrit_splits(l2, [0.30], seed=42).items():
        meta = {**meta, "variant": name, "cleaned": True}
        sha = save_split(SPLITS_OUT / f"L2_dcrit_p30_{name}.json", sp, meta)
        tr = {tuple(k) for k in sp["train"]}
        te = {tuple(k) for k in sp["test"]}
        assert not tr & te, f"{name}: train/test surface overlap"
        summary[f"L2_dcrit_p30_{name}"] = {
            **meta, "sizes": {k: len(v) for k, v in sp.items()},
            "test_sha256": sha[:12], "surface_overlap": 0,
        }
        print(f"  L2_dcrit_p30_{name}: " +
              " ".join(f"{k}={len(v)}" for k, v in sp.items()) + f" sha={sha[:12]}")


def main():
    SPLITS_OUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    build("v2clean", str(OUT / "roots_dataset.parquet"), summary)
    if (OUT / "roots_dataset_enriched.parquet").exists():   # optional enriched variant, not used in the paper
        build("enrclean", str(OUT / "roots_dataset_enriched.parquet"), summary)
    (SPLITS_OUT / "_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nfreeze written to {SPLITS_OUT}/_summary.json")


if __name__ == "__main__":
    main()
