"""Build artefacts/phase0/roots_dataset.parquet from data/raw/qabas.csv.

Usage:
    python scripts/data/build_dataset.py [--qabas PATH] [--out PATH] [--all-languages]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from grokkingroots.data.roots_dataset import build_from_qabas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qabas", default="data/raw/qabas.csv")
    ap.add_argument("--out", default="artefacts/phase0/roots_dataset.parquet")
    ap.add_argument("--all-languages", action="store_true",
                    help="do not restrict to MSA (فصحى حديثة)")
    args = ap.parse_args()

    main_df, ambig = build_from_qabas(args.qabas, msa_only=not args.all_languages)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    main_df.to_parquet(out, index=False)
    ambig.to_parquet(out.with_name("roots_dataset_ambiguous.parquet"), index=False)

    lemmas_per_root = main_df.groupby("root_clean").size()
    l1 = main_df[main_df["difficulty_level"] == "L1"]
    l1_lpr = l1.groupby("root_clean").size()
    stats = {
        "n_pairs": int(len(main_df)),
        "n_roots": int(main_df["root_clean"].nunique()),
        "n_ambiguous_pairs": int(len(ambig)),
        "difficulty_counts": main_df["difficulty_level"].value_counts().to_dict(),
        "root_type_counts": main_df["root_type"].value_counts().to_dict(),
        "lemmas_per_root_mean": round(float(lemmas_per_root.mean()), 2),
        "lemmas_per_root_median": int(lemmas_per_root.median()),
        "L1_pairs": int(len(l1)),
        "L1_roots": int(l1["root_clean"].nunique()),
        "L1_roots_with_ge5_lemmas": int((l1_lpr >= 5).sum()),
        "L1_lemmas_per_root_mean": round(float(l1_lpr.mean()), 2) if len(l1_lpr) else 0.0,
    }
    out.with_name("roots_dataset_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
