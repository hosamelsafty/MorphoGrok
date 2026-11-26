"""Root inventory with root types: artefacts/phase0/root_type_map.json.

The inventory is every root of the cleaned lexicon with its type (sound,
hollow, defective, ...). It is the candidate set of inventory-constrained
decoding and the label source of the auxiliary root-type loss.

Usage:
    python scripts/data/make_root_type_map.py \
        [--dataset artefacts/phase0/roots_dataset_v2clean.parquet] \
        [--out artefacts/phase0/root_type_map.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="artefacts/phase0/roots_dataset_v2clean.parquet")
    ap.add_argument("--out", default="artefacts/phase0/root_type_map.json")
    args = ap.parse_args()
    df = pd.read_parquet(args.dataset)
    mapping = dict(zip(df["root_clean"], df["root_type"]))
    payload = {"types": sorted(set(mapping.values())), "map": mapping}
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(f"{len(mapping)} roots, {len(payload['types'])} types -> {args.out}")


if __name__ == "__main__":
    main()
