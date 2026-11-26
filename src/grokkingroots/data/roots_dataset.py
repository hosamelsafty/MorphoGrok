"""Build the surface->root dataset from the Qabas lexicon (self-contained).

Gold roots come directly from Qabas' `root` column (deterministic; no analyzer).
Each lemma is a distinct surface form (a distinct pattern/wazn of its root), so
surface->root over Qabas tests templatic pattern abstraction; root-disjoint
splits then make it a genuine unseen-root generalization test.
"""
from __future__ import annotations

import pandas as pd

from .arabic_text import normalize_surface
from .difficulty import classify_root_type

MSA_LANGUAGE = "فصحى حديثة"  # Modern Standard Arabic tag in Qabas `language`


def load_qabas(path) -> pd.DataFrame:
    return pd.read_csv(path)


def build_from_qabas(path, msa_only: bool = True):
    """Return (main_df, ambiguous_df).

    main_df: deterministic (surface_clean -> single root) pairs.
    ambiguous_df: surfaces mapping to >1 root, held out of the main splits.
    """
    df = load_qabas(path)
    if msa_only:
        df = df[df["language"] == MSA_LANGUAGE]
    df = df[df["root"].notna() & df["lemma"].notna()]

    rows = []
    for lemma, root in zip(df["lemma"].astype(str), df["root"].astype(str)):
        radicals = root.split()
        if len(radicals) not in (3, 4):
            continue
        surface_clean = normalize_surface(lemma)
        if len(surface_clean) < 2:
            continue
        info = classify_root_type(root)
        root_clean = "".join(radicals)
        rows.append({
            "surface_clean": surface_clean,
            "surface_diacritized": lemma.strip(),
            "root": root.strip(),
            "root_clean": root_clean,
            "root_id": f"root_{root_clean}_{len(radicals)}",
            "root_type": info["root_type"],
            "difficulty_level": info["difficulty"],
            "hamzated": info["hamzated"],
            "n_radicals": len(radicals),
            "source": "qabas",
        })

    out = pd.DataFrame(rows).drop_duplicates(subset=["surface_clean", "root_clean"])

    n_roots_per_surface = out.groupby("surface_clean")["root_clean"].transform("nunique")
    ambiguous = out[n_roots_per_surface > 1].copy()
    main = out[n_roots_per_surface == 1].copy()
    return main.reset_index(drop=True), ambiguous.reset_index(drop=True)
