"""P1: build DIACRITIZED-input splits by swapping surface_clean -> surface_diacritized
in the existing frozen splits. Root partitioning is inherited exactly (same files),
so train/val/test roots are identical to the undiacritized cells — only the input
representation changes. Recomputes the test SHA freeze.

Usage: python scripts/data/make_diac_splits.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

SPLITS = Path("artefacts/phase0/splits")
OUT = Path("artefacts/phase0/splits_diac")
CELLS = ["L1_dcrit_p05", "L1_dcrit_p30", "L2_dcrit_p05", "L2_dcrit_p30"]


def sha256_of_keys(keys) -> str:
    blob = json.dumps(sorted("\t".join(k) for k in keys), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def main():
    m = pd.read_parquet("artefacts/phase0/roots_dataset.parquet")
    diac = {(s, r): d for s, r, d in
            zip(m["surface_clean"], m["root_clean"], m["surface_diacritized"])}
    OUT.mkdir(parents=True, exist_ok=True)

    def remap(keys):
        out = []
        for s, r in keys:
            d = diac.get((s, r))
            if d is None:
                raise KeyError(f"no diacritized surface for ({s!r},{r!r})")
            out.append([d, r])
        return out

    for cell in CELLS:
        sp = json.loads((SPLITS / f"{cell}.json").read_text())
        new = {"train": remap(sp["train"]), "val": remap(sp["val"]), "test": remap(sp["test"])}
        meta = {**sp["meta"], "input": "diacritized"}
        payload = {"meta": meta, "test_sha256": sha256_of_keys(new["test"]), **new}
        (OUT / f"{cell}.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        # sanity: same partition sizes as the source
        assert all(len(new[k]) == len(sp[k]) for k in ("train", "val", "test"))
        print(f"{cell}: train={len(new['train'])} val={len(new['val'])} test={len(new['test'])} "
              f"sha={payload['test_sha256'][:12]}")
    print(f"\nWrote {len(CELLS)} diacritized split cells to {OUT}/")


if __name__ == "__main__":
    main()
