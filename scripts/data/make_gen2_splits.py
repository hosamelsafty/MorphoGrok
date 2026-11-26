"""Gen-2: multi-form (paradigm) input splits — the information-supply law.

Design: a k-form example is ONE surface string
    f_target ␟ f_sup1 ␟ ... ␟ f_sup(k-1)
joined with DELIM ('؞', U+061E, absent from the data), so the existing
encoder/loss/decoder work unchanged. The model must output the root of the
TARGET (first) form; supports are context.

Conditions (all share the SAME test-root partition, restricted to roots with
>= MIN_FORMS forms in the enrclean dataset, so the accuracy-vs-k curve is not
confounded by root-set changes):
  k1        single form                     (baseline / anchors the law)
  k2_para   target + 1 true paradigm form   (dose midpoint)
  k3_para   target + 2 true paradigm forms  (treatment)
  k3_self   target repeated 3x              (length/compute control)
  k3_shuf   target + 2 forms of OTHER roots (paradigm-specificity control)

Honest caveat carried in the paper: at test time supports for UNSEEN roots
come from oracle root-clustering — gen-2 measures the LAW; gen-3
(consistency training) is the deployable method.

Support forms are sampled deterministically (seeded) from the root's other
forms ANYWHERE in the dataset. This never leaks labels: split is
root-disjoint, so a test root's forms appear in no training example at all;
supports are inputs, not labels.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path

import pandas as pd

DELIM = "؞"          # U+061E Arabic triple dot punctuation — not in the data
MIN_FORMS = 3
SEED = 42
# Base data variant: "enrclean" (default, original frozen splits) or "v2clean"
# (cleaned; the fair base — enrclean's higher ceiling is junk-inflated).
VARIANT = os.environ.get("GEN2_VARIANT", "v2clean")
_SUF = "" if VARIANT == "enrclean" else f"_{VARIANT}"
DATASET = f"artefacts/phase0/roots_dataset_{VARIANT}.parquet"
BASE_SPLIT = f"artefacts/phase0/splits_v2/L2_dcrit_p30_{VARIANT}.json"
OUT = Path(f"artefacts/phase0/splits_gen2{_SUF}")


def sha256_of_keys(keys) -> str:
    blob = json.dumps(sorted("\t".join(k) for k in keys), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def main():
    rng = random.Random(SEED)
    df = pd.read_parquet(DATASET)
    assert not df["surface_clean"].str.contains(DELIM).any(), "DELIM occurs in data"
    forms_by_root = df.groupby("root_clean")["surface_clean"].apply(list).to_dict()
    all_surfaces = df["surface_clean"].tolist()

    base = json.loads(Path(BASE_SPLIT).read_text(encoding="utf-8"))
    eligible = {r for r, fs in forms_by_root.items() if len(set(fs)) >= MIN_FORMS}

    def restrict(pairs):
        return [(s, r) for s, r in pairs if r in eligible]

    parts = {p: restrict(base[p]) for p in ("train", "val", "test")}
    print(f"eligible roots (>= {MIN_FORMS} forms): {len(eligible)}; "
          + " ".join(f"{p}={len(v)}" for p, v in parts.items()))

    def supports(surface, root, n, mode):
        if mode == "self":
            return [surface] * n
        if mode == "shuf":
            out = []
            while len(out) < n:
                c = rng.choice(all_surfaces)
                if c != surface and c not in out:
                    out.append(c)
            return out
        pool = [f for f in dict.fromkeys(forms_by_root[root]) if f != surface]
        rng.shuffle(pool)
        out = pool[:n]
        while len(out) < n:          # root has exactly MIN_FORMS incl. target
            out.append(pool[len(out) % max(1, len(pool))])
        return out

    def build(name, k, mode):
        cond = {}
        for part, pairs in parts.items():
            new = []
            for s, r in pairs:
                sup = supports(s, r, k - 1, mode) if k > 1 else []
                new.append([DELIM.join([s] + sup), r])
            cond[part] = new
        tr_targets = {p[0].split(DELIM)[0] for p in cond["train"]}
        te_targets = {p[0].split(DELIM)[0] for p in cond["test"]}
        assert not tr_targets & te_targets, f"{name}: target surface leakage"
        meta = {"family": "gen2_multiform", "k": k, "mode": mode,
                "min_forms": MIN_FORMS, "delim": DELIM, "base": BASE_SPLIT,
                "seed": SEED}
        payload = {"meta": meta, "test_sha256": sha256_of_keys(cond["test"]), **cond}
        (OUT / f"{name}.json").write_text(json.dumps(payload, ensure_ascii=False),
                                          encoding="utf-8")
        summary[name] = {**meta, "sizes": {p: len(v) for p, v in cond.items()},
                         "test_sha256": payload["test_sha256"][:12]}
        print(f"{name}: " + " ".join(f"{p}={len(v)}" for p, v in cond.items())
              + f" sha={payload['test_sha256'][:12]}")

    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    build("L2_p30_k1", 1, "para")
    build("L2_p30_k2_para", 2, "para")
    build("L2_p30_k3_para", 3, "para")
    build("L2_p30_k3_self", 3, "self")
    build("L2_p30_k3_shuf", 3, "shuf")
    (OUT / "_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                       encoding="utf-8")
    print(f"\nfreeze written to {OUT}/_summary.json")


if __name__ == "__main__":
    main()
