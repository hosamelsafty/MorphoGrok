"""Off-the-shelf Arabic root/stem tools on the frozen splits (paper Table 1 extension).

Extends scripts/isri_baseline.py from one stemmer to the runnable field:

  pattern/rule systems (no root dictionary -> no test-root leakage)
    skeleton    our consonant-copy floor
    isri        ISRI stemmer (Taghva et al., 2005; NLTK)
    arlstem     ARLSTem (Abainia et al., 2017; NLTK)
    arlstem2    ARLSTem2 (Abainia, 2020; NLTK) -- root-oriented
    snowball    Assem's Arabic light stemmer (Snowball)
    tashaphyne  Tashaphyne light stemmer, get_root() (Zerrouki)

  lexicon-backed analyzers (root read out of a dictionary -> see caveat)
    qalsadi     Qalsadi/Arramooz analyzer (Zerrouki)
    camel       CAMeL Tools calima-msa-r13 analyzer (Obeid et al., 2020)

Analyzers return a *set* of analyses, so they get two scores: top-1 under the
system's own prior (qalsadi `freq`, camel `pos_lex_logprob`) and an oracle
"gold root anywhere in the analysis set" upper bound, which also measures how
much of our held-out root inventory sits in their lexicon. calima-msa-r13
writes '#' for a radical it does not determine; scored strictly (# never
matches) and with # as a wildcard, an oracle upper bound on that system with
perfect weak-radical restoration.

Metric is the paper's: strict exact-root match, plus a hamza/ya-lenient
secondary, bucketed by the copy tag (root contiguous in the query vs restored).

Usage (needs the optional tools: pip install -e '.[baselines]', see README):
    python scripts/baselines/arabic_tools_baseline.py --out artefacts/phase0/arabic_tools.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Tashaphyne and Qalsadi iterate string sets internally, so Python's per-process
# hash randomization changes their output: unpinned, tashaphyne's L2 score moves
# over a ~0.03 range run to run and qalsadi's top-1 flips on ties. Pin the seed
# and re-exec once, so the script is reproducible however it is invoked.
if os.environ.get("PYTHONHASHSEED") != "0":
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable] + sys.argv)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from grokkingroots.baselines.skeleton import skeleton_root  # noqa: E402

HAMZA = str.maketrans({'أ': 'ا', 'إ': 'ا', 'آ': 'ا', 'ؤ': 'ء', 'ئ': 'ء', 'ى': 'ي'})
DIAC = str.maketrans('', '', 'ًٌٍَُِّْـ')

# (tag, path, n_radicals) -- the exact frozen sets the paper reports on.
SPLITS = [
    ("L1", "artefacts/phase0/splits/L1_dcrit_p30.json", 3),
    ("L2", "artefacts/phase0/splits_gen2_v2clean/L2_p30_k1.json", 3),
    ("L3", "artefacts/phase0/splits_l34_clean/L3_dcrit_p30.json", 3),
    ("L4", "artefacts/phase0/splits_l34_clean/L4_dcrit_p30.json", 4),
]


def norm(s: str) -> str:
    return s.translate(HAMZA)


def load(path: str, delim: str = "؞"):
    d = json.load(open(path))
    return [(x.split(delim)[0], y) for x, y in d["test"]], d.get("test_sha256")


# --- single-output systems: word -> one predicted root -------------------------

def build_single_systems(n: int):
    from nltk.stem.isri import ISRIStemmer
    from nltk.stem.arlstem import ARLSTem
    from nltk.stem.arlstem2 import ARLSTem2
    import snowballstemmer
    from tashaphyne.stemming import ArabicLightStemmer

    isri, arl, arl2 = ISRIStemmer(), ARLSTem(), ARLSTem2()
    snow = snowballstemmer.stemmer('arabic')
    tash = ArabicLightStemmer()

    def tash_root(w):
        tash.light_stem(w)
        return tash.get_root()

    return {
        "skeleton": lambda w: skeleton_root(w, n=n),
        "isri": isri.stem,
        "arlstem": arl.stem,
        "arlstem2": arl2.stem,
        "snowball": snow.stemWord,
        "tashaphyne": tash_root,
    }


# --- analyzers: word -> ranked list of candidate roots -------------------------

def build_qalsadi():
    import qalsadi.analex as qa
    an = qa.Analex()

    def cand(w):
        try:
            res = an.check_word(w)
        except Exception:
            return []
        seen, out = set(), []
        for r in sorted(res, key=lambda r: -(r.freq or 0)):
            root = (r.root or "").translate(DIAC).strip()
            if root and root not in seen:
                seen.add(root)
                out.append(root)
        return out

    return cand


def build_camel():
    from camel_tools.morphology.database import MorphologyDB
    from camel_tools.morphology.analyzer import Analyzer
    an = Analyzer(MorphologyDB.builtin_db('calima-msa-r13', 'a'))

    def cand(w):
        try:
            res = an.analyze(w)
        except Exception:
            return []
        seen, out = set(), []
        for a in sorted(res, key=lambda a: -(a.get('pos_lex_logprob') or -99)):
            root = (a.get('root') or "").replace('.', '').translate(DIAC).strip()
            # 'NTWS' and all-'#' are calima's explicit "no root determined"
            # sentinels; drop them so the system is scored on its real guesses.
            if not root or root == '#' * len(root) or not any('؀' <= c <= 'ۿ' for c in root):
                continue
            if root not in seen:
                seen.add(root)
                out.append(root)
        return out

    return cand


def wildcard_eq(pred: str, gold: str) -> bool:
    """calima writes '#' for an undetermined radical; treat it as a free match."""
    if len(pred) != len(gold):
        return False
    return all(p == '#' or p == g for p, g in zip(pred, gold))


def blank(keys):
    return {k: {b: [0, 0, 0] for b in ('all', 'substring', 'restoration')} for k in keys}


def score(hits, key, bucket, strict, lenient):
    for b in ('all', bucket):
        hits[key][b][0] += 1
        hits[key][b][1] += int(strict)
        hits[key][b][2] += int(lenient)


def evaluate(tag: str, path: str, n: int):
    test, sha = load(path)
    single = build_single_systems(n)
    qal, cam = build_qalsadi(), build_camel()

    keys = list(single) + ["qalsadi", "qalsadi_oracle", "camel", "camel_oracle",
                           "camel_wild", "camel_wild_oracle"]
    hits = blank(keys)
    cover = {"qalsadi": 0, "camel": 0}
    mask = []

    for i, (x, y) in enumerate(test):
        bucket = 'substring' if y in x else 'restoration'
        for k, fn in single.items():
            p = fn(x) or ""
            score(hits, k, bucket, p == y, norm(p) == norm(y))

        qc = qal(x)
        cover["qalsadi"] += bool(qc)
        score(hits, "qalsadi", bucket, bool(qc) and qc[0] == y,
              bool(qc) and norm(qc[0]) == norm(y))
        score(hits, "qalsadi_oracle", bucket, y in qc,
              norm(y) in [norm(c) for c in qc])

        cc = cam(x)
        cover["camel"] += bool(cc)
        score(hits, "camel", bucket, bool(cc) and cc[0] == y,
              bool(cc) and norm(cc[0]) == norm(y))
        score(hits, "camel_oracle", bucket, y in cc,
              norm(y) in [norm(c) for c in cc])
        score(hits, "camel_wild", bucket, bool(cc) and wildcard_eq(cc[0], y),
              bool(cc) and wildcard_eq(norm(cc[0]), norm(y)))
        score(hits, "camel_wild_oracle", bucket, any(wildcard_eq(c, y) for c in cc),
              any(wildcard_eq(norm(c), norm(y)) for c in cc))

        # per-item coverage mask: which analyzer returned any analysis at all.
        # Lets the model be re-scored on each analyzer's covered subset, so the
        # covered-item rates in the paper are a matched comparison, not a ratio.
        mask.append((int(bool(qc)), int(bool(cc)),
                     int(bool(qc) and qc[0] == y),
                     int(bool(cc) and wildcard_eq(cc[0], y))))

        if (i + 1) % 500 == 0:
            print(f"  [{tag}] {i + 1}/{len(test)}", file=sys.stderr, flush=True)

    out = {"split": tag, "path": path, "n": len(test), "test_sha256": sha,
           "coverage": {k: v / len(test) for k, v in cover.items()},
           "systems": {}}
    for k in keys:
        out["systems"][k] = {
            b: {"n": v[0], "strict": v[1] / v[0] if v[0] else None,
                "lenient": v[2] / v[0] if v[0] else None}
            for b, v in hits[k].items()}
    # boolean arrays aligned to the frozen test-set order; join by index (the
    # split is SHA-pinned, so index is a stable key and costs no disk).
    return out, {"split": tag, "path": path, "n": len(test), "test_sha256": sha,
                 "order": "aligned to test[] in the frozen split file",
                 "qalsadi_covered": [m[0] for m in mask],
                 "camel_covered": [m[1] for m in mask],
                 "qalsadi_hit": [m[2] for m in mask],
                 "camel_wild_hit": [m[3] for m in mask]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artefacts/phase0/arabic_tools.json")
    ap.add_argument("--mask-out", default="artefacts/phase0/arabic_tools_coverage.json",
                    help="per-item analyzer coverage mask (written alongside --out)")
    ap.add_argument("--splits", nargs="*", default=[s[0] for s in SPLITS])
    args = ap.parse_args()

    results, masks = [], []
    for tag, path, n in SPLITS:
        if tag not in args.splits:
            continue
        print(f"=== {tag} ({path}) ===", file=sys.stderr, flush=True)
        r, m = evaluate(tag, path, n)
        results.append(r)
        masks.append(m)
        print(f"--- {tag}  n={r['n']}  sha={str(r['test_sha256'])[:12]} ---")
        print(f"    {'system':20s} {'strict':>8s} {'lenient':>8s} "
              f"{'substr':>8s} {'restor':>8s}")
        for k, v in r["systems"].items():
            sub = v['substring']['strict']
            res = v['restoration']['strict']
            print(f"    {k:20s} {v['all']['strict']:8.4f} {v['all']['lenient']:8.4f} "
                  f"{(f'{sub:.4f}' if sub is not None else '  --  '):>8s} "
                  f"{(f'{res:.4f}' if res is not None else '  --  '):>8s}")
        print(f"    coverage: " + ", ".join(f"{k}={v:.3f}" for k, v in r["coverage"].items()))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(args.out, "w"), ensure_ascii=False, indent=2)
    json.dump(masks, open(args.mask_out, "w"), ensure_ascii=False, separators=(",", ":"))
    print(f"\nwrote {args.out} and {args.mask_out}", file=sys.stderr)


if __name__ == "__main__":
    main()
