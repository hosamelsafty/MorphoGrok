"""The ceiling procedure, formalized.

Definition. For a task mapping surfaces x to roots y with reference lexicon R
(the language, not the test labels), an *information model* phi maps x to the
evidence it logically carries about y. The phi-ceiling of a test set T is the
accuracy of the Bayes-optimal predictor that sees only phi(x):

    C_phi(T) = (1/|T|) * sum_{(x,y) in T}  max_y'  P_hat(y' | phi(x))

with P_hat estimated from R. No model whose input carries at most phi(x)'s
evidence can exceed C_phi in expectation; the observed plateau is read against
it. Reported with a deterministic seed-level bootstrap CI over test items.

Two instantiations (both functions of x alone — never of the gold label):

  --phi exact    phi(x) = x. Groups R by exact surface; ambiguity = surface
                 collisions. Generalizes scripts/ceiling/oracle_ceiling.py (surfaces
                 absent from R are *unconstrained*: contribute 1.0, upper bound).

  --phi compat   phi(x) = the set of inventory roots COMPATIBLE with x under
                 the deletion rules of Arabic weak morphology: every strong
                 radical must appear in order in x (hamza forms / ya~alif-maqsura
                 treated as carriers); weak/hamza radicals may be deleted.
                 P_hat(y|x) proportional to lexicon frequency of y among the
                 compatible set. This is the generalized, x-only form of
                 scripts/ceiling/ambiguity_audit.py's visible/hidden slot-prior audit.

  --phi template phi(x) = compatibility PLUS the pattern evidence. Decomposing
                 x under candidate y induces a template signature (the surface
                 with matched radicals replaced by slot digits, plus the set of
                 deleted slots) — e.g. qal/qwl -> "1a3"+del{2}. Signature
                 frequencies are estimated from R's gold decompositions, and
                 P_hat(y|x) proportional to  P(y) * max_decomp P(template).
                 Root-template independence is the classic Semitic assumption;
                 this is the tight(er) evidence model the bracket needs.

Diagnostics reported: candidate-set size histogram, gold-in-candidates rate
(violations of the deletion model), per-bucket ceilings (root-substring "copy"
items vs true "restoration" items), and the strict upper-bound convention
(model-violation or empty-candidate items contribute 1.0).

Sensitivity knobs, all defaulting to the published configuration so that the
frozen artefacts stay reproducible byte-for-byte: --smoothing (additive
pseudo-count), --score (best-explanation max vs the marginal sum over
decompositions), --no-loo (drop the leave-one-out correction), --prior (the
root prior: lexicon frequency, uniform, or frequency net of the test set's own
contribution), --template-source (estimate the template statistics from the
whole lexicon, or from the split's TRAINING pairs only, i.e. from what the
model itself was shown), --lenient (score under the و/ي-lenient criterion:
candidates differing only by waw<->ya are merged before the max). Each is
recorded in the output JSON; scripts/ceiling/ceiling_sensitivity_table.py tabulates a
sweep.

No GPU; pure lexicon analysis. Usage:
  uv run --with pandas --with pyarrow python scripts/ceiling/ceiling_method.py \
      --dataset artefacts/phase0/roots_dataset_v2clean.parquet \
      --split artefacts/phase0/splits_gen2_v2clean/L2_p30_k1.json \
      --phi compat --out artefacts/phase0/ceiling_k1_v2clean_compat.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))  # aggregate_ci
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from aggregate_ci import bootstrap_ci  # deterministic LCG bootstrap (no time deps)
from grokkingroots.data.arabic_text import lenient_key

HAMZA = "ءأإؤئآ"
CARRIERS = {**{h: HAMZA for h in HAMZA},
            "و": "وؤ", "ي": "يىئ", "ا": "اآأإى"}
WEAK = set("اوي") | set(HAMZA)
# Gemination absorption (opt-in, --gemination): a radical identical to its
# predecessor may be absorbed into the predecessor's surface letter (the shadda
# is a diacritic, absent from surface_clean) -- e.g. شاك realizes شكك with one
# visible kaf. OFF by default so every published L1/L2 ceiling stays
# byte-identical; required for L3 (doubled) roots, where 44% of test items are
# otherwise deletion-model violations.
GEMINATE = False


def compatible(root: str, surface: str) -> bool:
    """Could `surface` be a realization of `root` under weak-deletion rules?

    Strong radicals must appear (via carriers) in order; weak/hamza radicals may
    be deleted. Small backtracking search — roots are <= 5 chars, so the skip
    branching is trivial and greedy-matching pitfalls are avoided.
    """
    def rec(ri: int, si: int) -> bool:
        if ri == len(root):
            return True
        rad = root[ri]
        allowed = CARRIERS.get(rad, rad)
        for k in range(si, len(surface)):
            if surface[k] in allowed and rec(ri + 1, k + 1):
                return True
        if rad in WEAK:                       # deletable radical
            return rec(ri + 1, si)
        if GEMINATE and ri > 0 and rad == root[ri - 1]:
            return rec(ri + 1, si)            # absorbed into the geminate
        return False

    return rec(0, 0)


def decompositions(root: str, surface: str):
    """All template signatures under which `surface` realizes `root`.

    A decomposition assigns each radical either a surface position (via
    carriers, in order) or deletion (weak/hamza only). Its signature is
    (template, deleted_slots): the surface with matched positions replaced by
    the 1-based slot digit, plus the frozenset of deleted slot indices.
    E.g. qal / q-w-l -> ("1a3", {1});  maktub / k-t-b -> ("m12w3", set()).
    """
    sigs = set()

    def rec(ri: int, si: int, assign):
        if ri == len(root):
            tpl = list(surface)
            deleted = frozenset(i for i, p in enumerate(assign) if p is None)
            for i, p in enumerate(assign):
                if p is not None:
                    tpl[p] = str(i + 1)
            sigs.add(("".join(tpl), deleted))
            return
        rad = root[ri]
        allowed = CARRIERS.get(rad, rad)
        for k in range(si, len(surface)):
            if surface[k] in allowed:
                rec(ri + 1, k + 1, assign + [k])
        if rad in WEAK:
            rec(ri + 1, si, assign + [None])
        if GEMINATE and ri > 0 and rad == root[ri - 1]:
            rec(ri + 1, si, assign + [None])  # absorbed into the geminate

    rec(0, 0, [])
    return sigs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="reference lexicon parquet")
    ap.add_argument("--extra-dataset", default=None,
                    help="optional second parquet folded into R (e.g. ambiguous bin)")
    ap.add_argument("--split", required=True)
    ap.add_argument("--phi", choices=["exact", "compat", "template"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gemination", action="store_true",
                    help="extend the deletion model with gemination absorption "
                         "(a radical identical to its predecessor may surface "
                         "as one letter). Needed for L3/doubled roots; leave "
                         "OFF elsewhere so published ceilings are unchanged.")
    ap.add_argument("--smoothing", type=float, default=0.0,
                    help="phi-sensitivity check: additive "
                         "pseudo-count added to both the root-frequency and "
                         "template-frequency terms before scoring. 0.0 (default) "
                         "reproduces the exact unsmoothed estimator byte-for-byte. "
                         "Must be >= 0 (a negative value can drive individual "
                         "sc[r] terms negative, producing an out-of-[0,1] ceiling).")
    ap.add_argument("--score", choices=["max", "sum"], default="max",
                    help="how to combine a candidate's template signatures. "
                         "max (default, published): score y by its best "
                         "explanation. sum: marginalize over D(x,y), the "
                         "literal consequence of root-template independence.")
    ap.add_argument("--no-loo", action="store_true",
                    help="disable the leave-one-out correction, i.e. let the "
                         "test item's own gold pair count towards its own "
                         "template signatures and root frequency. Quantifies "
                         "what LOO buys; not a publishable configuration.")
    ap.add_argument("--prior", choices=["freq", "uniform", "matched"],
                    default="freq",
                    help="the root prior P(y). freq (default, published): "
                         "lexicon root frequency. uniform: 1 for every "
                         "candidate. matched: lexicon frequency net of the "
                         "test set's own contribution (floor 1).")
    ap.add_argument("--template-source", choices=["lexicon", "train"],
                    default="lexicon",
                    help="where the template-signature counts come from. "
                         "lexicon (default, published): every pair in R. "
                         "train: only the split's training pairs, so the "
                         "evidence model sees no more form-root pairs than "
                         "the neural model does (no LOO needed: a test pair "
                         "is never in train).")
    ap.add_argument("--lenient", action="store_true",
                    help="و/ي-lenient criterion: sum the posterior over "
                         "candidates that differ only by waw<->ya before "
                         "taking the max, and count the argmax class as "
                         "correct if it contains the gold root.")
    ap.add_argument("--items-out", default=None,
                    help="optional path for the per-item records "
                         "(surface, gold, argmax prediction, max posterior).")
    args = ap.parse_args()
    if args.smoothing < 0:
        ap.error("--smoothing must be >= 0")
    global GEMINATE
    GEMINATE = args.gemination

    import pandas as pd
    R = pd.read_parquet(args.dataset)
    if args.extra_dataset:
        R = pd.concat([R, pd.read_parquet(args.extra_dataset)], ignore_index=True)
    surf2roots: dict[str, Counter] = {}
    root_freq: Counter = Counter()
    for s, r in zip(R["surface_clean"], R["root_clean"]):
        surf2roots.setdefault(s, Counter())[r] += 1
        root_freq[r] += 1
    inventory = sorted(root_freq)

    split = json.loads(Path(args.split).read_text(encoding="utf-8"))
    # gen2 splits carry delimiter-joined context; the ceiling is about the
    # single query form = the FIRST field
    test = [(x.split("؞")[0], y) for x, y in split["test"]]

    # per-root test-item counts; used only by --prior matched
    test_count: Counter = Counter(y for _, y in test)

    def prior(r: str, loo: int) -> int:
        """P(y) up to normalization, per --prior. loo applies to freq only."""
        if args.prior == "uniform":
            return 1
        if args.prior == "matched":
            return max(root_freq[r] - test_count[r], 1)
        return max(root_freq[r] - loo, 0)

    # template-signature frequencies from R's gold decompositions (x-only at
    # test time: signatures are induced by CANDIDATE roots, never the gold)
    T_freq: Counter = Counter()
    if args.phi == "template":
        if args.template_source == "train":
            tpl_pairs = [(x.split("؞")[0], y) for x, y in split["train"]]
        else:
            tpl_pairs = list(zip(R["surface_clean"], R["root_clean"]))
        for s, r in tpl_pairs:
            for sig in decompositions(r, s):
                T_freq[sig] += 1
        print(f"template signatures: {len(T_freq)} distinct from {len(tpl_pairs)} pairs")

    def merged(scores):
        """Collapse و/ي-equivalent candidates (identity unless --lenient)."""
        if not args.lenient:
            return scores
        out: Counter = Counter()
        for r, v in scores.items():
            out[lenient_key(r)] += v
        return out

    per_item, records = [], []
    n_unconstrained = n_violation = n_template_fallback = 0
    cand_sizes, argmax_hits = [], []
    for x, gold in test:
        gold_key = lenient_key(gold) if args.lenient else gold
        pred = None
        if args.phi == "exact":
            c = surf2roots.get(x)
            if c is None:                       # unseen surface: unconstrained
                mp, csz, gold_in = 1.0, 0, False
                n_unconstrained += 1
            else:
                tot = sum(c.values())
                mp, csz, gold_in = max(merged(c).values()) / tot, len(c), gold in c
        else:  # compat / template
            cs = [r for r in inventory if compatible(r, x)]
            csz, gold_in = len(cs), gold in cs
            if not cs or not gold_in:           # deletion-model violation
                mp = 1.0
                n_violation += 1
            elif args.phi == "compat":
                pr = merged({r: prior(r, 0) for r in cs})
                tot = sum(pr.values())
                mp = max(pr.values()) / tot
            else:  # template: P(y|x) ∝ P(y) * best-explanation template freq.
                # Leave-one-out: this test pair itself contributed its gold
                # decompositions to T_freq and one count to root_freq[gold];
                # subtract both when scoring the gold candidate so the ceiling
                # never peeks at the item's own mapping.
                own_sigs = decompositions(gold, x)
                sc = {}
                a = args.smoothing
                combine = max if args.score == "max" else sum
                for r in cs:
                    loo = 0 if args.no_loo else (1 if r == gold else 0)
                    # a test pair never contributed to train-only counts
                    tloo = 0 if args.template_source == "train" else loo
                    adj = [max(T_freq[g] - (tloo if g in own_sigs else 0), 0)
                           for g in decompositions(r, x)]
                    ev = combine(adj) if adj else 0
                    sc[r] = (prior(r, loo) + a) * (ev + a)
                sc = merged(sc)
                tot = sum(sc.values())
                # NOTE: once a>0, every sc[r] has a floor of a*a > 0, so tot==0
                # can no longer happen and n_template_fallback is structurally
                # unreachable (it stops meaning "template evidence covers this
                # item" and just reads 0 for any smoothed run) -- diagnostic
                # only, doesn't affect ceiling/realized_argmax_acc correctness.
                if tot == 0:                    # no seen template explains x
                    n_template_fallback += 1
                    pr = merged({r: prior(r, 0) for r in cs})
                    tot = sum(pr.values())
                    mp = max(pr.values()) / tot
                    pred = max(pr, key=pr.get)
                else:
                    mp = max(sc.values()) / tot
                    pred = max(sc, key=sc.get)
                argmax_hits.append(pred == gold_key)
        cand_sizes.append(csz)
        per_item.append(mp)
        records.append({"surface": x, "gold": gold, "pred": pred, "max_post": round(mp, 4),
                        "n_cands": csz, "gold_in_cands": gold_in,
                        "bucket": "substring" if gold in x else "restoration"})

    mean, lo, hi = bootstrap_ci(per_item)
    buckets = {}
    for b in ("substring", "restoration"):
        vals = [r["max_post"] for r in records if r["bucket"] == b]
        if vals:
            m, l, h = bootstrap_ci(vals)
            buckets[b] = {"ceiling": round(m, 4), "ci95": [round(l, 4), round(h, 4)],
                          "n": len(vals)}
    gold_in_rate = sum(r["gold_in_cands"] for r in records) / len(records)
    out = {
        "phi": args.phi, "gemination": args.gemination,
        "score": args.score, "loo": not args.no_loo, "prior": args.prior,
        "smoothing": args.smoothing,
        "template_source": args.template_source, "lenient": args.lenient,
        "dataset": args.dataset, "split": args.split,
        "n_test": len(test), "n_inventory": len(inventory),
        "ceiling": round(mean, 4), "ceiling_ci95": [round(lo, 4), round(hi, 4)],
        "buckets": buckets,
        "gold_in_candidates_rate": round(gold_in_rate, 4),
        "n_unconstrained_items": n_unconstrained,
        "n_deletion_model_violations": n_violation,
        "n_template_fallback": n_template_fallback,
        "realized_argmax_acc": round(sum(argmax_hits) / len(argmax_hits), 4)
                               if argmax_hits else None,
        "cand_size_quartiles": sorted(cand_sizes)[:: max(1, len(cand_sizes) // 4)][:5],
        "note": "upper bound: unconstrained/violation items contribute 1.0",
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    if args.items_out:
        Path(args.items_out).write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
