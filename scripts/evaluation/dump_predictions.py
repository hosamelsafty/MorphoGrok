"""Dump per-item test predictions at each run's val-selected checkpoint.

Lenient و/ي scoring, the error taxonomy behind the
"why constrained decoding closes only part of the gap" paragraph, and any later
re-scoring all need the predictions themselves, which no earlier script kept
(fulltest_eval.py stores only the accuracy). This script re-decodes and saves:

  greedy       the unconstrained greedy root for every test item
  constrained  (--constrained) the inventory-constrained top-5 roots with their
               log-probabilities, plus the top-3 و/ي-merged classes, i.e. the
               ranking after summing the posterior over roots that differ only
               by waw<->ya (the same merge the lenient ceiling applies)

Checkpoint selection is the paper's: argmax val_exact over the logged history
(first maximum), snapped to the nearest ladder checkpoint, optionally restricted
to steps <= --max-step for the matched-budget comparison. Supervised baseline
runs (meta["arch"] set by train_supervised.py) use their ckpt_best.pt, which is
the same rule applied online.

Strict accuracies are printed so they can be checked against fulltest_eval.json
and calibrated_decode.json: the dump must reproduce those numbers exactly.

Usage:
  python scripts/evaluation/dump_predictions.py \
      --runs 'artefacts/runs/g5_L2_k1_*' --constrained \
      --out-dir artefacts/camera_ready/preds
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from constrained_topk_eval import build_candidates, make_scorer  # noqa: E402
from fulltest_eval import val_selected_step  # noqa: E402

from grokkingroots.data.arabic_text import lenient_key, lenient_match  # noqa: E402
from grokkingroots.data.encoding import build_vocab, load_split, max_input_len  # noqa: E402
from grokkingroots.model.char_transformer import CharTransformer  # noqa: E402
from grokkingroots.train.train_supervised import (  # noqa: E402
    build_model, encode_src, encode_tgt, predict,
)


def seq2seq_scorer(model, vocab, cands, device, chunk=4096):
    """score_item(surface) -> [n_cands] log P(cand + EOS | surface), enc-dec."""
    tin, tout = encode_tgt(vocab, cands, device)
    mask = tout != vocab.pad

    @torch.no_grad()
    def score_item(surface):
        src = encode_src(vocab, [surface], device)
        mem, src_pad = model.encode(src), src == vocab.pad
        rows = []
        for s in range(0, len(cands), chunk):
            n = tin[s:s + chunk].size(0)
            logp = torch.log_softmax(
                model.decode(tin[s:s + chunk], mem.expand(n, -1, -1),
                             src_pad.expand(n, -1)), dim=-1)
            lp = logp.gather(-1, tout[s:s + chunk].unsqueeze(-1)).squeeze(-1)
            rows.append((lp * mask[s:s + chunk]).sum(-1))
        return torch.cat(rows)

    return score_item


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="glob over run dirs")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-step", type=int, default=0,
                    help="restrict val-selection to steps <= this (matched budget)")
    ap.add_argument("--tag", default="", help="suffix for the output file name")
    ap.add_argument("--constrained", action="store_true")
    ap.add_argument("--inventory", default="artefacts/phase0/root_type_map.json")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    for run_dir in sorted(glob.glob(args.runs)):
        name = os.path.basename(run_dir.rstrip("/"))
        mp, hp = os.path.join(run_dir, "meta.json"), os.path.join(run_dir, "history.json")
        if not (os.path.exists(mp) and os.path.exists(hp)
                and os.path.exists(os.path.join(run_dir, "DONE"))):
            print(f"SKIP {name}: not a finished run", flush=True)
            continue
        meta, hist = json.load(open(mp)), json.load(open(hp))
        arch = meta.get("arch", "grok")
        tr, va, te, _ = load_split(meta["split"])
        allp = tr + va + te
        vocab = build_vocab(allp)
        mlen = max(max_input_len(allp), max(len(s) for s, _ in allp) + 5 + 3)

        want = val_selected_step(hist, args.max_step)
        if arch == "grok":
            ladder = {int(re.search(r"ckpt_(\d+)\.pt", p).group(1)): p
                      for p in glob.glob(os.path.join(run_dir, "ckpts", "ckpt_*.pt"))}
            got = min(ladder, key=lambda s: abs(s - want))
            ma = meta["args"]
            model = CharTransformer(len(vocab), ma["d_model"], ma["n_heads"], ma["n_layers"],
                                    ma["d_ff"], 0.0, max_len=mlen, pad_id=vocab.pad).to(device)
            sd = torch.load(ladder[got], map_location=device, weights_only=False)["model"]
            # aux-roottype runs carry a training-only head; it takes no part in decoding
            model.load_state_dict({k: v for k, v in sd.items() if not k.startswith("aux_head.")})
        else:
            ck = torch.load(os.path.join(run_dir, "ckpt_best.pt"), map_location=device,
                            weights_only=False)
            got = ck["step"]
            assert got == want, f"{name}: ckpt_best step {got} != val-selected {want}"
            model = build_model(arch, vocab, meta["args"], mlen, device)
            model.load_state_dict(ck["model"])
        model.eval()

        surfaces, golds = [s for s, _ in te], [r for _, r in te]
        greedy = predict(model, "encdec" if arch == "encdec" else "dec", vocab, surfaces, device)
        n = len(te)
        rec = {"run": name, "arch": arch, "split": meta["split"],
               "val_selected_step": want, "scored_step": got,
               "max_step": args.max_step or None, "n_test": n, "greedy": greedy,
               "greedy_strict": sum(p == g for p, g in zip(greedy, golds)) / n,
               "greedy_lenient": sum(lenient_match(p, g) for p, g in zip(greedy, golds)) / n}
        line = (f"{name:46s} step={got:>7} greedy strict={rec['greedy_strict']:.4f} "
                f"lenient={rec['greedy_lenient']:.4f}")

        if args.constrained:
            cands, skipped = build_candidates(args.inventory, te, vocab)
            score = (make_scorer(model, vocab, cands, device) if arch in ("grok", "dec")
                     else seq2seq_scorer(model, vocab, cands, device))
            keys = sorted({lenient_key(c) for c in cands})
            k2i = {k: i for i, k in enumerate(keys)}
            cls = torch.tensor([k2i[lenient_key(c)] for c in cands], device=device)
            top5, top5_lp, merged3 = [], [], []
            for s in surfaces:
                lp = score(s)
                order = torch.argsort(lp, descending=True)[:5]
                top5.append([cands[int(i)] for i in order])
                top5_lp.append([round(float(lp[i]), 4) for i in order])
                # posterior mass per و/ي class = sum over its member roots
                mass = torch.zeros(len(keys), device=device).scatter_add_(
                    0, cls, (lp - lp.max()).exp())
                merged3.append([keys[int(i)] for i in torch.argsort(mass, descending=True)[:3]])
            rec.update({
                "n_cands": len(cands), "n_cands_skipped_oov": skipped,
                "top5": top5, "top5_logp": top5_lp, "merged_top3": merged3,
                "constrained_top1_strict": sum(t[0] == g for t, g in zip(top5, golds)) / n,
                "constrained_top3_strict": sum(g in t[:3] for t, g in zip(top5, golds)) / n,
                "merged_top1_lenient":
                    sum(t[0] == lenient_key(g) for t, g in zip(merged3, golds)) / n,
                "merged_top3_lenient":
                    sum(lenient_key(g) in t for t, g in zip(merged3, golds)) / n,
            })
            line += (f" | constr top1={rec['constrained_top1_strict']:.4f} "
                     f"top3={rec['constrained_top3_strict']:.4f} "
                     f"merged top1={rec['merged_top1_lenient']:.4f}")
        print(line, flush=True)
        fn = name + (f".{args.tag}" if args.tag else "") + ".json"
        Path(args.out_dir, fn).write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
    print("DUMP_PREDICTIONS_DONE", flush=True)


if __name__ == "__main__":
    main()
