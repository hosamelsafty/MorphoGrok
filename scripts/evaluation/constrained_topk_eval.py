"""Does the model saturate the achievable posterior?

For a trained run's val-selected checkpoint, score EVERY root in the valid-root
inventory for each test item — log P(root | surface) summed over the root chars
+ EOS — and report:
  - constrained top-1/3/5 exact-root accuracy (argmax over the inventory)
  - unconstrained greedy top-1 (the training-time metric, for reference)
  - calibration of the normalized posterior (15-bin ECE, mean p(gold))
  - all of the above split by the copy-vs-restore buckets: root contiguous
    in the surface vs not (the true-restoration subset).

The claim this feeds: the model's calibrated posterior
saturates the computable information ceiling — errors are concentrated where
the input underdetermines the root.

CPU-friendly (model is ~400k params). Usage:
  python scripts/evaluation/constrained_topk_eval.py \
      --ckpt RUNS/<id>/ckpts/ckpt_<step>.pt --meta RUNS/<id>/meta.json \
      --split artefacts/phase0/splits_gen2_v2clean/L2_p30_k1.json \
      --inventory artefacts/phase0/root_type_map.json --out OUT.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from grokkingroots.data.encoding import build_vocab, load_split, max_input_len, greedy_decode
from grokkingroots.model.char_transformer import CharTransformer


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--meta", required=True, help="run meta.json (model dims)")
    p.add_argument("--split", required=True)
    p.add_argument("--inventory", default="artefacts/phase0/root_type_map.json")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default=None)
    p.add_argument("--max-items", type=int, default=0, help="0 = all test items")
    p.add_argument("--max-cands", type=int, default=0, help="0 = full inventory (smoke only)")
    p.add_argument("--batch-rows", type=int, default=2048)
    return p.parse_args()


def load_model(meta_path, split_path, ckpt_path, device):
    """Rebuild the training-time vocab/model and load a checkpoint (shared by
    constrained_topk_eval and calibrated_decode so numbers can never drift)."""
    meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    margs = meta["args"]
    train_pairs, val_pairs, test_pairs, _ = load_split(split_path)
    all_pairs = train_pairs + val_pairs + test_pairs
    vocab = build_vocab(all_pairs)          # identical to training
    max_in = max_input_len(all_pairs)
    max_s = max(len(s) for s, _ in all_pairs)
    model_max_len = max(max_in, max_s + 5 + 3)
    model = CharTransformer(len(vocab), margs["d_model"], margs["n_heads"],
                            margs["n_layers"], margs["d_ff"], 0.0,
                            max_len=model_max_len, pad_id=vocab.pad,
                            n_aux=0).to(device)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ck["model"] if "model" in ck else ck
    # tolerate aux_head keys from aux runs (analysis never uses the aux head)
    state = {k: v for k, v in state.items() if not k.startswith("aux_head")}
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, vocab, (train_pairs, val_pairs, test_pairs)


def build_candidates(inventory_path, test_pairs, vocab, max_cands=0):
    """Valid-root candidate list = inventory ∪ test golds, minus OOV-char roots."""
    inv = json.loads(Path(inventory_path).read_text(encoding="utf-8"))
    known = set(vocab.itos)
    cands, skipped = [], 0
    for r in sorted(set(inv["map"].keys()) | {r for _, r in test_pairs}):
        if set(r) <= known:
            cands.append(r)
        else:
            skipped += 1
    if max_cands:
        gold_set = {r for _, r in test_pairs}
        cands = sorted(gold_set | set(cands[:max_cands]))
    return cands, skipped


def make_scorer(model, vocab, cands, device, batch_rows=2048):
    """Returns score_item(surface) -> [n_cands] log P(cand | surface)."""
    cand_toks = [vocab.encode(r) + [vocab.eos] for r in cands]
    by_len = {}
    for ci, toks in enumerate(cand_toks):
        by_len.setdefault(len(toks), []).append(ci)
    groups = []
    for lg, idxs in sorted(by_len.items()):
        toks = torch.tensor([cand_toks[i] for i in idxs], dtype=torch.long, device=device)
        groups.append((lg, torch.tensor(idxs, dtype=torch.long, device=device), toks))
    n_cands = len(cands)

    @torch.no_grad()
    def score_item(surface):
        prefix = [vocab.bos] + vocab.encode(surface) + [vocab.sep]
        P = len(prefix)
        out = torch.full((n_cands,), -1e30, device=device)
        for lg, idxs, toks in groups:
            C = toks.size(0)
            L = P + lg                       # full seq len; input = seq[:-1]
            x = torch.full((C, L - 1), vocab.pad, dtype=torch.long, device=device)
            x[:, :P] = torch.tensor(prefix, device=device)
            x[:, P:] = toks[:, :-1]          # all but last target token
            lp_rows = []
            for s in range(0, C, batch_rows):
                logits = model(x[s:s + batch_rows])
                logp = F.log_softmax(logits[:, P - 1:L - 1], dim=-1)
                lp = logp.gather(-1, toks[s:s + batch_rows].unsqueeze(-1)).squeeze(-1).sum(-1)
                lp_rows.append(lp)
            out[idxs] = torch.cat(lp_rows)
        return out

    return score_item


def main():
    args = parse()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, vocab, (train_pairs, val_pairs, test_pairs) = load_model(
        args.meta, args.split, args.ckpt, device)
    cands, skipped = build_candidates(args.inventory, test_pairs, vocab, args.max_cands)
    c2i = {r: i for i, r in enumerate(cands)}
    test = test_pairs[: args.max_items] if args.max_items else test_pairs
    n_items, n_cands = len(test), len(cands)
    score_item = make_scorer(model, vocab, cands, device, args.batch_rows)

    top1 = top3 = top5 = 0
    ece_conf, ece_hit, p_gold_all = [], [], []
    bucket = {}   # substring / restoration
    golds = [r for _, r in test]
    for it, (s, gold) in enumerate(test):
        lp = score_item(s)
        post = F.softmax(lp, dim=0)
        order = torch.argsort(lp, descending=True)
        top = [cands[int(i)] for i in order[:5]]
        gi = c2i.get(gold)
        hit1, hit3, hit5 = gold == top[0], gold in top[:3], gold in top[:5]
        top1 += hit1; top3 += hit3; top5 += hit5
        conf = float(post[order[0]])
        pg = float(post[gi]) if gi is not None else 0.0
        ece_conf.append(conf); ece_hit.append(hit1); p_gold_all.append(pg)
        b = "substring" if gold in s else "restoration"
        bb = bucket.setdefault(b, {"n": 0, "top1": 0, "top3": 0, "top5": 0, "p_gold": 0.0})
        bb["n"] += 1; bb["top1"] += hit1; bb["top3"] += hit3; bb["top5"] += hit5
        bb["p_gold"] += pg
        if (it + 1) % 200 == 0:
            print(f"  {it + 1}/{n_items} items", flush=True)

    # unconstrained greedy top-1 (training-time metric) on the same items
    greedy = greedy_decode(model, vocab, [s for s, _ in test], device)
    greedy_top1 = sum(p == g for p, g in zip(greedy, golds)) / n_items

    # 15-bin ECE
    ece, nb = 0.0, 15
    conf_t = torch.tensor(ece_conf); hit_t = torch.tensor(ece_hit, dtype=torch.float32)
    for b in range(nb):
        lo, hi = b / nb, (b + 1) / nb
        m = (conf_t > lo) & (conf_t <= hi)
        if int(m.sum()):
            ece += float(m.float().mean() * abs(hit_t[m].mean() - conf_t[m].mean()))

    for bb in bucket.values():
        n = bb.pop("n")
        for k in list(bb):
            bb[k] = round(bb[k] / n, 4)
        bb["n"] = n

    result = {
        "ckpt": args.ckpt, "split": args.split, "n_items": n_items,
        "n_cands": n_cands, "n_cands_skipped_oov": skipped,
        "constrained_top1": round(top1 / n_items, 4),
        "top3": round(top3 / n_items, 4), "top5": round(top5 / n_items, 4),
        "greedy_top1_unconstrained": round(greedy_top1, 4),
        "mean_p_gold": round(sum(p_gold_all) / n_items, 4),
        "ece_15bin": round(ece, 4),
        "buckets": bucket,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
