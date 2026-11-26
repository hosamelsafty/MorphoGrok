"""Train a char transformer on root extraction and log the grokking signature.

Canonical recipe (Power et al.): full-batch AdamW, lr=1e-3, heavy weight decay,
no early stopping, up to ~1e6 steps. Logs train vs val vs test accuracy and the
parameter norm on a log-spaced step grid so delayed generalization is visible.

Usage:
    python -m grokkingroots.train.train_grok --split SPLIT.json --out OUT_DIR \
        --steps 100000 --weight-decay 1.0 --init-scale 1.0 --n-layers 2 --seed 0
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from grokkingroots.data.encoding import (
    build_vocab, encode_pairs, greedy_decode, load_split, max_input_len,
)
from grokkingroots.model.char_transformer import CharTransformer, param_count
from grokkingroots.model.init_scale import scale_init
from grokkingroots.train.metrics import exact_root_acc, per_radical_acc, weight_norm


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def log_grid(total: int, n_points: int) -> list:
    pts = np.unique(np.logspace(0, np.log10(max(total, 1)), n_points).astype(int))
    pts = sorted({int(p) for p in pts if 1 <= p <= total} | {total})
    return pts


def rng_states():
    st = {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def restore_rng(st):
    # ckpts are loaded with map_location=device, which moves the saved RNG
    # ByteTensors to CUDA; the generators require them on CPU
    torch.set_rng_state(st["torch"].cpu())
    np.random.set_state(st["numpy"])
    random.setstate(st["python"])
    if torch.cuda.is_available() and st.get("cuda"):
        try:
            torch.cuda.set_rng_state_all([t.cpu() for t in st["cuda"]])
        except (RuntimeError, TypeError):
            pass  # different GPU count after resume; harmless for full-batch runs


def atomic_save(obj, path: Path):
    """torch.save via tmp+rename so a mid-write kill never corrupts the file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def atomic_write_text(path: Path, text: str):
    """Same guarantee for JSON files — a kill mid-write must never leave a
    truncated history/meta that would crash every future --resume."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


@torch.no_grad()
def eval_split(model, X, Y, M, surfaces, golds, vocab, device, gen_subsample):
    model.eval()
    pr = per_radical_acc(model(X), Y, M)
    if gen_subsample and len(surfaces) > gen_subsample:
        surf, gold = surfaces[:gen_subsample], golds[:gen_subsample]
    else:
        surf, gold = surfaces, golds
    ex = exact_root_acc(greedy_decode(model, vocab, surf, device), gold)
    return pr, ex


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--split", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--d-ff", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1.0)
    p.add_argument("--init-scale", type=float, default=1.0)
    p.add_argument("--batch-size", default="full")  # 'full' or int
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--log-points", type=int, default=60)
    p.add_argument("--gen-subsample", type=int, default=2000)
    p.add_argument("--early-stop", action="store_true",
                   help="early-stopped twin baseline (stop when val exact-root stalls)")
    p.add_argument("--patience-steps", type=int, default=5000,
                   help="early-stop if val exact-root has not improved for this many steps")
    p.add_argument("--min-steps", type=int, default=500,
                   help="never early-stop before this step (clears the pre-generalization dead zone)")
    p.add_argument("--grad-clip", type=float, default=1.0,
                   help="gradient norm clip (0 = disabled)")
    p.add_argument("--save-ckpts", action="store_true", help="save a checkpoint at each log point")
    p.add_argument("--ckpt-dir", default=None,
                   help="where the --save-ckpts ladder goes (default: --out). "
                        "Point at node-local scratch to keep the run dir small.")
    p.add_argument("--resume", action="store_true",
                   help="resume from OUT/ckpt_last.pt if present (model+opt+step+RNG+history)")
    p.add_argument("--consistency-lambda", type=float, default=0.0,
                   help="gen-3: weight of the cross-form consistency loss (KL of each "
                        "training form's root-region distribution to its root-group mean). "
                        "Full-batch only; 0 disables.")
    p.add_argument("--aux-roottype-lambda", type=float, default=0.0,
                   help="auxiliary loss: predict the root TYPE (hollow/defective/... — "
                        "i.e. which radical is weak) from the surface representation. "
                        "Tests whether structural awareness helps reach the ceiling. "
                        "0 disables (model stays bit-identical to base).")
    p.add_argument("--aux-type-map", default="artefacts/phase0/root_type_map.json",
                   help="JSON {types:[...], map:{root: type}} for --aux-roottype-lambda.")
    p.add_argument("--ckpt-every-s", type=int, default=900,
                   help="also refresh ckpt_last.pt every N seconds between log points (0 = off)")
    return p.parse_args()


def main():
    args = parse()
    set_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    train_pairs, val_pairs, test_pairs, split_meta = load_split(args.split)
    all_pairs = train_pairs + val_pairs + test_pairs
    vocab = build_vocab(all_pairs)
    max_in = max_input_len(all_pairs)
    max_s = max(len(s) for s, _ in all_pairs)
    model_max_len = max(max_in, max_s + 5 + 3)

    def enc(pairs):
        x, y, m, g = encode_pairs(pairs, vocab, max_in)
        return x.to(device), y.to(device), m.to(device), g

    Xtr, Ytr, Mtr, _ = enc(train_pairs)
    Xva, Yva, Mva, _ = enc(val_pairs)
    Xte, Yte, Mte, _ = enc(test_pairs)
    tr_surf, tr_gold = [s for s, _ in train_pairs], [r for _, r in train_pairs]
    va_surf, va_gold = [s for s, _ in val_pairs], [r for _, r in val_pairs]
    te_surf, te_gold = [s for s, _ in test_pairs], [r for _, r in test_pairs]

    # --- optional auxiliary root-type supervision -------------------------------
    # Does knowing WHICH radical is weak (hollow=middle, defective=last, ...) help
    # the model reach the information ceiling? The type is predicted from the
    # SURFACE representation only (pre-root positions), so it can't shortcut the
    # root output. Training signal only — never used at test time.
    aux_lambda = args.aux_roottype_lambda
    aux_ids = aux_smask = None
    n_aux = 0
    if aux_lambda > 0:
        tm = json.loads(Path(args.aux_type_map).read_text(encoding="utf-8"))
        types, tmap = tm["types"], tm["map"]
        t2i = {t: i for i, t in enumerate(types)}
        n_aux = len(types)
        ids, miss = [], 0
        for _, r in train_pairs:
            t = tmap.get(r)
            if t is None:
                ids.append(0); miss += 1
            else:
                ids.append(t2i[t])
        aux_ids = torch.tensor(ids, dtype=torch.long, device=device)
        sm = torch.zeros((len(train_pairs), Xtr.size(1)), dtype=torch.bool)
        for i, (s, _) in enumerate(train_pairs):
            sm[i, :1 + len(s) + 1] = True    # BOS + surface chars + SEP (pre-root)
        aux_smask = sm.to(device)
        print(f"aux root-type: {n_aux} classes, {miss} roots unmapped "
              f"(lambda={aux_lambda})", flush=True)

    model = CharTransformer(len(vocab), args.d_model, args.n_heads, args.n_layers,
                            args.d_ff, args.dropout, max_len=model_max_len,
                            pad_id=vocab.pad, n_aux=n_aux).to(device)
    scale_init(model, args.init_scale)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay, betas=(0.9, 0.98))

    full = str(args.batch_size) == "full" or int_or(args.batch_size) >= len(train_pairs)
    bs = None if full else int(args.batch_size)
    grid = set(log_grid(args.steps, args.log_points))

    # gen-3 consistency: same-root training forms share an IDENTICAL target
    # sequence, so their root-region distributions are directly comparable.
    # Precompute per-root member groups, bucketed by root length (positions
    # per member differ, but counts match within a group).
    cons_buckets = {}
    if args.consistency_lambda > 0:
        assert full, "--consistency-lambda requires full-batch training"
        by_root = {}
        for i, (_, r) in enumerate(train_pairs):
            by_root.setdefault(r, []).append(i)
        raw = {}
        for r, idxs in by_root.items():
            if len(idxs) < 2:
                continue
            lg = int(Mtr[idxs[0]].sum().item())
            b = raw.setdefault(lg, {"rows": [], "gid": [], "counts": []})
            g = len(b["counts"])
            b["counts"].append(len(idxs))
            b["rows"] += idxs
            b["gid"] += [g] * len(idxs)
        n_groups = n_members = 0
        for lg, b in raw.items():
            rows = torch.tensor(b["rows"], dtype=torch.long, device=device)
            pos = torch.stack([Mtr[i].nonzero().squeeze(1) for i in b["rows"]]).to(device)
            cons_buckets[lg] = {
                "rows": rows, "pos": pos,
                "gid": torch.tensor(b["gid"], dtype=torch.long, device=device),
                "counts": torch.tensor(b["counts"], dtype=torch.float32,
                                       device=device).view(-1, 1, 1),
                "ngroups": len(b["counts"]),
            }
            n_groups += len(b["counts"]); n_members += len(b["rows"])
        print(f"consistency: {n_groups} root groups / {n_members} member forms "
              f"(lambda={args.consistency_lambda})", flush=True)

    def consistency_loss(logits):
        total, n = logits.new_zeros(()), 0
        for b in cons_buckets.values():
            lg = logits[b["rows"].unsqueeze(1), b["pos"]]        # [E, Lg, V]
            logp = F.log_softmax(lg, dim=-1)
            p = logp.exp()
            gm = torch.zeros(b["ngroups"], *p.shape[1:], device=p.device,
                             dtype=p.dtype).index_add_(0, b["gid"], p) / b["counts"]
            kl = (p * (logp - (gm[b["gid"]] + 1e-9).log())).sum(-1).mean()
            total = total + kl * len(b["rows"]); n += len(b["rows"])
        return total / max(n, 1)

    def aux_loss_fn(hidden, idx_batch):
        sm = aux_smask if idx_batch is None else aux_smask[idx_batch]
        ids = aux_ids if idx_batch is None else aux_ids[idx_batch]
        pooled = (hidden * sm.unsqueeze(-1)).sum(1) / sm.sum(1, keepdim=True).clamp(min=1)
        return F.cross_entropy(model.aux_head(pooled), ids)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(args.ckpt_dir) if args.ckpt_dir else out
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out / "meta.json"
    meta = {
        "split": args.split, "split_meta": split_meta, "test_sha256_prefix": None,
        "vocab_size": len(vocab), "n_params": param_count(model), "device": device,
        "n_train": len(train_pairs), "n_val": len(val_pairs), "n_test": len(test_pairs),
        "args": vars(args),
    }

    history = []
    best_val, best_step = -1.0, 0
    start_step, prior_elapsed = 0, 0.0
    last_ckpt = out / "ckpt_last.pt"
    if args.resume and last_ckpt.exists():
        # weights_only=False: the ckpt holds RNG states (numpy/python objects), not just tensors
        ck = torch.load(last_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_step = ck["step"]
        prior_elapsed = ck.get("elapsed_s", 0.0)
        if ck.get("rng"):
            restore_rng(ck["rng"])
        hist_path = out / "history.json"
        if hist_path.exists():
            try:
                history = [r for r in json.loads(hist_path.read_text(encoding="utf-8"))
                           if r["step"] <= start_step]
            except (json.JSONDecodeError, KeyError):
                history = []   # ckpt is authoritative; a corrupt history must not brick resume
        for r in history:  # rebuild early-stop state from the kept records
            if r["val_exact"] > best_val + 1e-4:
                best_val, best_step = r["val_exact"], r["step"]
        # provenance: record the resume event, keep the original meta otherwise
        if meta_path.exists():
            try:
                fresh_args = meta["args"]
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                meta["args_latest"] = fresh_args   # current invocation (steps may differ)
            except json.JSONDecodeError:
                pass   # keep the freshly built meta
        meta.setdefault("resumes", []).append(
            {"at_step": start_step, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        print(f"resumed from {last_ckpt} at step {start_step} "
              f"({len(history)} history records kept, {prior_elapsed:.0f}s prior)", flush=True)
    atomic_write_text(meta_path, json.dumps(meta, indent=2, ensure_ascii=False))

    def save_last(step):
        atomic_save({"step": step, "model": model.state_dict(), "opt": opt.state_dict(),
                     "rng": rng_states(), "elapsed_s": prior_elapsed + time.time() - t0},
                    last_ckpt)

    t0 = time.time()
    last_ckpt_t = t0
    for step in range(start_step + 1, args.steps + 1):
        model.train()
        if full:
            xb, yb, mb = Xtr, Ytr, Mtr
        else:
            idx = torch.randint(0, len(train_pairs), (bs,), device=device)
            xb, yb, mb = Xtr[idx], Ytr[idx], Mtr[idx]
        if aux_lambda > 0:
            logits, hidden = model(xb, return_hidden=True)
        else:
            logits = model(xb)
        loss = F.cross_entropy(logits[mb], yb[mb])
        if cons_buckets:
            loss = loss + args.consistency_lambda * consistency_loss(logits)
        if aux_lambda > 0:
            loss = loss + aux_lambda * aux_loss_fn(hidden, None if full else idx)
        opt.zero_grad()
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()

        if step in grid:
            tr_pr, tr_ex = eval_split(model, Xtr, Ytr, Mtr, tr_surf, tr_gold, vocab, device, args.gen_subsample)
            va_pr, va_ex = eval_split(model, Xva, Yva, Mva, va_surf, va_gold, vocab, device, args.gen_subsample)
            te_pr, te_ex = eval_split(model, Xte, Yte, Mte, te_surf, te_gold, vocab, device, args.gen_subsample)
            rec = {
                "step": step, "train_loss": loss.item(), "weight_norm": weight_norm(model),
                "train_pr_acc": tr_pr, "train_exact": tr_ex,
                "val_pr_acc": va_pr, "val_exact": va_ex,
                "test_pr_acc": te_pr, "test_exact": te_ex,
                "elapsed_s": round(prior_elapsed + time.time() - t0, 1),
            }
            history.append(rec)
            atomic_write_text(out / "history.json", json.dumps(history, ensure_ascii=False))
            save_last(step)
            last_ckpt_t = time.time()
            if args.save_ckpts:
                atomic_save({"step": step, "model": model.state_dict()}, ckpt_dir / f"ckpt_{step}.pt")
            print(f"step {step:>7} | loss {rec['train_loss']:.4f} | "
                  f"train_ex {tr_ex:.3f} val_ex {va_ex:.3f} test_ex {te_ex:.3f} | "
                  f"wnorm {rec['weight_norm']:.1f}", flush=True)

            if args.early_stop:
                if va_ex > best_val + 1e-4:
                    best_val, best_step = va_ex, step
                elif step >= args.min_steps and step - best_step >= args.patience_steps:
                    print(f"early stop at step {step} (best val_ex {best_val:.3f} @ {best_step})", flush=True)
                    break
        elif args.ckpt_every_s > 0 and time.time() - last_ckpt_t >= args.ckpt_every_s:
            # safety net between log points: a killed job loses at most ckpt-every-s
            save_last(step)
            last_ckpt_t = time.time()

    atomic_write_text(out / "history.json", json.dumps(history, ensure_ascii=False))
    # completion sentinel: a job scheduler can use this to tell "finished" from "needs resume"
    atomic_write_text(out / "DONE", json.dumps({
        "final_step": history[-1]["step"] if history else 0,
        "target_steps": args.steps,
        "early_stopped": bool(args.early_stop and history and history[-1]["step"] < args.steps),
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }))
    print(f"done | {len(history)} log points | {time.time() - t0:.1f}s", flush=True)


def int_or(x, default=10**9):
    try:
        return int(x)
    except (TypeError, ValueError):
        return default


if __name__ == "__main__":
    main()
