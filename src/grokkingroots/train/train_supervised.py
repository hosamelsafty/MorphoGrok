"""Conventional supervised baseline.

The grokking recipe in train_grok.py is full-batch, constant LR, no dropout, no
early stopping. This script trains on the SAME frozen split under the standard
supervised recipe instead, so the paper can say whether the Single-Form plateau
is specific to the grokking recipe or a property of character-level models on
this split.

Recipe = Wu, Cotterell & Hulden (EACL 2021), "Applying the Transformer to
Character-level Transduction", fixed a priori (no search beyond the two dropout
values that paper itself reports):
  mini-batch 400, Adam lr 1e-3 with beta2 0.98, inverse-square-root schedule
  with 4k warmup updates, label smoothing 0.1, dropout 0.3 (or 0.1), 20k
  updates, evaluated every 400 updates, best-of-50 checkpoint selected on the
  (root-disjoint) validation set, greedy decoding.

Two architectures:
  --arch encdec  their encoder-decoder (pre-norm, 4+4 layers, 4 heads,
                 d_model 256, d_ff 1024, sinusoidal positions).
  --arch dec     the paper's own decoder-only CharTransformer; pass the
                 MorphoGrok sizes to change the recipe and nothing else.

Details the paper does not state are taken from the authors' released code
(github.com/shijie-wu/neural-transducer: src/transformer.py and
example/transformer/trm-sig17.sh): gradient clipping at max-norm 1, embeddings
initialised N(0, d_model^-0.5) and scaled by sqrt(d_model), Xavier-uniform
linear layers, source wrapped in BOS/EOS.

Usage:
    python -m grokkingroots.train.train_supervised --split SPLIT.json \
        --out OUT_DIR --arch encdec --dropout 0.3 --seed 0 --resume
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from grokkingroots.data.encoding import (
    build_vocab, encode_pairs, greedy_decode, load_split, max_input_len,
)
from grokkingroots.model.char_transformer import CharTransformer, param_count
from grokkingroots.train.metrics import exact_root_acc
from grokkingroots.train.train_grok import (
    atomic_save, atomic_write_text, restore_rng, rng_states, set_seed,
)


class Seq2SeqTransformer(nn.Module):
    """Pre-norm encoder-decoder character transformer (Wu et al., 2021)."""

    def __init__(self, vocab_size, d_model=256, n_heads=4, n_layers=4, d_ff=1024,
                 dropout=0.3, pad_id=0, max_len=128):
        super().__init__()
        self.pad_id = pad_id
        self.scale = math.sqrt(d_model)
        self.src_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.tgt_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        # As in the reference code: with N(0, 1) embeddings the sqrt(d_model)
        # scale would bury the unit-norm sinusoids, i.e. the position signal.
        for emb in (self.src_emb, self.tgt_emb):
            nn.init.normal_(emb.weight, mean=0.0, std=d_model ** -0.5)
            nn.init.zeros_(emb.weight[pad_id])
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)
        self.drop = nn.Dropout(dropout)
        self.tf = nn.Transformer(d_model, n_heads, n_layers, n_layers, d_ff, dropout,
                                 activation="relu", batch_first=True, norm_first=True)
        self.out = nn.Linear(d_model, vocab_size)
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def _embed(self, emb, x):
        return self.drop(emb(x) * self.scale + self.pe[: x.size(1)])

    def encode(self, src):
        return self.tf.encoder(self._embed(self.src_emb, src),
                               src_key_padding_mask=src == self.pad_id)

    def decode(self, tgt_in, memory, src_pad):
        L = tgt_in.size(1)
        causal = torch.triu(torch.ones(L, L, dtype=torch.bool, device=tgt_in.device), 1)
        h = self.tf.decoder(self._embed(self.tgt_emb, tgt_in), memory, tgt_mask=causal,
                            tgt_key_padding_mask=tgt_in == self.pad_id,
                            memory_key_padding_mask=src_pad)
        return self.out(h)

    def forward(self, src, tgt_in):
        return self.decode(tgt_in, self.encode(src), src == self.pad_id)


def encode_src(vocab, surfaces, device):
    """[B, S] source ids: [BOS] surface chars [EOS], right-padded."""
    rows = [[vocab.bos] + vocab.encode(s) + [vocab.eos] for s in surfaces]
    x = torch.full((len(rows), max(len(r) for r in rows)), vocab.pad, dtype=torch.long)
    for i, r in enumerate(rows):
        x[i, : len(r)] = torch.tensor(r)
    return x.to(device)


def encode_tgt(vocab, roots, device):
    """Teacher-forcing pair: ([BOS]+root, root+[EOS]), right-padded."""
    rows = [vocab.encode(r) for r in roots]
    L = max(len(r) for r in rows) + 1
    tin = torch.full((len(rows), L), vocab.pad, dtype=torch.long)
    tout = torch.full((len(rows), L), vocab.pad, dtype=torch.long)
    for i, r in enumerate(rows):
        tin[i, : len(r) + 1] = torch.tensor([vocab.bos] + r)
        tout[i, : len(r) + 1] = torch.tensor(r + [vocab.eos])
    return tin.to(device), tout.to(device)


@torch.no_grad()
def seq2seq_greedy(model, vocab, surfaces, device, max_radicals: int = 5, batch: int = 1024):
    """Greedy roots from the encoder-decoder; same length cap as greedy_decode."""
    model.eval()
    out = []
    for s in range(0, len(surfaces), batch):
        src = encode_src(vocab, surfaces[s:s + batch], device)
        mem, src_pad = model.encode(src), src == vocab.pad
        ys = torch.full((src.size(0), 1), vocab.bos, dtype=torch.long, device=device)
        for _ in range(max_radicals + 1):
            nxt = model.decode(ys, mem, src_pad)[:, -1].argmax(-1)
            ys = torch.cat([ys, nxt.unsqueeze(1)], dim=1)
            if bool((ys == vocab.eos).any(dim=1).all()):
                break
        for row in ys[:, 1:].tolist():
            toks = row[: row.index(vocab.eos)] if vocab.eos in row else row
            out.append("".join(vocab.itos[t] for t in toks))
    return out


def build_model(arch, vocab, a, max_len, device):
    """`a` is a dict of model sizes (the run's meta['args'] at eval time)."""
    if arch == "dec":
        return CharTransformer(len(vocab), a["d_model"], a["n_heads"], a["n_layers"],
                               a["d_ff"], a["dropout"], max_len=max_len,
                               pad_id=vocab.pad).to(device)
    return Seq2SeqTransformer(len(vocab), a["d_model"], a["n_heads"], a["n_layers"],
                              a["d_ff"], a["dropout"], pad_id=vocab.pad).to(device)


def predict(model, arch, vocab, surfaces, device):
    if arch == "dec":
        return greedy_decode(model, vocab, surfaces, device)
    return seq2seq_greedy(model, vocab, surfaces, device)


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--split", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--arch", choices=["encdec", "dec"], default="encdec")
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--d-ff", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--beta2", type=float, default=0.98)
    p.add_argument("--warmup", type=int, default=4000)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=400)
    p.add_argument("--max-updates", type=int, default=20_000)
    p.add_argument("--eval-every", type=int, default=400)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--resume", action="store_true",
                   help="resume from OUT/ckpt_last.pt if present")
    return p.parse_args()


def main():
    args = parse()
    set_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    train_pairs, val_pairs, test_pairs, split_meta = load_split(args.split)
    all_pairs = train_pairs + val_pairs + test_pairs
    vocab = build_vocab(all_pairs)                      # identical to train_grok
    max_in = max_input_len(all_pairs)
    model_max_len = max(max_in, max(len(s) for s, _ in all_pairs) + 5 + 3)
    N = len(train_pairs)
    surf = {k: [s for s, _ in p] for k, p in
            (("train", train_pairs), ("val", val_pairs), ("test", test_pairs))}
    gold = {k: [r for _, r in p] for k, p in
            (("train", train_pairs), ("val", val_pairs), ("test", test_pairs))}

    if args.arch == "dec":
        X, Y, M, _ = encode_pairs(train_pairs, vocab, max_in)
        X, Y, M = X.to(device), Y.to(device), M.to(device)
    else:
        SRC = encode_src(vocab, surf["train"], device)
        TIN, TOUT = encode_tgt(vocab, gold["train"], device)

    model = build_model(args.arch, vocab, vars(args), model_max_len, device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, args.beta2))
    sched = torch.optim.lr_scheduler.LambdaLR(     # inverse-sqrt with linear warmup
        opt, lambda u: min((u + 1) / args.warmup, (args.warmup / (u + 1)) ** 0.5))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta_path, last, best = out / "meta.json", out / "ckpt_last.pt", out / "ckpt_best.pt"
    meta = {"arch": args.arch, "split": args.split, "split_meta": split_meta,
            "vocab_size": len(vocab), "n_params": param_count(model), "device": device,
            "n_train": N, "n_val": len(val_pairs), "n_test": len(test_pairs),
            "args": vars(args)}

    history, best_val, best_step, start, prior = [], -1.0, 0, 0, 0.0
    if args.resume and last.exists():
        ck = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start, prior = ck["step"], ck["elapsed_s"]
        best_val, best_step, history = ck["best_val"], ck["best_step"], ck["history"]
        restore_rng(ck["rng"])
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta.setdefault("resumes", []).append(
            {"at_step": start, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        print(f"resumed at step {start}", flush=True)
    atomic_write_text(meta_path, json.dumps(meta, indent=2, ensure_ascii=False))

    per_epoch = math.ceil(N / args.batch_size)
    t0 = time.time()
    perm, perm_epoch = None, -1
    for step in range(start + 1, args.max_updates + 1):
        # Epoch shuffling keyed on (seed, epoch): the batch for any step is a
        # pure function of the step, so resume needs no sampler state.
        epoch, i = divmod(step - 1, per_epoch)
        if epoch != perm_epoch:
            g = torch.Generator().manual_seed(args.seed * 1_000_003 + epoch)
            perm, perm_epoch = torch.randperm(N, generator=g).to(device), epoch
        idx = perm[i * args.batch_size:(i + 1) * args.batch_size]

        model.train()
        if args.arch == "dec":
            logits, mb = model(X[idx]), M[idx]
            loss = F.cross_entropy(logits[mb], Y[idx][mb],
                                   label_smoothing=args.label_smoothing)
        else:
            logits = model(SRC[idx], TIN[idx])
            loss = F.cross_entropy(logits.flatten(0, 1), TOUT[idx].flatten(),
                                   ignore_index=vocab.pad,
                                   label_smoothing=args.label_smoothing)
        opt.zero_grad()
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        sched.step()

        if step % args.eval_every == 0 or step == args.max_updates:
            acc = {k: exact_root_acc(predict(model, args.arch, vocab, surf[k], device),
                                     gold[k]) for k in ("train", "val", "test")}
            history.append({"step": step, "lr": sched.get_last_lr()[0],
                            "train_loss": loss.item(), "train_exact": acc["train"],
                            "val_exact": acc["val"], "test_exact": acc["test"],
                            "elapsed_s": round(prior + time.time() - t0, 1)})
            # strict '>' keeps the EARLIEST best-validation checkpoint, the
            # same tie-break as the paper's val-selection (first argmax).
            if acc["val"] > best_val:
                best_val, best_step = acc["val"], step
                atomic_save({"step": step, "model": model.state_dict()}, best)
            atomic_write_text(out / "history.json", json.dumps(history, ensure_ascii=False))
            atomic_save({"step": step, "model": model.state_dict(), "opt": opt.state_dict(),
                         "sched": sched.state_dict(), "rng": rng_states(),
                         "best_val": best_val, "best_step": best_step, "history": history,
                         "elapsed_s": prior + time.time() - t0}, last)
            print(f"step {step:>6} | loss {loss.item():.4f} | train_ex {acc['train']:.3f} "
                  f"val_ex {acc['val']:.3f} test_ex {acc['test']:.3f} | "
                  f"best val {best_val:.3f} @ {best_step}", flush=True)

    atomic_write_text(out / "DONE", json.dumps({
        "final_step": history[-1]["step"] if history else 0,
        "target_steps": args.max_updates, "best_val": best_val, "best_step": best_step,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}))
    print(f"done | best val {best_val:.4f} @ step {best_step} | {time.time() - t0:.1f}s",
          flush=True)


if __name__ == "__main__":
    main()
