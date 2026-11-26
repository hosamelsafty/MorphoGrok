"""Encode (surface, root) pairs for the decoder-only seq2seq model.

Sequence layout (closest analog to the Power/Nanda grokking setup):
    [BOS] surface_chars [SEP] root_radicals [EOS]
The model is trained as a causal LM; the loss is applied only on the root region
(predicting r1..rk and EOS), so the model must produce the root from the surface.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from .vocab import Vocab


def load_split(path):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return d["train"], d["val"], d["test"], d.get("meta", {})


def build_vocab(pairs) -> Vocab:
    chars = set()
    for s, r in pairs:
        chars.update(s)
        chars.update(r)
    return Vocab(chars)


def max_input_len(pairs) -> int:
    """Length of the model input = len([BOS]+surf+[SEP]+root+[EOS]) - 1."""
    return max(len(s) + len(r) for s, r in pairs) + 2


def encode_pairs(pairs, vocab: Vocab, max_in: int):
    """Return (inputs, targets, mask, gold_roots) padded to max_in.

    inputs[i], targets[i] are the causal-LM input/next-token sequences; mask[i]
    is True exactly on the root-region target positions (r1..rk, EOS).
    """
    B = len(pairs)
    inp = torch.full((B, max_in), vocab.pad, dtype=torch.long)
    tgt = torch.full((B, max_in), vocab.pad, dtype=torch.long)
    msk = torch.zeros((B, max_in), dtype=torch.bool)
    golds = []
    for i, (s, r) in enumerate(pairs):
        seq = [vocab.bos] + vocab.encode(s) + [vocab.sep] + vocab.encode(r) + [vocab.eos]
        sep_index = 1 + len(s)
        x, y = seq[:-1], seq[1:]
        L = len(x)
        inp[i, :L] = torch.tensor(x)
        tgt[i, :L] = torch.tensor(y)
        for j in range(L):
            if (j + 1) > sep_index:      # supervise only the root region
                msk[i, j] = True
        golds.append(r)
    return inp, tgt, msk, golds


@torch.no_grad()
def greedy_decode(model, vocab: Vocab, surfaces, device, max_radicals: int = 5):
    """Batched greedy generation of the root for each surface string."""
    model.eval()
    enc = [[vocab.bos] + vocab.encode(s) + [vocab.sep] for s in surfaces]
    B = len(enc)
    cur = torch.tensor([len(e) for e in enc], device=device)
    width = max(len(e) for e in enc) + max_radicals + 1
    seq = torch.full((B, width), vocab.pad, dtype=torch.long, device=device)
    for i, e in enumerate(enc):
        seq[i, :len(e)] = torch.tensor(e, device=device)
    done = torch.zeros(B, dtype=torch.bool, device=device)
    out = [[] for _ in range(B)]
    rows = torch.arange(B, device=device)
    for _ in range(max_radicals + 1):
        nxt = model(seq)[rows, (cur - 1).clamp(min=0)].argmax(-1)
        for i in range(B):
            if done[i]:
                continue
            t = int(nxt[i])
            if t == vocab.eos:
                done[i] = True
                continue
            if int(cur[i]) < width:
                seq[i, int(cur[i])] = t
                out[i].append(t)
            cur[i] += 1
        if bool(done.all()):
            break
    return ["".join(vocab.itos[t] for t in toks) for toks in out]
