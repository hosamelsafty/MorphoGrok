"""Shared character vocabulary for surfaces and roots (one alphabet).

Fixed special-token ids: pad=0, bos=1, eos=2, sep=3, unk=4. The vocabulary is
built over the whole dataset's characters — the alphabet is not the prediction
target (the root identity is), so seeing all characters is not leakage.
"""
from __future__ import annotations

SPECIALS = ["<pad>", "<bos>", "<eos>", "<sep>", "<unk>"]


class Vocab:
    def __init__(self, chars):
        self.itos = SPECIALS + sorted(chars)
        self.stoi = {c: i for i, c in enumerate(self.itos)}

    pad, bos, eos, sep, unk = 0, 1, 2, 3, 4

    def __len__(self):
        return len(self.itos)

    def encode(self, text):
        return [self.stoi.get(c, self.unk) for c in text]
