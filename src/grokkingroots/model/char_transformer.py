"""Small decoder-only character transformer (the Power/Nanda grokking regime)."""
from __future__ import annotations

import torch
import torch.nn as nn


class CharTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=2, d_ff=512,
                 dropout=0.0, max_len=64, pad_id=0, n_aux=0):
        super().__init__()
        self.pad_id = pad_id
        self.tok = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=d_ff, dropout=dropout,
            activation="relu", batch_first=True, norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        # Optional auxiliary classification head (e.g. root-type). n_aux=0 keeps
        # the model bit-identical to the base config.
        self.aux_head = nn.Linear(d_model, n_aux) if n_aux > 0 else None

    def forward(self, x, return_hidden=False):
        L = x.size(1)
        pos = torch.arange(L, device=x.device).unsqueeze(0)
        h = self.tok(x) + self.pos(pos)
        causal = torch.triu(torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1)
        pad_mask = x == self.pad_id
        h = self.blocks(h, mask=causal, src_key_padding_mask=pad_mask)
        hn = self.norm(h)
        logits = self.head(hn)
        return (logits, hn) if return_hidden else logits


def param_count(model) -> int:
    return sum(p.numel() for p in model.parameters())
