"""Metrics for grokking root-extraction runs."""
from __future__ import annotations

import torch

EOS_ID = 2  # matches Vocab.eos


def per_radical_acc(logits, target, mask) -> float:
    """Teacher-forced accuracy over root radical positions (excluding EOS)."""
    radical_mask = mask & (target != EOS_ID)
    total = int(radical_mask.sum())
    if total == 0:
        return 0.0
    pred = logits.argmax(-1)
    correct = int(((pred == target) & radical_mask).sum())
    return correct / total


def exact_root_acc(pred_strs, gold_strs) -> float:
    if not gold_strs:
        return 0.0
    return sum(p == g for p, g in zip(pred_strs, gold_strs)) / len(gold_strs)


@torch.no_grad()
def weight_norm(model) -> float:
    return float(sum(p.detach().float().pow(2).sum() for p in model.parameters()) ** 0.5)
