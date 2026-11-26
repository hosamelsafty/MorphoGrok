"""Omnigrok-style init-scale knob: multiply all weights by alpha after init.

Large alpha => large initial weight norm => the model overfits fast then slowly
walks toward the generalizing norm (delayed grokking), even without heavy weight
decay. This is sweep (c).
"""
from __future__ import annotations

import torch


@torch.no_grad()
def scale_init(model, alpha: float):
    if alpha == 1.0:
        return
    for p in model.parameters():
        p.mul_(alpha)
