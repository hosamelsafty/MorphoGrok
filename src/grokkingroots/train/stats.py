"""Bootstrap statistics for aggregating sweep results (paper-grade rigor)."""
from __future__ import annotations

import numpy as np


def bootstrap_ci(values, n_resamples=10_000, ci=0.95, seed=0):
    """Mean and (lo, hi) percentile bootstrap CI of a 1-D sample."""
    a = np.asarray(values, dtype=float)
    if a.size == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    means = a[rng.integers(0, a.size, size=(n_resamples, a.size))].mean(axis=1)
    lo, hi = np.quantile(means, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return float(a.mean()), float(lo), float(hi)


def paired_bootstrap_test(a, b, n_resamples=10_000, seed=0):
    """Two-sided paired bootstrap: return (observed mean diff a-b, p-value)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    assert a.shape == b.shape, "paired samples must align"
    d = a - b
    rng = np.random.default_rng(seed)
    boot = d[rng.integers(0, d.size, size=(n_resamples, d.size))].mean(axis=1)
    p = 2.0 * min((boot <= 0).mean(), (boot >= 0).mean())
    return float(d.mean()), float(min(p, 1.0))
