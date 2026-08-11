"""Metrics + bootstrap CIs.

Every headline number in the paper is `mean [lo, hi]` over >=5 seeds. Point estimates
without intervals do not go in tables.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def recall_at_k(retrieved: Sequence[str], gold: Sequence[str], k: int) -> float:
    return float(len(set(retrieved[:k]) & set(gold)) > 0)


def mrr(retrieved: Sequence[str], gold: Sequence[str]) -> float:
    g = set(gold)
    for rank, rid in enumerate(retrieved, start=1):
        if rid in g:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], gold: Sequence[str], k: int) -> float:
    g = set(gold)
    dcg = sum(
        (1.0 / np.log2(i + 2)) for i, rid in enumerate(retrieved[:k]) if rid in g
    )
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(g), k)))
    return float(dcg / idcg) if idcg > 0 else 0.0


def bootstrap_ci(
    values: Sequence[float],
    n_resamples: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Returns (mean, lo, hi). Report as ``mean [lo, hi]``."""
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    idx = rng.integers(0, arr.size, size=(n_resamples, arr.size))
    means = arr[idx].mean(axis=1)
    return (
        float(arr.mean()),
        float(np.quantile(means, alpha / 2)),
        float(np.quantile(means, 1 - alpha / 2)),
    )


def convergence_stats(results) -> dict:
    """Instrumentation for risk register R-5 -- embedding geometry vs attractor dynamics.

    A high blend rate is not a bug to hide; it is a finding. Report it.
    """
    steps = [r.n_steps for r in results if r.n_steps is not None]
    conv = [r.converged for r in results if r.converged is not None]
    margins = [
        r.energy_margin
        for r in results
        if r.energy_margin is not None and np.isfinite(r.energy_margin)
    ]
    return {
        "mean_steps": float(np.mean(steps)) if steps else None,
        "convergence_rate": float(np.mean(conv)) if conv else None,
        "mean_energy_margin": float(np.mean(margins)) if margins else None,
        # low margin => landed between attractors rather than in one
        "blend_rate": float(np.mean([m < 0.05 for m in margins])) if margins else None,
    }
