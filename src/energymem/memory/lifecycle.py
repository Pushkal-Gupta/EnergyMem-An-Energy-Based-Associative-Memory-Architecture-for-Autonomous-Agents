"""Memory lifecycle -- G2 (strength-modulated basin depth) and C4 (consolidation).

The claim under test (RQ2): reinforcement, decay and forgetting -- normally three
separate heuristics bolted onto a vector store -- are one mechanism, namely a bias
term ``b_i`` that deepens or shallows each memory's basin in the shared energy.

The honest counter-hypothesis, which MUST be run as a baseline (E2): a cosine
retriever with the same signals as a static score prior does just as well. If it does,
this is an engineering convenience rather than a contribution, and the paper says so.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..base import Memory


@dataclass
class StrengthParams:
    """Weights for the strength bias. Grid-searched in E2; knocked out one at a time."""

    w_usage: float = 0.3
    w_recency: float = 0.3
    w_surprise: float = 0.2
    tau: float = 7.0        # recency half-life, days
    decay_lambda: float = 0.98   # per-tick multiplicative decay for inactive traces


def strength_bias(
    memories: Sequence[Memory],
    now: float,
    params: StrengthParams | None = None,
    surprise: np.ndarray | None = None,
) -> np.ndarray:
    """Compute ``b`` for EnergyMemory.set_strength_bias.

    b_i = w_u log(1 + usage_i) + w_r exp(-dt_i / tau) + w_s surprise_i
    """
    p = params or StrengthParams()
    usage = np.array([m.usage_count for m in memories], dtype=np.float64)
    dt_days = np.array([(now - m.last_access) / 86_400.0 for m in memories])
    dt_days = np.clip(dt_days, 0.0, None)

    b = p.w_usage * np.log1p(usage) + p.w_recency * np.exp(-dt_days / p.tau)
    if surprise is not None:
        b = b + p.w_surprise * np.asarray(surprise, dtype=np.float64)
    return b


def reinforce(memories: Sequence[Memory], hit_ids: Sequence[str], now: float) -> None:
    """Retrieval is itself a write. Deepens the basins that got used."""
    hits = set(hit_ids)
    for m in memories:
        if m.id in hits:
            m.usage_count += 1
            m.last_access = now


def decay(memories: Sequence[Memory], params: StrengthParams | None = None) -> None:
    """Continuous forgetting -- unused traces lose basin depth rather than being deleted."""
    p = params or StrengthParams()
    for m in memories:
        m.salience *= p.decay_lambda


def consolidate(
    X: np.ndarray,
    memories: Sequence[Memory],
    threshold: float = 0.85,
) -> tuple[np.ndarray, list[list[int]]]:
    """C4 -- merge overlapping basins into prototypes.

    Returns the prototype matrix and the provenance groups (which originals each
    prototype absorbed). Provenance is kept because E4 needs to score
    specific-episode queries against the memories that were merged away.

    EXPECTED RESULT (and the interesting one): a crossover. Consolidation should help
    general-knowledge queries and hurt specific-episode queries. Finding where that
    crossover sits is a better contribution than a monotone win -- do not tune the
    threshold to hide it.

    NOTE: this is deliberately a simple greedy agglomeration. Do not replace it with
    something clever until E4 shows the simple version is the bottleneck.
    """
    n = X.shape[1]
    sims = X.T @ X
    unassigned = set(range(n))
    groups: list[list[int]] = []

    while unassigned:
        seed = min(unassigned)
        members = [j for j in unassigned if sims[seed, j] >= threshold]
        groups.append(sorted(members))
        unassigned -= set(members)

    protos = np.stack(
        [X[:, g].mean(axis=1) for g in groups], axis=1
    )
    protos = protos / (np.linalg.norm(protos, axis=0, keepdims=True) + 1e-12)
    return protos, groups
