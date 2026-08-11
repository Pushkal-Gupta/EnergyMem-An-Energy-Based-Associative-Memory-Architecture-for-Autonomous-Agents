"""Shared interfaces. FREEZE THIS FILE after the first standup.

Every retriever -- ours and every baseline -- implements `Retriever`. That is what lets
three people work in parallel without blocking each other, and it is what makes the
matched-encoder comparison exact by construction rather than by convention.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence

import numpy as np


@dataclass
class Memory:
    """One stored episode."""

    id: str
    text: str
    timestamp: float = 0.0
    source: str = "agent"
    salience: float = 1.0
    #: filled in by the encoder; L2-normalized
    embedding: np.ndarray | None = None
    #: lifecycle state (see memory/lifecycle.py)
    usage_count: int = 0
    last_access: float = 0.0
    metadata: dict = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """What every retriever returns.

    Baselines leave the energy-specific fields as ``None``. The evaluation harness
    treats ``None`` as "not applicable", never as zero.
    """

    ids: list[str]
    scores: list[float]
    latency_ms: float

    # --- energy-specific; None for non-iterative baselines -------------------
    n_steps: int | None = None
    converged: bool | None = None
    final_energy: float | None = None
    #: gap between the winning attractor and the runner-up. Low margin => blend state.
    energy_margin: float | None = None
    #: trajectory of the retrieval state, for the convergence figures
    trajectory: np.ndarray | None = None


class Retriever(Protocol):
    """The one interface. Do not add methods without a standup."""

    def index(self, memories: Sequence[Memory]) -> None:
        """Build whatever internal structure is needed. Called once."""
        ...

    def retrieve(self, query: str | np.ndarray, k: int = 5) -> RetrievalResult:
        """Return the top-k memory ids for a query cue."""
        ...


def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    """L2 normalization. Required everywhere -- the reduction result in
    Proposition 1 only holds for normalized patterns."""
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + eps)
