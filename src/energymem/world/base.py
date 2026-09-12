"""The predictor interface -- module P of the target architecture (docs/12_ARCHITECTURE.md).

A predictor answers "given this state, what tends to come next?" **in embedding space**
(the JEPA formulation -- never in token space), and it answers with a *distribution*, not
a point.

INVARIANT I4 -- NEVER AVERAGE MODES. The conditional next-state distribution is
multi-modal: one situation can legitimately lead to several mutually exclusive futures.
The average of two distinct memories is a location where nothing is stored -- a ridge
between basins rather than a basin -- so a point estimate is off-manifold by construction
whenever the cue branches. ``modes()`` is therefore the primary interface and ``predict()``
is kept only as the documented baseline that ``modes()`` is measured against.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np

#: One hypothesis: a unit-norm state and its (unnormalized-at-call-time) weight.
Hypothesis = tuple[np.ndarray, float]


def renormalize(hyps: list[Hypothesis]) -> list[Hypothesis]:
    """Rescale weights to sum to 1. Raises rather than silently dividing by zero."""
    tot = float(sum(w for _, w in hyps))
    if tot <= 0.0:
        raise ValueError("hypothesis weights sum to zero -- nothing to renormalize")
    return [(z, w / tot) for z, w in hyps]


class Predictor(Protocol):
    """Module P. Implemented by world/knn_wm.py (default) and world/jepa.py (negative result)."""

    def fit(self, Z: np.ndarray, transitions: list[tuple[int, int]]) -> "Predictor":
        """``transitions`` are (t, t+1) index pairs observed in the agent's own stream.

        Temporal adjacency is free supervision. INVARIANT I7: these pairs come from the
        stream, never from embedding similarity -- learning association from similarity
        would collapse the module back into the baseline.
        """
        ...

    def modes(self, q: np.ndarray, K: int) -> list[Hypothesis]:
        """Top-``K`` modes of P(next | q), weights summing to 1."""
        ...

    def rollout(self, q: np.ndarray, H: int, K: int, S: int) -> list[Hypothesis]:
        """Iterate ``modes`` ``H`` times, keeping the ``S`` heaviest hypotheses.

        ``H=0`` MUST return ``[(q, 1.0)]`` -- the cue itself, one hypothesis. That is the
        null setting under which the predictor is bypassed and the stack reduces to plain
        relaxation of the query (rung R0/R1 of the reduction ladder).
        """
        ...
