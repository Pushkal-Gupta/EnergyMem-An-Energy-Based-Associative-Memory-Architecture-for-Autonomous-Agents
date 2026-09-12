"""Module D -- the readout. Scores stored memories against a *set* of hypotheses.

    score(m) = logsumexp_s [ beta * <xi_s, z_m> + log w_s ]

and NEVER  <mean_s xi_s, z_m>.

WHY THIS FILE EXISTS. Every other stage of the stack can carry a distribution over
futures faithfully and still have it destroyed here, because averaging at the readout is
indistinguishable from averaging at the predictor as far as the final ranking is
concerned. INVARIANT I4 has to be enforced at the last step or it is not enforced at all.

TWO SILENT BUGS THIS FILE GUARDS AGAINST -- both written by us, both cost a day, both
score exactly 0.000 rather than crashing:

  1. Multiplying similarity by weight *before* the max-reduction instead of adding
     log w inside the logsumexp. Weights are < 1, so this shrinks every score toward
     zero and destroys the ranking.
  2. Applying a mode's displacement to *its own neighbour's* cue rather than to the
     query, which makes every candidate a literal stored training memory.

``tests/test_mixture_readout.py`` pins both with hand-computed fixtures.
"""
from __future__ import annotations

import numpy as np

from .base import l2_normalize
from .world.base import Hypothesis


def _stack(hyps: list[Hypothesis]) -> tuple[np.ndarray, np.ndarray]:
    if not hyps:
        raise ValueError("no hypotheses to read out")
    Xi = np.stack([np.asarray(z, dtype=np.float64) for z, _ in hyps])      # (S, d)
    w = np.asarray([w for _, w in hyps], dtype=np.float64)                 # (S,)
    if np.any(w <= 0.0):
        raise ValueError("hypothesis weights must be strictly positive")
    return Xi, w / w.sum()


def mixture_scores(hyps: list[Hypothesis], Z: np.ndarray, beta: float) -> np.ndarray:
    """score(m) = logsumexp_s [ beta * <xi_s, z_m> + log w_s ]   -> (N,)

    ``Z`` is (N, d), rows L2-normalized. Max-centred over the hypothesis axis, so the
    reduction is numerically exact at any beta.

    With S=1 this is ``beta * <xi, z_m>``, a strictly increasing function of the cosine
    similarity, so the induced RANKING is identical to plain cosine readout from that
    state. That is the null setting, and it is asserted in the test suite.
    """
    Xi, w = _stack(hyps)
    A = beta * (Xi @ Z.T) + np.log(w)[:, None]        # (S, N)
    m = A.max(axis=0)                                  # (N,)
    return m + np.log(np.exp(A - m).sum(axis=0))


def averaged_scores(hyps: list[Hypothesis], Z: np.ndarray) -> np.ndarray:
    """<mean_s w_s xi_s, z_m> -- the mode-averaging BASELINE. Not for the retrieval path.

    Present so the failure it causes can be measured rather than asserted.
    """
    Xi, w = _stack(hyps)
    return Z @ l2_normalize((w[:, None] * Xi).sum(0))


def rank(scores: np.ndarray, k: int, exclude: set[int] | None = None) -> list[int]:
    """Top-k indices by score, descending, with ``exclude`` removed FIRST.

    Exclusion happens before ranking and applies to every method identically. An
    asymmetry here -- excluding the cue for one method and not another -- once produced
    an exact 0.000 that looked like a finding.
    """
    order = np.argsort(-scores)
    if exclude:
        order = [i for i in order if i not in exclude]
    return [int(i) for i in order[:k]]
