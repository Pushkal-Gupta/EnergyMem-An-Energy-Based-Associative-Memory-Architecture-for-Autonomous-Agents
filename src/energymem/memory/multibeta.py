"""Per-pattern beta, correctly parameterized.

A heterogeneous-beta memory GENERALIZES the fixed-beta memory: if every beta_i is equal
it must reproduce the fixed-beta network exactly. Any implementation that scores worse
than fixed beta under a proper search is therefore broken, not informative. The first
version of this file was broken in two ways, both fixed here.

BUG 1 -- the constant solution was not in the search space.
    Old:  beta_i = beta_min + gain * log(1+N) / gap_i,  beta_min = 4, gain in {0..32}
    gain=0 collapses to a uniform beta of 4.0 -- and beta<=8 is already known to collapse
    to the corpus centroid (docs/09_DAY0_FINDINGS F2, reproduced on images). So the only
    uniform setting reachable by the search was a catastrophic one; uniform beta=32 could
    not be expressed at all. The search could not fall back to the baseline it was
    supposed to generalize.

    New:  beta_i = beta_base * (1 + lam * z_i),   z = standardized crowding score
    lam=0 gives beta_i = beta_base for every i -- the fixed-beta network exactly, for any
    beta_base. The special case is now interior to the parameter space and (beta_base,
    lam) are searched jointly.

BUG 2 -- heterogeneous beta rescaled patterns against each other.
    In ``softmax_i(beta_i * s_i)`` a larger beta_i does not merely sharpen pattern i's
    basin; it multiplies pattern i's score relative to every other candidate. beta was
    controlling basin DEPTH when only SHARPNESS was intended, so patterns in crowded
    regions won on multiplier magnitude rather than similarity.

    Fix: center the scores on the running maximum before applying beta,

        w_i = exp(beta_i * (s_i - max_j s_j)),      p = w / sum(w)

    The top-scoring pattern is pinned at w=1 regardless of its own beta, so no pattern can
    out-shout another by carrying a bigger multiplier. Each beta_i now controls only how
    fast pattern i decays once it is *behind* the leader -- sharpness, not depth. Because
    softmax is shift-invariant, a constant beta makes this identical to the standard
    update, which is what preserves the reduction in BUG 1.

The scientific question this file can now actually answer: at the jointly-optimal
(beta_base, lam), is lam > 0? A nonzero optimum means locally-varying sharpness earns its
place. lam = 0 means it does not -- and that is a real answer, not an artifact of a
search that could not represent the baseline.
"""

from __future__ import annotations

import time
from typing import Sequence

import numpy as np

from ..base import Memory, RetrievalResult, Retriever, l2_normalize

#: beta_i is clipped to this floor so a large lam cannot drive a pattern's beta negative
BETA_FLOOR = 0.1


def _nearest_neighbor_gap(X: np.ndarray) -> np.ndarray:
    """gap_i = 1 - cos_sim to the closest OTHER stored pattern. X is (d, N), unit columns."""
    S = X.T @ X
    n = S.shape[0]
    S[np.arange(n), np.arange(n)] = -np.inf
    nn_sim = S.max(axis=1)
    return np.clip(1.0 - nn_sim, 1e-6, None)


def crowding_z(X: np.ndarray) -> np.ndarray:
    """Standardized crowding: high where a pattern's nearest neighbour is close.

    Uses -log(gap) so the long tail of near-duplicates does not dominate, then z-scores
    it so that ``lam`` has a consistent meaning across corpora of different densities.
    """
    gap = _nearest_neighbor_gap(X)
    c = -np.log(gap)
    sd = c.std()
    if sd < 1e-12:
        return np.zeros_like(c)
    return (c - c.mean()) / sd


class MultiBetaEnergyMemory(Retriever):
    """Energy memory with per-pattern sharpness.

    Parameters
    ----------
    beta_base
        The uniform temperature. With ``lam=0`` this *is* the fixed-beta network.
    lam
        Heterogeneity strength. ``beta_i = beta_base * (1 + lam * z_i)`` with ``z`` the
        standardized crowding score. ``lam=0`` disables heterogeneity entirely.
    """

    def __init__(
        self,
        n_steps: int = 8,
        beta_base: float = 32.0,
        lam: float = 0.0,
        tol: float = 1e-5,
        encoder=None,
    ) -> None:
        self.n_steps = n_steps
        self.beta_base = beta_base
        self.lam = lam
        self.tol = tol
        self.encoder = encoder

        self.memories: list[Memory] = []
        self.X: np.ndarray | None = None
        self.b: np.ndarray | None = None
        self.beta_vec: np.ndarray | None = None

    # ------------------------------------------------------------------ index

    def index(self, memories: Sequence[Memory], z: np.ndarray | None = None) -> None:
        """Index the bank. ``z`` optionally supplies a precomputed crowding score --
        it depends only on the patterns, so a (beta_base, lam) sweep over one fixed bank
        should compute it once via ``crowding_z`` rather than N^2 times."""
        if not memories:
            raise ValueError("cannot index an empty memory set")
        embs = [m.embedding for m in memories]
        if any(e is None for e in embs):
            raise ValueError("all memories must be encoded before indexing")
        self.memories = list(memories)
        self.X = l2_normalize(np.stack(embs), axis=1).T.astype(np.float64)
        self.b = np.zeros(len(memories), dtype=np.float64)

        if self.lam == 0.0:
            self.beta_vec = np.full(self.X.shape[1], self.beta_base, dtype=np.float64)
        else:
            if z is None:
                z = crowding_z(self.X)
            self.beta_vec = np.clip(
                self.beta_base * (1.0 + self.lam * z), BETA_FLOOR, None
            )

    # --------------------------------------------------------------- dynamics

    def _weights(self, s: np.ndarray) -> np.ndarray:
        """Max-centered heterogeneous softmax.

        Centering on max_j s_j pins the leader's weight at 1 for every beta, so beta_i
        cannot inflate pattern i against the field -- it only sets how quickly pattern i
        falls away once behind. With constant beta this is exactly softmax(beta * s).
        """
        z = self.beta_vec * (s - s.max())
        w = np.exp(z)
        return w / w.sum()

    def _descend(self, q: np.ndarray) -> tuple[np.ndarray, int, bool]:
        X, b = self.X, self.b
        xi = l2_normalize(q.astype(np.float64))
        converged = False
        steps = 0
        for t in range(self.n_steps):
            nxt = l2_normalize(X @ self._weights(X.T @ xi + b))
            delta = float(np.linalg.norm(nxt - xi))
            xi = nxt
            steps = t + 1
            if delta < self.tol:
                converged = True
                break
        return xi, steps, converged

    # --------------------------------------------------------------- retrieve

    def retrieve(self, query: str | np.ndarray, k: int = 5) -> RetrievalResult:
        if self.X is None or self.beta_vec is None:
            raise RuntimeError("index() before retrieve()")
        t0 = time.perf_counter()
        xi, steps, converged = self._descend(self._encode(query))

        sims = self.X.T @ xi
        order = np.argsort(-sims)[:k]
        margin = float(sims[order[0]] - sims[order[1]]) if len(order) > 1 else float("inf")

        return RetrievalResult(
            ids=[self.memories[i].id for i in order],
            scores=[float(sims[j]) for j in order],
            latency_ms=(time.perf_counter() - t0) * 1e3,
            n_steps=steps,
            converged=converged,
            energy_margin=margin,
        )

    def _encode(self, query: str | np.ndarray) -> np.ndarray:
        if isinstance(query, np.ndarray):
            return query.astype(np.float64)
        if self.encoder is None:
            raise ValueError("pass an encoder to embed string queries")
        return np.asarray(self.encoder.encode(query), dtype=np.float64)
