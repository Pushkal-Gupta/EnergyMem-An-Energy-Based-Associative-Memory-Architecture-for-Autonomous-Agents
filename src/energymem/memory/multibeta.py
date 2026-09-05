"""Multi-beta energy memory -- a per-memory temperature instead of one global beta.

MOTIVATION (see experiments/PROTOCOL_CHANGES.md, 2026-09-03 capacity-sweep entry).
EnergyMemory uses a single global beta for every stored pattern. The capacity sweep
showed this is a real limitation: a beta tuned for a small, evenly-spread memory bank
stops being sharp enough once the bank grows and some regions become locally crowded
(e.g. several near-duplicate images), while other regions stay sparse. One shared
number cannot be simultaneously right for both.

THE IDEA. Give every stored memory its own beta_i, set from how crowded its own
neighborhood is:

    - a memory with a close look-alike nearby (small gap to its nearest neighbour)
      needs a SHARP, high beta_i so its own basin does not blur into its neighbour's
    - a memory sitting alone (large gap to its nearest neighbour) can use a LOW,
      forgiving beta_i, giving it a wide basin that still finds it from a heavily
      corrupted cue

This changes the SHAPE of the energy landscape (sculpting, done once at index time).
The retrieval loop itself (_descend) is unchanged in structure -- same iterative
"move toward the weighted blend of stored patterns, repeat until settled" process as
EnergyMemory. The only change is that each pattern's pull uses its own beta_i instead
of one shared beta.

SCULPTING RULE. For pattern i, let gap_i = 1 - max_j!=i cos_sim(x_i, x_j) (patterns
are L2-normalized, so cos_sim(x_i, x_i) = 1; gap_i is the similarity headroom before
the nearest competitor). This mirrors the modern-Hopfield-network stability
requirement: the beta needed to keep a pattern a well-separated attractor grows as its
separation from its nearest neighbour shrinks, and grows (logarithmically) with the
number of competing patterns N. We use:

    beta_i = beta_min + gain * log(1 + N) / gap_i

`gain` is the one free constant. It is NOT hand-picked: `index()` self-calibrates it
by corrupting a handful of the memory bank's own patterns (dimension dropout, not
domain-specific) and choosing whichever gain in a small grid recovers them best. No
external validation set or domain knowledge is required.
"""

from __future__ import annotations

import time
from typing import Sequence

import numpy as np

from ..base import Memory, RetrievalResult, Retriever, l2_normalize

GAIN_GRID = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
BETA_MIN = 4.0
CALIB_N_QUERIES = 50
CALIB_DROPOUT = 0.30


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def _nearest_neighbor_gap(X: np.ndarray) -> np.ndarray:
    """gap_i = 1 - similarity to the closest OTHER stored pattern. X is (d, N), L2-normalized columns."""
    S = X.T @ X  # (N, N) cosine similarities
    n = S.shape[0]
    S[np.arange(n), np.arange(n)] = -np.inf
    nn_sim = S.max(axis=1)
    return np.clip(1.0 - nn_sim, 1e-3, None)


def _corrupt_embedding(x: np.ndarray, rate: float, rng: np.random.Generator) -> np.ndarray:
    """Domain-agnostic corruption for self-calibration: zero out a random fraction of
    dimensions and renormalize. Mirrors the "partial/noisy cue" scenario without
    needing to know whether x came from an image, text, or anything else."""
    keep = rng.random(x.shape[0]) >= rate
    if not keep.any():
        keep[rng.integers(x.shape[0])] = True
    return l2_normalize(x * keep)


class MultiBetaEnergyMemory(Retriever):
    """EnergyMemory with a per-pattern beta instead of one global beta.

    Same retrieval loop and interface as EnergyMemory. See module docstring for the
    sculpting rule and self-calibration procedure.
    """

    def __init__(self, n_steps: int = 8, tol: float = 1e-5, encoder=None) -> None:
        self.n_steps = n_steps
        self.tol = tol
        self.encoder = encoder

        self.memories: list[Memory] = []
        self.X: np.ndarray | None = None
        self.b: np.ndarray | None = None
        self.beta_vec: np.ndarray | None = None
        self.gain_used: float | None = None

    # ------------------------------------------------------------------ index

    def index(self, memories: Sequence[Memory]) -> None:
        if not memories:
            raise ValueError("cannot index an empty memory set")
        embs = [m.embedding for m in memories]
        if any(e is None for e in embs):
            raise ValueError("all memories must be encoded before indexing")
        self.memories = list(memories)
        self.X = l2_normalize(np.stack(embs), axis=1).T.astype(np.float64)  # (d, N)
        self.b = np.zeros(len(memories), dtype=np.float64)

        gap = _nearest_neighbor_gap(self.X)
        n = self.X.shape[1]
        self.gain_used = self._calibrate_gain(gap, n)
        self.beta_vec = BETA_MIN + self.gain_used * np.log(1 + n) / gap

    def _beta_for_gain(self, gap: np.ndarray, n: int, gain: float) -> np.ndarray:
        return BETA_MIN + gain * np.log(1 + n) / gap

    def _calibrate_gain(self, gap: np.ndarray, n: int) -> float:
        """Self-test: corrupt a sample of the bank's own patterns, see which gain
        in GAIN_GRID recovers them best. No external labels or domain knowledge used."""
        rng = np.random.default_rng(0)
        m = min(CALIB_N_QUERIES, n)
        sample_idx = rng.choice(n, size=m, replace=False)

        best_gain, best_acc = GAIN_GRID[0], -1.0
        for gain in GAIN_GRID:
            beta_vec = self._beta_for_gain(gap, n, gain)
            hits = 0
            for i in sample_idx:
                q = _corrupt_embedding(self.X[:, i], CALIB_DROPOUT, rng)
                xi, *_ = self._descend(q, self.X, self.b, beta_vec)
                sims = self.X.T @ xi
                if int(np.argmax(sims)) == i:
                    hits += 1
            acc = hits / m
            if acc > best_acc:
                best_acc, best_gain = acc, gain
        return best_gain

    # --------------------------------------------------------------- dynamics

    def _descend(
        self, q: np.ndarray, X: np.ndarray, b: np.ndarray, beta_vec: np.ndarray
    ) -> tuple[np.ndarray, int, bool]:
        xi = l2_normalize(q.astype(np.float64))
        converged = False
        steps = 0
        for t in range(self.n_steps):
            p = _softmax(beta_vec * (X.T @ xi + b))
            nxt = l2_normalize(X @ p)
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
        q = self._encode(query)
        xi, steps, converged = self._descend(q, self.X, self.b, self.beta_vec)

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
