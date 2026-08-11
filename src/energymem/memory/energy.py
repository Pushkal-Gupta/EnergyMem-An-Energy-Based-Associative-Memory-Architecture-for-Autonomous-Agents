"""The energy core.

Implements

    E(xi) = -lse(beta, X^T xi + b) + 0.5 * ||xi||^2

with the concave-convex (CCCP) update

    xi <- X softmax(beta (X^T xi + b))                  # Hopfield step
    xi <- (1 - eta) xi + eta R xi                       # associative routing (G3)
    xi <- xi / ||xi||

Three generalization axes over dense top-k retrieval (see docs/01_RESEARCH_PLAN.md):

    G1  n_steps > 1, beta schedule      -- multi-step pattern completion
    G2  b != 0                          -- strength-modulated basin depth
    G3  R != I                          -- learned association topology

PROPOSITION 1 (the reduction). With ``n_steps=1``, ``beta -> inf``, ``b = 0``, ``R = I``
and L2-normalized patterns, this returns exactly the top-1 cosine neighbour. Dense
retrieval is a corner of our parameter space, not a separate system. ``tests/
test_reduction.py`` asserts this numerically; if it ever fails, no result downstream is
trustworthy.
"""

from __future__ import annotations

import time
from typing import Literal, Sequence

import numpy as np

from ..base import Memory, RetrievalResult, Retriever, l2_normalize

BetaSchedule = Literal["const", "linear", "cosine"]


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def _logsumexp(z: np.ndarray, beta: float) -> float:
    m = z.max()
    return float(m + np.log(np.exp(beta * (z - m)).sum()) / beta)


class EnergyMemory(Retriever):
    """Energy-based associative memory over embedded episodes.

    Parameters
    ----------
    n_steps
        Descent steps ``T``. ``1`` recovers one-shot retrieval (G1 off).
    beta
        Inverse temperature. Large beta sharpens toward hard argmax.
    beta_schedule
        Annealing over steps. Low-to-high beta explores broadly then commits --
        this is the mechanism we expect to help on composite cues (DegradeBench D5).
    eta
        Routing mix weight. ``0.0`` disables G3.
    tol
        Convergence threshold on ``||xi_{t+1} - xi_t||``.
    shortlist
        If set, restrict descent to the top-M ANN candidates. This is the latency
        mitigation from docs/06_RISK_REGISTER.md R-4 -- report it as a design choice
        and measure its accuracy cost, never hide it.
    """

    def __init__(
        self,
        n_steps: int = 8,
        beta: float = 16.0,
        beta_schedule: BetaSchedule = "const",
        beta_final: float | None = None,
        eta: float = 0.0,
        tol: float = 1e-5,
        shortlist: int | None = None,
        encoder=None,
    ) -> None:
        self.n_steps = n_steps
        self.beta = beta
        self.beta_schedule = beta_schedule
        self.beta_final = beta_final if beta_final is not None else beta
        self.eta = eta
        self.tol = tol
        self.shortlist = shortlist
        self.encoder = encoder

        self.memories: list[Memory] = []
        self.X: np.ndarray | None = None       # (d, N) normalized patterns
        self.b: np.ndarray | None = None       # (N,)   strength bias, G2
        self.R: np.ndarray | None = None       # (d, d) routing operator, G3

    # ------------------------------------------------------------------ index

    def index(self, memories: Sequence[Memory]) -> None:
        if not memories:
            raise ValueError("cannot index an empty memory set")
        embs = [m.embedding for m in memories]
        if any(e is None for e in embs):
            raise ValueError(
                "all memories must be encoded before indexing; run the encoder first "
                "so that every retriever sees identical vectors (matched-encoder rule)"
            )
        self.memories = list(memories)
        self.X = l2_normalize(np.stack(embs), axis=1).T.astype(np.float64)  # (d, N)
        self.b = np.zeros(len(memories), dtype=np.float64)

    def set_strength_bias(self, b: np.ndarray) -> None:
        """G2. Supplied by memory/lifecycle.py -- usage, recency, surprise."""
        if self.X is None:
            raise RuntimeError("index() before set_strength_bias()")
        if b.shape != (self.X.shape[1],):
            raise ValueError(f"expected shape {(self.X.shape[1],)}, got {b.shape}")
        self.b = b.astype(np.float64)

    def set_routing(self, R: np.ndarray) -> None:
        """G3. Learned from temporal co-occurrence -- never from cosine similarity.

        Training this on similarity would collapse G3 back into the baseline and
        silently invalidate RQ3.
        """
        self.R = R.astype(np.float64)

    # --------------------------------------------------------------- dynamics

    def _beta_at(self, t: int) -> float:
        if self.n_steps <= 1 or self.beta_schedule == "const":
            return self.beta
        frac = t / (self.n_steps - 1)
        if self.beta_schedule == "linear":
            return self.beta + frac * (self.beta_final - self.beta)
        # cosine
        return self.beta_final + 0.5 * (self.beta - self.beta_final) * (
            1 + np.cos(np.pi * frac)
        )

    def energy(self, xi: np.ndarray, X: np.ndarray, b: np.ndarray, beta: float) -> float:
        """E(xi) = -lse(beta, X^T xi + b) + 0.5 ||xi||^2.

        Must decrease monotonically under the CCCP update -- asserted in tests.
        """
        return -_logsumexp(X.T @ xi + b, beta) + 0.5 * float(xi @ xi)

    def _descend(
        self, q: np.ndarray, X: np.ndarray, b: np.ndarray
    ) -> tuple[np.ndarray, int, bool, float, np.ndarray]:
        xi = l2_normalize(q.astype(np.float64))
        traj = [xi.copy()]
        converged = False
        steps = 0

        for t in range(self.n_steps):
            beta_t = self._beta_at(t)
            p = _softmax(beta_t * (X.T @ xi + b))
            nxt = X @ p
            if self.R is not None and self.eta > 0.0:
                nxt = (1.0 - self.eta) * nxt + self.eta * (self.R @ nxt)
            nxt = l2_normalize(nxt)

            delta = float(np.linalg.norm(nxt - xi))
            xi = nxt
            traj.append(xi.copy())
            steps = t + 1
            if delta < self.tol:
                converged = True
                break

        final_e = self.energy(xi, X, b, self._beta_at(steps - 1))
        return xi, steps, converged, final_e, np.stack(traj)

    # --------------------------------------------------------------- retrieve

    def retrieve(self, query: str | np.ndarray, k: int = 5) -> RetrievalResult:
        if self.X is None or self.b is None:
            raise RuntimeError("index() before retrieve()")

        t0 = time.perf_counter()
        q = self._encode(query)

        X, b, idx_map = self.X, self.b, None
        if self.shortlist is not None and self.shortlist < self.X.shape[1]:
            sims = self.X.T @ l2_normalize(q)
            idx_map = np.argpartition(-sims, self.shortlist)[: self.shortlist]
            X, b = self.X[:, idx_map], self.b[idx_map]

        xi, steps, converged, final_e, traj = self._descend(q, X, b)

        # Read out: which stored patterns does the converged state sit on?
        sims = X.T @ xi
        order = np.argsort(-sims)[:k]
        # Blend detection: a small margin means we landed between attractors rather
        # than in one. Reported, not suppressed -- see risk register R-5.
        margin = float(sims[order[0]] - sims[order[1]]) if len(order) > 1 else float("inf")

        local = order if idx_map is None else idx_map[order]
        return RetrievalResult(
            ids=[self.memories[i].id for i in local],
            scores=[float(sims[j]) for j in order],
            latency_ms=(time.perf_counter() - t0) * 1e3,
            n_steps=steps,
            converged=converged,
            final_energy=final_e,
            energy_margin=margin,
            trajectory=traj,
        )

    def retrieve_multi(self, query: str | np.ndarray, k: int = 3) -> RetrievalResult:
        """Multi-target retrieval by residual peeling (matching pursuit over the landscape).

        Descend to an attractor, project it out of the residual cue, re-descend. Repeat k
        times.

        WHY THIS EXISTS -- see docs/09_DAY0_FINDINGS.md F4/F5. Plain ``retrieve`` converges
        to a *single* point, and reading top-k from that point ranks by proximity to one
        attractor. For a composite cue this is strictly worse than top-k on the original
        query, because convergence destroys the multiplicity the query still carries.
        Measured: Recall 0.25 vs 1.00 at k=4. Peeling restores it to 1.00.

        This is not a workaround -- it is the correct readout for multi-target queries under
        energy dynamics, and none of the systems in the citation audit do it.
        """
        if self.X is None or self.b is None:
            raise RuntimeError("index() before retrieve_multi()")

        t0 = time.perf_counter()
        resid = l2_normalize(self._encode(query))
        found: list[int] = []
        scores: list[float] = []
        total_steps = 0

        for _ in range(k):
            xi, steps, _conv, _e, _traj = self._descend(resid, self.X, self.b)
            total_steps += steps
            sims = self.X.T @ xi
            for j in np.argsort(-sims):
                if int(j) not in found:
                    found.append(int(j))
                    scores.append(float(sims[j]))
                    break
            else:
                break
            # project the committed attractor out of the residual cue
            v = self.X[:, found[-1]]
            resid = resid - (resid @ v) * v
            if np.linalg.norm(resid) < 1e-8:
                break
            resid = l2_normalize(resid)

        return RetrievalResult(
            ids=[self.memories[i].id for i in found],
            scores=scores,
            latency_ms=(time.perf_counter() - t0) * 1e3,
            n_steps=total_steps,
            converged=True,
        )

    def _encode(self, query: str | np.ndarray) -> np.ndarray:
        if isinstance(query, np.ndarray):
            return query.astype(np.float64)
        if self.encoder is None:
            raise ValueError("pass an encoder to embed string queries")
        return np.asarray(self.encoder.encode(query), dtype=np.float64)


def dense_topk_config(**kw) -> EnergyMemory:
    """The baseline, expressed as a configuration of our own model.

    This is deliberate: it makes the comparison in every table exact by construction
    rather than dependent on two codebases agreeing. See Proposition 1.
    """
    kw.setdefault("encoder", None)
    return EnergyMemory(n_steps=1, beta=1e6, eta=0.0, **kw)
