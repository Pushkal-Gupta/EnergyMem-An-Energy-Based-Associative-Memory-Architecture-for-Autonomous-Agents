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

    def set_routing(self, R) -> None:
        """G3. Learned from temporal co-occurrence -- never from cosine similarity.

        Training this on similarity would collapse G3 back into the baseline and
        silently invalidate RQ3.

        ``R`` may be either a (d, d) matrix or **any callable** ``z -> z`` -- which is how
        a learned world model enters the energy landscape (``world/knn_wm.py``,
        ``world/jepa.py``). The nonlinear case is the interesting one: association stops
        being a fixed reweighting and becomes a *prediction* about what follows, applied
        inside the descent rather than alongside it. That integration is the claim that
        separates this from PAM, which keeps its predictor external to retrieval.
        """
        self.R = R if callable(R) else np.asarray(R, dtype=np.float64)

    def _route(self, z: np.ndarray) -> np.ndarray:
        """Apply the routing operator, matrix or learned predictor."""
        if callable(self.R):
            return self.R(z)
        return self.R @ z

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
                nxt = (1.0 - self.eta) * nxt + self.eta * self._route(nxt)
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

    def retrieve_multi_parallel(
        self,
        query: str | np.ndarray,
        k: int = 3,
        n_particles: int | None = None,
        sigma: float = 0.35,   # perturbation norm as a FRACTION OF THE CUE (dim-normalized)
        gamma: float = 0.6,
        seed: int = 0,
    ) -> RetrievalResult:
        """Multi-target retrieval by REPULSIVE PARALLEL descent -- the ablation against
        ``retrieve_multi`` (residual peeling).

        Instead of descending k times sequentially, descend ``n_particles`` perturbed copies
        of the cue *simultaneously*, with a repulsion term that stops them collapsing onto
        the same attractor. Structurally this is Stein-variational descent over E: the
        particles cover the modes instead of all falling into the deepest one.

        WHY IT EXISTS -- see docs/09_DAY0_FINDINGS.md F7. Two naive versions of this idea
        do NOT work and must be reported as such:

          * perturbation alone (gamma=0) fails at every noise level. Small sigma -> every
            particle converges to the same attractor (1.0 distinct out of 16); large sigma
            -> particles scatter uniformly over the corpus and the gold set is no more
            likely to be hit than anything else. There is no sweet spot, because isotropic
            noise carries no information about where the *other* targets are.
          * seeding at the cosine top-m candidates is circular. Stored patterns are exact
            fixed points (self-recall 1.000 at every beta we tested), so descent is a no-op
            on those seeds, and any ranking over the landings is just cosine top-k again.

        Repulsion is the version that works: it concentrates the particles onto the gold
        set (1.000 of particles land on a gold memory at k=4) because the force is
        data-dependent -- particles push off each other's *current* positions, which are
        inside the cue's own neighbourhood.

        Trade-off vs peeling, to be reported honestly in the ablation table:
          + one batched matmul per step instead of k sequential descents (the advantage
            should grow with corpus size and on GPU; on CPU at N=1000 it is currently
            2-3x SLOWER, so measure, do not assume)
          - no ranking signal. Peeling gets an ordering for free from the residual each
            round explains; here every particle looks equally converged.
          - lower recall (0.93 vs 1.000 at k=4) and it degrades as beta rises
          - two extra hyperparameters (sigma, gamma) that peeling does not need
        """
        if self.X is None or self.b is None:
            raise RuntimeError("index() before retrieve_multi_parallel()")

        t0 = time.perf_counter()
        q = l2_normalize(self._encode(query))
        m = n_particles if n_particles is not None else 4 * k
        rng = np.random.default_rng(seed)

        X = self.X
        d = X.shape[0]
        # DIMENSION-NORMALIZED PERTURBATION. sigma is the perturbation norm as a
        # FRACTION OF THE CUE, not a per-component standard deviation. An isotropic
        # draw with per-component sd s has expected norm s*sqrt(d), so at d=768 a
        # "sigma" of 0.35 produces a perturbation ~10x the length of the unit cue and
        # destroys it outright. That is exactly what happened on the first real-embedding
        # run -- see docs/09_DAY0_FINDINGS.md F10. Tuning sigma on d=256 synthetic data
        # and carrying the number to d=768 real data is a silent, total failure.
        sigma_c = sigma / np.sqrt(d)
        Xi = l2_normalize(q[:, None] + sigma_c * rng.normal(size=(d, m)), axis=0)

        steps = 0
        for t in range(self.n_steps):
            beta_t = self._beta_at(t)
            Z = beta_t * (X.T @ Xi + self.b[:, None])
            Z = Z - Z.max(axis=0, keepdims=True)
            P = np.exp(Z)
            P /= P.sum(axis=0, keepdims=True)
            nxt = X @ P
            if self.R is not None and self.eta > 0.0:
                routed = np.stack([self._route(nxt[:, j]) for j in range(nxt.shape[1])], axis=1)
                nxt = (1.0 - self.eta) * nxt + self.eta * routed
            if gamma > 0.0 and m > 1:
                G = Xi.T @ Xi
                np.fill_diagonal(G, 0.0)
                W = np.exp(beta_t * 0.25 * G)
                np.fill_diagonal(W, 0.0)
                nxt = nxt - gamma * (Xi @ W) / (m - 1)
            nxt = l2_normalize(nxt, axis=0)
            delta = float(np.abs(nxt - Xi).max())
            Xi = nxt
            steps = t + 1
            if delta < self.tol:
                break

        # Readout: each particle votes for its nearest stored pattern. Strongest-matching
        # particles vote first. This is the weak point of the method -- see docstring.
        S = X.T @ Xi
        found: list[int] = []
        scores: list[float] = []
        for p_idx in np.argsort(-S.max(axis=0)):
            for j in np.argsort(-S[:, p_idx]):
                if int(j) not in found:
                    found.append(int(j))
                    scores.append(float(S[j, p_idx]))
                    break
            if len(found) == k:
                break
        for j in np.argsort(-S.max(axis=1)):        # backfill if particles collapsed
            if len(found) == k:
                break
            if int(j) not in found:
                found.append(int(j))
                scores.append(float(S[j].max()))

        return RetrievalResult(
            ids=[self.memories[i].id for i in found],
            scores=scores,
            latency_ms=(time.perf_counter() - t0) * 1e3,
            n_steps=steps,
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
