"""Perplexity-targeted adaptive beta -- a query-conditional, step-dynamic landscape.

WHY THIS EXISTS. Two prior negative results motivate it, both in
experiments/PROTOCOL_CHANGES.md:

  (a) Capacity sweep: a fixed global beta tuned at one corpus size degrades as the
      corpus grows. With beta held constant, adding memories makes the softmax
      relatively flatter -- more candidates share the mass -- so the descent commits
      less sharply at N=4000 than at N=300 despite identical settings.

  (b) Per-memory beta (memory/multibeta.py): assigning each stored pattern its own
      beta_i FAILED, and worse than baseline. Reason: in ``softmax_i(beta_i * s_i)``
      a heterogeneous beta_i does not merely sharpen pattern i's basin, it rescales
      pattern i's score against every other candidate. Crowded patterns (high beta_i)
      won the competition on multiplier size rather than on similarity. beta was made
      to control basin DEPTH when only basin SHARPNESS was intended.

THE FIX -- adapt along the query/time axis, not the memory axis. At every descent step
a SINGLE beta_t is applied to all candidates, so the competition stays fair (this is
what (b) violated), but beta_t is chosen freshly from the similarity distribution the
current state actually sees (this is what (a) lacked).

MECHANISM. Let s = X^T xi + b be the current scores. Define the softmax perplexity

    PP(beta) = exp(H(softmax(beta * s)))

the *effective number of memories* contributing to the next state. PP is monotonically
decreasing in beta: PP -> N as beta -> 0 (everything contributes), PP -> 1 as
beta -> inf (one attractor takes all). We binary-search beta_t (geometrically, on
log beta) so that PP(beta_t) hits a target, and anneal that target across steps:

    PP_target(t): pp_start  ->  pp_end     (geometric)

Early steps keep many candidates alive (explore the landscape from a degraded cue);
late steps force commitment to a single attractor.

WHY THIS SHOULD FIX THE CAPACITY PROBLEM. Perplexity targeting is *scale-free*. It
pins the effective competitor count, not the temperature. When N grows, holding beta
fixed lets the effective competitor count drift upward; targeting PP instead makes
beta rise automatically to hold commitment constant. The controlled quantity is the
one that matters for retrieval, and it is invariant to corpus size.

Precedent: this is t-SNE's per-point bandwidth calibration (binary search sigma to hit
a target perplexity) transplanted onto Hopfield descent. The novelty here is not the
binary search; it is applying commitment-targeting *per query and per descent step* to
an associative memory, and the claim that it is the right axis of adaptation because
the memory axis provably is not (see (b)).
"""

from __future__ import annotations

import time
from typing import Sequence

import numpy as np

from ..base import Memory, RetrievalResult, Retriever, l2_normalize


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def _perplexity(p: np.ndarray) -> float:
    """exp(entropy) -- the effective number of contributing memories."""
    nz = p[p > 0]
    return float(np.exp(-(nz * np.log(nz)).sum()))


def beta_for_perplexity(
    s: np.ndarray,
    target_pp: float,
    lo: float = 1e-4,
    hi: float = 1e5,
    iters: int = 50,
) -> float:
    """Binary-search (geometric) the beta whose softmax over ``s`` has perplexity ``target_pp``.

    PP is monotonically decreasing in beta, so if the current PP is too high we need a
    sharper (larger) beta.
    """
    target_pp = max(1.0 + 1e-9, min(target_pp, float(len(s))))
    for _ in range(iters):
        mid = float(np.sqrt(lo * hi))
        pp = _perplexity(_softmax(mid * s))
        if pp > target_pp:
            lo = mid          # too flat -> sharpen
        else:
            hi = mid          # too sharp -> soften
        if hi / lo < 1.0 + 1e-6:
            break
    return float(np.sqrt(lo * hi))


class AdaptiveBetaEnergyMemory(Retriever):
    """Energy memory whose temperature is re-solved per query and per descent step.

    Parameters
    ----------
    n_steps
        Descent steps T.
    pp_start, pp_end
        Perplexity (effective-competitor-count) target at the first and last step.
        Annealed geometrically. ``pp_start`` is additionally capped at N.
    tol
        Convergence threshold on the state update.
    """

    def __init__(
        self,
        n_steps: int = 8,
        pp_start: float = 32.0,
        pp_end: float = 1.2,
        anchor: float = 0.0,
        tol: float = 1e-5,
        encoder=None,
    ) -> None:
        self.n_steps = n_steps
        self.pp_start = pp_start
        self.pp_end = pp_end
        self.anchor = anchor
        self.tol = tol
        self.encoder = encoder

        self.memories: list[Memory] = []
        self.X: np.ndarray | None = None
        self.b: np.ndarray | None = None
        #: beta values actually solved for on the most recent retrieve() -- diagnostics
        self.last_betas: list[float] = []

    # ------------------------------------------------------------------ index

    def index(self, memories: Sequence[Memory]) -> None:
        if not memories:
            raise ValueError("cannot index an empty memory set")
        embs = [m.embedding for m in memories]
        if any(e is None for e in embs):
            raise ValueError("all memories must be encoded before indexing")
        self.memories = list(memories)
        self.X = l2_normalize(np.stack(embs), axis=1).T.astype(np.float64)
        self.b = np.zeros(len(memories), dtype=np.float64)

    # --------------------------------------------------------------- schedule

    def _pp_targets(self, n: int) -> list[float]:
        start = min(self.pp_start, float(n))
        end = max(1.0 + 1e-6, min(self.pp_end, start))
        if self.n_steps == 1:
            return [end]
        ratio = (end / start) ** (1.0 / (self.n_steps - 1))
        return [start * ratio**t for t in range(self.n_steps)]

    # --------------------------------------------------------------- dynamics

    def _descend(self, q: np.ndarray, X: np.ndarray, b: np.ndarray):
        """Anchored descent.

            xi_{t+1} = normalize( (1 - anchor) * X softmax(beta_t (X^T xi_t + b))
                                  + anchor * q_hat )

        The anchor term is not cosmetic. Without it (anchor=0) the first step already
        replaces the state with a blend of stored patterns, and the original cue never
        enters the computation again -- readout then ranks memories by proximity to that
        blend rather than to the query. That is defensible when the cue IS a degraded
        stored pattern (image completion) and structurally wrong when it is not
        (question -> evidence retrieval).

        anchor=1.0 pins the state at the query, so readout reduces to exact cosine
        ranking. Cosine is therefore a point in this model's parameter space, and
        sweeping anchor measures how much descent adds -- or costs -- against it.
        """
        q_hat = l2_normalize(q.astype(np.float64))
        xi = q_hat.copy()
        targets = self._pp_targets(X.shape[1])
        betas: list[float] = []
        converged = False
        steps = 0

        for t in range(self.n_steps):
            s = X.T @ xi + b
            beta_t = beta_for_perplexity(s, targets[t])
            betas.append(beta_t)
            step = X @ _softmax(beta_t * s)
            if self.anchor > 0.0:
                step = (1.0 - self.anchor) * step + self.anchor * q_hat
            nxt = l2_normalize(step)
            delta = float(np.linalg.norm(nxt - xi))
            xi = nxt
            steps = t + 1
            if delta < self.tol:
                converged = True
                break

        return xi, steps, converged, betas

    # --------------------------------------------------------------- retrieve

    def retrieve(self, query: str | np.ndarray, k: int = 5) -> RetrievalResult:
        if self.X is None or self.b is None:
            raise RuntimeError("index() before retrieve()")
        t0 = time.perf_counter()
        q = self._encode(query)
        xi, steps, converged, betas = self._descend(q, self.X, self.b)
        self.last_betas = betas

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

    def retrieve_multi(self, query: str | np.ndarray, k: int = 5) -> RetrievalResult:
        """Residual peeling under adaptive beta: descend, commit, project out, repeat.

        For multi-evidence questions (LoCoMo category 1 averages ~3.1 gold turns) a
        single converged state cannot represent several distinct targets; reading top-k
        off one attractor ranks by proximity to that attractor rather than by relevance
        to the remaining query content.
        """
        if self.X is None or self.b is None:
            raise RuntimeError("index() before retrieve_multi()")
        t0 = time.perf_counter()
        resid = l2_normalize(self._encode(query))
        found: list[int] = []
        scores: list[float] = []
        total_steps = 0

        for _ in range(k):
            xi, steps, _c, _b = self._descend(resid, self.X, self.b)
            total_steps += steps
            sims = self.X.T @ xi
            for j in np.argsort(-sims):
                if int(j) not in found:
                    found.append(int(j))
                    scores.append(float(sims[j]))
                    break
            else:
                break
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
