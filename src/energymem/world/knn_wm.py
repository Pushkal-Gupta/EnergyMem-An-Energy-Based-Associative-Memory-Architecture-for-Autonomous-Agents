"""A nonparametric world model: transfer displacements from remembered transitions.

WHY NOT THE PARAMETRIC JEPA. X17 trained a JEPA predictor at five capacities from rank-2
to rank-32. Every one of them memorised: 0.78 Recall@1 on storylines it had seen, 0.008 on
unseen ones, against a cosine baseline of 0.106. With ~107 training transitions in 768
dimensions there is no capacity at which a free-form map generalises, and the diagnostic
says why -- displacement vectors between consecutive episodes are near-orthogonal
(mean pairwise cosine 0.006), so there is no global "next step" direction to fit.

WHAT WORKS INSTEAD. Do not learn a global map; carry the transitions themselves and
transfer them locally. Given a cue, find the most similar cues among remembered
transitions and apply their displacements:

    P(q) = normalize( q + sum_i w_i * (z_next_i - z_cue_i) ),   w = softmax(tau * sim)

This is still a world model -- it predicts the representation of what comes next -- and it
is still JEPA-flavoured in that the prediction lives in embedding space. It is simply
nonparametric, which is the right inductive bias when transitions are numerous relative to
nothing and scarce relative to the dimension.

It also composes: iterate it and you roll the model forward, which is what recovers
multi-hop chains.
"""
from __future__ import annotations

import numpy as np

from ..base import l2_normalize
from .base import Hypothesis, renormalize


class KNNWorldModel:
    """Predicts the next episode representation by locally-weighted displacement transfer."""

    def __init__(self, k: int = 10, tau: float = 5.0):
        self.k = k
        self.tau = tau
        self.Zc: np.ndarray | None = None   # (M, d) cue representations
        self.D: np.ndarray | None = None    # (M, d) displacements to the next episode

    def fit(self, Z: np.ndarray, transitions: list[tuple[int, int]]) -> "KNNWorldModel":
        """transitions: (t, t+1) index pairs OBSERVED IN THE AGENT'S OWN STREAM.

        These are episodes the agent actually experienced in sequence. Nothing here is
        supervision in the usual sense -- temporal adjacency is free.
        """
        if not transitions:
            raise ValueError("a world model needs observed transitions")
        self.Zc = np.stack([Z[a] for a, _ in transitions])
        self.D = np.stack([Z[b] - Z[a] for a, b in transitions])
        return self

    # ------------------------------------------------------------- the kernel

    def _neighbours(self, q: np.ndarray, k: int | None = None):
        """The k nearest remembered cues and their softmax weights, sorted by weight."""
        if self.Zc is None or self.D is None:
            raise RuntimeError("fit() before predicting")
        sims = self.Zc @ q
        k = min(k or self.k, sims.size)
        nn = np.argpartition(-sims, k - 1)[:k]
        nn = nn[np.argsort(-sims[nn])]                  # descending, so mode i is the i-th heaviest
        w = np.exp(self.tau * (sims[nn] - sims[nn].max()))
        w /= w.sum()
        return nn, w

    # ------------------------------------------------------ modes (primary, I4)

    def modes(self, q: np.ndarray, K: int | None = None,
              alpha: float = 1.0) -> list[Hypothesis]:
        """Top-K modes of P(next | q) -- the kernel WITHOUT collapsing it.

        Each remembered transition contributes one mode: the displacement it took, applied
        to **the query**, not to its own cue. Applying it to its own cue would make every
        candidate a literal stored training memory, which scores zero and does so silently
        -- a bug we have written twice.

        ``alpha`` IS NOT OPTIONAL IN PRACTICE, and the reason is measured (X23b). A raw
        displacement has norm ~1.17 against a unit-norm cue, so ``q + Delta_i`` is
        *dominated by the displacement* and lands essentially on ``z_next(i)`` -- a stored
        training episode, 100% of the time on our corpus. Averaging the kernel is not only
        a collapse to a point; it also SHRINKS the displacement (to ~0.35 of its mean norm
        here), which is what keeps the query in control of where the prediction lands.

        Comparing modes at ``alpha=1`` against the average therefore confounds two
        differences -- distributional vs point, and step size. ``alpha`` exists so the
        comparison can be made at matched effective displacement, which is the only form
        in which it answers anything.
        """
        nn, w = self._neighbours(q)
        K = min(K if K is not None else self.k, len(nn))
        nn, w = nn[:K], w[:K]
        return renormalize([(l2_normalize(q + alpha * self.D[j]), float(wi))
                            for j, wi in zip(nn, w)])

    def mean_step_norm(self, q: np.ndarray) -> tuple[float, float]:
        """(mean per-mode |Delta|, |sum w_i Delta_i|) -- the shrinkage the average applies.

        Use it to set ``alpha`` so a mode's step matches the average's, rather than
        guessing. Diagnostic only; never in the retrieval path.
        """
        nn, w = self._neighbours(q)
        return (float(np.linalg.norm(self.D[nn], axis=1).mean()),
                float(np.linalg.norm((w[:, None] * self.D[nn]).sum(0))))

    def rollout(self, q: np.ndarray, H: int, K: int = 2, S: int = 8,
                alpha: float = 1.0) -> list[Hypothesis]:
        """Iterate ``modes`` H times, pruning to the S heaviest hypotheses each hop.

        H=0 returns [(q, 1.0)] -- the predictor bypassed, which is the null setting the
        reduction ladder needs.

        The model is only ever fitted on 1-step transitions, so H>1 is *composition*: the
        dynamics applied to their own output. That composition is what reaches memories
        similarity search structurally cannot (X19).
        """
        beam: list[Hypothesis] = [(l2_normalize(q), 1.0)]
        for _ in range(H):
            nxt: list[Hypothesis] = []
            for z, w in beam:
                for zi, wi in self.modes(z, K, alpha=alpha):
                    nxt.append((zi, w * wi))
            nxt.sort(key=lambda t: -t[1])
            beam = renormalize(nxt[:S])
        return beam

    # ---------------------------------------------- the averaging baseline (I4)

    def predict(self, q: np.ndarray) -> np.ndarray:
        """The MEAN of the kernel -- a single point.

        Kept deliberately: it is the baseline that ``modes`` is measured against, and the
        mechanism whose failure motivates the whole distributional treatment. Do not use
        it in the retrieval path.
        """
        nn, w = self._neighbours(q)
        return l2_normalize(q + (w[:, None] * self.D[nn]).sum(0))

    def rollout_mean(self, q: np.ndarray, steps: int) -> list[np.ndarray]:
        """Averaging rollout -- the baseline form of ``rollout``. See ``predict``."""
        out, cur = [], l2_normalize(q)
        for _ in range(steps):
            cur = self.predict(cur)
            out.append(cur)
        return out
