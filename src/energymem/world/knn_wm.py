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

    def predict(self, q: np.ndarray) -> np.ndarray:
        sims = self.Zc @ q
        k = min(self.k, sims.size)
        nn = np.argpartition(-sims, k - 1)[:k]
        w = np.exp(self.tau * (sims[nn] - sims[nn].max()))
        w /= w.sum()
        return l2_normalize(q + (w[:, None] * self.D[nn]).sum(0))

    def rollout(self, q: np.ndarray, steps: int) -> list[np.ndarray]:
        out, cur = [], l2_normalize(q)
        for _ in range(steps):
            cur = self.predict(cur)
            out.append(cur)
        return out
