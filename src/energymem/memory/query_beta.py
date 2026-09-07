"""Query-conditional beta: beta as a learned continuous function of the descent state.

MOTIVATED BY MEASUREMENT, not intuition. scripts/sweep_beta_transition.py shows the
optimal beta depends on how degraded the cue is:

    occlusion 30% -> best beta 48     occlusion 50% -> best beta 36
    occlusion 70% -> best beta 128 (i.e. give up and be cosine)

A single global beta has to compromise across all of them. But "how degraded is this
cue" is a property the system can *estimate at query time* from the shape of the
similarity distribution -- a clean cue has one clear winner, a degraded cue has a flat,
ambiguous spread. So beta should be a FUNCTION of that shape, not a constant.

WHAT IS LEARNED. A small function f with parameters theta:

    beta_t = beta_floor + softplus( w . phi(xi_t) + b )

evaluated at EVERY descent step from the current state xi_t, where phi is five cheap
statistics of the current similarity vector s = X^T xi_t:

    max(s), (top1 - top2) gap, mean(s), std(s), entropy of softmax(s)

This makes beta a continuous function over the landscape, sampled along the trajectory --
not a per-memory constant (which memory/multibeta.py showed is inert once beta is past
its leverage zone) and not a fixed schedule (which memory/adaptive.py used). Because a
single scalar beta_t applies to all candidates at each step, the competition between
memories stays fair -- the flaw that sank the per-memory version cannot recur here.

WHY IT SUBSUMES THE BASELINES. With w = 0 the model is a fixed global beta, so any
tuned scalar beta is inside this family; with a large learned bias it degenerates to
cosine (Proposition 1). Training can therefore only improve on those if there is signal
to find, and failing to beat them is meaningful rather than an artifact of a search
space that excluded them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

N_FEATURES = 5
BETA_FLOOR = 4.0


def state_features_torch(s: torch.Tensor) -> torch.Tensor:
    """phi(xi) from the similarity vector s = X^T xi. s is (N, B) -> returns (B, 5).

    These describe how *resolvable* the current state is: a clean cue produces a peaked
    distribution with a large top1-top2 gap and low entropy; a degraded cue produces a
    flat one. Cheap: no extra matmuls beyond the s already computed by the descent.
    """
    top2 = torch.topk(s, k=min(2, s.shape[0]), dim=0).values
    gap = top2[0] - top2[1] if top2.shape[0] > 1 else top2[0]
    p = torch.softmax(s, dim=0)
    ent = -(p * torch.log(p + 1e-12)).sum(dim=0)
    return torch.stack([s.max(dim=0).values, gap, s.mean(dim=0), s.std(dim=0), ent], dim=1)


@dataclass
class QBetaHistory:
    train_loss: list = field(default_factory=list)
    val_loss: list = field(default_factory=list)
    val_acc: list = field(default_factory=list)
    epochs: list = field(default_factory=list)


class QueryConditionalBeta:
    """Learned f: state -> beta. Holds the parameters and the (differentiable) descent."""

    def __init__(self, n_steps: int = 8, beta_init: float = 32.0, seed: int = 0):
        self.n_steps = n_steps
        g = torch.Generator().manual_seed(seed)
        # start near a constant beta: tiny weights, bias set so softplus(bias) ~ beta_init
        self.w = (0.01 * torch.randn(N_FEATURES, generator=g, dtype=torch.float64)
                  ).requires_grad_(True)
        inv_softplus = float(np.log(np.expm1(max(beta_init - BETA_FLOOR, 1e-3))))
        self.b = torch.tensor(inv_softplus, dtype=torch.float64, requires_grad=True)
        self.feat_mean = torch.zeros(N_FEATURES, dtype=torch.float64)
        self.feat_std = torch.ones(N_FEATURES, dtype=torch.float64)

    def parameters(self):
        return [self.w, self.b]

    def beta_from_state(self, s: torch.Tensor) -> torch.Tensor:
        phi = (state_features_torch(s) - self.feat_mean) / (self.feat_std + 1e-8)
        return BETA_FLOOR + torch.nn.functional.softplus(phi @ self.w + self.b)

    def descend(self, X: torch.Tensor, Q: torch.Tensor, record: bool = False):
        """X: (d, N) memories. Q: (d, B) queries. Returns final sims (N, B) [, betas]."""
        xi = Q
        betas = []
        for _ in range(self.n_steps):
            s = X.T @ xi                                    # (N, B)
            beta = self.beta_from_state(s)                  # (B,)
            if record:
                betas.append(beta.detach().numpy().copy())
            w = torch.softmax(beta[None, :] * s, dim=0)      # (N, B)
            nxt = X @ w
            xi = nxt / (nxt.norm(dim=0, keepdim=True) + 1e-12)
        sims = X.T @ xi
        return (sims, betas) if record else sims

    def calibrate_features(self, X: torch.Tensor, Q: torch.Tensor) -> None:
        """Standardize phi using the initial-state statistics of a sample of queries."""
        with torch.no_grad():
            phi = state_features_torch(X.T @ Q)
            self.feat_mean = phi.mean(dim=0)
            self.feat_std = phi.std(dim=0)
            self.feat_std[self.feat_std < 1e-8] = 1.0

    # ------------------------------------------------------------------ inference

    def retrieve_batch(self, X_np: np.ndarray, Q_np: np.ndarray, k: int = 1):
        """Returns (topk_indices (B,k), betas_per_step list)."""
        X = torch.tensor(X_np, dtype=torch.float64)
        Q = torch.tensor(Q_np, dtype=torch.float64)
        Q = Q / (Q.norm(dim=0, keepdim=True) + 1e-12)
        with torch.no_grad():
            sims, betas = self.descend(X, Q, record=True)
        idx = torch.topk(sims, k=k, dim=0).indices.T.numpy()
        return idx, betas


def train_query_beta(
    X_np: np.ndarray,
    train_Q: np.ndarray,
    train_y: np.ndarray,
    val_Q: np.ndarray,
    val_y: np.ndarray,
    n_steps: int = 8,
    epochs: int = 300,
    batch_size: int = 96,
    lr: float = 0.03,
    beta_init: float = 32.0,
    loss_scale: float = 20.0,
    seed: int = 0,
) -> tuple[QueryConditionalBeta, QBetaHistory]:
    """Train f on REAL corrupted queries (train_Q, columns are queries; train_y are the
    indices of the memory each was corrupted from)."""
    model = QueryConditionalBeta(n_steps=n_steps, beta_init=beta_init, seed=seed)
    X = torch.tensor(X_np, dtype=torch.float64)
    trQ = torch.tensor(train_Q, dtype=torch.float64)
    vaQ = torch.tensor(val_Q, dtype=torch.float64)
    trY = torch.tensor(train_y, dtype=torch.long)
    vaY = torch.tensor(val_y, dtype=torch.long)
    model.calibrate_features(X, trQ)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    hist = QBetaHistory()

    def evaluate(Q, Y):
        with torch.no_grad():
            sims = model.descend(X, Q)
            logits = (sims * loss_scale).T
            loss = torch.nn.functional.cross_entropy(logits, Y)
            acc = (logits.argmax(dim=1) == Y).double().mean().item()
        return float(loss.item()), float(acc)

    for epoch in range(epochs):
        sel = rng.choice(trQ.shape[1], size=min(batch_size, trQ.shape[1]), replace=False)
        opt.zero_grad()
        sims = model.descend(X, trQ[:, sel])
        loss = torch.nn.functional.cross_entropy((sims * loss_scale).T, trY[sel])
        loss.backward()
        opt.step()
        hist.train_loss.append(float(loss.item()))
        if epoch % 20 == 0 or epoch == epochs - 1:
            vl, va = evaluate(vaQ, vaY)
            hist.val_loss.append(vl)
            hist.val_acc.append(va)
            hist.epochs.append(epoch)

    return model, hist
