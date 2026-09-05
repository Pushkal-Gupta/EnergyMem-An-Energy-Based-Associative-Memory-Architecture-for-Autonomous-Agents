"""Gradient-trained, per-cluster beta.

Motivation: memory/multibeta.py's per-pattern beta used a hand-derived formula
(crowding -> beta via a fixed equation). That is a guess about the right shape of the
landscape. This module replaces the guess with optimization: partition the memory bank
into regions (k-means clusters -- the practical version of "fixed-size regions" in a
512-dim space, where a literal grid is combinatorially impossible), give each region its
own beta, and TRAIN those beta values with gradient descent against a real objective,
the same way a neural network's weights are trained.

WHY THIS IS LEGITIMATE AND NOT JUST A BIGGER SEARCH. The retrieval process (softmax,
weighted-average, renormalize, repeat) is entirely differentiable in beta. So instead of
trying a handful of hand-picked values, we can compute the *gradient* of the retrieval
loss with respect to every cluster's beta and follow it downhill for hundreds of steps --
this explores the space far more thoroughly than any grid we could afford to hand-run.

TRAINING SIGNAL (self-supervised, no external labels needed). For each stored memory,
manufacture a corrupted copy of it (embedding-dimension dropout -- a domain-agnostic
proxy for "a degraded cue"), run the descent, and penalize the model if the descent does
not end up closest to the TRUE source memory. This is exactly a classification loss
(cross-entropy over "which stored memory is this"), fully differentiable end to end.

USES THE ALREADY-VERIFIED MECHANISM. The per-step update here is the max-centered
heterogeneous softmax from multibeta.py (``w_i = exp(beta_i * (s_i - max_j s_j))``),
which was proven there to reduce exactly to the ordinary update when beta is uniform.
This module only learns the beta VALUES; the retrieval math is identical, so a trained
result can be dropped straight into ``MultiBetaEnergyMemory.beta_vec`` and evaluated with
the same, already-checked retrieve() code path.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch


def kmeans_regions(X: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    """Partition N unit-norm patterns (X is d x N) into k regions via k-means.

    This is the high-dimensional stand-in for "divide the landscape into fixed-size
    regions": a literal grid needs (bins)^d cells, which is astronomical for d=512;
    k-means instead groups patterns that are actually near each other in the space the
    data lives on.
    """
    from sklearn.cluster import KMeans

    n = X.shape[1]
    k = min(k, n)
    km = KMeans(n_clusters=k, n_init=4, random_state=seed)
    return km.fit_predict(X.T)


def _dropout_corrupt(x: np.ndarray, rate: float, rng: np.random.Generator) -> np.ndarray:
    keep = rng.random(x.shape[0]) >= rate
    if not keep.any():
        keep[rng.integers(x.shape[0])] = True
    out = x * keep
    return out / (np.linalg.norm(out) + 1e-12)


@dataclass
class TrainHistory:
    train_loss: list = field(default_factory=list)
    val_loss: list = field(default_factory=list)
    val_acc: list = field(default_factory=list)
    beta_history: list = field(default_factory=list)  # per-cluster beta at checkpoints


def train_cluster_beta(
    X: np.ndarray,
    cluster_id: np.ndarray,
    n_steps: int = 8,
    epochs: int = 400,
    batch_size: int = 128,
    lr: float = 0.05,
    beta_init: float = 32.0,
    loss_scale: float = 10.0,
    val_frac: float = 0.2,
    corrupt_rate_range: tuple[float, float] = (0.2, 0.7),
    seed: int = 0,
) -> tuple[np.ndarray, TrainHistory]:
    """Learn one beta per cluster by gradient descent on a self-supervised recall loss.

    Returns (beta_vec, history): beta_vec is (N,) -- each memory's beta, taken from its
    cluster -- ready to assign directly to MultiBetaEnergyMemory.beta_vec.
    """
    rng = np.random.default_rng(seed)
    d, n = X.shape
    k = int(cluster_id.max()) + 1

    Xt = torch.tensor(X, dtype=torch.float64)
    cluster_t = torch.tensor(cluster_id, dtype=torch.long)
    log_beta = torch.full((k,), float(np.log(beta_init)), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([log_beta], lr=lr)

    idx_all = np.arange(n)
    rng.shuffle(idx_all)
    n_val = max(1, int(n * val_frac))
    val_idx, train_idx = idx_all[:n_val], idx_all[n_val:]

    hist = TrainHistory()

    def run_batch(idx_batch: np.ndarray, beta_per_mem: torch.Tensor):
        rates = rng.uniform(*corrupt_rate_range, size=len(idx_batch))
        Q = np.stack([_dropout_corrupt(X[:, i], r, rng) for i, r in zip(idx_batch, rates)], axis=1)
        xi = torch.tensor(Q, dtype=torch.float64)  # (d, B)
        for _ in range(n_steps):
            s = Xt.T @ xi                                    # (N, B)
            centered = beta_per_mem[:, None] * (s - s.max(dim=0, keepdim=True).values)
            w = torch.exp(centered)
            p = w / w.sum(dim=0, keepdim=True)
            nxt = Xt @ p                                      # (d, B)
            xi = nxt / nxt.norm(dim=0, keepdim=True)
        sims = Xt.T @ xi                                       # (N, B)
        logits = (sims * loss_scale).T                         # (B, N)
        target = torch.tensor(idx_batch, dtype=torch.long)
        loss = torch.nn.functional.cross_entropy(logits, target)
        acc = (logits.argmax(dim=1) == target).float().mean().item()
        return loss, acc

    for epoch in range(epochs):
        beta_per_mem = torch.exp(log_beta)[cluster_t]
        batch = rng.choice(train_idx, size=min(batch_size, len(train_idx)), replace=False)
        opt.zero_grad()
        loss, _ = run_batch(batch, beta_per_mem)
        loss.backward()
        opt.step()
        hist.train_loss.append(float(loss.item()))

        if epoch % 20 == 0 or epoch == epochs - 1:
            with torch.no_grad():
                beta_per_mem = torch.exp(log_beta)[cluster_t]
                vloss, vacc = run_batch(val_idx, beta_per_mem)
            hist.val_loss.append(float(vloss.item()))
            hist.val_acc.append(float(vacc))
            hist.beta_history.append(torch.exp(log_beta).detach().numpy().copy())

    beta_per_cluster = torch.exp(log_beta).detach().numpy()
    beta_vec = beta_per_cluster[cluster_id]
    return beta_vec, hist
