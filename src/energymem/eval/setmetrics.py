"""Multi-target retrieval metrics and cluster-aware tests.

Everything that existed before this file assumed ONE gold item per query:
``eval/metrics.py:recall_at_k`` takes a ``gold`` sequence but scores a hit if *any*
gold appears, and ``experiments/assoc_harness.py:recall_at`` takes a single ``gold:
int``. LoCoMo category 1 averages 3.13 gold turns per question, so single-gold metrics
cannot see the thing we are trying to measure: whether a method retrieves the *whole*
evidence set.

Two metrics, deliberately both reported:

  full_set_recall_at_k   1.0 only if EVERY gold id is in the top k. This is what a
                         reader needs -- partial evidence answers the question wrong.
  mean_gold_recall_at_k  fraction of gold ids in the top k. Degrades gracefully, so it
                         still has resolution when full-set recall is near zero.

With |gold| == 1 the two coincide and both equal plain Recall@k, which is asserted in
the test suite -- that is the null setting that lets category 4 act as a control.

STATISTICS. LoCoMo has 10 conversations and 1977 questions, and questions inside a
conversation are dependent. The primary test is therefore a paired permutation that
flips signs only WITHIN a conversation (``blocked_permutation``); the interval is a
two-level bootstrap that resamples conversations and then questions inside them
(``hierarchical_bootstrap``). Ten clusters is few, and a one-level bootstrap over
questions would understate the interval by a lot.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def full_set_recall_at_k(ranked: Sequence, gold: Sequence, k: int) -> float:
    """1.0 iff every gold id appears in the first k of ``ranked``."""
    if not gold:
        raise ValueError("empty gold set")
    return float(set(gold).issubset(set(ranked[:k])))


def mean_gold_recall_at_k(ranked: Sequence, gold: Sequence, k: int) -> float:
    """Fraction of the gold set appearing in the first k of ``ranked``."""
    if not gold:
        raise ValueError("empty gold set")
    g = set(gold)
    return len(g & set(ranked[:k])) / len(g)


def reciprocal_rank(ranked: Sequence, gold: Sequence) -> float:
    """1/rank of the FIRST gold id, 0.0 if none present."""
    g = set(gold)
    for r, item in enumerate(ranked, 1):
        if item in g:
            return 1.0 / r
    return 0.0


def ndcg_at_k(ranked: Sequence, gold: Sequence, k: int) -> float:
    """Binary-relevance nDCG@k against a gold SET (ideal = all gold at the top)."""
    g = set(gold)
    dcg = sum(1.0 / np.log2(r + 1) for r, it in enumerate(ranked[:k], 1) if it in g)
    ideal = sum(1.0 / np.log2(r + 1) for r in range(1, min(len(g), k) + 1))
    return float(dcg / ideal) if ideal > 0 else 0.0


# ----------------------------------------------------------------- inference

def blocked_permutation(diffs, blocks, n: int = 100_000, seed: int = 0) -> float:
    """Two-sided p for mean(diffs) == 0, flipping signs only within a block.

    ``diffs`` are per-question paired differences (method A minus method B) and
    ``blocks`` the conversation each question belongs to. Permuting within blocks
    keeps the dependence structure that makes a naive test anticonservative.
    """
    d = np.asarray(diffs, dtype=np.float64)
    b = np.asarray(blocks)
    if d.size == 0:
        return float("nan")
    obs = abs(d.mean())
    rng = np.random.default_rng(seed)
    idx = [np.flatnonzero(b == u) for u in np.unique(b)]
    hits = 0
    for _ in range(n):
        s = np.empty_like(d)
        for g in idx:
            s[g] = rng.choice((-1.0, 1.0)) * d[g]   # one sign per block
        hits += abs(s.mean()) >= obs - 1e-15
    return (hits + 1) / (n + 1)


def hierarchical_bootstrap(values, clusters, n: int = 20_000, seed: int = 0):
    """(mean, lo, hi) 95% CI resampling clusters, then items within each drawn cluster."""
    v = np.asarray(values, dtype=np.float64)
    c = np.asarray(clusters)
    uniq = np.unique(c)
    idx = {u: np.flatnonzero(c == u) for u in uniq}
    rng = np.random.default_rng(seed)
    means = np.empty(n)
    for i in range(n):
        take = []
        for u in rng.choice(uniq, size=len(uniq), replace=True):
            g = idx[u]
            take.append(rng.choice(g, size=len(g), replace=True))
        means[i] = v[np.concatenate(take)].mean()
    return float(v.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_report(a, b, clusters, n_boot: int = 20_000, n_perm: int = 100_000, seed: int = 0):
    """Everything needed for one row: both means, the paired delta, its CI, and p."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    d = a - b
    mean, lo, hi = hierarchical_bootstrap(d, clusters, n=n_boot, seed=seed)
    return dict(a=float(a.mean()), b=float(b.mean()), delta=mean, lo=lo, hi=hi,
                p=blocked_permutation(d, clusters, n=n_perm, seed=seed), n=int(a.size))


def exact_blocked_permutation(diffs, blocks) -> float:
    """EXACT two-sided p for the same null, by enumerating all sign patterns.

    ``blocked_permutation`` draws one sign per block at random, so with ``B`` blocks the
    null has exactly ``2**B`` points. LoCoMo has ten conversations -- 1024 patterns --
    which is cheaper to enumerate than to sample, and the enumeration is the ground truth
    the Monte Carlo version is approximating. Refuses above 22 blocks.

    The floor is the honest one: with B blocks no result can be significant below
    ``2 / 2**B`` two-sided, because the all-flipped pattern always ties the observed
    statistic. At B=10 that is 0.00195 -- so a p of 0.002 on this corpus is the smallest
    attainable value, not a measurement of how strong the effect is.
    """
    d = np.asarray(diffs, dtype=np.float64)
    b = np.asarray(blocks)
    if d.size == 0:
        return float("nan")
    uniq = np.unique(b)
    B = len(uniq)
    if B > 22:
        raise ValueError(f"{B} blocks is 2**{B} patterns; use blocked_permutation instead")
    # (B, n) one-hot block membership, so a sign vector becomes a matrix product.
    M = np.stack([(b == u).astype(np.float64) * d for u in uniq])       # (B, n)
    sums = M.sum(axis=1)                                               # per-block sums
    signs = ((np.arange(2 ** B)[:, None] >> np.arange(B)) & 1) * 2.0 - 1.0   # (2**B, B)
    stats = np.abs(signs @ sums) / d.size
    return float((stats >= abs(d.mean()) - 1e-15).mean())
