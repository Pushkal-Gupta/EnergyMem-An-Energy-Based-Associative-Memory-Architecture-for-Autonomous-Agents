"""Fixtures for the multi-target metrics. Hand-computed; no metric trusts itself."""
import numpy as np
import pytest

from energymem.eval.setmetrics import (
    blocked_permutation, full_set_recall_at_k, hierarchical_bootstrap,
    mean_gold_recall_at_k, ndcg_at_k, reciprocal_rank,
)

RANKED = ["a", "b", "c", "d", "e"]


def test_full_set_needs_all_gold():
    assert full_set_recall_at_k(RANKED, ["a", "c"], 3) == 1.0
    assert full_set_recall_at_k(RANKED, ["a", "d"], 3) == 0.0      # d is at rank 4
    assert full_set_recall_at_k(RANKED, ["a", "d"], 4) == 1.0


def test_mean_gold_is_a_fraction():
    assert mean_gold_recall_at_k(RANKED, ["a", "d"], 3) == pytest.approx(0.5)
    assert mean_gold_recall_at_k(RANKED, ["a", "b", "c"], 3) == pytest.approx(1.0)
    assert mean_gold_recall_at_k(RANKED, ["x", "y"], 5) == 0.0


def test_single_gold_collapses_to_recall_at_k():
    """THE NULL SETTING: with one gold the two metrics coincide. This is what lets
    category 4 (mean 1.06 gold turns) serve as a control for category 1."""
    for k in (1, 2, 3, 5):
        for g in ("a", "c", "e", "zz"):
            assert full_set_recall_at_k(RANKED, [g], k) == mean_gold_recall_at_k(RANKED, [g], k)


def test_reciprocal_rank_uses_first_gold():
    assert reciprocal_rank(RANKED, ["c", "e"]) == pytest.approx(1 / 3)
    assert reciprocal_rank(RANKED, ["zz"]) == 0.0


def test_ndcg_perfect_and_zero():
    assert ndcg_at_k(RANKED, ["a", "b"], 5) == pytest.approx(1.0)
    assert ndcg_at_k(RANKED, ["zz"], 5) == 0.0
    mid = ndcg_at_k(RANKED, ["a", "e"], 5)
    assert 0.0 < mid < 1.0


def test_empty_gold_raises_not_silently_zero():
    with pytest.raises(ValueError):
        full_set_recall_at_k(RANKED, [], 3)
    with pytest.raises(ValueError):
        mean_gold_recall_at_k(RANKED, [], 3)


def test_blocked_permutation_detects_and_ignores():
    blocks = np.repeat(np.arange(10), 20)
    assert blocked_permutation(np.full(200, 0.5), blocks, n=2000) < 0.01   # all positive
    rng = np.random.default_rng(0)
    assert blocked_permutation(rng.normal(0, 1, 200), blocks, n=2000) > 0.05


def test_blocked_permutation_is_not_fooled_by_one_cluster():
    """A real effect inside ONE conversation is not evidence across conversations.
    A question-level test would call this p<0.001; the blocked test must not."""
    blocks = np.repeat(np.arange(10), 20)
    d = np.zeros(200)
    d[:20] = 1.0
    assert blocked_permutation(d, blocks, n=2000) > 0.1


def test_hierarchical_bootstrap_brackets_the_mean():
    blocks = np.repeat(np.arange(10), 20)
    rng = np.random.default_rng(1)
    v = rng.normal(0.3, 1.0, 200)
    m, lo, hi = hierarchical_bootstrap(v, blocks, n=2000)
    assert lo < m < hi
    assert m == pytest.approx(v.mean())


def test_hierarchical_interval_is_wider_than_naive():
    """Clustered data: the two-level interval must be wider than an iid one, or the
    clustering is not being respected."""
    blocks = np.repeat(np.arange(10), 20)
    rng = np.random.default_rng(2)
    v = np.repeat(rng.normal(0, 1, 10), 20) + rng.normal(0, 0.05, 200)  # heavy cluster effect
    _, lo, hi = hierarchical_bootstrap(v, blocks, n=2000)
    naive = np.percentile([rng.choice(v, 200).mean() for _ in range(2000)], [2.5, 97.5])
    assert (hi - lo) > (naive[1] - naive[0])
