"""Sanity tests. These gate everything else.

Test 1 (the reduction) is the most important test in the repository. It asserts
Proposition 1: dense top-k retrieval is the T=1, beta->inf, b=0, R=I corner of the
EnergyMem parameter space. If it fails, every number in the paper is meaningless and
the central framing (docs/01_RESEARCH_PLAN.md section 2) is wrong.

Run: pytest -q
"""

from __future__ import annotations

import numpy as np
import pytest

from energymem.base import Memory, l2_normalize
from energymem.baselines.dense import DenseTopK
from energymem.memory.energy import EnergyMemory, dense_topk_config


@pytest.fixture
def corpus():
    rng = np.random.default_rng(0)
    d, n = 64, 500
    embs = l2_normalize(rng.normal(size=(n, d)), axis=1)
    return [
        Memory(id=f"m{i}", text=f"memory {i}", embedding=embs[i]) for i in range(n)
    ], embs


def test_reduction_to_cosine_topk(corpus):
    """PROPOSITION 1. T=1, beta->inf, b=0, R=I must reproduce top-1 cosine exactly."""
    mems, embs = corpus
    rng = np.random.default_rng(1)
    queries = l2_normalize(rng.normal(size=(1000, embs.shape[1])), axis=1)

    em = dense_topk_config()
    em.index(mems)
    dense = DenseTopK()
    dense.index(mems)

    mismatches = [
        i
        for i, q in enumerate(queries)
        if em.retrieve(q, k=1).ids[0] != dense.retrieve(q, k=1).ids[0]
    ]
    assert not mismatches, (
        f"{len(mismatches)}/1000 queries disagree. Proposition 1 is violated -- "
        "stop and fix this before running any experiment."
    )


def test_energy_decreases_monotonically(corpus):
    """CCCP guarantees descent. Violation means the update rule is wrong."""
    mems, embs = corpus
    em = EnergyMemory(n_steps=16, beta=8.0, tol=0.0)  # tol=0 forces all steps
    em.index(mems)
    rng = np.random.default_rng(2)

    for _ in range(50):
        q = l2_normalize(rng.normal(size=embs.shape[1]))
        res = em.retrieve(q, k=1)
        energies = [
            em.energy(xi, em.X, em.b, em.beta) for xi in res.trajectory
        ]
        # allow a small numerical tolerance on the normalization step
        diffs = np.diff(energies)
        assert (diffs <= 1e-8).all(), f"energy increased: {diffs[diffs > 0]}"


def test_orthogonal_capacity():
    """Perfect recall for well-separated patterns. Finds where capacity breaks.

    Real sentence embeddings are NOT well-separated (risk register R-5) -- this test
    establishes the ceiling that the real-data numbers should be compared against.
    """
    d = 128
    n = 64
    eye = np.eye(d)[:n]
    mems = [Memory(id=f"m{i}", text="", embedding=eye[i]) for i in range(n)]

    em = EnergyMemory(n_steps=8, beta=32.0)
    em.index(mems)
    for i in range(n):
        assert em.retrieve(eye[i], k=1).ids[0] == f"m{i}"


def test_determinism(corpus):
    """Same seed => bitwise identical. Required for reproducibility claims."""
    mems, embs = corpus
    q = l2_normalize(np.random.default_rng(3).normal(size=embs.shape[1]))
    a = EnergyMemory(n_steps=8, beta=16.0)
    a.index(mems)
    b = EnergyMemory(n_steps=8, beta=16.0)
    b.index(mems)
    ra, rb = a.retrieve(q, k=5), b.retrieve(q, k=5)
    assert ra.ids == rb.ids
    assert np.allclose(ra.scores, rb.scores)


def test_multistep_differs_from_onestep(corpus):
    """Sanity check on G1: multi-step descent must actually do something different.

    If this passes trivially (identical results) then n_steps is not wired up and
    every G1 result would be a null by construction.
    """
    mems, embs = corpus
    rng = np.random.default_rng(4)
    one = EnergyMemory(n_steps=1, beta=4.0)
    many = EnergyMemory(n_steps=16, beta=4.0, tol=0.0)
    one.index(mems)
    many.index(mems)

    differ = sum(
        one.retrieve(q, k=1).ids[0] != many.retrieve(q, k=1).ids[0]
        for q in l2_normalize(rng.normal(size=(200, embs.shape[1])), axis=1)
    )
    assert differ > 0, "multi-step descent changed nothing -- check the update loop"
