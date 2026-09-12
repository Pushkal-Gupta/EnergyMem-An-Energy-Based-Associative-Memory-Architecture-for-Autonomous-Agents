"""Fixture tests for module D and module P's mode interface.

Failure mode 10 in docs/12_ARCHITECTURE.md: readout bugs of this shape score exactly
0.000 instead of crashing, so they are only caught by comparing against numbers computed
by hand. Every expected value below is derived independently of the implementation.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from energymem.base import l2_normalize                     # noqa: E402
from energymem.readout import averaged_scores, mixture_scores, rank   # noqa: E402
from energymem.world.knn_wm import KNNWorldModel            # noqa: E402


def test_mixture_matches_hand_computation():
    """Two hypotheses, two memories, every number computed by hand."""
    xi1 = np.array([1.0, 0.0])
    xi2 = np.array([0.0, 1.0])
    Z = np.array([[1.0, 0.0], [0.0, 1.0]])
    w1, w2, beta = 0.75, 0.25, 2.0

    got = mixture_scores([(xi1, w1), (xi2, w2)], Z, beta)

    # memory 0: <xi1,z0>=1, <xi2,z0>=0
    exp0 = math.log(math.exp(2.0 * 1.0 + math.log(0.75)) + math.exp(2.0 * 0.0 + math.log(0.25)))
    exp1 = math.log(math.exp(2.0 * 0.0 + math.log(0.75)) + math.exp(2.0 * 1.0 + math.log(0.25)))
    assert abs(got[0] - exp0) < 1e-12
    assert abs(got[1] - exp1) < 1e-12


def test_single_hypothesis_ranks_exactly_like_cosine():
    """S=1 is the null setting: the ranking must be plain cosine from that state.

    This is the rung the reduction ladder needs -- if it ever fails, no mixture number
    downstream can be compared against the dense baseline.
    """
    rng = np.random.default_rng(0)
    Z = l2_normalize(rng.normal(size=(200, 32)), axis=1)
    xi = l2_normalize(rng.normal(size=32))
    for beta in (1.0, 16.0, 1e6):
        mix = np.argsort(-mixture_scores([(xi, 1.0)], Z, beta))
        cos = np.argsort(-(Z @ xi))
        assert np.array_equal(mix, cos), f"ranking diverged from cosine at beta={beta}"


def test_weights_are_renormalized_not_taken_raw():
    """Unnormalized weights shift every score by the same constant, so the RANKING
    is unchanged. A readout that multiplies by the weight instead would reorder."""
    rng = np.random.default_rng(1)
    Z = l2_normalize(rng.normal(size=(50, 16)), axis=1)
    h = [(l2_normalize(rng.normal(size=16)), 3.0), (l2_normalize(rng.normal(size=16)), 1.0)]
    a = mixture_scores(h, Z, 8.0)
    b = mixture_scores([(z, w / 4.0) for z, w in h], Z, 8.0)
    assert np.allclose(a - b, (a - b)[0])
    assert np.array_equal(np.argsort(-a), np.argsort(-b))


def test_mixture_separates_from_averaging_when_modes_disagree():
    """The whole point. Two opposed modes: their mean is orthogonal to both, so the
    averaging readout cannot rank either of them first."""
    d = 8
    z_a = np.zeros(d); z_a[0] = 1.0
    z_b = np.zeros(d); z_b[1] = 1.0
    decoy = l2_normalize(z_a + z_b)             # sits exactly between them
    Z = np.stack([z_a, z_b, decoy])

    hyps = [(z_a, 0.5), (z_b, 0.5)]
    mix_top = int(np.argmax(mixture_scores(hyps, Z, 8.0)))
    avg_top = int(np.argmax(averaged_scores(hyps, Z)))

    assert mix_top in (0, 1), "mixture must rank an actual mode first"
    assert avg_top == 2, "averaging must be seduced by the between-modes decoy"


def test_rank_excludes_before_truncating():
    s = np.array([5.0, 4.0, 3.0, 2.0])
    assert rank(s, k=2) == [0, 1]
    assert rank(s, k=2, exclude={0}) == [1, 2]


# ------------------------------------------------------------------ module P

def _toy_model():
    """Three remembered transitions, orthogonal displacements."""
    d = 6
    Z = np.zeros((6, d))
    Z[0, 0] = 1.0; Z[1, 1] = 1.0          # transition 0: e0 -> e1
    Z[2, 2] = 1.0; Z[3, 3] = 1.0          # transition 1: e2 -> e3
    Z[4, 4] = 1.0; Z[5, 5] = 1.0          # transition 2: e4 -> e5
    wm = KNNWorldModel(k=3, tau=5.0).fit(Z, [(0, 1), (2, 3), (4, 5)])
    return wm, Z


def test_modes_weights_sum_to_one_and_are_sorted():
    wm, Z = _toy_model()
    ms = wm.modes(Z[0], K=3)
    assert abs(sum(w for _, w in ms) - 1.0) < 1e-12
    assert [w for _, w in ms] == sorted([w for _, w in ms], reverse=True)


def test_modes_apply_displacement_to_the_query_not_to_the_neighbour():
    """The second silent bug. Mode 0 from cue e0 must be normalize(e0 + (e1-e0)) = e1,
    which is a PREDICTION -- not the neighbour's own cue e0."""
    wm, Z = _toy_model()
    top, _ = wm.modes(Z[0], K=1)[0]
    assert np.allclose(top, Z[1], atol=1e-9), "mode is not the predicted next state"
    assert not np.allclose(top, Z[0]), "mode collapsed back onto the cue"


def test_rollout_h0_is_the_bypass():
    """H=0 must return the cue with weight 1 -- the predictor switched off."""
    wm, Z = _toy_model()
    out = wm.rollout(Z[0], H=0, K=2, S=4)
    assert len(out) == 1
    assert out[0][1] == 1.0
    assert np.allclose(out[0][0], Z[0])


def test_rollout_prunes_to_S_and_renormalizes():
    wm, Z = _toy_model()
    out = wm.rollout(Z[0], H=2, K=3, S=4)
    assert len(out) <= 4
    assert abs(sum(w for _, w in out) - 1.0) < 1e-12
    assert all(abs(np.linalg.norm(z) - 1.0) < 1e-9 for z, _ in out)


def test_modes_do_not_reduce_to_predict_when_the_kernel_is_split():
    """If modes ever collapses to the mean, this is the test that notices."""
    wm, Z = _toy_model()
    q = l2_normalize(Z[0] + Z[2])            # equidistant from two remembered cues
    ms = wm.modes(q, K=2)
    mean = wm.predict(q)
    assert all(not np.allclose(z, mean, atol=1e-6) for z, _ in ms)
