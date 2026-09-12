"""Shared harness for association-probe experiments (module P / rungs R3-R4).

Everything that is easy to get wrong and fatal when you do lives here, once:

  * THREAD-DISJOINT SPLITS. X17 is the standing demonstration that this is not a detail:
    the parametric predictor scored 0.784 on threads it had seen and 0.008 on threads it
    had not. Same model, same numbers, opposite conclusions -- only the split tells them
    apart. The world model is fitted on TRAIN-thread transitions only; hyperparameters are
    chosen on VAL; TEST is touched once.
  * CLUSTER BOOTSTRAP OVER THREADS. Probes inside one storyline are not independent
    observations. Resampling probes instead of threads is the pseudo-replication that
    once turned p=0.076 into p=0.0005.
  * SYMMETRIC EXCLUSION. The cue is excluded from the candidate set for EVERY method
    identically. An asymmetry here once produced an exact 0.000 that looked like a result.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from energymem.encoders import SentenceEncoder          # noqa: E402
from energymem.eval.association import build_probes     # noqa: E402
from energymem.eval.corpus import build                 # noqa: E402
from energymem.world.knn_wm import KNNWorldModel        # noqa: E402

SPLIT_SEED = 12345          # fixed for the life of the project; changing it is a protocol deviation


@dataclass
class Bench:
    episodes: list
    Z: np.ndarray                       # (N, d), L2-normalized, one row per episode
    train: set[str]
    val: set[str]
    test: set[str]

    def probes(self, split: str, hop: int):
        keep = {"train": self.train, "val": self.val, "test": self.test}[split]
        return [p for p in build_probes(self.episodes, hop=hop) if p.thread in keep]

    def transitions(self, split: str = "train") -> list[tuple[int, int]]:
        """1-step pairs the model is ALLOWED to see. Train threads only, by default."""
        return [(p.cue_idx, p.gold_idx) for p in self.probes(split, hop=1)]

    def world_model(self, k: int, tau: float, split: str = "train") -> KNNWorldModel:
        return KNNWorldModel(k=k, tau=tau).fit(self.Z, self.transitions(split))


def _corpus(seed: int, n_filler: int):
    eps, _ = build(seed, n_filler=n_filler)
    Z = SentenceEncoder().encode([e.text for e in eps])
    threads = sorted({e.thread for e in eps
                      if e.thread != "filler" and not e.thread.endswith("_nearmiss")})
    rng = np.random.default_rng(SPLIT_SEED)
    rng.shuffle(threads)
    return eps, Z, threads


def load(seed: int = 0, n_filler: int = 900) -> Bench:
    """Single 50/25/25 thread split. Kept for quick diagnostics."""
    eps, Z, threads = _corpus(seed, n_filler)
    n = len(threads)
    a, b = int(0.5 * n), int(0.75 * n)
    return Bench(eps, Z, set(threads[:a]), set(threads[a:b]), set(threads[b:]))


def folds(seed: int = 0, n_filler: int = 900, n_folds: int = 4) -> list[Bench]:
    """Thread-level K-fold: every thread is in TEST exactly once, and is never in the
    train or val of the fold that tests it.

    WHY. A single 50/25/25 split of 122 storylines leaves ~60 hop-1 and ~30 hop-2 test
    probes, which is not enough to distinguish any of the effects at stake here -- an
    interval on 30 probes spans most of the plausible range before the experiment starts.
    Rotating the folds puts every thread in the test set once, so the reported number is
    over the whole corpus while each individual prediction is still made by a model that
    never saw that storyline. Cluster-bootstrap over threads remains the correct interval
    because each thread still contributes exactly once.
    """
    eps, Z, threads = _corpus(seed, n_filler)
    parts = [set(threads[i::n_folds]) for i in range(n_folds)]
    out = []
    for f in range(n_folds):
        test = parts[f]
        val = parts[(f + 1) % n_folds]
        train = set().union(*[parts[j] for j in range(n_folds) if j not in (f, (f + 1) % n_folds)])
        out.append(Bench(eps, Z, train, val, test))
    return out


# ------------------------------------------------------------------ statistics

def cluster_ci(values, clusters, n: int = 20000, seed: int = 0):
    """Bootstrap the mean, resampling CLUSTERS (threads) with replacement.

    Returns (mean, lo, hi, p) where p is the one-sided mass at or below zero.
    """
    a = np.asarray(values, dtype=np.float64)
    c = np.asarray(clusters)
    groups = [a[c == u] for u in np.unique(c)]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(groups), size=(n, len(groups)))
    means = np.array([np.concatenate([groups[j] for j in row]).mean() for row in idx])
    return float(a.mean()), float(np.quantile(means, .025)), float(np.quantile(means, .975)), \
        float((means <= 0).mean())


def recall_at(ranked: list[int], gold: int, k: int) -> float:
    return float(gold in ranked[:k])
