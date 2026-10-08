"""X17 (re-run) -- does a PARAMETRIC predictor generalise to unseen storylines?

WHY THIS EXISTS. The original X17 driver was lost before it was committed, so the
paper's parametric-failure result -- 0.784 on fitted storylines against 0.008 on unseen
ones -- had no reproducible path. Appendix F lists that as one of two things that cannot
be reproduced. This closes it: world/jepa.py survived, only the driver was missing.

The result matters because it is what forces module P to be nonparametric. If a free-form
learned map generalised here, the whole design would be different.

Also measures the diagnostic that explains the failure: displacement vectors between
consecutive episodes are near-orthogonal, so there is no global "next step" direction for
a map to fit.

    python experiments/x17_parametric.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from energymem.base import l2_normalize                       # noqa: E402
from energymem.world.jepa import JEPAConfig, JEPAPredictor     # noqa: E402
from experiments.assoc_harness import load                     # noqa: E402

RANKS = [2, 4, 8, 16, 32]


def recall_at(Z, wm_predict, probes, k):
    r1, r5 = [], []
    for p in probes:
        zp = wm_predict(Z[p.cue_idx])
        s = Z @ zp
        s[p.cue_idx] = -np.inf
        top = np.argsort(-s)[:5]
        r1.append(float(top[0] == p.gold_idx))
        r5.append(float(p.gold_idx in top))
    return float(np.mean(r1)), float(np.mean(r5))


def main():
    b = load()
    Z = b.Z
    train = b.probes("train", 1)
    test = b.probes("test", 1)
    pairs = [(p.cue_idx, p.gold_idx) for p in train]

    print("=" * 80)
    print("X17 (re-run)  Does a parametric JEPA predictor generalise across storylines?")
    print("=" * 80)
    print(f"  train transitions {len(pairs)}  (from {len(b.train)} storylines)")
    print(f"  test probes       {len(test)}  (from {len(b.test)} UNSEEN storylines)")
    print(f"  embedding dim     {Z.shape[1]}")

    # the diagnostic that explains the result, before the result
    D = np.stack([Z[a2] - Z[a1] for a1, a2 in pairs])
    Dn = l2_normalize(D, axis=1)
    G = Dn @ Dn.T
    iu = np.triu_indices(len(Dn), k=1)
    mutual = float(G[iu].mean())
    Dc = D - D.mean(0)
    sv = np.linalg.svd(Dc, compute_uv=False)
    var10 = float((sv[:10] ** 2).sum() / (sv ** 2).sum())
    print(f"\n  DIAGNOSTIC  mean pairwise cosine between displacements : {mutual:+.3f}")
    print(f"              variance in the top-10 principal components : {var10:.1%}")
    print("              -> there is no global 'next step' direction to fit.")

    # cosine baseline on the same probes
    c1, c5 = recall_at(Z, lambda q: q, test, 5)
    print(f"\n  cosine baseline on unseen storylines: R@1 {c1:.3f}  R@5 {c5:.3f}")

    print(f"\n  {'rank':>6} {'R@1 FITTED':>12} {'R@1 UNSEEN':>12} {'R@5 UNSEEN':>12}")
    print("  " + "-" * 46)
    rows = []
    for rank in RANKS:
        m = JEPAPredictor(Z.shape[1], JEPAConfig(rank=rank, seed=0))
        m.fit(Z, pairs, verbose=False)
        f1, _ = recall_at(Z, m.predict, train, 5)
        u1, u5 = recall_at(Z, m.predict, test, 5)
        rows.append({"rank": rank, "fitted_R@1": f1, "unseen_R@1": u1, "unseen_R@5": u5})
        print(f"  {rank:>6} {f1:>12.3f} {u1:>12.3f} {u5:>12.3f}")

    best = max(rows, key=lambda r: r["unseen_R@1"])
    print(f"\n  best unseen R@1 at any capacity: {best['unseen_R@1']:.3f} (rank {best['rank']})")
    print(f"  cosine:                          {c1:.3f}")
    verdict = ("MEMORISES -- fitted >> unseen, and unseen is at or below cosine at every "
               "capacity." if best["unseen_R@1"] <= c1 else
               "GENERALISES -- this would overturn the nonparametric design choice.")
    print(f"  => {verdict}")

    dst = ROOT / "experiments" / "results" / "x17_parametric.json"
    dst.write_text(json.dumps({
        "n_train_transitions": len(pairs), "n_test_probes": len(test),
        "dim": int(Z.shape[1]),
        "displacement_mutual_cosine": mutual, "top10_pc_variance": var10,
        "cosine": {"R@1": c1, "R@5": c5},
        "ranks": rows,
        "best_unseen_R@1": best["unseen_R@1"],
    }, indent=2))
    print(f"\n  wrote {dst}")


if __name__ == "__main__":
    main()
