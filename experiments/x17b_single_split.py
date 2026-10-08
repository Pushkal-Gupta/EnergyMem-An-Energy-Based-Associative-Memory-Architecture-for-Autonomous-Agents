"""X17b (re-run) -- the one-hop result under the ORIGINAL single 50/50 storyline split.

WHY THIS EXISTS. Table 3 row (i) came from a driver that was lost before it was
committed, so the paper carried a row no one could regenerate. The claim itself is
corroborated by rows (ii) and (iii), so the honest options were to drop the row or to
reproduce it. This reproduces it, under the protocol the original used: a single 50/50
split by storyline rather than 4-fold rotation.

Keeping the row is worth the few lines, because the three rows together make a point the
paper relies on: the effect holds under a small-n split, a rotated split, and a public
benchmark, and the interval narrows as n grows exactly as it should.

    python experiments/x17b_single_split.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from energymem.eval.association import build_probes   # noqa: E402
from experiments.assoc_harness import (               # noqa: E402
    SPLIT_SEED, _corpus, cluster_ci,
)
from energymem.world.knn_wm import KNNWorldModel      # noqa: E402

KP_GRID = [5, 10, 20]
TAU_GRID = [2.0, 5.0, 10.0]


def score(Z, wm, probes):
    r1 = []
    for p in probes:
        zp = wm.predict(Z[p.cue_idx]) if wm is not None else Z[p.cue_idx]
        s = Z @ zp
        s[p.cue_idx] = -np.inf
        r1.append(float(int(np.argmax(s)) == p.gold_idx))
    return np.array(r1)


def main():
    eps, Z, threads = _corpus(0, 900)
    half = len(threads) // 2
    dev, test = set(threads[:half]), set(threads[half:])
    allp = build_probes(eps, hop=1)
    devp = [p for p in allp if p.thread in dev]
    testp = [p for p in allp if p.thread in test]
    pairs = [(p.cue_idx, p.gold_idx) for p in devp]

    print("=" * 76)
    print("X17b (re-run)  one hop, single 50/50 storyline split")
    print("=" * 76)
    print(f"  storylines   {len(dev)} dev / {len(test)} test")
    print(f"  transitions  {len(pairs)} (dev only)")
    print(f"  test probes  {len(testp)} over {len({p.thread for p in testp})} storylines")

    # (k, tau) chosen on DEV probes only -- never on the reported half
    best, arg = -1.0, (KP_GRID[0], TAU_GRID[0])
    for kp in KP_GRID:
        for tau in TAU_GRID:
            wm = KNNWorldModel(k=kp, tau=tau).fit(Z, pairs)
            v = score(Z, wm, devp).mean()
            if v > best:
                best, arg = float(v), (kp, tau)
    print(f"  selected on dev: k_P={arg[0]}, tau={arg[1]:g}  (dev R@1 {best:.3f})")

    wm = KNNWorldModel(k=arg[0], tau=arg[1]).fit(Z, pairs)
    base = score(Z, None, testp)
    got = score(Z, wm, testp)
    clusters = [p.thread for p in testp]
    d, lo, hi, p = cluster_ci(got - base, clusters)

    print(f"\n  cosine       R@1 {base.mean():.3f}")
    print(f"  world model  R@1 {got.mean():.3f}")
    print(f"  paired delta {d:+.3f} [{lo:+.3f}, {hi:+.3f}]  p={p:.3f}"
          f"   ({(got.mean()/base.mean()-1)*100:+.0f}% relative)")

    dst = ROOT / "experiments" / "results" / "x17b_single_split.json"
    dst.write_text(json.dumps({
        "n_probes": len(testp), "n_clusters": len(set(clusters)),
        "n_train_transitions": len(pairs),
        "selected": {"k_P": arg[0], "tau": arg[1], "dev_r1": best},
        "cosine_R@1": float(base.mean()), "wm_R@1": float(got.mean()),
        "delta": d, "lo": lo, "hi": hi, "p": p,
        "split_seed": SPLIT_SEED,
    }, indent=2))
    print(f"\n  wrote {dst}")


if __name__ == "__main__":
    main()
