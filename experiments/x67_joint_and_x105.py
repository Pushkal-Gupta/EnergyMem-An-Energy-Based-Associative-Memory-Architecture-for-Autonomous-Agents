"""X67-X69 -- the full beta x T x K joint grid, and X105 -- cross-corpus transfer.

X67-X69. Family C mapped beta, T and K one axis at a time and every one was neutral or
harmful. The joint grid is the remaining possibility that the axes interact: that some
interior cell beats the corner even though no single axis does. It is volume rather than
insight, which is why it was P2, but it is the last thing standing between "no knob helps"
and "no COMBINATION of knobs helps".

X105. Tune on LoCoMo, evaluate on HotpotQA, no retuning. The honest external-validity
test: a hyperparameter chosen on one corpus and applied to another, which is what anyone
deploying this would actually do.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from experiments.locomo_qa import RESULTS, banks, evaluate, folds, s_cosine, s_mixture
from experiments.x6x_energy import descend                             # noqa: E402
from energymem.encoders import SentenceEncoder                         # noqa: E402
from energymem.eval.setmetrics import (                                # noqa: E402
    exact_blocked_permutation, mean_gold_recall_at_k, paired_report,
)

OUT = RESULTS / "x67_joint_and_x105.json"
R: dict = {}


def hr(t):
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


# ======================================================== X67-X69 joint grid
def x67(bks):
    hr("X67-X69  the joint beta x T x K grid -- does any interior cell beat the corner?")
    BET = (16.0, 32.0, 64.0, 128.0, 256.0, 512.0)
    TS = (1, 2, 3, 5, 8)
    KS_ = (1, 5, 10)
    cl = np.concatenate([np.full(b.m, b.sample_id) for b in bks])
    base = {k: np.concatenate([[mean_gold_recall_at_k(np.argsort(-(b.Z @ b.Q[i])),
                                                      b.gold[i], k)
                                for i in range(b.m)] for b in bks]) for k in KS_}
    print("  corner (T=1, beta->inf) = cosine:  " +
          "  ".join(f"gold@{k}={base[k].mean():.4f}" for k in KS_))
    print(f"\n  {'beta':>7}{'T':>4}" + "".join(f"{f'd gold@{k}':>13}" for k in KS_))
    rows, best = {}, ("", -np.inf)
    for beta in BET:
        for T in TS:
            S = [np.stack([b.Z @ descend(b.Z, b.Q[i], beta, T) for i in range(b.m)])
                 for b in bks]
            ds = []
            for k in KS_:
                v = np.concatenate([[mean_gold_recall_at_k(np.argsort(-S[bi][i]),
                                                           b.gold[i], k)
                                     for i in range(b.m)]
                                    for bi, b in enumerate(bks)])
                ds.append(float(v.mean() - base[k].mean()))
                if k == 10:
                    v10 = v
            r = paired_report(v10, base[10], cl, n_boot=2000, n_perm=5000)
            rows[f"beta={beta:g},T={T}"] = dict(
                d_gold=[*ds], gold10=float(v10.mean()), delta=r["delta"],
                lo=r["lo"], hi=r["hi"], p_exact=exact_blocked_permutation(v10 - base[10], cl))
            print(f"  {beta:>7g}{T:>4}" + "".join(f"{x:>+13.4f}" for x in ds))
            if ds[-1] > best[1]:
                best = (f"beta={beta:g},T={T}", ds[-1])
    npos = sum(1 for v in rows.values() if v["lo"] > 0)
    print(f"\n  cells with delta gold@10 > 0 and interval excluding 0: {npos}/{len(rows)}")
    print(f"  best cell: {best[0]} at {best[1]:+.4f}")
    print("  (T=1 at the largest beta IS the corner by Corollary 1, so a delta of exactly")
    print("   0.0000 there is the reduction working, not an effect.)")
    R["X67_X69"] = dict(rows=rows, n_positive=npos, n_cells=len(rows),
                        best_cell=best[0], best_delta=best[1],
                        corner={str(k): float(v.mean()) for k, v in base.items()})


# ============================================================== X105 transfer
def x105(bks):
    hr("X105  cross-corpus transfer: hyperparameters chosen on LoCoMo, applied to HotpotQA")
    GRID = [dict(K0=k, beta=b, with_query=w)
            for k in (0, 1, 2, 3, 5) for b in (4.0, 16.0, 64.0) for w in (True, False)]
    # pick the LoCoMo-best mixture configuration on ALL of LoCoMo (that is the point:
    # the deploying party has no HotpotQA labels to tune on)
    best, bs = None, -np.inf
    for p in GRID:
        s = float(np.mean([evaluate(b, s_mixture, **p)["gold@10"].mean() for b in bks]))
        if s > bs:
            best, bs = p, s
    print(f"  LoCoMo-selected mixture config: {best}  (LoCoMo gold@10 {bs:.4f})")

    try:
        from experiments.hotpot import load, split
        from experiments import x102_hotpot as H
    except Exception as e:
        print(f"  HotpotQA unavailable: {e}")
        R["X105"] = dict(error=str(e)); return
    items = load(SentenceEncoder(), verbose=False)
    _, test = split(items)
    cos = np.asarray([mean_gold_recall_at_k(np.argsort(-H.s_cosine(items[i])),
                                            items[i].gold, 5) for i in test])
    tr = np.asarray([mean_gold_recall_at_k(
        np.argsort(-H.s_mixture(items[i], **best)), items[i].gold, 5) for i in test])
    # what HotpotQA's OWN tuning would have chosen, for the contrast
    own, os_ = None, -np.inf
    trn, _ = split(items)
    for p in GRID:
        s = float(np.mean([mean_gold_recall_at_k(
            np.argsort(-H.s_mixture(items[i], **p)), items[i].gold, 5) for i in trn]))
        if s > os_:
            own, os_ = p, s
    ow = np.asarray([mean_gold_recall_at_k(
        np.argsort(-H.s_mixture(items[i], **own)), items[i].gold, 5) for i in test])
    d = H.boot(tr - cos)
    d2 = H.boot(ow - cos)
    print(f"\n  on HotpotQA held-out (n={len(test)}), gold@5:")
    print(f"    cosine                                      {cos.mean():.4f}")
    print(f"    mixture, LoCoMo-transferred config          {tr.mean():.4f}  "
          f"delta {d[0]:+.4f} [{d[1]:+.4f},{d[2]:+.4f}] p={d[3]:.4f}")
    print(f"    mixture, HotpotQA's own selection {str(own):<10} {ow.mean():.4f}  "
          f"delta {d2[0]:+.4f} [{d2[1]:+.4f},{d2[2]:+.4f}] p={d2[3]:.4f}")
    print("\n  Transfer can only be interesting if the method has something to transfer.")
    R["X105"] = dict(locomo_config=best, locomo_gold10=bs, hotpot_own_config=own,
                     n_test=len(test), cosine=float(cos.mean()),
                     transferred=float(tr.mean()), transferred_delta=d[0],
                     transferred_lo=d[1], transferred_hi=d[2], transferred_p=d[3],
                     own_selected=float(ow.mean()), own_delta=d2[0])


if __name__ == "__main__":
    t0 = time.time()
    bks = banks()
    x67(bks)
    x105(bks)
    R["_meta"] = dict(seconds=round(time.time() - t0, 1))
    OUT.write_text(json.dumps(R, indent=2, default=str))
    print(f"\nwrote {OUT}  ({R['_meta']['seconds']}s)")
