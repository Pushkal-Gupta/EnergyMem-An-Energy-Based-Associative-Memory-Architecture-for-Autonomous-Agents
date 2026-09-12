"""X23-X25 -- the mode-averaging gate (docs/12_ARCHITECTURE.md, build step 1).

The Theta>0 sampler, the mixture readout and invariant I4 all rest on one claim:

    the conditional next-state distribution is MULTI-MODAL, so a point estimate lands
    between basins, where nothing is stored.

That claim had been measured in a previous session and never written into the ledger, so
it was supporting three design decisions from outside the record. This script re-measures
it from scratch and prints numbers in ledger form.

  X23  Are averaged predictions off-manifold?
       Compare the nearest-neighbour cosine of a predicted state against the same
       statistic for real stored episodes. If prediction sits further from everything
       than a real memory does, it is in empty space.

  X23b Does averaging do something OTHER than collapse the distribution?
       Yes: it shrinks the displacement. A raw mode q+Delta lands on a stored TRAINING
       episode, because |Delta| exceeds the norm of the cue. Any mixture-vs-average
       comparison that does not match step size measures step size.

  X24  Is the conditional distribution actually multi-modal?
       Mutual cosine among the top-K modes of P(.|q). Low mutual cosine = the futures
       disagree = there is something for averaging to destroy.

  X25  Does keeping the modes beat averaging them, at 1 hop and at 2?
       Mixture readout vs averaged rollout vs cosine. Hyperparameters on VAL threads,
       reported on TEST threads, cluster bootstrap over threads.

Protocol: world model fitted on TRAIN-thread transitions only; cue excluded from the
candidate set for every method identically.

    python experiments/x23_mode_diagnostics.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from energymem.readout import averaged_scores, mixture_scores, rank   # noqa: E402
from experiments.assoc_harness import cluster_ci, folds, load        # noqa: E402

KP_GRID = [5, 10, 20]      # world-model neighbourhood
TAU_GRID = [2.0, 5.0, 10.0]
K_GRID = [1, 2, 3, 5]
BETA_GRID = [4.0, 16.0, 32.0, 64.0]
ALPHA_GRID = [0.25, 0.35, 0.5, 1.0]   # 1.0 == the raw transplant; ~0.35 matches the average
TAU, K_P = 5.0, 10
S_MAX = 8


# ---------------------------------------------------------------------- X23

def x23_off_manifold(bench, wm, probes):
    """Nearest-neighbour cosine: real episodes vs averaged predictions vs single modes."""
    Z = bench.Z
    real, avg, mode = [], [], []
    for p in probes:
        q = Z[p.cue_idx]
        sims = Z @ q
        sims[p.cue_idx] = -np.inf
        real.append(float(sims.max()))                       # how close a REAL memory sits

        zm = wm.predict(q)                                   # the averaged point estimate
        s = Z @ zm; s[p.cue_idx] = -np.inf
        avg.append(float(s.max()))

        top, _ = wm.modes(q, K=1)[0]                         # a single un-averaged mode
        s = Z @ top; s[p.cue_idx] = -np.inf
        mode.append(float(s.max()))
    return (float(np.mean(real)), float(np.mean(avg)), float(np.mean(mode)))


def x23b_step_size(bench, wm, probes):
    """Is a raw mode dominated by the displacement, and where does it land?"""
    Z = bench.Z
    train_eps = {i for p in bench.probes("train", 1) for i in (p.cue_idx, p.gold_idx)}
    raw, shrunk, on_train_mode, on_train_avg = [], [], [], []
    for p in probes:
        q = Z[p.cue_idx]
        a, b = wm.mean_step_norm(q)
        raw.append(a); shrunk.append(b)
        top, _ = wm.modes(q, K=1, alpha=1.0)[0]
        s = Z @ top; s[p.cue_idx] = -np.inf
        on_train_mode.append(int(np.argmax(s)) in train_eps)
        zm = wm.predict(q)
        s = Z @ zm; s[p.cue_idx] = -np.inf
        on_train_avg.append(int(np.argmax(s)) in train_eps)
    prior = len(train_eps) / len(bench.episodes)
    return (float(np.mean(raw)), float(np.mean(shrunk)),
            float(np.mean(on_train_mode)), float(np.mean(on_train_avg)), float(prior))


# ---------------------------------------------------------------------- X24

def x24_multimodality(wm, probes, bench, K=3, thresh=0.5):
    """Mean pairwise cosine among the top-K modes, per cue."""
    mutual = []
    for p in probes:
        ms = wm.modes(bench.Z[p.cue_idx], K=K)
        if len(ms) < 2:
            continue
        V = np.stack([z for z, _ in ms])
        G = V @ V.T
        iu = np.triu_indices(len(ms), k=1)
        mutual.append(float(G[iu].mean()))
    mutual = np.array(mutual)
    return float(mutual.mean()), float((mutual < thresh).mean()), len(mutual)


# ---------------------------------------------------------------------- X25

def score_probes(bench, wm, probes, method: str, hop: int, K: int, beta: float,
                 alpha: float = 1.0):
    """Per-probe (R@1, R@5) for one method. Cue excluded for every method alike."""
    Z, r1, r5 = bench.Z, [], []
    for p in probes:
        q = Z[p.cue_idx]
        ex = {p.cue_idx}
        if method == "cosine":
            s = Z @ q
        elif method == "averaged":
            cur = q
            for _ in range(hop):
                cur = wm.predict(cur)
            s = Z @ cur
        elif method == "mixture":
            hyps = wm.rollout(q, H=hop, K=K, S=S_MAX, alpha=alpha)
            s = mixture_scores(hyps, Z, beta)
        elif method == "mixture_then_average":       # the I4 violation, measured
            hyps = wm.rollout(q, H=hop, K=K, S=S_MAX, alpha=alpha)
            s = averaged_scores(hyps, Z)
        else:
            raise ValueError(method)
        ranked = rank(s, k=5, exclude=ex)
        r1.append(float(ranked[0] == p.gold_idx))
        r5.append(float(p.gold_idx in ranked))
    return np.array(r1), np.array(r5)


def select_wm_on_val(bench, hop):
    """Choose the world model's own (k_P, tau) on VAL, by the averaged predictor's R@5.

    This is the X17b protocol. Fixing them by hand -- which the first version of this
    script did -- silently advantages or handicaps every method built on top.
    """
    val = bench.probes("val", hop)
    best, arg = -1.0, (KP_GRID[0], TAU_GRID[0])
    for kp in KP_GRID:
        for tau in TAU_GRID:
            wm = bench.world_model(k=kp, tau=tau)
            _, r5 = score_probes(bench, wm, val, "averaged", hop, 1, 1.0, 1.0)
            if r5.mean() > best:
                best, arg = float(r5.mean()), (kp, tau)
    return arg


def select_mixture_on_val(bench, wm, hop):
    """Choose (K, beta, alpha) for the mixture on VAL. TEST is not touched here.

    Selection is on R@5: at these probe counts R@1 on val is a handful of hits and argmax
    over it is noise-fitting.
    """
    val = bench.probes("val", hop)
    best, arg = -1.0, (K_GRID[0], BETA_GRID[0], ALPHA_GRID[0])
    for K in K_GRID:
        for beta in BETA_GRID:
            for alpha in ALPHA_GRID:
                _, r5 = score_probes(bench, wm, val, "mixture", hop, K, beta, alpha)
                if r5.mean() > best:
                    best, arg = float(r5.mean()), (K, beta, alpha)
    return arg, best, len(val)


METHODS = ("cosine", "averaged", "mixture", "mixture_then_average")


def run_folds(benches, hop):
    """Every thread is tested exactly once, by a model that never saw it."""
    acc = {m: {"r1": [], "r5": []} for m in METHODS}
    threads, chosen = [], []
    for b in benches:
        kp, tau = select_wm_on_val(b, hop)
        wm = b.world_model(k=kp, tau=tau)
        (K, beta, alpha), vbest, nval = select_mixture_on_val(b, wm, hop)
        chosen.append({"k_P": kp, "tau": tau, "K": K, "beta": beta, "alpha": alpha,
                       "val_r5": vbest, "n_val": nval})
        probes = b.probes("test", hop)
        threads.extend(p.thread for p in probes)
        for m in METHODS:
            r1, r5 = score_probes(b, wm, probes, m, hop, K, beta, alpha)
            acc[m]["r1"].append(r1); acc[m]["r5"].append(r5)
    for m in METHODS:
        acc[m] = {kk: np.concatenate(v) for kk, v in acc[m].items()}
    return acc, np.array(threads), chosen


def main():
    benches = folds()
    b0 = benches[0]
    wm0 = b0.world_model(k=10, tau=5.0)
    diag_probes = b0.probes("test", 1)

    print("=" * 84)
    print("SETUP")
    print("=" * 84)
    print(f"  episodes                 {len(b0.episodes)}")
    print(f"  thread-level {len(benches)}-fold: every storyline is in TEST exactly once")
    print(f"  per fold                 {len(b0.train)} train / {len(b0.val)} val / {len(b0.test)} test threads")
    print(f"  transitions fitted       {len(b0.transitions('train'))} (fold 0, train threads only)")
    out = {"setup": {"n_episodes": len(b0.episodes), "n_folds": len(benches),
                     "fold0_threads": [len(b0.train), len(b0.val), len(b0.test)],
                     "fold0_transitions": len(b0.transitions("train"))}}

    print("\n" + "=" * 84)
    print("X23  Do averaged predictions land off-manifold?   (fold 0, descriptive)")
    print("=" * 84)
    real, avg, mode = x23_off_manifold(b0, wm0, diag_probes)
    print(f"  mean nearest-neighbour cosine")
    print(f"    a real stored episode      {real:.3f}")
    print(f"    an AVERAGED prediction     {avg:.3f}   <- the point estimate")
    print(f"    a single un-averaged mode  {mode:.3f}")
    print(f"  averaging costs {avg - real:+.3f} of proximity to the manifold")
    out["X23"] = {"real": real, "averaged": avg, "single_mode": mode}

    print("\n" + "=" * 84)
    print("X23b Does averaging also SHRINK the step, not just collapse the distribution?")
    print("=" * 84)
    raw, shr, ontr_m, ontr_a, prior = x23b_step_size(b0, wm0, diag_probes)
    print(f"  mean |Delta_i| (one remembered displacement)   {raw:.3f}   vs cue norm 1.000")
    print(f"  mean |sum_i w_i Delta_i| (the average's step)  {shr:.3f}   shrinkage x{shr/raw:.2f}")
    print(f"  top-1 of a RAW mode (alpha=1) is a train episode   {ontr_m:.1%}")
    print(f"  top-1 of the AVERAGE is a train episode            {ontr_a:.1%}   (bank prior {prior:.1%})")
    print("  verdict: a raw transplanted displacement outweighs the cue, so the mode lands on")
    print("           the source transition's destination. Step size must be matched or the")
    print("           mixture-vs-average comparison measures step size instead.")
    out["X23b"] = {"delta_norm": raw, "avg_step_norm": shr, "shrinkage": shr / raw,
                   "raw_mode_on_train": ontr_m, "avg_on_train": ontr_a, "bank_prior": prior}

    print("\n" + "=" * 84)
    print("X24  Is the conditional next-state distribution multi-modal?")
    print("=" * 84)
    mu, frac, n = x24_multimodality(wm0, diag_probes, b0, K=3)
    print(f"  mean mutual cosine among top-3 modes   {mu:.3f}   (n={n} cues)")
    print(f"  cues whose modes disagree (<0.5)       {frac:.1%}")
    print(f"  verdict: {'multi-modal -- averaging has something to destroy' if mu < 0.6 else 'largely uni-modal'}")
    out["X24"] = {"mutual_cosine": mu, "frac_multimodal": frac, "n": n}

    print("\n" + "=" * 84)
    print("X25  Mixture readout vs averaging vs cosine -- whole corpus, thread-level K-fold")
    print("=" * 84)
    for hop in (1, 2):
        acc, threads, chosen = run_folds(benches, hop)
        n = len(threads)
        print(f"\n  hop {hop}   n={n} probes over {len(set(threads))} threads")
        print("   per-fold selection: " + "; ".join(
            f"f{i}: k_P={c['k_P']} tau={c['tau']:g} K={c['K']} b={c['beta']:g} a={c['alpha']:g}"
            for i, c in enumerate(chosen)))
        print(f"  {'method':>22} {'R@1':>8} {'R@5':>8}   {'dR@1 vs cosine':>30}")
        base = acc["cosine"]["r1"]
        res = {}
        for m in METHODS:
            r1, r5 = acc[m]["r1"], acc[m]["r5"]
            if m == "cosine":
                print(f"  {m:>22} {r1.mean():>8.3f} {r5.mean():>8.3f}   {'--':>30}")
                res[m] = {"R@1": float(r1.mean()), "R@5": float(r5.mean())}
            else:
                d, lo, hi, p = cluster_ci(r1 - base, threads)
                print(f"  {m:>22} {r1.mean():>8.3f} {r5.mean():>8.3f}   "
                      f"{d:>+8.3f} [{lo:+.3f}, {hi:+.3f}] p={p:.3f}")
                res[m] = {"R@1": float(r1.mean()), "R@5": float(r5.mean()),
                          "dR1": d, "lo": lo, "hi": hi, "p": p}
        dm, lo, hi, p = cluster_ci(acc["mixture"]["r1"] - acc["averaged"]["r1"], threads)
        d5, lo5, hi5, p5 = cluster_ci(acc["mixture"]["r5"] - acc["averaged"]["r5"], threads)
        print(f"  >> I4 test, mixture - averaged:  R@1 {dm:+.3f} [{lo:+.3f}, {hi:+.3f}] p={p:.3f}"
              f"   R@5 {d5:+.3f} [{lo5:+.3f}, {hi5:+.3f}] p={p5:.3f}")
        out[f"X25_hop{hop}"] = {"n": n, "n_threads": len(set(threads)), "selected": chosen,
                                "methods": res,
                                "mixture_minus_averaged": {
                                    "R@1": {"d": dm, "lo": lo, "hi": hi, "p": p},
                                    "R@5": {"d": d5, "lo": lo5, "hi": hi5, "p": p5}}}

    dst = Path(__file__).resolve().parent / "results" / "x23_mode_diagnostics.json"
    dst.parent.mkdir(exist_ok=True)
    dst.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {dst}")


if __name__ == "__main__":
    main()
