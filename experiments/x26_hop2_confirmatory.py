"""X26 -- pre-registered, multi-realization confirmatory test of the 2-hop claim (X19/X25b).

WHY THIS EXISTS. docs/10_EXPERIMENT_LEDGER.md Phase 4 found X19 (2-hop Recall@5 +70%,
p<0.001, "the strongest result in the project") does not reproduce under a rigorous
protocol (X25b: cosine 0.207, averaged 0.230, mixture 0.237, every CI spans zero). The
ledger calls this "suspended, not withdrawn" and names a pre-registered reproduction as
gate 1 of the build order. This is that gate.

WHAT "PRE-REGISTERED" MEANS HERE. Every choice below -- which methods, which hyperparameter
grids, which metric is primary, and the decision rule -- is fixed in this file BEFORE any
result is computed, reusing verbatim the grids and scoring functions from
experiments/x23_mode_diagnostics.py (already gated, already tested) rather than inventing
new ones that could be shopped post-hoc.

WHY MULTIPLE REALIZATIONS, NOT ONE MORE RUN. eval/corpus.py's storylines (THREADS) are
fixed, hand-authored content -- build(seed, ...) only varies (a) how filler distractor
sentences are sampled and (b) the temporal interleaving of threads. A single seed is
therefore still one specific realization of distractor noise. X25b is ONE point; a single
confirmatory rerun would still be one point. This script runs the identical protocol across
FILLER_SEEDS x SPLIT_SEEDS realizations and reports how many independently show an effect,
which is what actually distinguishes "a real, if small, effect" from "one lucky/unlucky
draw of filler sentences and fold partition."

PRE-REGISTERED DECISION RULE (fixed before running):
  - Primary metric: R@5 (the ledger's stated choice, since R@1 counts are small at this
    probe density -- ~130-260 probes per single run).
  - A realization counts as a POSITIVE hop-2 replication of X19 if the cluster-bootstrap
    95% CI of (averaged - cosine) OR (mixture - cosine) on R@5 excludes zero on the
    positive side.
  - X19 is CONFIRMED only if a majority of realizations replicate AND the meta-analytic
    mean effect across realizations (treating each realization as one observation) has a
    CI excluding zero.
  - Otherwise X19 remains NOT CONFIRMED, regardless of any single realization's p-value --
    a single significant draw among many realizations is exactly the multiple-comparisons
    trap the project has already been burned by once (X14).

Run:
    python experiments/x26_hop2_confirmatory.py
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from energymem.encoders import SentenceEncoder                     # noqa: E402
from energymem.eval.association import build_probes                # noqa: E402
from energymem.eval.corpus import build                            # noqa: E402
from energymem.world.knn_wm import KNNWorldModel                   # noqa: E402
from experiments.assoc_harness import Bench, cluster_ci            # noqa: E402
from experiments.x23_mode_diagnostics import (                     # noqa: E402
    KP_GRID, TAU_GRID, K_GRID, BETA_GRID, ALPHA_GRID, S_MAX, METHODS,
    score_probes, select_wm_on_val, select_mixture_on_val,
)

# ---------------------------------------------------------------- pre-registration
FILLER_SEEDS = [0, 1, 2, 3, 4]        # which draw of distractor filler sentences
SPLIT_SEEDS = [12345, 7, 99]          # which thread-to-fold partition (12345 = X23's own)
N_FOLDS = 4
N_FILLER = 900
HOPS = (1, 2)                          # 1 kept as a sanity replication of the surviving result
PRIMARY_METRIC = "R@5"


def custom_folds(filler_seed: int, split_seed: int, n_folds: int = N_FOLDS) -> list[Bench]:
    """Identical to assoc_harness.folds(), except the thread-to-fold partition seed is a
    parameter instead of the module's frozen SPLIT_SEED. assoc_harness.py's own SPLIT_SEED
    is a declared protocol constant ("fixed for the life of the project") and is
    deliberately left untouched; this is a separate, explicitly-labelled robustness sweep,
    not a silent change to it.
    """
    eps, _ = build(filler_seed, n_filler=N_FILLER)
    Z = SentenceEncoder().encode([e.text for e in eps])
    threads = sorted({e.thread for e in eps
                      if e.thread != "filler" and not e.thread.endswith("_nearmiss")})
    rng = np.random.default_rng(split_seed)
    rng.shuffle(threads)
    parts = [set(threads[i::n_folds]) for i in range(n_folds)]
    out = []
    for f in range(n_folds):
        test = parts[f]
        val = parts[(f + 1) % n_folds]
        train = set().union(*[parts[j] for j in range(n_folds) if j not in (f, (f + 1) % n_folds)])
        out.append(Bench(eps, Z, train, val, test))
    return out


@dataclass
class RunResult:
    filler_seed: int
    split_seed: int
    hop: int
    n_probes: int
    n_threads: int
    cosine_r5: float
    averaged_r5: float
    mixture_r5: float
    d_avg: float; lo_avg: float; hi_avg: float; p_avg: float
    d_mix: float; lo_mix: float; hi_mix: float; p_mix: float


def run_one(filler_seed: int, split_seed: int, hop: int) -> RunResult:
    benches = custom_folds(filler_seed, split_seed)
    acc = {m: {"r5": []} for m in METHODS}
    threads = []
    for b in benches:
        kp, tau = select_wm_on_val(b, hop)
        wm = b.world_model(k=kp, tau=tau)
        (K, beta, alpha), _, _ = select_mixture_on_val(b, wm, hop)
        probes = b.probes("test", hop)
        threads.extend(p.thread for p in probes)
        for m in METHODS:
            _, r5 = score_probes(b, wm, probes, m, hop, K, beta, alpha)
            acc[m]["r5"].append(r5)
    for m in METHODS:
        acc[m]["r5"] = np.concatenate(acc[m]["r5"])
    threads = np.array(threads)

    base = acc["cosine"]["r5"]
    d_avg, lo_avg, hi_avg, p_avg = cluster_ci(acc["averaged"]["r5"] - base, threads)
    d_mix, lo_mix, hi_mix, p_mix = cluster_ci(acc["mixture"]["r5"] - base, threads)

    return RunResult(
        filler_seed=filler_seed, split_seed=split_seed, hop=hop,
        n_probes=len(threads), n_threads=len(set(threads.tolist())),
        cosine_r5=float(base.mean()), averaged_r5=float(acc["averaged"]["r5"].mean()),
        mixture_r5=float(acc["mixture"]["r5"].mean()),
        d_avg=d_avg, lo_avg=lo_avg, hi_avg=hi_avg, p_avg=p_avg,
        d_mix=d_mix, lo_mix=lo_mix, hi_mix=hi_mix, p_mix=p_mix,
    )


def main() -> int:
    print("=" * 92)
    print("X26 -- PRE-REGISTERED confirmatory test of the 2-hop composition claim (X19)")
    print("=" * 92)
    print(f"\nrealizations: {len(FILLER_SEEDS)} filler seeds x {len(SPLIT_SEEDS)} fold-partition "
          f"seeds = {len(FILLER_SEEDS) * len(SPLIT_SEEDS)} runs per hop")
    print(f"decision rule fixed in advance: replicated iff 95% CI of (averaged-cosine) or "
          f"(mixture-cosine) on {PRIMARY_METRIC} excludes zero on the positive side")

    all_results: list[RunResult] = []
    for hop in HOPS:
        print(f"\n{'-' * 92}\nHOP {hop}\n{'-' * 92}")
        print(f"  {'filler':>7}{'split':>8}{'n':>6} | {'cosine':>8}{'avg':>8}{'mix':>8} | "
              f"{'d_avg [CI]':>26}{'d_mix [CI]':>26}")
        for fs in FILLER_SEEDS:
            for ss in SPLIT_SEEDS:
                r = run_one(fs, ss, hop)
                all_results.append(r)
                sig_avg = "*" if r.lo_avg > 0 else (" " if r.hi_avg > 0 else "v")
                sig_mix = "*" if r.lo_mix > 0 else (" " if r.hi_mix > 0 else "v")
                print(f"  {fs:>7}{ss:>8}{r.n_probes:>6} | {r.cosine_r5:>8.3f}"
                      f"{r.averaged_r5:>8.3f}{r.mixture_r5:>8.3f} | "
                      f"{r.d_avg:>+7.3f}[{r.lo_avg:+.3f},{r.hi_avg:+.3f}]{sig_avg} "
                      f"{r.d_mix:>+7.3f}[{r.lo_mix:+.3f},{r.hi_mix:+.3f}]{sig_mix}")

    print("\n" + "=" * 92)
    print("META-ANALYSIS (each realization is one observation)")
    print("=" * 92)
    for hop in HOPS:
        rs = [r for r in all_results if r.hop == hop]
        n_pos_avg = sum(1 for r in rs if r.lo_avg > 0)
        n_neg_avg = sum(1 for r in rs if r.hi_avg < 0)
        n_pos_mix = sum(1 for r in rs if r.lo_mix > 0)
        n_neg_mix = sum(1 for r in rs if r.hi_mix < 0)
        d_avgs = np.array([r.d_avg for r in rs])
        d_mixs = np.array([r.d_mix for r in rs])
        # meta CI: bootstrap over realizations themselves
        rng = np.random.default_rng(0)
        boot_avg = rng.choice(d_avgs, size=(20000, len(d_avgs)), replace=True).mean(axis=1)
        boot_mix = rng.choice(d_mixs, size=(20000, len(d_mixs)), replace=True).mean(axis=1)
        meta_avg = (float(d_avgs.mean()), float(np.quantile(boot_avg, .025)), float(np.quantile(boot_avg, .975)))
        meta_mix = (float(d_mixs.mean()), float(np.quantile(boot_mix, .025)), float(np.quantile(boot_mix, .975)))

        print(f"\n  hop {hop}  ({len(rs)} realizations)")
        print(f"    averaged: {n_pos_avg}/{len(rs)} realizations positive, {n_neg_avg}/{len(rs)} negative, "
              f"{len(rs) - n_pos_avg - n_neg_avg}/{len(rs)} null")
        print(f"      mean effect across realizations: {meta_avg[0]:+.3f}  95% CI [{meta_avg[1]:+.3f}, {meta_avg[2]:+.3f}]")
        print(f"    mixture:  {n_pos_mix}/{len(rs)} realizations positive, {n_neg_mix}/{len(rs)} negative, "
              f"{len(rs) - n_pos_mix - n_neg_mix}/{len(rs)} null")
        print(f"      mean effect across realizations: {meta_mix[0]:+.3f}  95% CI [{meta_mix[1]:+.3f}, {meta_mix[2]:+.3f}]")

        majority = max(n_pos_avg, n_pos_mix) > len(rs) / 2
        meta_excludes_zero = meta_avg[1] > 0 or meta_mix[1] > 0
        verdict = "CONFIRMED" if (majority and meta_excludes_zero) else "NOT CONFIRMED"
        print(f"    PRE-REGISTERED VERDICT for hop {hop}: {verdict}")

    out = {
        "pre_registration": {
            "filler_seeds": FILLER_SEEDS, "split_seeds": SPLIT_SEEDS, "n_folds": N_FOLDS,
            "hops": list(HOPS), "primary_metric": PRIMARY_METRIC,
            "decision_rule": "replicated iff 95% CI of (method-cosine) R@5 excludes zero; "
                             "CONFIRMED iff majority of realizations replicate AND meta CI excludes zero",
        },
        "runs": [r.__dict__ for r in all_results],
    }
    dst = Path(__file__).resolve().parent / "results" / "x26_hop2_confirmatory.json"
    dst.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
