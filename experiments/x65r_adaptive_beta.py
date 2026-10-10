"""X65-r -- re-run X65 against a baseline that is actually the best fixed beta.

WHY THIS EXISTS. X65 printed its result column as "delta vs best fixed" while hard-coding
the reference to `score_all(b, 16.0, 1)`. X60, in the same driver and the same JSON, shows
beta=16 is NOT the best fixed beta at T=1: it scores gold@10 0.241 where beta=32 scores
0.397. So family C's one nominally positive cell (+0.067, p=0.004 for beta ~ 1/std) was
measured against a handicapped reference, which is the same failure mode as the X31 false
positive -- a method beating a baseline that was denied its own best setting.

This driver fixes it two ways and reports both:
  (a) against the best fixed beta on the SAME data (the generous reading, still a
      comparison the adaptive rule should win if it is doing anything);
  (b) against a fixed beta SELECTED ON TRAINING FOLDS ONLY, leave-one-conversation-out,
      which is the comparison the protocol actually requires.

Reported as intervals, not p-values: with ten conversation clusters the blocked
permutation bottoms out at 2/2**10 = 0.00195, so a "p = 0.002" here carries no information
beyond "as extreme as the blocking permits" (Phase 8, X99).
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

from experiments.locomo_qa import RESULTS, banks, folds                  # noqa: E402
from experiments.x6x_energy import descend                               # noqa: E402
from energymem.eval.setmetrics import (                                  # noqa: E402
    exact_blocked_permutation, mean_gold_recall_at_k, paired_report,
)

OUT = RESULTS / "x65r_adaptive_beta.json"
BETAS = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0)
RULES = {
    "beta ~ 1/margin": lambda s: 1.0 / (s[0] - s[1] + 1e-3),
    "beta ~ 1/std": lambda s: 1.0 / (s.std() + 1e-3),
    "beta ~ margin (sign-reversed control)": lambda s: 20.0 * (s[0] - s[1]),
}


def fixed_per_query(bks, beta: float):
    """gold@10 per question for a fixed beta, T=1 descent."""
    out = []
    for b in bks:
        for i in range(b.m):
            xi = descend(b.Z, b.Q[i], beta, 1)
            out.append(mean_gold_recall_at_k(np.argsort(-(b.Z @ xi)), b.gold[i], 10))
    return np.asarray(out)


def adaptive_per_query(bks, rule):
    out = []
    for b in bks:
        for i in range(b.m):
            s = np.sort(b.Z @ b.Q[i])[::-1]
            beta = float(np.clip(rule(s[:20]), 1.0, 512.0))
            xi = descend(b.Z, b.Q[i], beta, 1)
            out.append(mean_gold_recall_at_k(np.argsort(-(b.Z @ xi)), b.gold[i], 10))
    return np.asarray(out)


if __name__ == "__main__":
    t0 = time.time()
    bks = banks()
    clus = np.concatenate([np.full(b.m, b.sample_id) for b in bks])
    bank_of = np.concatenate([np.full(b.m, j) for j, b in enumerate(bks)])
    cos = np.concatenate([
        [mean_gold_recall_at_k(np.argsort(-(b.Z @ b.Q[i])), b.gold[i], 10)
         for i in range(b.m)] for b in bks])

    print(f"{len(bks)} conversations, {len(clus)} questions, "
          f"{len(np.unique(clus))} clusters.  cosine gold@10 = {cos.mean():.4f}\n")
    print(f"{'fixed beta':>12}{'gold@10':>10}")
    fixed = {}
    for beta in BETAS:
        fixed[beta] = fixed_per_query(bks, beta)
        print(f"{beta:>12g}{fixed[beta].mean():>10.4f}")
    best_beta = max(BETAS, key=lambda b: fixed[b].mean())
    print(f"\n  best fixed beta on the pooled data: {best_beta:g} "
          f"(gold@10 {fixed[best_beta].mean():.4f})")
    print(f"  the reference X65 actually used:    16 "
          f"(gold@10 {fixed[16.0].mean():.4f})  <-- handicapped by "
          f"{fixed[best_beta].mean() - fixed[16.0].mean():+.4f}")

    # fold-selected fixed beta: chosen on the nine training conversations per fold
    sel = np.empty_like(cos)
    picks = []
    for train, test in folds(bks):
        tr = np.isin(bank_of, train)
        pick = max(BETAS, key=lambda b: fixed[b][tr].mean())
        te = bank_of == test
        sel[te] = fixed[pick][te]
        picks.append((bks[test].sample_id, pick))
    print(f"  fold-selected fixed beta: gold@10 {sel.mean():.4f}, "
          f"picks {sorted({p for _, p in picks})}")

    rows = {}
    print(f"\n{'rule':<40}{'gold@10':>9}{'vs beta=16':>26}{'vs best fixed':>26}"
          f"{'vs fold-selected':>26}")
    print("-" * 127)
    for name, rule in RULES.items():
        v = adaptive_per_query(bks, rule)
        r = {}
        for label, base in (("beta16", fixed[16.0]), ("best_fixed", fixed[best_beta]),
                            ("fold_selected", sel)):
            pr = paired_report(v, base, clus, n_boot=4000, n_perm=20000)
            pr["p_exact"] = exact_blocked_permutation(v - base, clus)
            r[label] = pr
        rows[name] = dict(gold10=float(v.mean()), **r)
        print(f"{name:<40}{v.mean():>9.4f}" + "".join(
            f"{r[k]['delta']:>+13.4f} [{r[k]['lo']:>+.3f},{r[k]['hi']:>+.3f}]"
            for k in ("beta16", "best_fixed", "fold_selected")))
    print("-" * 127)
    best_rule = max(rows, key=lambda k: rows[k]["gold10"])
    print(f"\n  best adaptive rule: {best_rule} at {rows[best_rule]['gold10']:.4f}")
    print(f"  best fixed beta:                  {fixed[best_beta].mean():.4f}")
    print(f"  fold-selected fixed beta:         {sel.mean():.4f}")
    print(f"  plain cosine:                     {cos.mean():.4f}")
    verdict = rows[best_rule]["fold_selected"]["lo"] > 0
    print(f"\n  VERDICT: query-conditional beta beats a properly selected fixed beta: "
          f"{verdict}")
    print("  (X65 as published compared against beta=16 and reported a win. Against the")
    print("   fixed beta the protocol requires, that win " +
          ("survives." if verdict else "does not exist."))

    json.dump(dict(
        rows=rows, cosine=float(cos.mean()),
        fixed={str(k): float(v.mean()) for k, v in fixed.items()},
        best_fixed_beta=best_beta, best_fixed=float(fixed[best_beta].mean()),
        x65_reference_beta=16.0, x65_reference=float(fixed[16.0].mean()),
        fold_selected=float(sel.mean()), fold_picks=[[a, b] for a, b in picks],
        verdict_beats_selected_fixed=bool(verdict),
        note="X65 printed 'delta vs best fixed' while hard-coding beta=16, which X60 in "
             "the same JSON shows is not the best fixed beta. Intervals, not p-values: "
             "ten clusters floor the blocked permutation at 2/1024.",
        seconds=round(time.time() - t0, 1)), open(OUT, "w"), indent=2)
    print(f"\nwrote {OUT}  ({round(time.time() - t0, 1)}s)")
