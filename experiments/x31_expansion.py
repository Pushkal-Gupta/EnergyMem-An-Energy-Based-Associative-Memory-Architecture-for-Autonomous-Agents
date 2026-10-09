"""X31 + X33 + X32 -- two-stage associative expansion, its averaging baseline, and the
pre-registered interaction.

One driver because all three share the expensive part (the sweep); it writes three
result files.

X31  Mixture expansion. Stage 1 retrieves top-K0 by cosine; stage 2 treats those turns
     as SEPARATE modes and reads out with logsumexp. Deployable -- no gold labels
     anywhere, unlike X28's oracle anchor.
X33  Rocchio / pseudo-relevance feedback: the SAME feedback turns AVERAGED into the
     query. The prior art, and the mode-averaging baseline.
X32  The interaction, predicted in experiment-backlog/PREREG_X32.md before this ran:
     mixture beats averaging on multi-hop (m=3.13) and not on single-hop (m=1.06).
     `s_mixture` and `s_mixture_avg` build their hypotheses with identical code and
     differ ONLY in the combination rule, so the gap is attributable to the readout.

Null settings, asserted: K0=0 leaves one mode, and one mode makes the mixture readout a
strictly increasing function of cosine -- identical ranking. Rocchio at gamma=0 likewise.
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from energymem.eval.setmetrics import hierarchical_bootstrap, paired_report
from experiments.locomo_qa import (KS, METHODS, RESULTS, banks, fold_select_all, sweep)

SHA = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                     capture_output=True, text=True).stdout.strip()

# K0 MUST STAY BELOW THE EVALUATION CUTOFF. A first version of this swept K0 up to 10
# while selecting on gold@10, and selection chose K0=10 for every fold. With
# with_query=True the K0 feedback modes ARE cosine's top-10, each scoring 1.0 against
# itself, so they are lifted above everything else and the top-10 SET is pinned to
# cosine's top-10 by construction -- the metric becomes mathematically blind to the
# method and reports delta = +0.0000 with a zero-width interval. That is an artifact of
# K0 == k, not a null result. K0 is capped at 5 against a cutoff of 10, and K0=0 (and
# gamma=0) are included so EVERY expansion method can back off to the cosine null if
# nothing beats it -- otherwise the comparison forces one method to be non-trivial while
# the other escapes to the null, and the gap measures the grid rather than the readout.
GRIDS = {
    "cosine": {},
    # X29 found RRF the strongest baseline by a wide margin (gold@10 0.651 vs cosine
    # 0.527, p=0.002). Comparing expansion only against cosine would be measuring it
    # against a baseline we already know is beaten, so RRF is carried here too.
    "rrf": {"k_rrf": [10.0, 60.0, 120.0]},
    "rocchio":     {"K0": [0, 1, 2, 3, 5], "gamma": [0.0, 0.25, 0.5, 1.0]},
    "mixture":     {"K0": [0, 1, 2, 3, 5], "beta": [4.0, 16.0, 64.0], "with_query": [True, False]},
    "mixture_avg": {"K0": [0, 1, 2, 3, 5], "beta": [4.0, 16.0, 64.0], "with_query": [True, False]},
}
K_CUTOFF = 10
CATS = {"1 multi-hop": (1,), "4 single-hop": (4,), "2 temporal": (2,),
        "3 open-domain": (3,), "5 adversarial": (5,), "all": None}
PRIMARY = "gold@10"


def main():
    bks = banks()
    # ---- null-setting gate: refuse to produce numbers if the ladder rung is broken
    for b in (bks[0], bks[4]):
        for i in (0, 3, 11):
            c = np.argsort(-METHODS["cosine"](b, i))[:25]
            assert (np.argsort(-METHODS["mixture"](b, i, K0=0, beta=16.0))[:25] == c).all()
            assert (np.argsort(-METHODS["mixture_avg"](b, i, K0=0, beta=16.0))[:25] == c).all()
            assert (np.argsort(-METHODS["rocchio"](b, i, K0=5, gamma=0.0))[:25] == c).all()
    print("null settings OK: mixture(K0=0) = mixture_avg(K0=0) = rocchio(gamma=0) = cosine")
    for nm, gr in GRIDS.items():
        if "K0" in gr:
            assert max(gr["K0"]) < K_CUTOFF, (
                f"{nm}: K0={max(gr['K0'])} >= cutoff {K_CUTOFF} pins the top-{K_CUTOFF} "
                "set to cosine's and makes the primary metric blind to the method")
            assert 0 in gr["K0"], f"{nm}: K0=0 must be available so the method can back off"
    print(f"grid gate OK: every K0 < cutoff {K_CUTOFF}, and K0=0 reachable by all\n")

    out = dict(id="X31+X33", git_sha=SHA, utc=datetime.now(timezone.utc).isoformat(),
               encoder="all-mpnet-base-v2", bank="per-conversation", n_clusters=len(bks),
               selection="leave-one-conversation-out, train side only", primary=PRIMARY,
               grids={k: {kk: list(map(str, vv)) for kk, vv in v.items()} for k, v in GRIDS.items()},
               results={})
    inter = dict(id="X32", git_sha=SHA, utc=datetime.now(timezone.utc).isoformat(),
                 prereg="experiment-backlog/PREREG_X32.md", primary=PRIMARY,
                 prediction="delta_1 > 0; delta_4 ~ 0; delta_1 - delta_4 > 0",
                 per_category={}, interaction={})

    keep = {}
    for label, cats in CATS.items():
        print(f"{'=' * 100}")
        print(f"X31/X33  expansion -- {label}")
        print(f"{'=' * 100}")
        hdr = (f"{'method':<13}{'n':>5}" + "".join(f"{'gold@' + str(k):>9}" for k in KS)
               + f"{'full@10':>9}{'mrr':>8}{'ndcg10':>8}")
        print(hdr)
        print("-" * len(hdr))
        store, vals = {}, {}
        for name, grid in GRIDS.items():
            table = sweep(bks, METHODS[name], grid, cats=cats)
            mets, clus, qcats, picks = fold_select_all(bks, table, score_key=PRIMARY)
            row = {k: float(v.mean()) for k, v in mets.items()}
            m, lo, hi = hierarchical_bootstrap(mets[PRIMARY], clus, n=4000)
            nulls = sum(1 for pk in picks
                        if "K0=0" in pk["picked"] or "gamma=0.0" in pk["picked"])
            store[name] = dict(metrics=row, primary_ci=[lo, hi], n=int(len(clus)),
                               picks=picks, folds_backed_off_to_null=nulls)
            vals[name] = (mets[PRIMARY], clus)
            print(f"{name:<13}{len(clus):>5}"
                  + "".join(f"{row['gold@' + str(k)]:>9.3f}" for k in KS)
                  + f"{row['full@10']:>9.3f}{row['mrr']:>8.3f}{row['ndcg@10']:>8.3f}"
                  + (f"   null {nulls}/10" if grid else ""))
        print("-" * len(hdr))

        cv, cc = vals["cosine"]
        for name in ("rrf", "rocchio", "mixture", "mixture_avg"):
            r = paired_report(vals[name][0], cv, cc, n_boot=4000, n_perm=20000)
            print(f"  {name + ' - cosine':<22}{r['delta']:>+8.4f}  "
                  f"[{r['lo']:>+.4f}, {r['hi']:>+.4f}]  p={r['p']:.4f}"
                  f"{'  *' if r['p'] < 0.05 else ''}")
            store[name]["vs_cosine"] = r
        # the readout contrast: identical modes, identical weights, different rule
        r = paired_report(vals["mixture"][0], vals["mixture_avg"][0], cc,
                          n_boot=4000, n_perm=20000)
        print(f"  {'mixture - mixture_avg':<22}{r['delta']:>+8.4f}  "
              f"[{r['lo']:>+.4f}, {r['hi']:>+.4f}]  p={r['p']:.4f}"
              f"{'  *' if r['p'] < 0.05 else ''}   <-- I4 contrast")
        store["I4_contrast"] = r
        rb = paired_report(vals["mixture"][0], vals["rrf"][0], cc, n_boot=4000, n_perm=20000)
        print(f"  {'mixture - rrf':<22}{rb['delta']:>+8.4f}  "
              f"[{rb['lo']:>+.4f}, {rb['hi']:>+.4f}]  p={rb['p']:.4f}"
              f"{'  *' if rb['p'] < 0.05 else ''}   <-- vs strongest baseline")
        store["vs_strongest"] = rb
        inter["per_category"][label] = dict(
            delta=r["delta"], lo=r["lo"], hi=r["hi"], p=r["p"], n=r["n"],
            mixture=r["a"], averaged=r["b"],
            mean_m=float(np.mean([len(g) for b in bks for j, g in enumerate(b.gold)
                                  if cats is None or b.cats[j] in cats])))
        keep[label] = (vals["mixture"][0], vals["mixture_avg"][0], cc)
        out["results"][label] = store
        print()

    # ---- X32: the pre-registered interaction
    print(f"{'=' * 100}")
    print("X32  THE INTERACTION  (prediction made in PREREG_X32.md before this ran)")
    print(f"{'=' * 100}")
    print(f"{'category':<16}{'mean m':>8}{'mixture':>10}{'averaged':>10}{'delta':>10}"
          f"{'95% CI':>22}{'p':>9}")
    print("-" * 85)
    for label in ("1 multi-hop", "4 single-hop", "2 temporal", "3 open-domain",
                  "5 adversarial", "all"):
        d = inter["per_category"][label]
        print(f"{label:<16}{d['mean_m']:>8.2f}{d['mixture']:>10.3f}{d['averaged']:>10.3f}"
              f"{d['delta']:>+10.4f}  [{d['lo']:>+.4f}, {d['hi']:>+.4f}]{d['p']:>9.4f}")
    print("-" * 85)

    m1, a1, c1 = keep["1 multi-hop"]
    m4, a4, c4 = keep["4 single-hop"]
    d1, d4 = m1 - a1, m4 - a4
    from energymem.eval.setmetrics import blocked_permutation
    rng = np.random.default_rng(0)
    # difference-of-differences: cluster bootstrap over conversations, both cats at once
    convs = np.unique(np.concatenate([c1, c4]))
    boot = np.empty(8000)
    for t in range(8000):
        take = rng.choice(convs, size=len(convs), replace=True)
        e1 = np.concatenate([d1[c1 == u] for u in take if (c1 == u).any()])
        e4 = np.concatenate([d4[c4 == u] for u in take if (c4 == u).any()])
        boot[t] = (e1.mean() if e1.size else 0.0) - (e4.mean() if e4.size else 0.0)
    dod = float(d1.mean() - d4.mean())
    lo, hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
    # clamp: with an all-zero bootstrap both tails are 1.0 and 2*min() returns 2.0
    p_dod = float(min(1.0, 2 * min((boot <= 0).mean(), (boot >= 0).mean())))
    inter["interaction"] = dict(delta_1=float(d1.mean()), delta_4=float(d4.mean()),
                                dod=dod, lo=lo, hi=hi, p=p_dod,
                                n_1=int(d1.size), n_4=int(d4.size))
    print(f"interaction  (delta_1 - delta_4) = {dod:+.4f}  [{lo:+.4f}, {hi:+.4f}]  p={p_dod:.4f}")
    print(f"  delta_1 = {d1.mean():+.4f}  (n={d1.size}, mean m=3.13)")
    print(f"  delta_4 = {d4.mean():+.4f}  (n={d4.size}, mean m=1.06)")
    print()
    pred = (d1.mean() > 0, abs(d4.mean()) < abs(d1.mean()) or d4.mean() <= 0, dod > 0)
    print(f"PREDICTION 1  delta_1 > 0            : {'MET' if pred[0] else 'NOT MET'}")
    print(f"PREDICTION 2  delta_4 ~ 0            : {'MET' if pred[1] else 'NOT MET'}")
    print(f"PREDICTION 3  delta_1 - delta_4 > 0  : {'MET' if pred[2] else 'NOT MET'}")
    inter["predictions_met"] = dict(d1_positive=bool(pred[0]), d4_null=bool(pred[1]),
                                    interaction_positive=bool(pred[2]))

    (RESULTS / "x31_expansion.json").write_text(json.dumps(out, indent=2, default=float))
    (RESULTS / "x32_interaction.json").write_text(json.dumps(inter, indent=2, default=float))
    print(f"\nwrote {RESULTS / 'x31_expansion.json'}")
    print(f"wrote {RESULTS / 'x32_interaction.json'}")


if __name__ == "__main__":
    main()
