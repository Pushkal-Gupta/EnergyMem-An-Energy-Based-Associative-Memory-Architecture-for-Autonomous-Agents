"""X29 + X30 -- the baseline grid, and whether set metrics change the ordering.

X29. Every retriever the repo has, matched encoder, on LoCoMo QA evidence retrieval.
     The table every later claim is measured against.
X30. The same rankings scored by full-set recall vs single-gold Recall@k. If the method
     ORDERING changes between them, R@1 was hiding the multi-target problem.

MMR IS THE CRITICAL BASELINE. If diversity reranking captures whatever mode expansion
gets, the mechanism is diversity and not association, and X31/X32 have to be reframed.
It is here, tuned per fold, with the same budget as everything else.
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
from experiments.locomo_qa import (KS, METHODS, RESULTS, banks, fold_select_all,
                                   fold_select, sweep)

SHA = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                     capture_output=True, text=True).stdout.strip()

GRIDS = {
    "cosine": {},
    "bm25": {},
    "rrf": {"k_rrf": [10.0, 60.0, 120.0]},
    "mmr": {"lam": [0.3, 0.5, 0.7, 0.9], "pool": [20, 50]},
}
CATS = {"1 multi-hop": (1,), "4 single-hop": (4,), "all": None}


def main():
    bks = banks()
    out = dict(id="X29+X30", git_sha=SHA, utc=datetime.now(timezone.utc).isoformat(),
               encoder="all-mpnet-base-v2", bank="per-conversation",
               n_clusters=len(bks), selection="leave-one-conversation-out, train side only",
               primary="gold@10", results={})

    for label, cats in CATS.items():
        print(f"\n{'=' * 104}")
        print(f"X29  baseline grid -- {label}")
        print(f"{'=' * 104}")
        hdr = (f"{'method':<12}{'n':>5}" + "".join(f"{'gold@' + str(k):>9}" for k in KS)
               + "".join(f"{'full@' + str(k):>9}" for k in KS) + f"{'mrr':>8}{'ndcg10':>8}")
        print(hdr)
        print("-" * len(hdr))
        store = {}
        for name, grid in GRIDS.items():
            table = sweep(bks, METHODS[name], grid, cats=cats)
            mets, clus, _, picks = fold_select_all(bks, table, score_key="gold@10")
            row = {k: float(v.mean()) for k, v in mets.items()}
            m, lo, hi = hierarchical_bootstrap(mets["gold@10"], clus, n=4000)
            store[name] = dict(metrics=row, gold10_ci=[lo, hi], picks=picks,
                               n=int(len(clus)), _vals=mets["gold@10"], _clus=clus)
            print(f"{name:<12}{len(clus):>5}"
                  + "".join(f"{row['gold@' + str(k)]:>9.3f}" for k in KS)
                  + "".join(f"{row['full@' + str(k)]:>9.3f}" for k in KS)
                  + f"{row['mrr']:>8.3f}{row['ndcg@10']:>8.3f}")
        print("-" * len(hdr))

        base = store["cosine"]
        print(f"{'vs cosine':<12}  paired delta on gold@10, blocked within conversation")
        for name in GRIDS:
            if name == "cosine":
                continue
            r = paired_report(store[name]["_vals"], base["_vals"], base["_clus"],
                              n_boot=4000, n_perm=20000)
            flag = "" if r["p"] >= 0.05 else ("  *" if r["delta"] > 0 else "  * (worse)")
            print(f"  {name:<10}{r['delta']:>+8.4f}  [{r['lo']:>+.4f}, {r['hi']:>+.4f}]"
                  f"  p={r['p']:.4f}{flag}")
            store[name]["vs_cosine"] = {k: v for k, v in r.items()}

        for v in store.values():
            v.pop("_vals", None)
            v.pop("_clus", None)
        out["results"][label] = store

    # ---- X30: does the metric choice change the ordering?
    print(f"\n{'=' * 104}")
    print("X30  does the method ORDERING change between single-gold and set metrics?")
    print(f"{'=' * 104}")
    for label in CATS:
        r = out["results"][label]
        rank_by = lambda k: [n for n, _ in sorted(r.items(), key=lambda x: -x[1]["metrics"][k])]
        o1, o10, of = rank_by("gold@1"), rank_by("gold@10"), rank_by("full@10")
        print(f"  {label:<14} by gold@1 (=R@1): {' > '.join(o1)}")
        print(f"  {'':<14} by gold@10:        {' > '.join(o10)}")
        print(f"  {'':<14} by full@10:        {' > '.join(of)}"
              f"{'   <-- ORDERING CHANGES' if of != o1 else ''}")
        out["results"][label]["_orderings"] = dict(gold1=o1, gold10=o10, full10=of,
                                                   changes=bool(of != o1))

    p = RESULTS / "x29_baselines.json"
    p.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
