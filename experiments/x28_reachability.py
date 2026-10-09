"""X28 -- is the headroom X27 certifies actually REACHABLE by association?

X27 shows the single query leaves a gold turn below the 10th distractor on 87.9% of
multi-hop questions, and that this is not an expressivity limit. This driver asks
whether the missed evidence can be reached from what the query DID retrieve.

Anchor on the gold turn the query ranks best -- the hop the query gets right -- exclude
it, and try to reach the residual gold under:
    query    cos(q, .)                     what dense retrieval does
    anchor   cos(z_a, .)                   pure association, the query discarded
    max      max of the two                the beta->inf limit of a 2-mode mixture
    lse(b)   logsumexp at finite beta       the actual mixture readout

THE ANCHOR IS AN ORACLE: it is chosen using the gold labels, so these numbers are an
upper bound on what a deployed system could do. The non-oracle version is X31. Reported
as an upper bound and labelled as such everywhere it appears.
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

from energymem.eval.setmetrics import hierarchical_bootstrap
from experiments.locomo_qa import CATNAME, RESULTS, banks

SHA = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                     capture_output=True, text=True).stdout.strip()
BETAS = (4.0, 16.0, 64.0)


def _ranks(scores, targets, exclude):
    order = [i for i in np.argsort(-scores) if i not in exclude]
    where = {t: r + 1 for r, t in enumerate(order)}
    return np.array([where[t] for t in targets])


def main():
    bks = banks()
    per = {}
    rows = []
    for b in bks:
        S = b.Q @ b.Z.T
        ZZ = b.Z @ b.Z.T
        for i in range(b.m):
            g = b.gold[i]
            if len(g) < 2:
                continue
            s = S[i]
            a = g[int(np.argmax(s[g]))]            # ORACLE anchor
            R = [x for x in g if x != a]
            ex = {a}
            variants = {"query": s, "anchor": ZZ[a], "max": np.maximum(s, ZZ[a])}
            for beta in BETAS:
                A = np.stack([beta * s, beta * ZZ[a]])
                mx = A.max(axis=0)
                variants[f"lse{int(beta)}"] = mx + np.log(np.exp(A - mx).sum(axis=0))
            rows.append(dict(conv=b.sample_id, cat=b.cats[i], m=len(g),
                             ranks={k: _ranks(v, R, ex).tolist() for k, v in variants.items()}))

    names = ["query", "anchor", "max"] + [f"lse{int(b)}" for b in BETAS]
    hdr = f"{'category':<16}{'nq':>5}{'resid':>7}" + "".join(f"{n:>9}" for n in names)
    print("=" * len(hdr))
    print("X28  reachability of residual gold evidence (ORACLE anchor -- upper bound)")
    print("     residual Recall@10, anchor excluded from every ranking")
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for cat in list(sorted(CATNAME)) + ["ALL"]:
        r = rows if cat == "ALL" else [x for x in rows if x["cat"] == cat]
        if len(r) < 10:
            continue
        label = "ALL multi-target" if cat == "ALL" else f"{cat} {CATNAME[cat]}"
        cells, store = "", {}
        for nm in names:
            v = np.concatenate([np.asarray(x["ranks"][nm]) <= 10 for x in r]).astype(float)
            cl = np.concatenate([[x["conv"]] * len(x["ranks"][nm]) for x in r])
            mean, lo, hi = hierarchical_bootstrap(v, cl, n=4000)
            store[nm] = dict(r10=mean, lo=lo, hi=hi,
                             median_rank=float(np.median(np.concatenate(
                                 [np.asarray(x["ranks"][nm]) for x in r]))))
            cells += f"{mean:>8.1%} "
        n_res = sum(len(x["ranks"]["query"]) for x in r)
        per[str(cat)] = dict(name=("all" if cat == "ALL" else CATNAME[cat]),
                             n_questions=len(r), n_residual=n_res, variants=store)
        print(f"{label:<16}{len(r):>5}{n_res:>7}{cells}")
    print("-" * len(hdr))
    a = per["ALL"]["variants"]
    print(f"query  R@10 {a['query']['r10']:.1%}  [{a['query']['lo']:.1%}, {a['query']['hi']:.1%}]"
          f"   median rank {a['query']['median_rank']:.0f}")
    print(f"anchor R@10 {a['anchor']['r10']:.1%}  [{a['anchor']['lo']:.1%}, {a['anchor']['hi']:.1%}]"
          f"   median rank {a['anchor']['median_rank']:.0f}")
    print("ORACLE anchor: an upper bound, not a method. X31 is the deployable version.")

    out = dict(id="X28", git_sha=SHA, utc=datetime.now(timezone.utc).isoformat(),
               encoder="all-mpnet-base-v2", anchor="oracle (gold turn the query ranks best)",
               n_clusters=len(bks), betas=list(BETAS), per_category=per, rows=rows)
    p = RESULTS / "x28_reachability.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
