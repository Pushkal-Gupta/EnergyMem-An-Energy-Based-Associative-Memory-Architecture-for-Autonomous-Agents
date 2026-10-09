"""X27 -- the headroom certificate.

Is dense retrieval's failure on multi-target questions an EXPRESSIVITY limit or a
QUERY-FORMATION limit? The two call for completely different work, so the question is
worth settling before any method is built.

BOUND. For unit gold vectors z_1..z_m with mean pairwise inner product eps, any unit q:
    min_i <q,z_i>  <=  (1/m) sum_i <q,z_i>  <=  ||sum_i z_i|| / m
                    =  sqrt( (1 + (m-1)*eps) / m )   =:  B(m, eps)
A mixture with K >= m modes attains 1 by taking xi_s = z_s. So B is the ceiling for the
whole single-query class, independent of how good the encoder is.

The bound only MATTERS where it sits below what distractors already reach. This driver
measures both: where the CEILING binds, and where the ACTUAL query fails.
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

from experiments.locomo_qa import CATNAME, RESULTS, banks

SHA = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                     capture_output=True, text=True).stdout.strip()


def main():
    bks = banks()
    rows = []
    for b in bks:
        S = b.Q @ b.Z.T
        for i in range(b.m):
            g = b.gold[i]
            m = len(g)
            G = b.Z[g]
            if m > 1:
                C = G @ G.T
                eps = float((C.sum() - m) / (m * (m - 1)))
            else:
                eps = 1.0
            s = S[i]
            mask = np.ones(b.n, bool)
            mask[g] = False
            dist = np.sort(s[mask])[::-1]
            rows.append(dict(
                conv=b.sample_id, cat=b.cats[i], m=m, eps=eps,
                bound=float(np.sqrt((1.0 + (m - 1) * eps) / m)),
                gold_min=float(s[g].min()), gold_max=float(s[g].max()),
                d1=float(dist[0]), d10=float(dist[9]),
                dm=float(dist[min(m, len(dist)) - 1]), n_bank=b.n,
            ))

    out = dict(id="X27", git_sha=SHA, utc=datetime.now(timezone.utc).isoformat(),
               encoder="all-mpnet-base-v2", n_queries=len(rows), n_clusters=len(bks),
               bank="per-conversation", per_category={}, rows=rows)

    hdr = (f"{'category':<16}{'n':>5}{'mean m':>8}{'mean eps':>10}{'B(m,eps)':>10}"
           f"{'gold_min':>10}{'d10':>8}{'B<d10':>8}{'gm<d10':>9}")
    print("=" * len(hdr))
    print("X27  headroom certificate -- LoCoMo QA evidence retrieval, per-conversation bank")
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for cat in sorted(CATNAME):
        r = [x for x in rows if x["cat"] == cat]
        if not r:
            continue
        f = lambda k: float(np.mean([x[k] for x in r]))
        binds = float(np.mean([x["bound"] < x["d10"] for x in r]))
        fails = float(np.mean([x["gold_min"] < x["d10"] for x in r]))
        out["per_category"][str(cat)] = dict(
            name=CATNAME[cat], n=len(r), mean_m=f("m"), mean_eps=f("eps"),
            mean_bound=f("bound"), mean_gold_min=f("gold_min"), mean_d10=f("d10"),
            frac_bound_binds=binds, frac_query_fails=fails)
        print(f"{str(cat) + ' ' + CATNAME[cat]:<16}{len(r):>5}{f('m'):>8.2f}{f('eps'):>10.3f}"
              f"{f('bound'):>10.3f}{f('gold_min'):>10.3f}{f('d10'):>8.3f}"
              f"{binds:>8.1%}{fails:>9.1%}")
    print("-" * len(hdr))
    print("B<d10   the single-query CEILING is below the 10th-best distractor:")
    print("        no query whatsoever can put every gold turn in the top 10.")
    print("gm<d10  the ACTUAL encoder query already leaves a gold turn below it.")

    p = RESULTS / "x27_headroom.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
