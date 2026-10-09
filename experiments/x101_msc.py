"""X101 -- the X26 association probe on Multi-Session Chat. External replication of the
project's one positive claim on a second public corpus.

MSC ships no QA evidence, so the QA families cannot run on it. What it does have is
exactly what X26 needs: multi-session dialogues between two speakers. The task, protocol
and code are X26's (experiments/locomo_assoc.py), imported rather than copied, so the
only thing that changes is the corpus:

  cue    a turn
  gold   the turn that followed it in the same session (hop 1), or two later (hop 2)
  bank   every turn of the MSC test split, pooled (a cue must beat turns from every
         other dialogue as well as its own)
  split  session; 4-fold rotation; world model fitted on train-session transitions only;
         (k_P, tau) and (K, beta, alpha) selected on that fold's val sessions

Pre-registered in experiment-backlog/PREREG_HI.md (P-H4): world model beats cosine at
R@1 on hop 1 with a CI excluding 0.
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import experiments.locomo_assoc as A                    # noqa: E402
from experiments.assoc_harness import cluster_ci        # noqa: E402
from experiments.corpus_qa import encode_unique, stamp  # noqa: E402


def build():
    import pyarrow.parquet as pq
    rows = pq.read_table(ROOT / "data/msc/test.parquet").to_pylist()
    texts, sess_of = [], []
    for r in rows:
        utts = r["dialogue"] if isinstance(r["dialogue"], list) else ast.literal_eval(r["dialogue"])
        spk = r["speaker"] if isinstance(r["speaker"], list) else ast.literal_eval(r["speaker"])
        for u, s in zip(utts, spk):
            texts.append(f"{s}: {u}")
            sess_of.append(f"{r['dialoug_id']}/{r['session_id']}")
    Z = encode_unique(texts)
    sess_of = np.array(sess_of)
    probes = []
    for s in np.unique(sess_of):
        members = np.flatnonzero(sess_of == s)
        for hop in (1, 2):
            for a in range(len(members) - hop):
                probes.append((int(members[a]), int(members[a + hop]), s, hop))
    return Z, sess_of, probes


def main():
    Z, sess_of, all_probes = build()
    n_dlg = len({s.split("/")[0] for s in sess_of})
    print(f"X101  MSC association: {len(Z)} turns, {len(set(sess_of))} sessions, {n_dlg} dialogues")
    out = stamp("X101", encoder="all-mpnet-base-v2", corpus="MSC test",
                bank=int(len(Z)), sessions=len(set(sess_of)), dialogues=n_dlg,
                folds=A.N_FOLDS, protocol="experiments/locomo_assoc.py (X26), imported")
    for hop in (1, 2):
        acc = {m: {"r1": [], "r5": []} for m in ("cosine", "averaged", "mixture")}
        clusters, chosen = [], []
        for train, val, test in A.folds(sess_of):
            tr = [(a, b) for a, b, s, h in all_probes if h == 1 and s in train]
            vp = A.subsample([p for p in all_probes if p[3] == hop and p[2] in val],
                             A.MAX_PROBES_PER_SESSION)[:A.VAL_CAP]
            tp = A.subsample([p for p in all_probes if p[3] == hop and p[2] in test],
                             A.MAX_PROBES_PER_SESSION)
            best, kt = -1.0, (A.KP_GRID[0], A.TAU_GRID[0])
            for kp in A.KP_GRID:
                for tau in A.TAU_GRID:
                    _, r5 = A.evaluate(Z, A.WM(Z, tr, kp, tau), vp, "averaged", hop, 1, 1.0, 1.0)
                    if r5.mean() > best:
                        best, kt = float(r5.mean()), (kp, tau)
            wm = A.WM(Z, tr, *kt)
            bb, kba = -1.0, (A.K_GRID[0], A.BETA_GRID[0], A.ALPHA_GRID[0])
            for K in A.K_GRID:
                for beta in A.BETA_GRID:
                    for alpha in A.ALPHA_GRID:
                        _, r5 = A.evaluate(Z, wm, vp, "mixture", hop, K, beta, alpha)
                        if r5.mean() > bb:
                            bb, kba = float(r5.mean()), (K, beta, alpha)
            chosen.append({"k_P": kt[0], "tau": kt[1], "K": kba[0], "beta": kba[1],
                           "alpha": kba[2], "n_train_transitions": len(tr), "n_val": len(vp)})
            clusters.extend(p[2] for p in tp)
            for m in acc:
                r1, r5 = A.evaluate(Z, wm, tp, m, hop, *kba)
                acc[m]["r1"].append(r1); acc[m]["r5"].append(r5)
        acc = {m: {k: np.concatenate(v) for k, v in d.items()} for m, d in acc.items()}
        clusters = np.array(clusters)
        print(f"\n  hop {hop}  n={len(clusters)} probes over {len(set(clusters))} sessions")
        res = {}
        for m in acc:
            r1, r5 = acc[m]["r1"], acc[m]["r5"]
            res[m] = {"R@1": float(r1.mean()), "R@5": float(r5.mean())}
            if m != "cosine":
                d, lo, hi, p = cluster_ci(r1 - acc["cosine"]["r1"], clusters)
                d5, lo5, hi5, p5 = cluster_ci(r5 - acc["cosine"]["r5"], clusters)
                res[m] |= {"dR1": d, "lo": lo, "hi": hi, "p": p,
                           "dR5": d5, "lo5": lo5, "hi5": hi5, "p5": p5}
                print(f"  {m:>10} R@1 {r1.mean():.3f} R@5 {r5.mean():.3f}  "
                      f"dR@1 {d:+.4f} [{lo:+.4f}, {hi:+.4f}] p={p:.4f}  "
                      f"dR@5 {d5:+.4f} [{lo5:+.4f}, {hi5:+.4f}]")
            else:
                print(f"  {m:>10} R@1 {r1.mean():.3f} R@5 {r5.mean():.3f}")
        out[f"hop{hop}"] = {"n": int(len(clusters)), "n_sessions": len(set(clusters)),
                            "selected": chosen, "methods": res}
    dst = ROOT / "experiments/results/x101_msc.json"
    dst.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nwrote {dst}")


if __name__ == "__main__":
    main()
