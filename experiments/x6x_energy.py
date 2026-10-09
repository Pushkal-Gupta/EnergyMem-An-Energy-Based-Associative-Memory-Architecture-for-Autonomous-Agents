"""Family C -- the energy axis (X60-X67).

Proposition 1: dense top-k retrieval is EXACTLY the (T=1, beta->inf, b=0, R=I) corner of
the modern Hopfield update. Corollary 1: at high beta stored patterns are exact fixed
points, so multi-step descent == one-step == cosine. Every experiment here looks for a
regime where that degeneracy LIFTS -- i.e. where the energy machinery buys something that
cosine cannot have.

X60 beta leverage band on LoCoMo QA, by top-1 DECISION AGREEMENT (not accuracy)
X61 multi-step descent at low beta -- the only place T can matter
X62 strength bias b: recency / frequency / session priors
X63 routing R: the least-explored knob; R=I is the corner
X64 iterated / parallel multi-retrieval and matching pursuit
X65 per-query adaptive beta
X66 descent trace -- is descent doing anything at all?
X67 joint beta x T sweep on accuracy
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

from energymem.base import l2_normalize
from energymem.eval.setmetrics import mean_gold_recall_at_k, paired_report
from experiments.locomo_qa import CATNAME, RESULTS, banks

SHA = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                     capture_output=True, text=True).stdout.strip()
OUT = dict(git_sha=SHA, utc=datetime.now(timezone.utc).isoformat(),
           encoder="all-mpnet-base-v2", n_clusters=10)
BETAS = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0)


def hr(t):
    print(f"\n{'=' * 96}\n{t}\n{'=' * 96}")


def descend(Z, q, beta, T, b=None, R=None, eta=0.0, trace=False):
    """One cue down the energy landscape. Z is (N,d) unit rows; returns the final state.

    xi <- normalize( Z^T softmax( beta (Z xi + b) ) ),  T times.
    T=1 with beta->inf returns the nearest stored pattern; T=1 at finite beta returns a
    softmax-weighted blend. This is the same update as EnergyMemory._descend, vectorized
    over a single query for speed.
    """
    xi = l2_normalize(q.astype(np.float64))
    tr = [xi.copy()] if trace else None
    bb = 0.0 if b is None else b
    for _ in range(T):
        a = beta * (Z @ xi + bb)
        a -= a.max()
        p = np.exp(a)
        p /= p.sum()
        nxt = Z.T @ p
        if R is not None and eta > 0:
            nxt = (1 - eta) * nxt + eta * (R @ nxt)
        nxt = l2_normalize(nxt)
        if trace:
            tr.append(nxt.copy())
        if np.linalg.norm(nxt - xi) < 1e-9:
            xi = nxt
            break
        xi = nxt
    return (xi, tr) if trace else xi


def score_all(b, beta, T, bias=None, R=None, eta=0.0):
    return np.stack([b.Z @ descend(b.Z, b.Q[i], beta, T, bias, R, eta) for i in range(b.m)])


# ----------------------------------------------------------------------- X60
def x60(bks):
    hr("X60  beta leverage band on LoCoMo QA -- top-1 DECISION AGREEMENT with cosine")
    print("Agreement, not accuracy: Corollary 1 says the energy path must CONVERGE to")
    print("cosine as beta grows. Where agreement is <100% is the only place beta can act.\n")
    res = {}
    print(f"{'beta':>8}{'agree T=1':>12}{'agree T=8':>12}{'gold@10 T=1':>14}{'gold@10 T=8':>14}")
    print("-" * 60)
    base_top, base_g = {}, []
    for b in bks:
        S = b.Q @ b.Z.T
        base_top[b.sample_id] = np.argmax(S, axis=1)
        base_g += [mean_gold_recall_at_k(np.argsort(-S[i]), b.gold[i], 10) for i in range(b.m)]
    res["cosine_gold10"] = float(np.mean(base_g))
    for beta in BETAS:
        row = {}
        for T in (1, 8):
            ag, gg = [], []
            for b in bks:
                S = score_all(b, beta, T)
                ag.append(np.argmax(S, axis=1) == base_top[b.sample_id])
                gg += [mean_gold_recall_at_k(np.argsort(-S[i]), b.gold[i], 10) for i in range(b.m)]
            row[T] = dict(agreement=float(np.concatenate(ag).mean()), gold10=float(np.mean(gg)))
        res[str(beta)] = row
        print(f"{beta:>8.0f}{row[1]['agreement']:>12.1%}{row[8]['agreement']:>12.1%}"
              f"{row[1]['gold10']:>14.3f}{row[8]['gold10']:>14.3f}")
    print("-" * 60)
    print(f"cosine reference gold@10 = {res['cosine_gold10']:.3f}")
    OUT["X60"] = res


# ----------------------------------------------------------------------- X61
def x61(bks):
    hr("X61  does T>1 ever beat T=1?  (Corollary 1 says only possibly at LOW beta)")
    res = {}
    print(f"{'beta':>8}" + "".join(f"{'T=' + str(t):>10}" for t in (1, 2, 3, 5, 10)))
    print("-" * 58)
    for beta in (1.0, 2.0, 4.0, 8.0, 16.0, 64.0):
        row, vals = {}, {}
        for T in (1, 2, 3, 5, 10):
            gg, cl = [], []
            for b in bks:
                S = score_all(b, beta, T)
                gg += [mean_gold_recall_at_k(np.argsort(-S[i]), b.gold[i], 10) for i in range(b.m)]
                cl += [b.sample_id] * b.m
            row[f"T={T}"] = float(np.mean(gg))
            vals[T] = (np.asarray(gg), np.asarray(cl))
        r = paired_report(vals[10][0], vals[1][0], vals[1][1], n_boot=2000, n_perm=5000)
        row["T10_minus_T1"] = r
        res[str(beta)] = row
        print(f"{beta:>8.0f}" + "".join(f"{row['T=' + str(t)]:>10.3f}" for t in (1, 2, 3, 5, 10))
              + f"   T10-T1 {r['delta']:+.4f} p={r['p']:.3f}")
    print("-" * 58)
    OUT["X61"] = res


# ----------------------------------------------------------------------- X62
def x62(bks):
    hr("X62  strength bias b -- recency, frequency and session priors (never swept before)")
    res = {}
    print(f"{'prior':<20}{'scale':>8}{'gold@10':>10}{'delta vs b=0':>26}{'p':>9}")
    print("-" * 75)
    base_v, base_c = [], []
    for b in bks:
        S = b.Q @ b.Z.T
        base_v += [mean_gold_recall_at_k(np.argsort(-S[i]), b.gold[i], 10) for i in range(b.m)]
        base_c += [b.sample_id] * b.m
    base_v, base_c = np.asarray(base_v), np.asarray(base_c)
    res["cosine"] = float(base_v.mean())
    for name in ("recency", "centrality", "session_size", "length"):
        for scale in (0.05, 0.15, 0.5):
            vv = []
            for b in bks:
                if name == "recency":
                    pr = np.arange(b.n) / b.n
                elif name == "centrality":
                    pr = (b.Z @ b.Z.mean(0))
                elif name == "session_size":
                    _, inv, cnt = np.unique(b.sessions, return_inverse=True, return_counts=True)
                    pr = cnt[inv] / cnt.max()
                else:
                    L = np.array([len(t) for t in b.texts], float)
                    pr = L / L.max()
                pr = (pr - pr.mean()) / (pr.std() + 1e-9)
                S = b.Z @ b.Q.T                       # (N,M)
                S = (S + scale * pr[:, None]).T
                vv += [mean_gold_recall_at_k(np.argsort(-S[i]), b.gold[i], 10) for i in range(b.m)]
            r = paired_report(np.asarray(vv), base_v, base_c, n_boot=2000, n_perm=5000)
            res[f"{name}@{scale}"] = r
            print(f"{name:<20}{scale:>8.2f}{r['a']:>10.3f}"
                  f"  {r['delta']:>+8.4f} [{r['lo']:>+.4f},{r['hi']:>+.4f}]{r['p']:>9.3f}"
                  f"{'  *' if r['p'] < 0.05 and r['delta'] > 0 else ''}")
    print("-" * 75)
    OUT["X62"] = res


# ----------------------------------------------------------------------- X63
def x63(bks):
    hr("X63  routing R -- R=I is the corner; does any learned R beat it?")
    print("R is fit on the NINE training conversations (ridge from question to gold-turn")
    print("centroid) and applied to the held-out one. Leave-one-conversation-out.\n")
    res = {}
    print(f"{'variant':<22}{'lambda':>9}{'gold@10':>10}{'delta vs R=I':>26}{'p':>9}")
    print("-" * 78)
    base_v, base_c = [], []
    for b in bks:
        S = b.Q @ b.Z.T
        base_v += [mean_gold_recall_at_k(np.argsort(-S[i]), b.gold[i], 10) for i in range(b.m)]
        base_c += [b.sample_id] * b.m
    base_v, base_c = np.asarray(base_v), np.asarray(base_c)
    for lam in (1.0, 10.0, 100.0):
        for mode in ("full", "diag"):
            vv, cc = [], []
            for ti in range(len(bks)):
                A, B = [], []
                for tj, b in enumerate(bks):
                    if tj == ti:
                        continue
                    for i in range(b.m):
                        A.append(b.Q[i]); B.append(b.Z[b.gold[i]].mean(0))
                A, B = np.asarray(A), np.asarray(B)
                d = A.shape[1]
                if mode == "full":
                    R = np.linalg.solve(A.T @ A + lam * np.eye(d), A.T @ B)
                else:
                    num = (A * B).sum(0); den = (A * A).sum(0) + lam
                    R = np.diag(num / den)
                b = bks[ti]
                Qr = np.stack([l2_normalize(R.T @ q) for q in b.Q])
                S = Qr @ b.Z.T
                vv += [mean_gold_recall_at_k(np.argsort(-S[i]), b.gold[i], 10) for i in range(b.m)]
                cc += [b.sample_id] * b.m
            r = paired_report(np.asarray(vv), base_v, base_c, n_boot=2000, n_perm=5000)
            res[f"{mode}@{lam}"] = r
            print(f"{'R ' + mode:<22}{lam:>9.0f}{r['a']:>10.3f}"
                  f"  {r['delta']:>+8.4f} [{r['lo']:>+.4f},{r['hi']:>+.4f}]{r['p']:>9.3f}"
                  f"{'  *' if r['p'] < 0.05 and r['delta'] > 0 else ''}")
    print("-" * 78)
    print(f"R=I reference gold@10 = {base_v.mean():.3f}")
    OUT["X63"] = res


# ----------------------------------------------------------------------- X64
def x64(bks):
    hr("X64  matching pursuit: retrieve, subtract, repeat -- does deflation beat top-k?")
    res = {}
    print(f"{'variant':<24}{'gold@10':>10}{'delta vs cosine':>26}{'p':>9}")
    print("-" * 70)
    base_v, base_c = [], []
    for b in bks:
        S = b.Q @ b.Z.T
        base_v += [mean_gold_recall_at_k(np.argsort(-S[i]), b.gold[i], 10) for i in range(b.m)]
        base_c += [b.sample_id] * b.m
    base_v, base_c = np.asarray(base_v), np.asarray(base_c)
    for rounds in (3, 5, 10):
        vv = []
        for b in bks:
            for i in range(b.m):
                q = b.Q[i].copy()
                picked, pool = [], set()
                for _ in range(rounds):
                    s = b.Z @ q
                    for j in picked:
                        s[j] = -np.inf
                    j = int(np.argmax(s))
                    picked.append(j); pool.add(j)
                    q = q - (q @ b.Z[j]) * b.Z[j]      # deflate the found direction
                    nq = np.linalg.norm(q)
                    if nq < 1e-8:
                        break
                    q = q / nq
                rest = [j for j in np.argsort(-(b.Z @ b.Q[i])) if j not in pool]
                vv.append(mean_gold_recall_at_k(picked + rest, b.gold[i], 10))
        r = paired_report(np.asarray(vv), base_v, base_c, n_boot=2000, n_perm=5000)
        res[f"rounds={rounds}"] = r
        print(f"{'matching pursuit r=' + str(rounds):<24}{r['a']:>10.3f}"
              f"  {r['delta']:>+8.4f} [{r['lo']:>+.4f},{r['hi']:>+.4f}]{r['p']:>9.3f}"
              f"{'  *' if r['p'] < 0.05 and r['delta'] > 0 else ''}")
    print("-" * 70)
    OUT["X64"] = res


# ----------------------------------------------------------------------- X65
def x65(bks):
    hr("X65  per-query adaptive beta -- set beta from the query's own score spread")
    res = {}
    print(f"{'rule':<26}{'gold@10':>10}{'delta vs best fixed':>26}{'p':>9}")
    print("-" * 72)
    fixed_v, fixed_c = [], []
    for b in bks:
        S = score_all(b, 16.0, 1)
        fixed_v += [mean_gold_recall_at_k(np.argsort(-S[i]), b.gold[i], 10) for i in range(b.m)]
        fixed_c += [b.sample_id] * b.m
    fixed_v, fixed_c = np.asarray(fixed_v), np.asarray(fixed_c)
    for rule, f in (("beta ~ 1/margin", lambda s: 1.0 / (s[0] - s[1] + 1e-3)),
                    ("beta ~ 1/std", lambda s: 1.0 / (s.std() + 1e-3)),
                    ("beta ~ margin", lambda s: 20.0 * (s[0] - s[1]))):
        vv = []
        for b in bks:
            for i in range(b.m):
                s = np.sort(b.Z @ b.Q[i])[::-1]
                beta = float(np.clip(f(s[:20]), 1.0, 512.0))
                xi = descend(b.Z, b.Q[i], beta, 1)
                vv.append(mean_gold_recall_at_k(np.argsort(-(b.Z @ xi)), b.gold[i], 10))
        r = paired_report(np.asarray(vv), fixed_v, fixed_c, n_boot=2000, n_perm=5000)
        res[rule] = r
        print(f"{rule:<26}{r['a']:>10.3f}  {r['delta']:>+8.4f} "
              f"[{r['lo']:>+.4f},{r['hi']:>+.4f}]{r['p']:>9.3f}"
              f"{'  *' if r['p'] < 0.05 and r['delta'] > 0 else ''}")
    print("-" * 72)
    print(f"fixed beta=16, T=1 reference gold@10 = {fixed_v.mean():.3f}")
    OUT["X65"] = res


# ----------------------------------------------------------------------- X66
def x66(bks):
    hr("X66  descent trace -- is descent doing anything, or is it a fixed point from step 1?")
    res = {}
    print(f"{'beta':>8}{'|xi_1-xi_0|':>14}{'|xi_8-xi_1|':>14}{'steps to converge':>20}"
          f"{'top1 changed':>14}")
    print("-" * 72)
    for beta in (1.0, 4.0, 16.0, 64.0, 256.0):
        d01, d18, nst, chg = [], [], [], []
        for b in bks[:4]:
            for i in range(0, b.m, 3):
                xi, tr = descend(b.Z, b.Q[i], beta, 8, trace=True)
                d01.append(np.linalg.norm(tr[1] - tr[0]))
                d18.append(np.linalg.norm(tr[-1] - tr[1]))
                nst.append(len(tr) - 1)
                chg.append(int(np.argmax(b.Z @ tr[1]) != np.argmax(b.Z @ tr[-1])))
        res[str(beta)] = dict(step1_move=float(np.mean(d01)), after_step1_move=float(np.mean(d18)),
                              mean_steps=float(np.mean(nst)), top1_changed=float(np.mean(chg)))
        print(f"{beta:>8.0f}{np.mean(d01):>14.4f}{np.mean(d18):>14.4f}{np.mean(nst):>20.2f}"
              f"{np.mean(chg):>14.1%}")
    print("-" * 72)
    print("If |xi_8 - xi_1| ~ 0 the extra steps are doing nothing: that is Corollary 1,")
    print("measured on the trajectory rather than argued from the algebra.")
    OUT["X66"] = res


if __name__ == "__main__":
    bks = banks()
    for f in (x60, x61, x62, x63, x64, x65, x66):
        try:
            f(bks)
        except Exception as e:
            print(f"\n!! {f.__name__} FAILED: {type(e).__name__}: {e}")
            OUT[f.__name__.upper()] = {"error": f"{type(e).__name__}: {e}"}
    p = RESULTS / "x6x_energy.json"
    p.write_text(json.dumps(OUT, indent=2, default=float))
    print(f"\nwrote {p}")
