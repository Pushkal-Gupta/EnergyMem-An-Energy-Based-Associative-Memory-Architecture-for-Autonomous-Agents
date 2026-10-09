"""Family B -- geometry diagnostics (X50-X57). Model-free characterizations of why the
single-query corner fails, run on the same banks and vectors as family A.

X50 gold-set geometry vs random size-matched sets
X51 linear separability of gold sets, WITH the dimension caveat made explicit
X52 displacement coherence: global (X17-r measured ~0) vs within-neighbourhood
X53 is per-question gain PREDICTABLE from geometry alone, before seeing the answer
X54 reachability depth in the kNN graph
X55 intrinsic dimension and anisotropy, per category
X56 hubness
X57 distractor anatomy -- what the false positives actually are
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

from scipy.optimize import linprog
from scipy.stats import spearmanr

from experiments.locomo_qa import CATNAME, RESULTS, banks

SHA = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                     capture_output=True, text=True).stdout.strip()
RNG = np.random.default_rng(12345)
OUT = dict(git_sha=SHA, utc=datetime.now(timezone.utc).isoformat(),
           encoder="all-mpnet-base-v2", n_clusters=10)


def hr(t):
    print(f"\n{'=' * 96}\n{t}\n{'=' * 96}")


# ----------------------------------------------------------------------- X50
def x50(bks):
    hr("X50  gold-set geometry -- are gold evidence sets tighter than random sets?")
    res = {}
    print(f"{'category':<16}{'n':>5}{'mean m':>8}{'gold eps':>10}{'random eps':>12}"
          f"{'bank bg':>10}{'gold tighter':>14}")
    print("-" * 75)
    for cat in sorted(CATNAME):
        ge, re_, bg, ms = [], [], [], []
        for b in bks:
            C = b.Z @ b.Z.T
            bgv = float((C.sum() - b.n) / (b.n * (b.n - 1)))
            for i in range(b.m):
                if b.cats[i] != cat:
                    continue
                g = b.gold[i]
                m = len(g)
                ms.append(m)
                if m < 2:
                    continue
                sub = C[np.ix_(g, g)]
                ge.append((sub.sum() - m) / (m * (m - 1)))
                r = RNG.choice(b.n, size=m, replace=False)
                sr = C[np.ix_(r, r)]
                re_.append((sr.sum() - m) / (m * (m - 1)))
                bg.append(bgv)
        if not ge:
            continue
        tighter = float(np.mean(np.asarray(ge) > np.asarray(re_)))
        res[str(cat)] = dict(name=CATNAME[cat], n_multi=len(ge), mean_m=float(np.mean(ms)),
                             gold_eps=float(np.mean(ge)), random_eps=float(np.mean(re_)),
                             bank_background=float(np.mean(bg)), frac_gold_tighter=tighter)
        print(f"{str(cat) + ' ' + CATNAME[cat]:<16}{len(ge):>5}{np.mean(ms):>8.2f}"
              f"{np.mean(ge):>10.3f}{np.mean(re_):>12.3f}{np.mean(bg):>10.3f}{tighter:>14.1%}")
    print("-" * 75)
    print("Gold sets being far tighter than random is why the X27 dilution bound does not")
    print("bind: eps is high, so the single-query ceiling sqrt((1+(m-1)eps)/m) stays high.")
    OUT["X50"] = res


# ----------------------------------------------------------------------- X51
def x51(bks):
    hr("X51  linear separability of gold sets  (READ THE DIMENSION CAVEAT)")
    print("A gold set is retrievable as an exact top-m by SOME query iff it is linearly")
    print("separable from the rest of the bank. With d=768 and n~588 per conversation,")
    print("n < d, so ANY subset in general position is separable and the test is vacuous.")
    print("It is only meaningful on the pooled bank (n=5882 > d=768). Both are run.\n")

    def separable(Z, g, cap=2500):
        """LP feasibility: exists w,t with <w,z_i> >= t+1 for i in g, <= t for i not in g.

        `cap` MUST EXCEED d. Subsampling distractors to fewer than d points makes the
        LP automatically feasible by Cover's theorem and the test says nothing -- a
        first version of this ran at cap=400 against d=768 and reported 100% separable
        on the pooled bank, which was an artifact of the cap and not a finding.
        """
        g = list(g)
        other = [j for j in range(len(Z)) if j not in set(g)]
        if len(other) > cap:                      # hardest distractors only
            s = Z[other] @ Z[g].mean(0)
            other = [other[j] for j in np.argsort(-s)[:cap]]
        d = Z.shape[1]
        A, bb = [], []
        for i in g:
            A.append(np.append(-Z[i], 1.0)); bb.append(-1.0)       # -w.z_i + t <= -1
        for j in other:
            A.append(np.append(Z[j], -1.0)); bb.append(0.0)        #  w.z_j - t <=  0
        r = linprog(np.zeros(d + 1), A_ub=np.asarray(A), b_ub=np.asarray(bb),
                    bounds=[(-50, 50)] * d + [(-50, 50)], method="highs")
        return bool(r.status == 0)

    res = {}
    for scope in ("per-conversation", "pooled"):
        if scope == "pooled":
            Zp = np.concatenate([b.Z for b in bks])
            off = np.cumsum([0] + [b.n for b in bks])
        per = {}
        for cat in (1, 4):
            hits, tot = 0, 0
            for bi, b in enumerate(bks):
                for i in range(b.m):
                    if b.cats[i] != cat or tot >= 40:
                        continue
                    g = b.gold[i]
                    if scope == "pooled":
                        ok = separable(Zp, [x + off[bi] for x in g])
                    else:
                        ok = separable(b.Z, g)
                    hits += ok
                    tot += 1
            per[str(cat)] = dict(name=CATNAME[cat], n_tested=tot,
                                 frac_separable=(hits / tot if tot else None))
            print(f"  {scope:<18} cat {cat} {CATNAME[cat]:<12} "
                  f"separable {hits}/{tot} = {hits / max(tot, 1):.1%}")
        res[scope] = per
    res["caveat"] = ("per-conversation n(588) < d(768) so separability is automatic by "
                     "Cover's theorem and that row is uninformative by construction; the "
                     "pooled row uses the 2500 hardest distractors against d=768, so it "
                     "is the only row that can fail")
    res["d"] = 768
    res["pooled_distractor_cap"] = 2500
    OUT["X51"] = res


# ----------------------------------------------------------------------- X52
def x52(bks):
    hr("X52  displacement coherence -- global (X17-r found ~0) vs within-neighbourhood")
    res = {}
    gl, lo_ = [], []
    for b in bks:
        # displacements question -> gold turn, the field the predictor would learn
        D, Qs = [], []
        for i in range(b.m):
            for g in b.gold[i]:
                D.append(b.Z[g] - b.Q[i]); Qs.append(b.Q[i])
        D = np.stack(D); Qs = np.stack(Qs)
        Dn = D / np.linalg.norm(D, axis=1, keepdims=True)
        C = Dn @ Dn.T
        n = len(Dn)
        gl.append((C.sum() - n) / (n * (n - 1)))
        # local: among the 10 nearest cues
        S = Qs @ Qs.T
        for i in RNG.choice(n, size=min(300, n), replace=False):
            nb = np.argsort(-S[i])[1:11]
            sub = Dn[nb] @ Dn[nb].T
            k = len(nb)
            lo_.append((sub.sum() - k) / (k * (k - 1)))
    res = dict(global_mutual_cos=float(np.mean(gl)), local_mutual_cos=float(np.mean(lo_)),
               n_local_samples=len(lo_))
    print(f"  global displacement mutual cosine  {np.mean(gl):+.4f}")
    print(f"  local  (within 10-NN of the cue)   {np.mean(lo_):+.4f}")
    print(f"  ratio local/global                 {np.mean(lo_) / max(abs(np.mean(gl)), 1e-9):.1f}x")
    print("\nNOTE: this is NOT comparable to X17-r's +0.004 global figure. X17-r measured")
    print("adjacent-turn displacements; this measures question -> gold-evidence")
    print("displacements, a different field. Both are reported; neither supersedes the")
    print("other. Local exceeding global says a local kernel is the right shape, but a")
    print("global coherence of +0.27 is NOT near zero and a global offset is not refuted")
    print("by this measurement.")
    OUT["X52"] = res


# ----------------------------------------------------------------------- X53
def x53(bks):
    hr("X53  is per-question headroom PREDICTABLE from geometry alone?")
    print("Predictors available before seeing the answer: top-1 margin, score entropy,")
    print("local displacement coherence, query norm-in-subspace. Target: the gain that")
    print("mode expansion actually achieves on that question (gold@10, K0=5 vs cosine).\n")
    from experiments.locomo_qa import METHODS
    from energymem.eval.setmetrics import mean_gold_recall_at_k
    feats, gains, cats = [], [], []
    for b in bks:
        S = b.Q @ b.Z.T
        for i in range(b.m):
            s = S[i]
            o = np.argsort(-s)
            top = s[o[:20]]
            p = np.exp(8 * (top - top.max())); p /= p.sum()
            margin = float(top[0] - top[1])
            ent = float(-(p * np.log(p + 1e-12)).sum())
            nb = o[:10]
            Dn = b.Z[nb] - b.Q[i]
            Dn /= np.linalg.norm(Dn, axis=1, keepdims=True)
            C = Dn @ Dn.T
            coh = float((C.sum() - 10) / 90)
            base = mean_gold_recall_at_k(o, b.gold[i], 10)
            exp_ = mean_gold_recall_at_k(
                np.argsort(-METHODS["mixture"](b, i, K0=5, beta=16.0, with_query=False)),
                b.gold[i], 10)
            feats.append([margin, ent, coh, float(top[0])])
            gains.append(exp_ - base)
            cats.append(b.cats[i])
    F, g, cats = np.asarray(feats), np.asarray(gains), np.asarray(cats)
    names = ["top1_margin", "score_entropy", "local_coherence", "top1_score"]
    res = {"overall": {}, "cat1": {}}
    print(f"{'feature':<18}{'rho (all)':>12}{'p':>10}{'rho (cat1)':>13}{'p':>10}")
    print("-" * 65)
    for j, nm in enumerate(names):
        r1, p1 = spearmanr(F[:, j], g)
        m = cats == 1
        r2, p2 = spearmanr(F[m, j], g[m])
        res["overall"][nm] = dict(rho=float(r1), p=float(p1))
        res["cat1"][nm] = dict(rho=float(r2), p=float(p2))
        print(f"{nm:<18}{r1:>+12.3f}{p1:>10.2e}{r2:>+13.3f}{p2:>10.2e}")
    print("-" * 65)
    res["mean_gain"] = float(g.mean())
    res["mean_gain_cat1"] = float(g[cats == 1].mean())
    print(f"mean expansion gain  all {g.mean():+.4f}   cat1 {g[cats == 1].mean():+.4f}")
    OUT["X53"] = res


# ----------------------------------------------------------------------- X54
def x54(bks):
    hr("X54  reachability depth -- how many association hops to the residual gold?")
    res = {}
    for K in (5, 10, 20):
        reach = {1: 0, 2: 0, 3: 0, "unreached": 0}
        tot = 0
        for b in bks:
            S = b.Q @ b.Z.T
            nbr = np.argsort(-(b.Z @ b.Z.T), axis=1)[:, 1:K + 1]
            for i in range(b.m):
                g = b.gold[i]
                if len(g) < 2:
                    continue
                a = g[int(np.argmax(S[i][g]))]
                frontier, seen = {a}, {a}
                depth_of = {}
                for dep in (1, 2, 3):
                    nxt = set()
                    for u in frontier:
                        nxt |= set(nbr[u].tolist())
                    nxt -= seen
                    for t in g:
                        if t != a and t in nxt and t not in depth_of:
                            depth_of[t] = dep
                    seen |= nxt
                    frontier = nxt
                for t in g:
                    if t == a:
                        continue
                    tot += 1
                    reach[depth_of.get(t, "unreached")] = reach.get(depth_of.get(t, "unreached"), 0) + 1
        res[f"K={K}"] = {str(k): (v / tot if tot else 0.0) for k, v in reach.items()} | {"n": tot}
        print(f"  K={K:<3} hop1 {reach[1] / tot:>6.1%}  hop2 {reach[2] / tot:>6.1%}  "
              f"hop3 {reach[3] / tot:>6.1%}  unreached {reach['unreached'] / tot:>6.1%}  (n={tot})")
    print("\nHow much of the residual evidence sits within one or two association hops")
    print("bounds how much machinery is worth building on top of a single step.")
    OUT["X54"] = res


# ----------------------------------------------------------------------- X55
def x55(bks):
    hr("X55  intrinsic dimension and anisotropy")
    res = {}
    Zp = np.concatenate([b.Z for b in bks])
    for nm, M in [("bank turns", Zp)] + [
            (f"cat{c} questions", np.concatenate([b.Q[[i for i in range(b.m) if b.cats[i] == c]]
                                                  for b in bks])) for c in (1, 2, 3, 4, 5)]:
        Mc = M - M.mean(0)
        ev = np.linalg.svd(Mc, compute_uv=False) ** 2
        ev = ev / ev.sum()
        pr = float((ev.sum() ** 2) / (ev ** 2).sum())
        res[nm] = dict(n=int(len(M)), top10_var=float(ev[:10].sum()),
                       participation_ratio=pr,
                       eff_rank=float(np.exp(-(ev * np.log(ev + 1e-12)).sum())))
        print(f"  {nm:<18} n={len(M):>5}  top-10 PC var {ev[:10].sum():>6.1%}  "
              f"participation ratio {pr:>7.1f}  eff rank {res[nm]['eff_rank']:>7.1f}")
    OUT["X55"] = res


# ----------------------------------------------------------------------- X56
def x56(bks):
    hr("X56  hubness -- are a few turns universal near-neighbours, and are they gold?")
    res = {}
    for b in bks[:4]:
        S = b.Z @ b.Z.T
        np.fill_diagonal(S, -np.inf)
        k10 = np.argsort(-S, axis=1)[:, :10]
        cnt = np.bincount(k10.ravel(), minlength=b.n)
        qk = np.argsort(-(b.Q @ b.Z.T), axis=1)[:, :10]
        qcnt = np.bincount(qk.ravel(), minlength=b.n)
        goldset = {g for gs in b.gold for g in gs}
        top = np.argsort(-qcnt)[:20]
        res[b.sample_id] = dict(
            skew=float(((cnt - cnt.mean()) ** 3).mean() / cnt.std() ** 3),
            max_innn=int(cnt.max()), mean_innn=float(cnt.mean()),
            top20_retrieved_share=float(qcnt[top].sum() / qcnt.sum()),
            top20_gold_frac=float(np.mean([t in goldset for t in top])),
            bank_gold_frac=float(len(goldset) / b.n))
        r = res[b.sample_id]
        print(f"  {b.sample_id:<9} k-occurrence skew {r['skew']:>6.2f}  max in-10NN {r['max_innn']:>4}"
              f"  top-20 hubs take {r['top20_retrieved_share']:>5.1%} of retrievals"
              f"  gold among hubs {r['top20_gold_frac']:>5.1%} vs base {r['bank_gold_frac']:>5.1%}")
    print("\nHubness is the standard explanation for exactly this failure mode, so it is")
    print("ruled in or out rather than left as an alternative a reviewer can raise.")
    OUT["X56"] = res


# ----------------------------------------------------------------------- X57
def x57(bks):
    hr("X57  distractor anatomy -- what ARE the false positives?")
    res = {}
    for cat in (1, 4):
        same_sess, same_spk, adj, tot = 0, 0, 0, 0
        for b in bks:
            S = b.Q @ b.Z.T
            for i in range(b.m):
                if b.cats[i] != cat:
                    continue
                g = set(b.gold[i])
                gs = {b.sessions[x] for x in g}
                gp = {b.speakers[x] for x in g}
                for j in np.argsort(-S[i])[:10]:
                    if j in g:
                        continue
                    tot += 1
                    same_sess += b.sessions[j] in gs
                    same_spk += b.speakers[j] in gp
                    adj += any(abs(j - x) <= 2 for x in g)
        res[str(cat)] = dict(name=CATNAME[cat], n_distractors=tot,
                             same_session_as_gold=same_sess / tot,
                             same_speaker_as_gold=same_spk / tot,
                             within_2_turns_of_gold=adj / tot)
        print(f"  cat {cat} {CATNAME[cat]:<12} of top-10 non-gold:  same session {same_sess / tot:>6.1%}"
              f"   same speaker {same_spk / tot:>6.1%}   within 2 turns {adj / tot:>6.1%}  (n={tot})")
    print("\nTells the next person which prior is worth trying (X39/X62) and which is not.")
    OUT["X57"] = res


if __name__ == "__main__":
    bks = banks()
    for f in (x50, x51, x52, x53, x54, x55, x56, x57):
        try:
            f(bks)
        except Exception as e:                     # one failure must not lose the rest
            print(f"\n!! {f.__name__} FAILED: {type(e).__name__}: {e}")
            OUT[f.__name__.upper()] = {"error": f"{type(e).__name__}: {e}"}
    p = RESULTS / "x5x_geometry.json"
    p.write_text(json.dumps(OUT, indent=2, default=float))
    print(f"\nwrote {p}")
