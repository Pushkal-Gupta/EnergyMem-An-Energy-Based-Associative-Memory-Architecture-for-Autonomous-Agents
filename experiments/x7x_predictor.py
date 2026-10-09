"""Family D -- predictor variants (X70-X76) on LoCoMo QA.

The world model is fit on the NINE training conversations' (question -> gold evidence)
displacements and applied to the held-out tenth. Conversations are disjoint banks, so
this is a genuine generalization test and not a within-bank fit: X17-r found the
parametric predictor at fitted 1.000 / unseen 0.000, so the distinction is the whole
point.

X70 kNN world model vs the parametric JEPA predictor
X71 modes() vs predict() -- distribution preserved vs collapsed, at matched alpha
X72 rollout horizon H (X26 found 2-hop unsupported on a CONSTRUCTED task; cat 1 is a
    real multi-evidence task with gold labels)
X73 step-size calibration -- the alpha confound, with mean_step_norm reported
X74 transition definition: QA pairs vs adjacent turns vs same-session pairs
X75 repulsive (Stein-like) modes -- the principled version of diversity
X76 matching-pursuit readout over predicted modes
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
from energymem.readout import averaged_scores, mixture_scores
from energymem.world.knn_wm import KNNWorldModel
from experiments.locomo_qa import RESULTS, banks, folds

SHA = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                     capture_output=True, text=True).stdout.strip()
OUT = dict(git_sha=SHA, utc=datetime.now(timezone.utc).isoformat(),
           encoder="all-mpnet-base-v2", n_clusters=10,
           protocol="leave-one-conversation-out; predictor fit on train conversations only")


def hr(t):
    print(f"\n{'=' * 96}\n{t}\n{'=' * 96}")


def qa_pairs(bks, idxs, kind="qa"):
    """Cue/target vector pairs from the given conversations."""
    C, D = [], []
    for j in idxs:
        b = bks[j]
        if kind == "qa":
            for i in range(b.m):
                for g in b.gold[i]:
                    C.append(b.Q[i]); D.append(b.Z[g] - b.Q[i])
        elif kind == "adjacent":
            for t in range(b.n - 1):
                C.append(b.Z[t]); D.append(b.Z[t + 1] - b.Z[t])
        elif kind == "same_session":
            sess = np.asarray(b.sessions)
            for s in np.unique(sess):
                ix = np.flatnonzero(sess == s)
                for a in ix[:40]:
                    for c in ix[:40]:
                        if a != c:
                            C.append(b.Z[a]); D.append(b.Z[c] - b.Z[a])
    return np.asarray(C), np.asarray(D)


def eval_pred(bks, fit_fn, score_fn, cats=None):
    """Leave-one-conversation-out: fit on train, score held-out."""
    vv, cc, qc = [], [], []
    for train, test in folds(bks):
        model = fit_fn(bks, train)
        b = bks[test]
        for i in range(b.m):
            if cats is not None and b.cats[i] not in cats:
                continue
            order = np.argsort(-score_fn(model, b, i))
            vv.append(mean_gold_recall_at_k(order, b.gold[i], 10))
            cc.append(b.sample_id); qc.append(b.cats[i])
    return np.asarray(vv), np.asarray(cc), np.asarray(qc)


def baseline(bks, cats=None):
    vv, cc = [], []
    for b in bks:
        S = b.Q @ b.Z.T
        for i in range(b.m):
            if cats is not None and b.cats[i] not in cats:
                continue
            vv.append(mean_gold_recall_at_k(np.argsort(-S[i]), b.gold[i], 10))
            cc.append(b.sample_id)
    return np.asarray(vv), np.asarray(cc)


def fit_knn(k, tau, kind="qa"):
    def f(bks, train):
        C, D = qa_pairs(bks, train, kind)
        wm = KNNWorldModel(k=k, tau=tau)
        wm.Zc, wm.D = C, D
        return wm
    return f


def mk_score(K, alpha, beta, rule="mixture"):
    def f(wm, b, i):
        hyps = wm.modes(b.Q[i], K=K, alpha=alpha)
        if rule == "mixture":
            return mixture_scores(hyps, b.Z, beta)
        if rule == "averaged":
            return averaged_scores(hyps, b.Z)
        if rule == "predict":
            return b.Z @ l2_normalize(sum(w * x for x, w in hyps))
        raise ValueError(rule)
    return f


# ----------------------------------------------------------------------- X70/X71/X73
def x70_71_73(bks):
    hr("X70/X71/X73  kNN world model on QA displacements: modes vs collapse, alpha swept")
    print("alpha scales the displacement. A raw displacement has norm ~1.17 against a unit")
    print("cue, so alpha=1 lands on a stored training turn; the kernel average implicitly")
    print("applies alpha~0.34. Comparing modes to the average at alpha=1 confounds")
    print("distribution-vs-point with step size, so both are swept.\n")
    base, bc = baseline(bks)
    res = {"cosine_gold10": float(base.mean())}
    print(f"{'rule':<12}{'K':>4}{'alpha':>7}{'gold@10':>10}{'delta vs cosine':>28}{'p':>9}")
    print("-" * 72)
    for rule in ("mixture", "averaged", "predict"):
        for K in (1, 3, 5):
            for alpha in (0.1, 0.34, 1.0):
                if rule == "predict" and K == 1:
                    continue
                v, c, _ = eval_pred(bks, fit_knn(10, 5.0), mk_score(K, alpha, 16.0, rule))
                r = paired_report(v, base, bc, n_boot=2000, n_perm=5000)
                res[f"{rule}/K={K}/a={alpha}"] = r
                star = "  *" if r["p"] < 0.05 and r["delta"] > 0 else ""
                print(f"{rule:<12}{K:>4}{alpha:>7.2f}{r['a']:>10.3f}"
                      f"  {r['delta']:>+8.4f} [{r['lo']:>+.4f},{r['hi']:>+.4f}]{r['p']:>9.3f}{star}")
    print("-" * 72)
    print(f"cosine reference gold@10 = {base.mean():.3f}")
    # step-size diagnostic (X73)
    C, D = qa_pairs(bks, list(range(9)), "qa")
    wm = KNNWorldModel(k=10, tau=5.0); wm.Zc, wm.D = C, D
    ms = [wm.mean_step_norm(bks[9].Q[i]) for i in range(0, bks[9].m, 7)]
    per = float(np.mean([a for a, _ in ms])); avg = float(np.mean([b for _, b in ms]))
    res["step_norms"] = dict(mean_per_mode=per, mean_of_average=avg,
                             implied_alpha=avg / max(per, 1e-9))
    print(f"X73  mean per-mode |Delta| = {per:.3f}   |sum w Delta| = {avg:.3f}"
          f"   implied alpha = {avg / per:.3f}")
    OUT["X70_X71_X73"] = res


# ----------------------------------------------------------------------- X72
def x72(bks):
    hr("X72  rollout horizon -- does a second predicted hop help on REAL multi-evidence?")
    base, bc = baseline(bks, cats=(1,))
    res = {"cosine_gold10_cat1": float(base.mean())}
    print(f"{'H':>4}{'K':>4}{'gold@10 (cat1)':>17}{'delta vs cosine':>28}{'p':>9}")
    print("-" * 62)
    for H in (1, 2, 3):
        for K in (2, 3):
            def sc(wm, b, i, H=H, K=K):
                hyps = wm.rollout(b.Q[i], H=H, K=K, S=8, alpha=0.34)
                return mixture_scores(hyps, b.Z, 16.0)
            try:
                v, c, _ = eval_pred(bks, fit_knn(10, 5.0), sc, cats=(1,))
                r = paired_report(v, base, bc, n_boot=2000, n_perm=5000)
                res[f"H={H}/K={K}"] = r
                print(f"{H:>4}{K:>4}{r['a']:>17.3f}  {r['delta']:>+8.4f} "
                      f"[{r['lo']:>+.4f},{r['hi']:>+.4f}]{r['p']:>9.3f}"
                      f"{'  *' if r['p'] < 0.05 and r['delta'] > 0 else ''}")
            except Exception as e:
                print(f"{H:>4}{K:>4}   FAILED {type(e).__name__}: {e}")
                res[f"H={H}/K={K}"] = {"error": str(e)}
    print("-" * 62)
    print("X26 found 2-hop composition unsupported on a constructed task "
          "(-0.002 [-0.008,+0.004] p=0.77).")
    OUT["X72"] = res


# ----------------------------------------------------------------------- X74
def x74(bks):
    hr("X74  which transitions should the world model learn from?")
    base, bc = baseline(bks)
    res = {}
    print(f"{'transitions':<16}{'pairs':>9}{'gold@10':>10}{'delta vs cosine':>28}{'p':>9}")
    print("-" * 74)
    for kind in ("qa", "adjacent", "same_session"):
        C, _ = qa_pairs(bks, [0], kind)
        v, c, _ = eval_pred(bks, fit_knn(10, 5.0, kind), mk_score(3, 0.34, 16.0, "mixture"))
        r = paired_report(v, base, bc, n_boot=2000, n_perm=5000)
        res[kind] = r | {"pairs_per_conv": int(len(C))}
        print(f"{kind:<16}{len(C):>9}{r['a']:>10.3f}  {r['delta']:>+8.4f} "
              f"[{r['lo']:>+.4f},{r['hi']:>+.4f}]{r['p']:>9.3f}"
              f"{'  *' if r['p'] < 0.05 and r['delta'] > 0 else ''}")
    print("-" * 74)
    print("'qa' uses question->gold displacements: LABEL-DERIVED, so it is fit on the nine")
    print("training conversations only. 'adjacent' and 'same_session' need no labels.")
    OUT["X74"] = res


# ----------------------------------------------------------------------- X75
def x75(bks):
    hr("X75  repulsive modes -- principled diversity instead of MMR's heuristic")
    base, bc = baseline(bks)
    res = {}
    print(f"{'repulsion':<12}{'K':>4}{'gold@10':>10}{'delta vs cosine':>28}{'p':>9}")
    print("-" * 68)
    for rep in (0.0, 0.3, 0.6):
        for K in (3, 5):
            def sc(wm, b, i, rep=rep, K=K):
                hyps = wm.modes(b.Q[i], K=K, alpha=0.34)
                X = np.stack([h for h, _ in hyps]); w = np.array([x for _, x in hyps])
                if rep > 0 and len(X) > 1:
                    G = X @ X.T
                    np.fill_diagonal(G, 0.0)
                    X = np.stack([l2_normalize(X[s] - rep * (G[s][:, None] * X).sum(0) / max(len(X) - 1, 1))
                                  for s in range(len(X))])
                return mixture_scores(list(zip(X, w)), b.Z, 16.0)
            v, c, _ = eval_pred(bks, fit_knn(10, 5.0), sc)
            r = paired_report(v, base, bc, n_boot=2000, n_perm=5000)
            res[f"rep={rep}/K={K}"] = r
            print(f"{rep:<12.1f}{K:>4}{r['a']:>10.3f}  {r['delta']:>+8.4f} "
                  f"[{r['lo']:>+.4f},{r['hi']:>+.4f}]{r['p']:>9.3f}"
                  f"{'  *' if r['p'] < 0.05 and r['delta'] > 0 else ''}")
    print("-" * 68)
    OUT["X75"] = res


if __name__ == "__main__":
    bks = banks()
    for f in (x70_71_73, x72, x74, x75):
        try:
            f(bks)
        except Exception as e:
            print(f"\n!! {f.__name__} FAILED: {type(e).__name__}: {e}")
            OUT[f.__name__.upper()] = {"error": f"{type(e).__name__}: {e}"}
    p = RESULTS / "x7x_predictor.json"
    p.write_text(json.dumps(OUT, indent=2, default=float))
    print(f"\nwrote {p}")
