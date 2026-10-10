"""Family A, the entries Phase 7 did not reach: X34-X37, X39-X41, and X42-X48 consolidated.

Every comparison here is against the fold-selected baseline on the HELD-OUT conversation,
with K0 kept strictly below the metric cutoff (the seventh protocol error) and the null
reachable in every grid (the X31 lesson). Intervals, not p-values, wherever the cluster is
the conversation: ten clusters floor the blocked permutation at 2/1024 (X99).

X34 displacement modes      X35 answer coverage        X36 cat-5 specificity
X37 bank scope              X39 metadata priors        X40 cross-session structure
X41 paraphrase modes        X42-X48 per-category table
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from experiments.locomo_qa import (                                   # noqa: E402
    CATNAME, KS, RESULTS, Bank, banks, evaluate, folds, fold_select, grid_points,
    s_cosine, s_mixture, s_rrf, sweep,
)
from energymem.base import l2_normalize                               # noqa: E402
from energymem.encoders import SentenceEncoder                        # noqa: E402
from energymem.eval.setmetrics import (                               # noqa: E402
    exact_blocked_permutation, hierarchical_bootstrap, mean_gold_recall_at_k,
    paired_report,
)
from energymem.readout import mixture_scores                          # noqa: E402
from energymem.world.knn_wm import KNNWorldModel                      # noqa: E402

OUT = RESULTS / "x3x_family_a.json"
K_CUTOFF = 10
R: dict = {}
STOP = set("a an the is are was were do does did what when where who whom which how why "
           "of in on at to for with about from and or if that this these those did's "
           "has have had been being be as by it its their they them he she his her".split())


def hr(t):
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


def clus(bks, cats=None):
    return np.concatenate([np.full(sum(1 for c in b.cats if cats is None or c in cats),
                                   b.sample_id) for b in bks])


def rep(a, b, c):
    r = paired_report(a, b, c, n_boot=4000, n_perm=20000)
    r["p_exact"] = exact_blocked_permutation(a - b, c)
    return r


def line(name, r):
    print(f"  {name:<34}{r['a']:>9.4f}{r['b']:>9.4f}{r['delta']:>+10.4f}"
          f"   [{r['lo']:>+.4f},{r['hi']:>+.4f}]{r['p_exact']:>9.4f}")


# ============================================================ X34 displacement
def fit_wm(bks, train, k=10, tau=5.0):
    """One world model over adjacent-turn transitions of the TRAINING conversations."""
    Z, tr = [], []
    off = 0
    for j in train:
        b = bks[j]
        Z.append(b.Z)
        tr += [(off + i, off + i + 1) for i in range(b.n - 1)]
        off += b.n
    return KNNWorldModel(k=k, tau=tau).fit(np.concatenate(Z), tr)


def x34(bks):
    hr("X34  displacement modes -- the X26 world-model offset added to the query")
    print("  alpha=0 is the null (the query alone). ||Delta||~1.157 against a unit cue, so")
    print("  alpha~0.34 is where the kernel average implicitly sits (X23b).\n")
    print(f"  {'alpha / cat':<34}{'disp':>9}{'cosine':>9}{'delta':>10}"
          f"{'95% CI':>22}{'p':>9}")
    rows = {}
    for alpha in (0.0, 0.17, 0.34, 0.68, 1.0):
        for cats, label in ((None, "all"), ({1}, "cat 1")):
            vals, base, cl = [], [], []
            for train, test in folds(bks):
                wm = fit_wm(bks, train)
                b = bks[test]
                for i in range(b.m):
                    if cats is not None and b.cats[i] not in cats:
                        continue
                    q = b.Q[i]
                    if alpha == 0.0:
                        qq = q
                    else:
                        nn, w = wm._neighbours(q)
                        qq = l2_normalize(q + alpha * (w[:, None] * wm.D[nn]).sum(0))
                    o = np.argsort(-(b.Z @ qq))
                    vals.append(mean_gold_recall_at_k(o, b.gold[i], 10))
                    base.append(mean_gold_recall_at_k(np.argsort(-(b.Z @ q)),
                                                      b.gold[i], 10))
                    cl.append(b.sample_id)
            r = rep(np.asarray(vals), np.asarray(base), np.asarray(cl))
            rows[f"alpha={alpha},{label}"] = r
            line(f"alpha={alpha:<5} {label}", r)
    pos = [k for k, v in rows.items() if v["lo"] > 0]
    print(f"\n  cells where the displacement beats the query alone: {len(pos)}/{len(rows)}")
    R["X34"] = dict(rows=rows, n_positive=len(pos))


# ============================================================ X35 answer coverage
def _norm(s):
    return re.sub(r"[^a-z0-9 ]", " ", s.lower())


def x35(bks):
    hr("X35  answer-string coverage -- a deterministic, API-free downstream PROXY")
    print("  Reported as COVERAGE, not QA accuracy: a substring hit is not an answer.")
    print("  The question that matters: does RRF's retrieval win survive into coverage?\n")
    print(f"  {'method':<14}" + "".join(f"{f'cov@{k}':>10}" for k in KS))
    cov = {}
    for name, fn, par in (("cosine", s_cosine, {}), ("rrf", s_rrf, {"k_rrf": 60.0}),
                          ("mixture K0=3", s_mixture, {"K0": 3, "beta": 16.0})):
        per = {k: [] for k in KS}
        cl = []
        for b in bks:
            for i in range(b.m):
                ans = _norm(b.answers[i]).strip()
                o = np.argsort(-fn(b, i, **par))
                for k in KS:
                    ctx = _norm(" ".join(b.texts[j] for j in o[:k]))
                    per[k].append(float(bool(ans) and ans in ctx))
                cl.append(b.sample_id)
        cov[name] = {k: np.asarray(v) for k, v in per.items()}
        print(f"  {name:<14}" + "".join(f"{cov[name][k].mean():>10.4f}" for k in KS))
    cl = np.asarray(cl)
    print(f"\n  {'vs cosine':<14}" + "".join(f"{f'd cov@{k}':>24}" for k in (5, 10)))
    out = {}
    for name in ("rrf", "mixture K0=3"):
        s = f"  {name:<14}"
        out[name] = {}
        for k in (5, 10):
            r = rep(cov[name][k], cov["cosine"][k], cl)
            out[name][f"cov@{k}"] = r
            s += f"{r['delta']:>+10.4f} [{r['lo']:>+.3f},{r['hi']:>+.3f}]"
        print(s)
    R["X35"] = dict(coverage={n: {str(k): float(v.mean()) for k, v in d.items()}
                              for n, d in cov.items()},
                    vs_cosine={n: {k: v for k, v in d.items()} for n, d in out.items()})


# ============================================================ X36 cat-5 specificity
def x36(bks):
    hr("X36  adversarial specificity (cat 5) -- does expansion make us confidently wrong?")
    print("  SCALE CONFOUND, HANDLED. A logsumexp score is not a cosine, so a raw top-1")
    print("  margin cannot be compared ACROSS methods and an absolute abstention")
    print("  threshold is meaningless across them. Two comparisons are valid and both are")
    print("  reported: the margin Z-SCORED within each query (cross-method), and the")
    print("  cat-5-minus-cat-4 contrast WITHIN a method (same scale by construction).")
    print("  The scale-free ground truth is the ranking itself, in X42-X48.\n")
    rows = {}
    print(f"  {'method':<16}{'cat':<14}{'z-margin':>10}{'raw margin':>12}"
          f"{'gold@10':>10}")
    for name, fn, par in (("cosine", s_cosine, {}),
                          ("mixture K0=3", s_mixture, {"K0": 3, "beta": 16.0})):
        for cats, label in (({5}, "5 adversarial"), ({1}, "1 multi-hop"),
                            ({4}, "4 single-hop")):
            zm, raw, gl, n = [], [], [], 0
            for b in bks:
                for i in range(b.m):
                    if b.cats[i] not in cats:
                        continue
                    sc = fn(b, i, **par)
                    ss = np.sort(sc)[::-1]
                    sd = float(sc.std()) or 1.0
                    zm.append(float((ss[0] - ss[1]) / sd))
                    raw.append(float(ss[0] - ss[1]))
                    gl.append(mean_gold_recall_at_k(np.argsort(-sc), b.gold[i], 10))
                    n += 1
            rows[f"{name}/{label}"] = dict(n=n, z_margin=float(np.mean(zm)),
                                           raw_margin=float(np.mean(raw)),
                                           gold10=float(np.mean(gl)))
            print(f"  {name:<16}{label:<14}{np.mean(zm):>10.4f}{np.mean(raw):>12.4f}"
                  f"{np.mean(gl):>10.4f}")
    print(f"\n  WITHIN-METHOD contrast (cat 5 minus cat 4), the comparison that is")
    print(f"  immune to the scale confound:")
    for name in ("cosine", "mixture K0=3"):
        dz = rows[f"{name}/5 adversarial"]["z_margin"] - rows[f"{name}/4 single-hop"]["z_margin"]
        dg = rows[f"{name}/5 adversarial"]["gold10"] - rows[f"{name}/4 single-hop"]["gold10"]
        rows[f"{name}/cat5_minus_cat4"] = dict(d_z_margin=dz, d_gold10=dg)
        print(f"    {name:<16} d z-margin {dz:>+8.4f}   d gold@10 {dg:>+8.4f}")
    print("\n  Read it as: a method is more confidently wrong on the adversarial set if")
    print("  its z-margin holds up there while its gold@10 falls further than cosine's.")
    R["X36"] = dict(rows=rows,
                    confound="raw margins and absolute abstention are not comparable "
                             "across methods; z-margin and within-method contrasts are")


# ============================================================ X37 bank scope
def x37(bks):
    hr("X37  bank scope -- per-conversation (588) vs pooled (5,882, namespaced ids)")
    Zp = np.concatenate([b.Z for b in bks])
    txt = [t for b in bks for t in b.texts]
    off = np.cumsum([0] + [b.n for b in bks])
    rows = {}
    for cats, label in ((None, "all"), ({1}, "cat 1"), ({4}, "cat 4")):
        for name, fn, par in (("cosine", s_cosine, {}),
                              ("mixture K0=3", s_mixture, {"K0": 3, "beta": 16.0})):
            per_conv, pooled, cl = [], [], []
            for bi, b in enumerate(bks):
                big = Bank(**{**b.__dict__, "_bm25": None})
                big.Z, big.texts = Zp, txt
                big.ids = [f"p{j}" for j in range(len(txt))]
                for i in range(b.m):
                    if cats is not None and b.cats[i] not in cats:
                        continue
                    per_conv.append(mean_gold_recall_at_k(
                        np.argsort(-fn(b, i, **par)), b.gold[i], 10))
                    g = [x + off[bi] for x in b.gold[i]]
                    pooled.append(mean_gold_recall_at_k(
                        np.argsort(-fn(big, i, **par)), g, 10))
                    cl.append(b.sample_id)
            r = rep(np.asarray(pooled), np.asarray(per_conv), np.asarray(cl))
            rows[f"{name}/{label}"] = r
            line(f"{name} {label} pooled-vs-own", r)
    print("\n  A 10x bank with the same gold set must cost accuracy. The question is")
    print("  whether expansion loses LESS than cosine does -- the scaling argument.")
    R["X37"] = rows


# ============================================================ X39 metadata priors
def x39(bks):
    hr("X39  metadata priors as a strength bias b -- the two X62 did not try")
    print("  X62 tested recency, centrality, session size and turn length: all hurt,")
    print("  monotonically. Same-speaker and session-adjacency are new, and cat 2")
    print("  (temporal) is the category they should help if anything does.\n")
    print(f"  {'prior / cat':<34}{'biased':>9}{'cosine':>9}{'delta':>10}"
          f"{'95% CI':>22}{'p':>9}")
    rows = {}
    for strength in (0.05, 0.15):
        for pname in ("same_speaker", "session_adjacent"):
            for cats, label in (({2}, "cat 2"), (None, "all")):
                vals, base, cl = [], [], []
                for b in bks:
                    sess = np.asarray([int(s.split("_")[1]) if "_" in s else 0
                                       for s in b.sessions])
                    spk = np.asarray(b.speakers)
                    for i in range(b.m):
                        if cats is not None and b.cats[i] not in cats:
                            continue
                        s = b.Z @ b.Q[i]
                        top = int(np.argmax(s))
                        if pname == "same_speaker":
                            bias = (spk == spk[top]).astype(float)
                        else:
                            bias = np.exp(-np.abs(sess - sess[top]).astype(float))
                        vals.append(mean_gold_recall_at_k(
                            np.argsort(-(s + strength * bias)), b.gold[i], 10))
                        base.append(mean_gold_recall_at_k(np.argsort(-s), b.gold[i], 10))
                        cl.append(b.sample_id)
                r = rep(np.asarray(vals), np.asarray(base), np.asarray(cl))
                rows[f"{pname}@{strength}/{label}"] = r
                line(f"{pname}@{strength} {label}", r)
    pos = [k for k, v in rows.items() if v["lo"] > 0]
    print(f"\n  priors that help with the interval excluding 0: {len(pos)}/{len(rows)}")
    R["X39"] = dict(rows=rows, n_positive=len(pos))


# ============================================================ X40 cross-session
def x40(bks):
    hr("X40  cross-session structure -- does anything concentrate where gold spans sessions?")
    rows = {}
    print(f"  {'subset':<26}{'n':>6}{'cosine':>9}{'rrf':>9}{'mixture':>9}")
    for span, label in ((True, "cat 1, gold spans >1 session"),
                        (False, "cat 1, gold within 1 session")):
        c, rr, mx, n = [], [], [], 0
        for b in bks:
            sess = [b.sessions[j] for j in range(b.n)]
            for i in range(b.m):
                if b.cats[i] != 1:
                    continue
                multi = len({sess[j] for j in b.gold[i]}) > 1
                if multi != span:
                    continue
                n += 1
                for arr, fn, par in ((c, s_cosine, {}), (rr, s_rrf, {"k_rrf": 60.0}),
                                     (mx, s_mixture, {"K0": 3, "beta": 16.0})):
                    arr.append(mean_gold_recall_at_k(
                        np.argsort(-fn(b, i, **par)), b.gold[i], 10))
        rows[label] = dict(n=n, cosine=float(np.mean(c)), rrf=float(np.mean(rr)),
                           mixture=float(np.mean(mx)))
        print(f"  {label:<26}{n:>6}{np.mean(c):>9.4f}{np.mean(rr):>9.4f}"
              f"{np.mean(mx):>9.4f}")
    small = min(v["n"] for v in rows.values())
    print(f"\n  the within-session cell has n={small}: UNDERPOWERED, descriptive only")
    R["X40"] = rows


# ============================================================ X41 paraphrase modes
def paraphrases(q: str) -> list[str]:
    """Deterministic template rewrites -- no LLM, so this stays reproducible."""
    core = re.sub(r"^(what|when|where|who|whom|which|how|why|did|does|do|is|are|was|were)\b",
                  "", q.strip(), flags=re.I).strip(" ?")
    kw = " ".join(w for w in re.findall(r"[A-Za-z0-9']+", q) if w.lower() not in STOP)
    return [f"In the conversation, {core}.", kw, f"Tell me about {core}.",
            core if core else q]


def x41(bks, enc):
    hr("X41  paraphrase modes -- the strongest alternative explanation on the table")
    print("  If expanding from PARAPHRASES of the question matches expanding from")
    print("  retrieved turns, the mechanism is query diversity, not memory association.\n")
    P = {}
    for b in bks:
        texts = [p for q in b.questions for p in paraphrases(q)]
        E = enc.encode(texts)
        P[b.sample_id] = E.reshape(len(b.questions), 4, -1)
    print(f"  {'variant / cat':<34}{'method':>9}{'cosine':>9}{'delta':>10}"
          f"{'95% CI':>22}{'p':>9}")
    rows = {}
    for beta in (16.0, 64.0):
        for cats, label in ((None, "all"), ({1}, "cat 1")):
            for src in ("paraphrase", "retrieved"):
                vals, base, cl = [], [], []
                for b in bks:
                    for i in range(b.m):
                        if cats is not None and b.cats[i] not in cats:
                            continue
                        q = b.Q[i]
                        if src == "paraphrase":
                            hyps = [(q, 1.0)] + [(l2_normalize(v), 0.25)
                                                 for v in P[b.sample_id][i]]
                        else:
                            t = np.argsort(-(b.Z @ q))[:3]
                            w = np.exp(beta * (b.Z[t] @ q)); w = w / w.sum()
                            hyps = [(q, 1.0)] + [(b.Z[j], float(wj) + 1e-12)
                                                 for j, wj in zip(t, w)]
                        vals.append(mean_gold_recall_at_k(
                            np.argsort(-mixture_scores(hyps, b.Z, beta)), b.gold[i], 10))
                        base.append(mean_gold_recall_at_k(
                            np.argsort(-(b.Z @ q)), b.gold[i], 10))
                        cl.append(b.sample_id)
                r = rep(np.asarray(vals), np.asarray(base), np.asarray(cl))
                rows[f"{src},beta={beta},{label}"] = r
                line(f"{src} beta={beta:g} {label}", r)
    R["X41"] = dict(rows=rows,
                    note="both mode sources compared against the same cosine null")


# ====================================================== X42-X48 per-category table
def x42_48(bks):
    hr("X42-X48  per-category table (consolidated -- seven IDs, one grid)")
    GRID = {"K0": [0, 1, 2, 3, 5], "beta": [4.0, 16.0, 64.0], "with_query": [True, False]}
    assert max(GRID["K0"]) < K_CUTOFF and 0 in GRID["K0"]
    tab = sweep(bks, s_mixture, GRID)
    rtab = sweep(bks, s_rrf, {"k_rrf": [10.0, 60.0, 120.0]})
    print(f"  {'category':<18}{'n':>6}{'cosine':>9}{'rrf':>9}{'mixture':>9}"
          f"{'null folds':>12}")
    rows = {}
    for cat in (1, 2, 3, 4, 5, None):
        mf = (lambda c: np.ones(len(c), bool)) if cat is None else \
             (lambda c, k=cat: np.asarray(c) == k)
        cos = np.concatenate([evaluate(b, s_cosine)["gold@10"][mf(np.asarray(b.cats))]
                              for b in bks])
        if cos.size == 0:
            continue
        def sel(t):
            v, pk = [], []
            for train, test in folds(bks):
                best, bs = None, -np.inf
                for k, pb in t.items():
                    parts = [pb[j]["gold@10"][mf(pb[j]["cat"])] for j in train]
                    sc = float(np.mean(np.concatenate(parts)))
                    if sc > bs:
                        best, bs = k, sc
                r = t[best][test]
                v.append(r["gold@10"][mf(r["cat"])]); pk.append(best)
            return np.concatenate(v), pk
        mv, mpk = sel(tab)
        rv, _ = sel(rtab)
        nulls = sum(1 for p in mpk if "K0=0" in p)
        name = CATNAME.get(cat, "all pooled") if cat else "all pooled"
        rows[str(cat)] = dict(name=name, n=int(cos.size), cosine=float(cos.mean()),
                              rrf=float(rv.mean()), mixture=float(mv.mean()),
                              null_folds=nulls)
        print(f"  {name:<18}{cos.size:>6}{cos.mean():>9.4f}{rv.mean():>9.4f}"
              f"{mv.mean():>9.4f}{nulls:>9}/10")
    # X48: the questions with the largest gold sets, where a single query must fail worst
    big = [(b, i) for b in bks for i in range(b.m) if len(b.gold[i]) >= 8]
    if big:
        c = [mean_gold_recall_at_k(np.argsort(-s_cosine(b, i)), b.gold[i], 10)
             for b, i in big]
        m = [mean_gold_recall_at_k(np.argsort(-s_mixture(b, i, K0=3, beta=16.0)),
                                   b.gold[i], 10) for b, i in big]
        print(f"\n  X48  the {len(big)} questions with >= 8 gold turns (anecdotal): "
              f"cosine {np.mean(c):.4f}  mixture {np.mean(m):.4f}  "
              f"mean |gold| {np.mean([len(b.gold[i]) for b, i in big]):.1f}")
        rows["X48_large_gold"] = dict(n=len(big), cosine=float(np.mean(c)),
                                      mixture=float(np.mean(m)))
    R["X42_X48"] = rows


if __name__ == "__main__":
    t0 = time.time()
    enc = SentenceEncoder()
    bks = banks(encoder=enc)
    print(f"{len(bks)} conversations, {sum(b.n for b in bks)} turns, "
          f"{sum(b.m for b in bks)} questions")
    x34(bks); x35(bks); x36(bks); x37(bks); x39(bks); x40(bks); x41(bks, enc); x42_48(bks)
    R["_meta"] = dict(seconds=round(time.time() - t0, 1), k_cutoff=K_CUTOFF,
                      note="X38 ran in Phase 8 (x9x_controls.json); X42-X48 consolidated")
    OUT.write_text(json.dumps(R, indent=2, default=str))
    print(f"\nwrote {OUT}  ({R['_meta']['seconds']}s)")
