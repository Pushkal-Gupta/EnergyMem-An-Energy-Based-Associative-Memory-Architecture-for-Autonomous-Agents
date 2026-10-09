"""Family G -- controls and falsification -- plus X38, the control that gates family A.

The backlog says of family G: "run these *before* believing anything in A or D." They
were not run before, which is why Phase 7 shipped with a hole in it. Nothing in family A
is citable until X38 comes back at chance, and nothing in A/D is citable until the
methods are shown not to see the labels.

X38  shuffled gold         -- the metric must not reward a method for anything.
X95  random modes          -- association, or just more mass on the bank?
X96  label-leakage audit   -- static AND functional: scores must be gold-invariant.
X97  exclusion symmetry    -- every method ranks the identical candidate set.
X98  reduction-ladder null -- every rung must reproduce the rung below, exactly.
X99  metric cross-check    -- independent re-implementations, and the EXACT p-value.
"""
from __future__ import annotations

import ast
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from experiments.locomo_qa import (                              # noqa: E402
    KS, METHODS, RESULTS, Bank, banks, evaluate, s_cosine, s_mixture, s_mixture_avg,
    s_mmr, s_rocchio, s_rrf, _top,
)
from energymem.base import l2_normalize                          # noqa: E402
from energymem.eval.setmetrics import (                          # noqa: E402
    blocked_permutation, exact_blocked_permutation, full_set_recall_at_k,
    hierarchical_bootstrap, mean_gold_recall_at_k, ndcg_at_k, paired_report,
    reciprocal_rank,
)

OUT = RESULTS / "x9x_controls.json"
R: dict = {}


def head(t: str) -> None:
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


# ============================================================ X38 shuffled gold
# Two nulls, because they break different things.
#   random:   gold replaced by a uniformly random set of the SAME SIZE. Tests the metric
#             and the plumbing. Chance level is computable in closed form.
#   permuted: gold sets swapped BETWEEN questions of the same conversation. Keeps the
#             real gold-set geometry (so hubness, length, session spread all survive) and
#             breaks only the question-to-evidence link. This is the one that catches a
#             metric inflated by gold turns being intrinsically easy to retrieve.

def x38(bks: list[Bank], seeds=range(5)) -> None:
    head("X38 -- shuffled-gold control (P0: family A is void if this beats chance)")
    methods = {"cosine": (s_cosine, {}), "rrf": (s_rrf, {"k_rrf": 60.0}),
               "mixture": (s_mixture, {"K0": 3, "beta": 16.0})}
    rows: dict = {}

    # closed-form chance for mean-gold recall@k with a uniformly random gold set:
    # each gold item lands in the top k with probability k/N, independently of m.
    chance = {f"gold@{k}": float(np.mean([k / b.n for b in bks for _ in range(b.m)]))
              for k in KS}
    print("closed-form chance, mean over questions: " +
          "  ".join(f"gold@{k}={chance[f'gold@{k}']:.4f}" for k in KS))

    for name, (fn, par) in methods.items():
        # rank once per query per method; the gold sets are what varies.
        orders = [[np.argsort(-fn(b, i, **par)) for i in range(b.m)] for b in bks]
        for mode in ("random", "permuted"):
            per_seed = []
            for sd in seeds:
                rng = np.random.default_rng(1000 + sd)
                vals = []
                for b, ords in zip(bks, orders):
                    if mode == "random":
                        fake = [list(rng.choice(b.n, size=len(g), replace=False))
                                for g in b.gold]
                    else:
                        perm = rng.permutation(b.m)
                        fake = [b.gold[j] for j in perm]
                    vals += [mean_gold_recall_at_k(o, g, 10)
                             for o, g in zip(ords, fake)]
                per_seed.append(float(np.mean(vals)))
            rows[f"{name}/{mode}"] = dict(mean=float(np.mean(per_seed)),
                                          sd=float(np.std(per_seed)),
                                          seeds=per_seed)
            print(f"  {name:8s} {mode:9s} gold@10 = {np.mean(per_seed):.4f} "
                  f"(sd {np.std(per_seed):.4f} over {len(per_seed)} seeds)")

    real = {n: float(np.mean([v for b in bks
                              for v in evaluate(b, f, **p)["gold@10"]]))
            for n, (f, p) in methods.items()}
    print("\n  real gold:  " + "  ".join(f"{n}={v:.4f}" for n, v in real.items()))
    verdict = all(rows[f"{n}/random"]["mean"] < chance["gold@10"] * 3 for n in methods)
    perm_ok = all(rows[f"{n}/permuted"]["mean"] < real[n] / 3 for n in methods)
    print(f"\n  random-gold at chance: {verdict}")
    print(f"  permuted-gold far below real: {perm_ok}")
    print("  VERDICT: " + ("PASS -- family A's metric is not rewarding structure"
                           if verdict and perm_ok else "FAIL -- family A is void"))
    R["x38"] = dict(chance=chance, shuffled=rows, real=real,
                    pass_random=bool(verdict), pass_permuted=bool(perm_ok))


# ============================================================= X95 random modes
# The mixture at its SELECTED configuration is the cosine null (X31), so the question
# "do random modes beat real modes" has no gap to interrogate. The question that still
# bites: at a FORCED non-trivial configuration, do real modes beat random ones? If they
# do not, "association" is doing nothing that extra mass on the bank would not do.

def s_random_modes(b: Bank, i: int, K0: int = 3, beta: float = 16.0,
                   with_query: bool = True, seed: int = 0, **_) -> np.ndarray:
    from energymem.readout import mixture_scores
    q = b.Q[i]
    rng = np.random.default_rng(seed * 100003 + i)
    hyps = [(q, 1.0)] if with_query else []
    if K0 > 0:
        t = rng.choice(b.n, size=K0, replace=False)
        w = np.exp(beta * (b.Z[t] @ q)); w = w / w.sum()
        hyps += [(b.Z[j], float(wj) + 1e-12) for j, wj in zip(t, w)]
    if not hyps:
        hyps = [(q, 1.0)]
    return mixture_scores(hyps, b.Z, beta)


def x95(bks: list[Bank]) -> None:
    head("X95 -- random modes at matched K (must not beat real modes)")
    clus = np.concatenate([np.full(b.m, b.sample_id) for b in bks])
    cats = np.concatenate([np.asarray(b.cats) for b in bks])
    base = np.concatenate([evaluate(b, s_cosine)["gold@10"] for b in bks])
    rows = []
    for K0 in (1, 3, 5):
        for beta in (4.0, 16.0, 64.0):
            real = np.concatenate([evaluate(b, s_mixture, K0=K0, beta=beta)["gold@10"]
                                   for b in bks])
            rnd = np.mean([[np.concatenate(
                [evaluate(b, s_random_modes, K0=K0, beta=beta, seed=sd)["gold@10"]
                 for b in bks])] for sd in range(3)], axis=0)[0]
            rr = paired_report(real, rnd, clus, n_boot=4000, n_perm=20000)
            rows.append(dict(K0=K0, beta=beta, real=rr["a"], random=rr["b"],
                             delta=rr["delta"], lo=rr["lo"], hi=rr["hi"], p=rr["p"],
                             cosine=float(base.mean()),
                             real_vs_cos=float(real.mean() - base.mean()),
                             rand_vs_cos=float(rnd.mean() - base.mean())))
            print(f"  K0={K0} beta={beta:>5}: real {rr['a']:.4f}  random {rr['b']:.4f}  "
                  f"delta {rr['delta']:+.4f} [{rr['lo']:+.4f},{rr['hi']:+.4f}] p={rr['p']:.4f}"
                  f"   (cosine {base.mean():.4f})")
    beats = [r for r in rows if r["rand_vs_cos"] > 0 and r["real_vs_cos"] <= r["rand_vs_cos"]]
    print(f"\n  cells where random modes do at least as well as real ones: "
          f"{len(beats)}/{len(rows)}")
    print("  (a cell with real <= random is a cell where association is not the mechanism)")
    R["x95"] = dict(rows=rows, cosine=float(base.mean()),
                    n_cells_random_ge_real=len(beats),
                    cats_n=int(len(cats)))


# ======================================================= X96 label-leakage audit

GOLD_NAMES = {"gold", "gold_ids", "answers", "answer", "cats"}


def x96(bks: list[Bank]) -> None:
    head("X96 -- label-leakage audit (static + functional)")

    # --- static: no scoring function may mention a label-bearing attribute.
    src = (ROOT / "experiments" / "locomo_qa.py").read_text()
    tree = ast.parse(src)
    static = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name.startswith("s_"):
            touched = sorted({n.attr for n in ast.walk(node)
                              if isinstance(n, ast.Attribute) and n.attr in GOLD_NAMES})
            static.append(dict(fn=node.name, label_attrs=touched))
            print(f"  static  {node.name:14s} label attrs: {touched or 'none'}")
    static_ok = all(not s["label_attrs"] for s in static)

    # --- functional: the scores a method returns must be bit-identical after the gold
    # sets are destroyed. This catches leakage the grep cannot see -- through a helper,
    # a closure, or a cached array.
    func = []
    b = bks[0]
    probe = list(range(0, b.m, 7))
    for name, fn in METHODS.items():
        par = {"K0": 3} if name in ("mixture", "mixture_avg", "rocchio") else {}
        before = np.stack([fn(b, i, **par) for i in probe])
        saved_g, saved_a, saved_c = b.gold, b.answers, b.cats
        rng = np.random.default_rng(7)
        b.gold = [list(rng.choice(b.n, size=len(g), replace=False)) for g in saved_g]
        b.answers = ["" for _ in saved_a]
        b.cats = [0 for _ in saved_c]
        after = np.stack([fn(b, i, **par) for i in probe])
        b.gold, b.answers, b.cats = saved_g, saved_a, saved_c
        same = bool(np.array_equal(before, after))
        func.append(dict(method=name, gold_invariant=same))
        print(f"  functional {name:12s} gold-invariant: {same}")
    func_ok = all(f["gold_invariant"] for f in func)

    # --- selection: fold_select must aggregate the TRAIN side only.
    sel_src = ast.parse(src)
    sel_ok = True
    for node in ast.walk(sel_src):
        if isinstance(node, ast.FunctionDef) and node.name.startswith("fold_select"):
            body = ast.unparse(node)
            # the selection loop must index `train`, and must not index `test` while scoring
            sel_ok &= ("for j in train" in body) and ("per_bank[test]" not in body)
    print(f"\n  fold_select scores on train indices only: {sel_ok}")
    print("  VERDICT: " + ("PASS" if static_ok and func_ok and sel_ok else "FAIL"))
    R["x96"] = dict(static=static, functional=func, static_ok=static_ok,
                    functional_ok=func_ok, selection_ok=bool(sel_ok))


# ====================================================== X97 exclusion symmetry

def x97(bks: list[Bank]) -> None:
    head("X97 -- exclusion symmetry and tie hygiene")
    b = bks[0]
    probe = list(range(0, b.m, 5))
    rows = []
    for name, fn in METHODS.items():
        par = {"K0": 3} if name in ("mixture", "mixture_avg", "rocchio") else {}
        n_finite, n_nan, n_ranked, tie_at_k = [], 0, set(), 0
        for i in probe:
            s = fn(b, i, **par)
            n_finite.append(int(np.isfinite(s).sum()))
            n_nan += int(np.isnan(s).sum())
            n_ranked.add(int(len(np.argsort(-s))))
            o = np.argsort(-s)
            # a tie straddling the k=10 boundary means argsort's index order decides a hit
            tie_at_k += int(s[o[9]] == s[o[10]]) if len(s) > 10 else 0
        ok = (set(n_finite) == {b.n}) and n_nan == 0 and n_ranked == {b.n}
        rows.append(dict(method=name, finite=sorted(set(n_finite)), nans=n_nan,
                         ranked=sorted(n_ranked), ties_at_k10=tie_at_k,
                         n_probe=len(probe), symmetric=bool(ok)))
        print(f"  {name:12s} finite={sorted(set(n_finite))} nan={n_nan} "
              f"ranked={sorted(n_ranked)} ties@10={tie_at_k}/{len(probe)}  ok={ok}")
    print("\n  every method scores all N items, no masking, no NaN: "
          f"{all(r['symmetric'] for r in rows)}")
    print("  (the QA task has nothing to exclude -- the query is not a bank item. The")
    print("   asymmetry risk here is ties, not exclusion.)")
    R["x97"] = dict(rows=rows, bank=b.sample_id, n=b.n,
                    all_symmetric=bool(all(r["symmetric"] for r in rows)))


# ==================================================== X98 reduction-ladder nulls

def x98(bks: list[Bank]) -> None:
    head("X98 -- every rung reproduces the rung below at its null setting")
    rows = []

    def ranking_identical(fa, pa, fb, pb, label):
        hits = tot = 0
        for b in bks:
            for i in range(b.m):
                ra = np.argsort(-fa(b, i, **pa))
                rb = np.argsort(-fb(b, i, **pb))
                hits += int(np.array_equal(ra, rb)); tot += 1
        rows.append(dict(rung=label, identical=hits, total=tot,
                         exact=bool(hits == tot)))
        print(f"  {label:52s} {hits}/{tot}  {'OK' if hits == tot else 'BROKEN'}")

    ranking_identical(s_mixture, {"K0": 0}, s_cosine, {}, "mixture(K0=0) == cosine")
    ranking_identical(s_mixture_avg, {"K0": 0}, s_cosine, {}, "mixture_avg(K0=0) == cosine")
    ranking_identical(s_rocchio, {"K0": 0}, s_cosine, {}, "rocchio(K0=0) == cosine")
    ranking_identical(s_rocchio, {"K0": 5, "gamma": 0.0}, s_cosine, {},
                      "rocchio(gamma=0) == cosine")
    ranking_identical(s_mixture, {"K0": 0, "beta": 4.0}, s_mixture,
                      {"K0": 0, "beta": 64.0}, "mixture(K0=0) beta-invariant")
    # single-mode mixture is a monotone function of cosine at ANY beta
    for beta in (1.0, 16.0, 256.0):
        ranking_identical(s_mixture, {"K0": 0, "beta": beta}, s_cosine, {},
                          f"single-mode mixture(beta={beta:g}) == cosine")
    # MMR at lam=1 is pure relevance -> cosine order within the pool
    hits = tot = 0
    for b in bks:
        for i in range(0, b.m, 3):
            s = s_cosine(b, i)
            top = np.argsort(-s)[:50]
            got = np.argsort(-s_mmr(b, i, lam=1.0, pool=50))[:50]
            hits += int(np.array_equal(top, got)); tot += 1
    rows.append(dict(rung="mmr(lam=1) == cosine on the pool", identical=hits,
                     total=tot, exact=bool(hits == tot)))
    print(f"  {'mmr(lam=1) == cosine on the pool':52s} {hits}/{tot} "
          f"{'OK' if hits == tot else 'BROKEN'}")

    # The energy rung. PROPOSITION 1 IS A TOP-1 STATEMENT -- `test_reduction_to_cosine_topk`
    # is named for a k it does not test; its own docstring says "top-1 cosine exactly".
    # At T=1, beta->inf the descent returns the nearest stored pattern, so ranking by
    # similarity to the CONVERGED STATE gives that pattern's neighbourhood, not the
    # query's. Both are measured below, because the gap between them is the mechanism
    # behind family C's finding that T>1 hurts.
    from experiments.x6x_energy import descend
    h1 = h10 = tot = 0
    for b in bks[:3]:
        for i in range(0, b.m, 5):
            xi = descend(b.Z, b.Q[i], beta=512.0, T=1, b=None, R=None)
            a = np.argsort(-(b.Z @ xi))
            c = np.argsort(-s_cosine(b, i))
            h1 += int(a[0] == c[0])
            h10 += int(np.array_equal(a[:10], c[:10]))
            tot += 1
    rows.append(dict(rung="descend(T=1,beta=512,b=0,R=I) top-1 == cosine top-1",
                     identical=h1, total=tot, exact=bool(h1 == tot)))
    print(f"  {'descend(T=1,b=0,R=I) top-1 == cosine top-1':52s} {h1}/{tot} "
          f"{'OK' if h1 == tot else 'BROKEN'}")
    print(f"  {'...and its top-10 == cosine top-10':52s} {h10}/{tot} "
          f"(expected to FAIL: Prop 1 is k=1 only)")
    R["x98_energy_top10"] = dict(agree=h10, total=tot)
    print(f"\n  rungs exact: {sum(r['exact'] for r in rows)}/{len(rows)}")
    R["x98"] = dict(rows=rows, all_exact=bool(all(r["exact"] for r in rows)),
                energy_top10_agree=R.pop("x98_energy_top10"))


# ==================================================== X99 metric cross-check
# Independent re-implementations, written in a different style on purpose: loops over
# positions instead of set algebra, so a shared misconception cannot pass both.

def alt_full(ranked, gold, k):
    found = 0
    for g in gold:
        for pos in range(min(k, len(ranked))):
            if ranked[pos] == g:
                found += 1
                break
    return 1.0 if found == len(set(gold)) and len(gold) > 0 else 0.0


def alt_gold(ranked, gold, k):
    uniq, hit = [], 0
    for g in gold:
        if g not in uniq:
            uniq.append(g)
    for g in uniq:
        for pos in range(min(k, len(ranked))):
            if ranked[pos] == g:
                hit += 1
                break
    return hit / len(uniq)


def alt_rr(ranked, gold):
    for pos in range(len(ranked)):
        for g in gold:
            if ranked[pos] == g:
                return 1.0 / (pos + 1)
    return 0.0


def alt_ndcg(ranked, gold, k):
    import math
    uniq = {g for g in gold}
    dcg = 0.0
    for pos in range(min(k, len(ranked))):
        if ranked[pos] in uniq:
            dcg += 1.0 / math.log2(pos + 2)
    ideal = sum(1.0 / math.log2(r + 1) for r in range(1, min(len(uniq), k) + 1))
    return dcg / ideal if ideal else 0.0


def x99(bks: list[Bank]) -> None:
    head("X99 -- independent metric re-implementations, and the EXACT p-value")
    mism = {m: 0 for m in ("full", "gold", "rr", "ndcg")}
    n = 0
    for b in bks:
        for name, fn in (("cosine", s_cosine), ("rrf", s_rrf)):
            for i in range(0, b.m, 3):
                o = list(np.argsort(-fn(b, i)))
                g = b.gold[i]
                n += 1
                for k in KS:
                    if abs(full_set_recall_at_k(o, g, k) - alt_full(o, g, k)) > 1e-12:
                        mism["full"] += 1
                    if abs(mean_gold_recall_at_k(o, g, k) - alt_gold(o, g, k)) > 1e-12:
                        mism["gold"] += 1
                if abs(reciprocal_rank(o, g) - alt_rr(o, g)) > 1e-12:
                    mism["rr"] += 1
                if abs(ndcg_at_k(o, g, 10) - alt_ndcg(o, g, 10)) > 1e-12:
                    mism["ndcg"] += 1
    print(f"  compared {n} (query, method) rankings on real data")
    for k, v in mism.items():
        print(f"    {k:5s} mismatches: {v}   {'OK' if v == 0 else 'BROKEN'}")

    # --- the inference cross-check. With B conversations the blocked null has exactly
    # 2**B points, so the exact p is cheaper than the Monte Carlo approximation to it.
    clus = np.concatenate([np.full(b.m, b.sample_id) for b in bks])
    cos = np.concatenate([evaluate(b, s_cosine)["gold@10"] for b in bks])
    rrf = np.concatenate([evaluate(b, s_rrf)["gold@10"] for b in bks])
    mix = np.concatenate([evaluate(b, s_mixture, K0=0)["gold@10"] for b in bks])
    pairs = {"rrf-cosine": (rrf, cos), "mixture(null)-cosine": (mix, cos)}
    inf = {}
    for label, (a, bb) in pairs.items():
        d = a - bb
        p_mc = blocked_permutation(d, clus, n=100_000)
        p_ex = exact_blocked_permutation(d, clus)
        inf[label] = dict(delta=float(d.mean()), p_mc=p_mc, p_exact=p_ex)
        print(f"  {label:22s} delta {d.mean():+.4f}  p_mc={p_mc:.5f}  p_exact={p_ex:.5f}")
    B = len(np.unique(clus))
    floor = 2.0 / (2 ** B)
    print(f"\n  {B} conversations -> {2 ** B} sign patterns -> smallest attainable "
          f"two-sided p = {floor:.5f}")
    print("  EVERY 'p=0.002' in Phase 7 is this floor. It means 'as extreme as the")
    print("  blocking permits', NOT 'p = 0.002 worth of evidence'.")
    R["x99"] = dict(mismatches=mism, n_rankings=n, inference=inf,
                    n_blocks=int(B), p_floor=floor,
                    all_metrics_agree=bool(all(v == 0 for v in mism.values())))


if __name__ == "__main__":
    t0 = time.time()
    bks = banks()
    print(f"{len(bks)} conversations, {sum(b.n for b in bks)} turns, "
          f"{sum(b.m for b in bks)} gold-annotated questions")
    x38(bks)
    x95(bks)
    x96(bks)
    x97(bks)
    x98(bks)
    x99(bks)
    R["_meta"] = dict(seconds=round(time.time() - t0, 1),
                      n_conversations=len(bks), n_queries=sum(b.m for b in bks))
    OUT.write_text(json.dumps(R, indent=2, default=str))
    print(f"\nwrote {OUT}  ({R['_meta']['seconds']}s)")
