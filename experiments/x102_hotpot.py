"""X102 -- HotpotQA. The backlog's last P0, and the geometric excuse's last refuge.

THE ARGUMENT THIS TESTS. X32's pre-registered interaction never appeared on LoCoMo. The
most charitable explanation left standing was geometric: LoCoMo's gold evidence turns have
mean pairwise inner product eps = 0.527 -- mutually SIMILAR -- so one query vector can
already sit close to all of them and a multi-mode readout has nothing to add. The
single-query dilution bound says the same thing from the other side: at eps = 0.53 it is
vacuous, binding for 0.0% of questions (X27).

HotpotQA's supporting sentences come from two different Wikipedia articles about different
entities, and `type == bridge` questions must traverse from one to the other. If eps is
much lower here and the effect STILL does not appear, the geometric excuse is gone and the
null is a fact about the readout rather than about LoCoMo.

Runs the LoCoMo trio on the new corpus: the headroom certificate (X27), the baseline grid
(X29), and the interaction (X31/X32). Primary metric gold@5, because m = 2.4 and n = 41 --
gold@10 would be a quarter of the bank. K0 < 5 is asserted for the same reason it is on
LoCoMo: a feedback depth at or above the metric cutoff makes the metric blind to the
method (the seventh protocol error).

INFERENCE. HotpotQA questions are independent and carry their own banks, so the question is
the resampling unit and the bootstrap is ordinary over ~600 held-out units. This is the one
experiment in the project whose p-values are not floored by ten clusters.
"""
from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from experiments.hotpot import HotpotItem, gold_eps, load, split          # noqa: E402
from energymem.base import l2_normalize                                   # noqa: E402
from energymem.encoders import SentenceEncoder                            # noqa: E402
from energymem.readout import averaged_scores, mixture_scores             # noqa: E402
from energymem.eval.setmetrics import (                                   # noqa: E402
    full_set_recall_at_k, mean_gold_recall_at_k, ndcg_at_k, reciprocal_rank,
)

OUT = ROOT / "experiments" / "results" / "x102_hotpot.json"
K_CUTOFF = 5
KS = (1, 2, 5, 10)
R: dict = {}


def hr(t: str) -> None:
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


# ------------------------------------------------------------------ methods
def s_cosine(it: HotpotItem, **_):
    return it.Z @ it.q


def s_bm25(it: HotpotItem, **_):
    return it.bm25()


def _ranks(s):
    o = np.argsort(-s)
    r = np.empty(len(s), dtype=np.int64)
    r[o] = np.arange(1, len(s) + 1)
    return r


def s_rrf(it: HotpotItem, k_rrf: float = 60.0, **_):
    return 1.0 / (k_rrf + _ranks(s_cosine(it))) + 1.0 / (k_rrf + _ranks(s_bm25(it)))


def s_mmr(it: HotpotItem, lam: float = 0.7, pool: int = 20, **_):
    s = s_cosine(it)
    cand = list(np.argsort(-s)[:pool])
    sel, out = [], s.copy()
    out[cand] = -np.inf
    while cand:
        j = cand[0] if not sel else cand[int(np.argmax(
            lam * s[cand] - (1 - lam) * (it.Z[cand] @ it.Z[sel].T).max(axis=1)))]
        out[j] = 1e6 - len(sel)
        sel.append(j)
        cand.remove(j)
    return out


def _top(it, K0):
    return np.argsort(-(it.Z @ it.q))[:K0]


def s_rocchio(it: HotpotItem, K0: int = 3, gamma: float = 0.5, **_):
    if K0 <= 0 or gamma == 0.0:
        return s_cosine(it)
    fb = it.Z[_top(it, K0)].mean(axis=0)
    return it.Z @ l2_normalize(it.q + gamma * fb)


def _hyps(it, K0, beta, gamma, with_query):
    h = [(it.q, 1.0)] if with_query else []
    if K0 > 0:
        t = _top(it, K0)
        w = np.exp(beta * (it.Z[t] @ it.q)); w = w / w.sum()
        h += [(it.Z[j], float(gamma * wj) + 1e-12) for j, wj in zip(t, w)]
    return h or [(it.q, 1.0)]


def s_mixture(it, K0=3, beta=16.0, gamma=1.0, with_query=True, **_):
    return mixture_scores(_hyps(it, K0, beta, gamma, with_query), it.Z, beta)


def s_mixture_avg(it, K0=3, beta=16.0, gamma=1.0, with_query=True, **_):
    return averaged_scores(_hyps(it, K0, beta, gamma, with_query), it.Z)


METHODS = {"cosine": s_cosine, "bm25": s_bm25, "rrf": s_rrf, "mmr": s_mmr,
           "rocchio": s_rocchio, "mixture": s_mixture, "mixture_avg": s_mixture_avg}

GRIDS = {
    "cosine": {}, "bm25": {},
    "rrf": {"k_rrf": [10.0, 60.0, 120.0]},
    "mmr": {"lam": [0.3, 0.5, 0.7, 0.9], "pool": [10, 20]},
    "rocchio": {"K0": [0, 1, 2, 3, 4], "gamma": [0.0, 0.25, 0.5, 1.0]},
    "mixture": {"K0": [0, 1, 2, 3, 4], "beta": [4.0, 16.0, 64.0], "with_query": [True, False]},
    "mixture_avg": {"K0": [0, 1, 2, 3, 4], "beta": [4.0, 16.0, 64.0],
                    "with_query": [True, False]},
}
for _m, _g in GRIDS.items():                       # the gate that X31 earned
    if "K0" in _g:
        assert max(_g["K0"]) < K_CUTOFF, f"{_m}: K0 must stay below the k={K_CUTOFF} cutoff"
        assert 0 in _g["K0"], f"{_m}: K0=0 must be reachable or the null is not in the grid"


def points(g):
    if not g:
        return [{}]
    keys = sorted(g)
    return [dict(zip(keys, v)) for v in itertools.product(*(g[k] for k in keys))]


def key(p):
    return ",".join(f"{k}={p[k]}" for k in sorted(p)) or "default"


def evaluate(items, idxs, fn, **par):
    rows = {f"gold@{k}": [] for k in KS} | {f"full@{k}": [] for k in KS}
    rows |= {"mrr": [], "ndcg@5": []}
    for i in idxs:
        it = items[i]
        o = np.argsort(-fn(it, **par))
        for k in KS:
            rows[f"gold@{k}"].append(mean_gold_recall_at_k(o, it.gold, k))
            rows[f"full@{k}"].append(full_set_recall_at_k(o, it.gold, k))
        rows["mrr"].append(reciprocal_rank(o, it.gold))
        rows["ndcg@5"].append(ndcg_at_k(o, it.gold, 5))
    return {k: np.asarray(v) for k, v in rows.items()}


def boot(d, n=20_000, seed=0):
    d = np.asarray(d, float)
    rng = np.random.default_rng(seed)
    m = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)])
    p = 2 * min((m <= 0).mean(), (m >= 0).mean())
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)), \
        float(min(1.0, p))


def selected(items, train, test, name, score_key="gold@5"):
    """Select on the train half only, report on the held-out half. Returns (vals, pick)."""
    best, best_s = None, -np.inf
    for p in points(GRIDS[name]):
        s = evaluate(items, train, METHODS[name], **p)[score_key].mean()
        if s > best_s:
            best, best_s = p, float(s)
    return evaluate(items, test, METHODS[name], **best), key(best), best_s


if __name__ == "__main__":
    t0 = time.time()
    enc = SentenceEncoder()
    print("loading HotpotQA (distractor, validation):")
    items = load(enc)
    train, test = split(items)
    bridge = [i for i in test if items[i].qtype == "bridge"]
    comp = [i for i in test if items[i].qtype == "comparison"]
    print(f"  train {len(train)}  held-out {len(test)} "
          f"(bridge {len(bridge)}, comparison {len(comp)})")
    print(f"  mean bank n = {np.mean([it.n for it in items]):.1f}, "
          f"mean |gold| = {np.mean([len(it.gold) for it in items]):.2f}")

    # ---------------------------------------------------------- the geometry
    hr("X102a -- the geometry: is eps actually lower than LoCoMo's 0.527?")
    e_all, n_all = gold_eps(items)
    e_br, _ = gold_eps(items, [i for i, it in enumerate(items) if it.qtype == "bridge"])
    e_cp, _ = gold_eps(items, [i for i, it in enumerate(items) if it.qtype == "comparison"])
    print(f"  mean pairwise inner product within gold sets, {n_all} pairs")
    print(f"    HotpotQA all          eps = {e_all:.4f}")
    print(f"    HotpotQA bridge       eps = {e_br:.4f}")
    print(f"    HotpotQA comparison   eps = {e_cp:.4f}")
    print(f"    LoCoMo cat 1 (X27)    eps = 0.5270   <-- the number this corpus was "
          f"chosen to beat")
    # the dilution bound: B(m, eps) = sqrt((1+(m-1)eps)/m); binds if B < cosine's own
    # similarity to its hardest gold item. Reported exactly as X27 reported it.
    binds = 0
    for it in items:
        m = len(it.gold)
        if m < 2:
            continue
        G = it.Z[it.gold]
        iu = np.triu_indices(m, 1)
        eps_i = float((G @ G.T)[iu].mean())
        B = np.sqrt((1 + (m - 1) * eps_i) / m)
        worst = float((G @ it.q).min())
        binds += int(B < worst)
    print(f"\n  single-query dilution bound binds for {binds}/{len(items)} questions "
          f"= {binds / len(items):.1%}")
    print("  (LoCoMo: 0.0%. A bound that does not bind is a certificate that the failure")
    print("   is query formation, not expressivity -- never an impossibility result.)")
    R["X102a_geometry"] = dict(eps_all=e_all, eps_bridge=e_br, eps_comparison=e_cp,
                               eps_locomo_cat1=0.527, n_pairs=n_all,
                               bound_binds=binds, n_items=len(items),
                               bound_binds_frac=binds / len(items))

    # --------------------------------------------------------- the baselines
    hr("X102b -- the baseline grid, selected on train, reported on held-out")
    base = {}
    print(f"  {'method':<14}{'pick':<34}{'gold@5':>9}{'full@5':>9}{'gold@2':>9}"
          f"{'mrr':>8}{'ndcg@5':>8}")
    for name in ("cosine", "bm25", "rrf", "mmr", "rocchio", "mixture", "mixture_avg"):
        r, pick, _ = selected(items, train, test, name)
        base[name] = {k: r[k] for k in r}
        print(f"  {name:<14}{pick:<34}{r['gold@5'].mean():>9.4f}{r['full@5'].mean():>9.4f}"
              f"{r['gold@2'].mean():>9.4f}{r['mrr'].mean():>8.4f}{r['ndcg@5'].mean():>8.4f}")
        R.setdefault("X102b_baselines", {})[name] = dict(
            pick=pick, **{k: float(v.mean()) for k, v in r.items()})

    cos = base["cosine"]["gold@5"]
    print(f"\n  {'vs cosine':<14}{'delta gold@5':>16}{'95% CI':>26}{'p':>10}")
    for name in ("bm25", "rrf", "mmr", "rocchio", "mixture", "mixture_avg"):
        d, lo, hi, p = boot(base[name]["gold@5"] - cos)
        print(f"  {name:<14}{d:>+16.4f}   [{lo:>+.4f},{hi:>+.4f}]{p:>10.4f}")
        R["X102b_baselines"][name]["vs_cosine"] = dict(delta=d, lo=lo, hi=hi, p=p)

    # -------------------------------------------------- the interaction, X32
    hr("X102c -- the interaction: bridge (multi-hop) vs comparison (parallel lookup)")
    print("  X32 predicted on LoCoMo: delta > 0 on multi-hop, ~0 on single-hop, and the")
    print("  DIFFERENCE positive. It was not met there. HotpotQA is multi-hop by")
    print("  construction, so this is the same prediction with the geometry it wanted.\n")
    inter = {}
    print(f"  {'subset':<14}{'n':>6}{'cosine':>9}{'mixture':>9}{'delta':>10}"
          f"{'95% CI':>24}{'p':>9}   pick")
    for label, idxs in (("bridge", bridge), ("comparison", comp)):
        rc = evaluate(items, idxs, s_cosine)
        rm, pick, _ = (lambda t: t)(None) or (None, None, None)
        # select the mixture on the TRAIN half restricted to the same question type
        tr = [i for i in train if items[i].qtype == label]
        best, best_s = None, -np.inf
        for p_ in points(GRIDS["mixture"]):
            s = evaluate(items, tr, s_mixture, **p_)["gold@5"].mean()
            if s > best_s:
                best, best_s = p_, float(s)
        rm = evaluate(items, idxs, s_mixture, **best)
        d, lo, hi, p = boot(rm["gold@5"] - rc["gold@5"])
        inter[label] = dict(n=len(idxs), cosine=float(rc["gold@5"].mean()),
                            mixture=float(rm["gold@5"].mean()), delta=d, lo=lo, hi=hi,
                            p=p, pick=key(best), null_selected="K0=0" in key(best))
        print(f"  {label:<14}{len(idxs):>6}{rc['gold@5'].mean():>9.4f}"
              f"{rm['gold@5'].mean():>9.4f}{d:>+10.4f}   [{lo:>+.4f},{hi:>+.4f}]"
              f"{p:>9.4f}   {key(best)}")
    dod = inter["bridge"]["delta"] - inter["comparison"]["delta"]
    print(f"\n  difference of deltas (bridge - comparison) = {dod:+.4f}")
    print(f"  X32's prediction was that this is POSITIVE. Met: {dod > 0}")
    R["X102c_interaction"] = dict(**inter, difference_of_deltas=dod,
                                  prediction_met=bool(dod > 0))

    # the readout comparison, I4, with matched modes
    hr("X102d -- I4 with matched modes: logsumexp vs averaging, same hypotheses")
    for label, idxs in (("bridge", bridge), ("comparison", comp), ("all", test)):
        tr = [i for i in train if label == "all" or items[i].qtype == label]
        best, best_s = None, -np.inf
        for p_ in points(GRIDS["mixture"]):
            if p_["K0"] == 0:
                continue                     # force a non-trivial readout comparison
            s = evaluate(items, tr, s_mixture, **p_)["gold@5"].mean()
            if s > best_s:
                best, best_s = p_, float(s)
        a = evaluate(items, idxs, s_mixture, **best)["gold@5"]
        b = evaluate(items, idxs, s_mixture_avg, **best)["gold@5"]
        c = evaluate(items, idxs, s_cosine)["gold@5"]
        d, lo, hi, p = boot(a - b)
        # THE COLUMN THAT DECIDES WHAT THIS MEANS. A gap between two mode combiners is
        # only evidence for I4 if at least one of them beats doing nothing. Free
        # selection picks K0=0 (X102b/X102c), so this cell is forced and may well sit
        # below cosine -- in which case the gap says "averaging loses more", not
        # "mode-keeping wins".
        dmc, lmc, hmc, pmc = boot(a - c)
        dac, _, _, _ = boot(b - c)
        print(f"  {label:<12} mixture {a.mean():.4f}  averaged {b.mean():.4f}  "
              f"delta {d:+.4f} [{lo:+.4f},{hi:+.4f}] p={p:.4f}   ({key(best)}, K0>0 forced)")
        print(f"  {'':<12} cosine  {c.mean():.4f}  -> mixture vs cosine {dmc:+.4f} "
              f"[{lmc:+.4f},{hmc:+.4f}] p={pmc:.4f};  averaged vs cosine {dac:+.4f}"
              f"   {'BOTH BELOW THE NULL' if dmc < 0 and dac < 0 else ''}")
        R.setdefault("X102d_I4", {})[label] = dict(
            mixture=float(a.mean()), averaged=float(b.mean()),
            cosine=float(c.mean()), delta=d, lo=lo, hi=hi, p=p, pick=key(best),
            mixture_vs_cosine=dmc, mixture_vs_cosine_lo=lmc, mixture_vs_cosine_hi=hmc,
            mixture_vs_cosine_p=pmc, averaged_vs_cosine=dac,
            both_below_null=bool(dmc < 0 and dac < 0))

    R["_meta"] = dict(seconds=round(time.time() - t0, 1), n_items=len(items),
                      n_train=len(train), n_test=len(test), k_cutoff=K_CUTOFF,
                      encoder="all-mpnet-base-v2", corpus="hotpot_qa distractor validation",
                      cluster="question (independent banks, ordinary bootstrap)")
    OUT.write_text(json.dumps(R, indent=2, default=str))
    print(f"\nwrote {OUT}  ({R['_meta']['seconds']}s)")
