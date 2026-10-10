"""X104 -- BEIR / SciFact. A validity check on OUR OWN code, not on the architecture.

WHY THIS MATTERS MORE THAN IT LOOKS. Rocchio / pseudo-relevance feedback is the prior art a
reviewer names first, and in this project it has never helped: worse than cosine on LoCoMo
when forced non-trivial (X33), and selected straight to its null (gamma=0) everywhere it
was given the choice. There are two explanations and they have very different consequences:

  (a) PRF genuinely does not help on these tasks, or
  (b) OUR PRF is weak, in which case every comparison against it in this project flatters
      the architecture and the "prior art does not help either" line is worthless.

SciFact is a standard ad-hoc IR benchmark where PRF is well calibrated and known to help.
If our Rocchio helps here, (a) stands and the implementation is exonerated. If it helps
nowhere, (b) is live and we have to say so.

Secondary: this is the project's fourth corpus, with single-target gold (mean |gold| 1.1)
and a shared 5,183-document bank -- a different shape from everything above.
"""
from __future__ import annotations

import csv
import itertools
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from energymem.base import Memory, l2_normalize                       # noqa: E402
from energymem.baselines.lexical import BM25                          # noqa: E402
from energymem.encoders import SentenceEncoder                        # noqa: E402
from energymem.readout import averaged_scores, mixture_scores         # noqa: E402
from energymem.eval.setmetrics import (                               # noqa: E402
    full_set_recall_at_k, mean_gold_recall_at_k, ndcg_at_k, reciprocal_rank,
)

DATA = ROOT / "data" / "external" / "beir_scifact"
OUT = ROOT / "experiments" / "results" / "x104_beir.json"
K_CUTOFF = 10
SPLIT_SEED = 12345
R: dict = {}


def hr(t):
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


def load():
    import pyarrow.parquet as pq
    corpus = pq.read_table(DATA / "corpus.parquet").to_pylist()
    queries = {r["_id"]: r["text"] for r in
               pq.read_table(DATA / "queries.parquet").to_pylist()}
    rows = list(csv.reader(open(DATA / "qrels_test.tsv"), delimiter="\t"))[1:]
    gold = defaultdict(list)
    for qid, cid, sc in rows:
        if int(sc) > 0:
            gold[qid].append(cid)
    pos = {d["_id"]: i for i, d in enumerate(corpus)}
    texts = [f"{d['title']}: {d['text']}" if d["title"] else d["text"] for d in corpus]
    qids = [q for q in gold if q in queries and all(c in pos for c in gold[q])]
    qids.sort()
    G = [[pos[c] for c in gold[q]] for q in qids]
    return texts, [queries[q] for q in qids], G, qids


class Bank:
    def __init__(self, texts, Z, questions, Q, gold):
        self.texts, self.Z, self.questions, self.Q, self.gold = texts, Z, questions, Q, gold
        self._bm = None

    def bm25(self):
        if self._bm is None:
            idx = BM25()
            idx.index([Memory(id=str(j), text=t, timestamp=float(j))
                       for j, t in enumerate(self.texts)])
            self._bm = np.stack([idx.scores(q) for q in self.questions])
        return self._bm


def s_cosine(b, i, **_):
    return b.Z @ b.Q[i]


def s_bm25(b, i, **_):
    return b.bm25()[i]


def _ranks(s):
    o = np.argsort(-s); r = np.empty(len(s), np.int64); r[o] = np.arange(1, len(s) + 1)
    return r


def s_rrf(b, i, k_rrf=60.0, **_):
    return 1.0 / (k_rrf + _ranks(s_cosine(b, i))) + 1.0 / (k_rrf + _ranks(s_bm25(b, i)))


def _top(b, i, K0):
    return np.argsort(-(b.Z @ b.Q[i]))[:K0]


def s_rocchio(b, i, K0=5, gamma=0.5, **_):
    """THE METHOD UNDER TEST HERE. gamma=0 or K0=0 reduces exactly to cosine."""
    if K0 <= 0 or gamma == 0.0:
        return s_cosine(b, i)
    fb = b.Z[_top(b, i, K0)].mean(axis=0)
    return b.Z @ l2_normalize(b.Q[i] + gamma * fb)


def _hyps(b, i, K0, beta, with_query):
    h = [(b.Q[i], 1.0)] if with_query else []
    if K0 > 0:
        t = _top(b, i, K0)
        w = np.exp(beta * (b.Z[t] @ b.Q[i])); w = w / w.sum()
        h += [(b.Z[j], float(wj) + 1e-12) for j, wj in zip(t, w)]
    return h or [(b.Q[i], 1.0)]


def s_mixture(b, i, K0=3, beta=16.0, with_query=True, **_):
    return mixture_scores(_hyps(b, i, K0, beta, with_query), b.Z, beta)


def s_mixture_avg(b, i, K0=3, beta=16.0, with_query=True, **_):
    return averaged_scores(_hyps(b, i, K0, beta, with_query), b.Z)


METHODS = {"cosine": s_cosine, "bm25": s_bm25, "rrf": s_rrf, "rocchio": s_rocchio,
           "mixture": s_mixture, "mixture_avg": s_mixture_avg}
GRIDS = {"cosine": {}, "bm25": {}, "rrf": {"k_rrf": [10.0, 60.0, 120.0]},
         "rocchio": {"K0": [0, 1, 3, 5, 9], "gamma": [0.0, 0.25, 0.5, 1.0, 2.0]},
         "mixture": {"K0": [0, 1, 3, 5, 9], "beta": [4.0, 16.0, 64.0],
                     "with_query": [True, False]},
         "mixture_avg": {"K0": [0, 1, 3, 5, 9], "beta": [4.0, 16.0, 64.0],
                         "with_query": [True, False]}}
for _m, _g in GRIDS.items():
    if "K0" in _g:
        assert max(_g["K0"]) < K_CUTOFF and 0 in _g["K0"], _m


def pts(g):
    if not g:
        return [{}]
    k = sorted(g)
    return [dict(zip(k, v)) for v in itertools.product(*(g[i] for i in k))]


def key(p):
    return ",".join(f"{k}={p[k]}" for k in sorted(p)) or "default"


def ev(b, idxs, fn, **par):
    out = defaultdict(list)
    for i in idxs:
        o = np.argsort(-fn(b, i, **par))
        for k in (1, 5, 10, 20):
            out[f"gold@{k}"].append(mean_gold_recall_at_k(o, b.gold[i], k))
        out["full@10"].append(full_set_recall_at_k(o, b.gold[i], 10))
        out["ndcg@10"].append(ndcg_at_k(o, b.gold[i], 10))
        out["mrr"].append(reciprocal_rank(o, b.gold[i]))
    return {k: np.asarray(v) for k, v in out.items()}


def boot(d, n=20_000, seed=0):
    d = np.asarray(d, float)
    rng = np.random.default_rng(seed)
    m = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)])
    return (float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)),
            float(min(1.0, 2 * min((m <= 0).mean(), (m >= 0).mean()))))


if __name__ == "__main__":
    t0 = time.time()
    enc = SentenceEncoder()
    texts, questions, gold, qids = load()
    print(f"SciFact: {len(texts)} documents, {len(questions)} test queries with gold, "
          f"mean |gold| = {np.mean([len(g) for g in gold]):.2f}")
    b = Bank(texts, enc.encode(texts), questions, enc.encode(questions), gold)
    rng = np.random.default_rng(SPLIT_SEED)
    perm = rng.permutation(len(questions))
    train, test = sorted(perm[: len(questions) // 2]), sorted(perm[len(questions) // 2:])
    print(f"  train {len(train)}  held-out {len(test)}  "
          f"(queries are independent, so the bootstrap is ordinary over the held-out set)")

    hr("X104  SciFact: selected on train, reported on held-out")
    print(f"  {'method':<14}{'pick':<28}{'ndcg@10':>9}{'gold@10':>9}{'gold@5':>9}"
          f"{'mrr':>8}")
    res = {}
    for name in ("cosine", "bm25", "rrf", "rocchio", "mixture", "mixture_avg"):
        best, bs = None, -np.inf
        for p in pts(GRIDS[name]):
            s = ev(b, train, METHODS[name], **p)["ndcg@10"].mean()
            if s > bs:
                best, bs = p, float(s)
        r = ev(b, test, METHODS[name], **best)
        res[name] = (r, key(best))
        print(f"  {name:<14}{key(best):<28}{r['ndcg@10'].mean():>9.4f}"
              f"{r['gold@10'].mean():>9.4f}{r['gold@5'].mean():>9.4f}{r['mrr'].mean():>8.4f}")
        R.setdefault("methods", {})[name] = dict(
            pick=key(best), **{k: float(v.mean()) for k, v in r.items()})

    cos = res["cosine"][0]
    print(f"\n  {'vs cosine':<14}{'d ndcg@10':>14}{'95% CI':>24}{'p':>9}")
    for name in ("bm25", "rrf", "rocchio", "mixture", "mixture_avg"):
        d, lo, hi, p = boot(res[name][0]["ndcg@10"] - cos["ndcg@10"])
        print(f"  {name:<14}{d:>+14.4f}   [{lo:>+.4f},{hi:>+.4f}]{p:>9.4f}")
        R["methods"][name]["vs_cosine_ndcg10"] = dict(delta=d, lo=lo, hi=hi, p=p)

    # ------------------------------------------------ the question this corpus answers
    hr("X104b  is our Rocchio sound? Forced non-trivial, swept, on its home turf")
    print(f"  {'K0':>4}{'gamma':>7}{'ndcg@10':>10}{'d vs cosine':>14}{'95% CI':>24}")
    rowsr, any_pos = {}, False
    for K0 in (1, 3, 5, 9):
        for gamma in (0.25, 0.5, 1.0, 2.0):
            r = ev(b, test, s_rocchio, K0=K0, gamma=gamma)
            d, lo, hi, p = boot(r["ndcg@10"] - cos["ndcg@10"])
            rowsr[f"K0={K0},gamma={gamma}"] = dict(ndcg10=float(r["ndcg@10"].mean()),
                                                   delta=d, lo=lo, hi=hi, p=p)
            any_pos |= lo > 0
            print(f"  {K0:>4}{gamma:>7g}{r['ndcg@10'].mean():>10.4f}{d:>+14.4f}"
                  f"   [{lo:>+.4f},{hi:>+.4f}]")
    npos = sum(1 for v in rowsr.values() if v["lo"] > 0)
    print(f"\n  cells where PRF beats cosine with the interval excluding 0: "
          f"{npos}/{len(rowsr)}")
    print(f"  VERDICT: our Rocchio " +
          ("DOES help on its home turf -- the implementation is exonerated, and its "
           "failure elsewhere is a property of those tasks."
           if any_pos else
           "does NOT help even here. Either SciFact's dense baseline is already too "
           "strong for PRF, or our PRF is weak -- and until that is resolved, "
           "'prior art does not help either' is not a claim this project can make."))
    R["X104b_prf"] = dict(rows=rowsr, n_positive=npos, n_cells=len(rowsr),
                          prf_helps_somewhere=bool(any_pos))
    R["_meta"] = dict(seconds=round(time.time() - t0, 1), corpus="BEIR/scifact",
                      n_docs=len(texts), n_queries=len(questions),
                      mean_gold=float(np.mean([len(g) for g in gold])),
                      n_train=len(train), n_test=len(test), k_cutoff=K_CUTOFF,
                      cluster="query (independent, ordinary bootstrap)")
    OUT.write_text(json.dumps(R, indent=2, default=str))
    print(f"\nwrote {OUT}  ({R['_meta']['seconds']}s)")
