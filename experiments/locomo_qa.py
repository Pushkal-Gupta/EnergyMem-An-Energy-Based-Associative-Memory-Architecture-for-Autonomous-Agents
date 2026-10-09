"""Shared machinery for the LoCoMo QA evidence-retrieval task (backlog families A-D).

NOT a driver -- it computes no result. Drivers import from here so that every method in
every family sees identical vectors, identical folds and identical exclusion sets.

THE TASK. LoCoMo ships 1986 QA pairs annotated with the dialogue turns that answer them
(``evidence``). 1977 of them resolve to turns that exist. Bank = one conversation's turns
(419-689, mean 588); query = the question text; gold = the evidence turn ids. Category 1
is multi-hop (mean 3.13 gold turns, 95.4% spanning >1 session) and category 4 is
single-hop (1.06) -- so category 4 is a built-in single-target control for anything
claimed about category 1.

FOLDS. Leave-one-conversation-out, 10 folds. Hyperparameters are selected on the nine
training conversations and applied to the held-out one. Conversation is also the
bootstrap cluster. Ten clusters is few; drivers say so.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import os                                                       # noqa: E402
os.environ.setdefault("ENERGYMEM_CACHE", str(ROOT / ".cache/emb"))

from energymem.baselines.lexical import BM25                    # noqa: E402
from energymem.base import l2_normalize                         # noqa: E402
from energymem.encoders import SentenceEncoder                  # noqa: E402
from energymem.eval.locomo import load_locomo                   # noqa: E402
from energymem.readout import averaged_scores, mixture_scores   # noqa: E402

DATA = ROOT / "data" / "locomo10.json"
RESULTS = ROOT / "experiments" / "results"
SPLIT_SEED = 12345            # fixed for the life of the project
KS = (1, 5, 10, 20)
CATNAME = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}


@dataclass
class Bank:
    sample_id: str
    ids: list[str]
    texts: list[str]
    Z: np.ndarray                      # (N, d) unit rows
    Q: np.ndarray                      # (M, d) unit rows
    gold: list[list[int]]              # per query, bank indices
    cats: list[int]
    answers: list[str]
    questions: list[str]
    sessions: list[str]
    speakers: list[str]
    _bm25: np.ndarray | None = field(default=None, repr=False)

    @property
    def n(self) -> int:
        return len(self.ids)

    @property
    def m(self) -> int:
        return len(self.gold)

    def bm25(self) -> np.ndarray:
        """(M, N) BM25 scores, computed once."""
        if self._bm25 is None:
            idx = BM25()
            from energymem.base import Memory
            idx.index([Memory(id=i, text=t, timestamp=float(j))
                       for j, (i, t) in enumerate(zip(self.ids, self.texts))])
            self._bm25 = np.stack([idx.scores(q) for q in self.questions])
        return self._bm25


def banks(encoder: SentenceEncoder | None = None, model: str | None = None) -> list[Bank]:
    enc = encoder or SentenceEncoder(model) if model else (encoder or SentenceEncoder())
    out = []
    for c in load_locomo(DATA):
        pos = {m.id: i for i, m in enumerate(c.memories)}
        keep = [q for q in c.queries if any(g in pos for g in q.gold_ids)]
        out.append(Bank(
            sample_id=c.sample_id,
            ids=[m.id for m in c.memories],
            texts=[m.text for m in c.memories],
            Z=enc.encode([m.text for m in c.memories]),
            Q=enc.encode([q.question for q in keep]),
            gold=[[pos[g] for g in q.gold_ids if g in pos] for q in keep],
            cats=[q.category for q in keep],
            answers=[q.answer for q in keep],
            questions=[q.question for q in keep],
            sessions=[str(m.metadata.get("session")) for m in c.memories],
            speakers=[str(m.metadata.get("speaker")) for m in c.memories],
        ))
    return out


def folds(bks: list[Bank]) -> list[tuple[list[int], int]]:
    """Leave-one-conversation-out: (train indices, test index)."""
    return [([j for j in range(len(bks)) if j != i], i) for i in range(len(bks))]


# ------------------------------------------------------------------ methods
# Every method maps (bank, query index, params) -> scores over the bank, (N,).
# `cosine` with no params is the (T=1, beta->inf, b=0, R=I) corner of Proposition 1,
# and every expansion method must reduce to it exactly at its null setting.

def s_cosine(b: Bank, i: int, **_) -> np.ndarray:
    return b.Z @ b.Q[i]


def s_bm25(b: Bank, i: int, **_) -> np.ndarray:
    return b.bm25()[i]


def _ranks_from(scores: np.ndarray) -> np.ndarray:
    """1-based rank of every item under `scores` (descending)."""
    order = np.argsort(-scores)
    r = np.empty(len(scores), dtype=np.int64)
    r[order] = np.arange(1, len(scores) + 1)
    return r


def s_rrf(b: Bank, i: int, k_rrf: float = 60.0, **_) -> np.ndarray:
    rc, rl = _ranks_from(s_cosine(b, i)), _ranks_from(s_bm25(b, i))
    return 1.0 / (k_rrf + rc) + 1.0 / (k_rrf + rl)


def s_mmr(b: Bank, i: int, lam: float = 0.7, pool: int = 50, **_) -> np.ndarray:
    """Greedy MMR over the top-`pool` cosine candidates.

    THE CRITICAL BASELINE. If diversity reranking captures the gain that mode
    expansion gets, the mechanism is diversity and not association. Scores are
    descending-by-selection-order so the induced ranking is what MMR would return.
    """
    s = s_cosine(b, i)
    cand = list(np.argsort(-s)[:pool])
    sel: list[int] = []
    out = s.copy()
    out[cand] = -np.inf
    while cand:
        if not sel:
            j = cand[0]
        else:
            sub = b.Z[cand] @ b.Z[sel].T
            j = cand[int(np.argmax(lam * s[cand] - (1 - lam) * sub.max(axis=1)))]
        out[j] = 1e6 - len(sel)
        sel.append(j)
        cand.remove(j)
    return out


def _top(b: Bank, i: int, K0: int) -> np.ndarray:
    return np.argsort(-(b.Z @ b.Q[i]))[:K0]


def s_rocchio(b: Bank, i: int, K0: int = 5, gamma: float = 0.5, **_) -> np.ndarray:
    """Pseudo-relevance feedback: AVERAGE the feedback turns into the query.

    This is the mode-averaging baseline and the prior art a reviewer will name.
    gamma=0 (or K0=0) reduces exactly to cosine.
    """
    if K0 <= 0 or gamma == 0.0:
        return s_cosine(b, i)
    fb = b.Z[_top(b, i, K0)].mean(axis=0)
    return b.Z @ l2_normalize(b.Q[i] + gamma * fb)


def s_mixture(b: Bank, i: int, K0: int = 5, beta: float = 16.0, gamma: float = 1.0,
              with_query: bool = True, **_) -> np.ndarray:
    """Keep the feedback turns as SEPARATE modes; read out with logsumexp.

    score(m) = logsumexp_s [ beta * <xi_s, z_m> + log w_s ],  w_s ∝ exp(beta*cos(q,z_s))

    NULL SETTING: K0<=0 leaves only the query mode, and a single mode makes
    mixture_scores a strictly increasing function of cosine -- identical ranking.
    """
    q = b.Q[i]
    hyps = [(q, 1.0)] if with_query else []
    if K0 > 0:
        t = _top(b, i, K0)
        w = np.exp(beta * (b.Z[t] @ q))
        w = w / w.sum()
        hyps += [(b.Z[j], float(gamma * wj) + 1e-12) for j, wj in zip(t, w)]
    if not hyps:
        hyps = [(q, 1.0)]
    return mixture_scores(hyps, b.Z, beta)


def s_mixture_avg(b: Bank, i: int, K0: int = 5, beta: float = 16.0, gamma: float = 1.0,
                  with_query: bool = True, **_) -> np.ndarray:
    """Identical hypotheses to s_mixture, collapsed by AVERAGING instead of logsumexp.

    The matched comparison for invariant I4: same modes, same weights, only the
    combination rule differs. Any gap between this and s_mixture is attributable to
    the readout alone.
    """
    q = b.Q[i]
    hyps = [(q, 1.0)] if with_query else []
    if K0 > 0:
        t = _top(b, i, K0)
        w = np.exp(beta * (b.Z[t] @ q))
        w = w / w.sum()
        hyps += [(b.Z[j], float(gamma * wj) + 1e-12) for j, wj in zip(t, w)]
    if not hyps:
        hyps = [(q, 1.0)]
    return averaged_scores(hyps, b.Z)


METHODS = {
    "cosine": s_cosine, "bm25": s_bm25, "rrf": s_rrf, "mmr": s_mmr,
    "rocchio": s_rocchio, "mixture": s_mixture, "mixture_avg": s_mixture_avg,
}


# --------------------------------------------------------------- evaluation

from energymem.eval.setmetrics import (                          # noqa: E402
    full_set_recall_at_k, mean_gold_recall_at_k, ndcg_at_k, reciprocal_rank,
)


def evaluate(b: Bank, fn, ks=KS, cats=None, **params) -> dict:
    """Per-question metrics for one method on one bank.

    Returns arrays aligned to the (filtered) query list so drivers can pair methods
    question-by-question. No exclusion is applied: for QA the query is not a bank
    item, so there is nothing to exclude -- and the exclusion must be identical
    across methods or an asymmetry masquerades as a finding.
    """
    rows = {f"full@{k}": [] for k in ks}
    rows |= {f"gold@{k}": [] for k in ks}
    rows |= {"mrr": [], "ndcg@10": [], "cat": [], "m": []}
    for i in range(b.m):
        if cats is not None and b.cats[i] not in cats:
            continue
        g = b.gold[i]
        order = np.argsort(-fn(b, i, **params))
        for k in ks:
            rows[f"full@{k}"].append(full_set_recall_at_k(order, g, k))
            rows[f"gold@{k}"].append(mean_gold_recall_at_k(order, g, k))
        rows["mrr"].append(reciprocal_rank(order, g))
        rows["ndcg@10"].append(ndcg_at_k(order, g, 10))
        rows["cat"].append(b.cats[i])
        rows["m"].append(len(g))
    return {k: np.asarray(v) for k, v in rows.items()}


def pooled(bks, fn, cats=None, which="gold@10", **params):
    """(values, clusters) across banks -- the arrays a paired test consumes."""
    vals, clus = [], []
    for b in bks:
        r = evaluate(b, fn, cats=cats, **params)
        vals.append(r[which])
        clus.append(np.full(len(r[which]), b.sample_id))
    return np.concatenate(vals), np.concatenate(clus)


# ------------------------------------------------- sweeps and fold selection
# Pattern used by every driver that has a hyperparameter: evaluate each grid point on
# each conversation ONCE, then do leave-one-conversation-out selection by aggregating
# only the TRAINING conversations' already-computed numbers. Selection never touches the
# held-out conversation, and the per-fold choices are reported.

import itertools                                                 # noqa: E402


def grid_points(grid: dict) -> list[dict]:
    if not grid:
        return [{}]
    keys = sorted(grid)
    return [dict(zip(keys, v)) for v in itertools.product(*(grid[k] for k in keys))]


def _key(p: dict) -> str:
    return ",".join(f"{k}={p[k]}" for k in sorted(p)) or "default"


def sweep(bks: list[Bank], fn, grid: dict, cats=None, ks=KS) -> dict:
    """{combo_key: [per-bank metric dicts]} -- every grid point on every conversation."""
    return {_key(p): [evaluate(b, fn, ks=ks, cats=cats, **p) for b in bks]
            for p in grid_points(grid)}


def fold_select(bks: list[Bank], table: dict, score_key: str = "gold@10"):
    """Leave-one-conversation-out selection over a sweep table.

    Returns (values, clusters, cats, picks): per-question metric values on the held-out
    conversations under the hyperparameter chosen from that fold's training side only.
    """
    vals, clus, cats, picks = [], [], [], []
    for train, test in folds(bks):
        best, best_s = None, -np.inf
        for k, per_bank in table.items():
            s = float(np.mean(np.concatenate([per_bank[j][score_key] for j in train])))
            if s > best_s:
                best, best_s = k, s
        r = table[best][test]
        vals.append(r[score_key])
        clus.append(np.full(len(r[score_key]), bks[test].sample_id))
        cats.append(r["cat"])
        picks.append(dict(test=bks[test].sample_id, picked=best, train_score=best_s))
    return (np.concatenate(vals), np.concatenate(clus), np.concatenate(cats), picks)


def fold_select_all(bks: list[Bank], table: dict, score_key: str = "gold@10", keys=None):
    """Same selection, but return EVERY metric on the held-out side, not just one."""
    keys = keys or [f"full@{k}" for k in KS] + [f"gold@{k}" for k in KS] + ["mrr", "ndcg@10"]
    out = {k: [] for k in keys}
    clus, cats, picks = [], [], []
    for train, test in folds(bks):
        best, best_s = None, -np.inf
        for k, per_bank in table.items():
            s = float(np.mean(np.concatenate([per_bank[j][score_key] for j in train])))
            if s > best_s:
                best, best_s = k, s
        r = table[best][test]
        for k in keys:
            out[k].append(r[k])
        clus.append(np.full(len(r["cat"]), bks[test].sample_id))
        cats.append(r["cat"])
        picks.append(dict(test=bks[test].sample_id, picked=best, train_score=best_s))
    return ({k: np.concatenate(v) for k, v in out.items()},
            np.concatenate(clus), np.concatenate(cats), picks)
