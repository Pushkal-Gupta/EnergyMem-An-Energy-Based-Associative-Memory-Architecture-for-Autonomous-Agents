"""HotpotQA loader for family H -- the corpus LoCoMo could not be.

WHY THIS CORPUS. X32's pre-registered interaction never appeared on LoCoMo, and the single
most likely innocent explanation was the geometry: LoCoMo's gold evidence turns have mean
pairwise inner product eps = 0.527, i.e. they are mutually SIMILAR, so a single query
vector can already sit near all of them and there is nothing for a multi-mode readout to
do. HotpotQA's supporting sentences come from two different Wikipedia articles about
different entities. If eps is much lower here and the effect still does not appear, the
geometric excuse is gone.

TASK. Each question ships 10 paragraphs: 2 gold and 8 distractors (the "distractor"
setting). We retrieve at SENTENCE level, because that is the granularity
`supporting_facts` annotates -- (title, sent_id) pairs. Bank = every sentence of the 10
paragraphs; gold = the annotated supporting sentences.

CLUSTERS. HotpotQA questions are independent and carry their own banks, so the question is
the resampling unit and the bootstrap is an ordinary one over thousands of units. This is
the one place in the project where the inference is not floored by ten clusters.

CATEGORIES. `type` is the built-in difficulty axis and it is the analogue of LoCoMo's
cat 1 vs cat 4: `bridge` questions must traverse from one entity to another (genuine
multi-hop), `comparison` questions name both entities up front (parallel lookup).
"""
from __future__ import annotations

import hashlib
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
from energymem.base import Memory                               # noqa: E402
from energymem.encoders import SentenceEncoder                  # noqa: E402

PARQUET = ROOT / "data" / "external" / "hotpotqa" / "validation.parquet"
N_QUESTIONS = 1200              # deterministic subsample; CPU encoding is the binding cost
SPLIT_SEED = 12345              # fixed for the life of the project


@dataclass
class HotpotItem:
    qid: str
    question: str
    answer: str
    qtype: str                  # 'bridge' | 'comparison'
    level: str                  # 'easy' | 'medium' | 'hard'
    texts: list[str]            # bank: every sentence of the 10 paragraphs
    gold: list[int]             # bank indices of the supporting sentences
    Z: np.ndarray = field(repr=False, default=None)     # (n, d)
    q: np.ndarray = field(repr=False, default=None)     # (d,)
    _bm25: np.ndarray | None = field(default=None, repr=False)

    @property
    def n(self) -> int:
        return len(self.texts)

    def bm25(self) -> np.ndarray:
        if self._bm25 is None:
            idx = BM25()
            idx.index([Memory(id=str(j), text=t, timestamp=float(j))
                       for j, t in enumerate(self.texts)])
            self._bm25 = idx.scores(self.question)
        return self._bm25


def _rows():
    import pyarrow.parquet as pq
    t = pq.read_table(PARQUET)
    # deterministic subsample by hash of the id, so the choice does not depend on row order
    rows = t.to_pylist()
    rows.sort(key=lambda r: hashlib.sha256(r["id"].encode()).hexdigest())
    return rows[:N_QUESTIONS]


def load(encoder: SentenceEncoder | None = None, verbose: bool = True) -> list[HotpotItem]:
    enc = encoder or SentenceEncoder()
    raw = _rows()
    items: list[HotpotItem] = []
    dropped = 0
    for r in raw:
        titles = r["context"]["title"]
        sents = r["context"]["sentences"]
        texts, where = [], {}
        for ti, (title, para) in enumerate(zip(titles, sents)):
            for si, s in enumerate(para):
                s = s.strip()
                if not s:
                    continue
                where[(title, si)] = len(texts)
                # the title is part of the evidence: a bare sentence often has no subject
                texts.append(f"{title}: {s}")
        g = [where[(t, s)] for t, s in zip(r["supporting_facts"]["title"],
                                           r["supporting_facts"]["sent_id"])
             if (t, s) in where]
        g = sorted(set(g))
        if not g or len(texts) < 10:
            dropped += 1
            continue
        items.append(HotpotItem(qid=r["id"], question=r["question"], answer=r["answer"],
                                qtype=r["type"], level=r["level"], texts=texts, gold=g))
    # encode every unique sentence once, then hand out views
    uniq: dict[str, int] = {}
    for it in items:
        for t in it.texts:
            uniq.setdefault(t, len(uniq))
    keys = list(uniq)
    if verbose:
        print(f"  {len(items)} questions, {dropped} dropped (unresolvable gold), "
              f"{len(keys)} unique sentences to encode")
    E = enc.encode(keys)
    Q = enc.encode([it.question for it in items])
    for i, it in enumerate(items):
        it.Z = E[[uniq[t] for t in it.texts]]
        it.q = Q[i]
    return items


def split(items: list[HotpotItem]) -> tuple[list[int], list[int]]:
    """Disjoint train/test halves by hash of the question id. Selection sees train only."""
    rng = np.random.default_rng(SPLIT_SEED)
    perm = rng.permutation(len(items))
    half = len(items) // 2
    return sorted(perm[:half].tolist()), sorted(perm[half:].tolist())


def gold_eps(items: list[HotpotItem], idxs=None) -> tuple[float, int]:
    """Mean pairwise inner product within gold sets -- the number LoCoMo put at 0.527."""
    vals = []
    for i in (idxs if idxs is not None else range(len(items))):
        it = items[i]
        if len(it.gold) < 2:
            continue
        G = it.Z[it.gold]
        M = G @ G.T
        iu = np.triu_indices(len(it.gold), 1)
        vals.extend(M[iu].tolist())
    return (float(np.mean(vals)) if vals else float("nan")), len(vals)
