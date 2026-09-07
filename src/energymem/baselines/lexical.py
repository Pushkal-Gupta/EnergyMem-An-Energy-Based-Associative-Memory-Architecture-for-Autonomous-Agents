"""Lexical and hybrid baselines. Owner: Rajit (docs/03_TASK_ALLOCATION.md).

A dense-only baseline is a weak baseline. Reviewers of retrieval papers expect BM25 and
a hybrid, because on short factual cues BM25 is frequently the thing that wins. See
docs/06_RISK_REGISTER.md R-7.
"""
from __future__ import annotations

import re
import time
from typing import Sequence

import numpy as np

from ..base import Memory, RetrievalResult, l2_normalize

_TOK = re.compile(r"[a-z0-9']+")


def tokenize(s: str) -> list[str]:
    return _TOK.findall(s.lower())


class BM25(object):
    """Okapi BM25 over the raw text. No embeddings involved."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.memories: list[Memory] = []
        self._bm = None

    def index(self, memories: Sequence[Memory]) -> None:
        from rank_bm25 import BM25Okapi
        self.memories = list(memories)
        self._bm = BM25Okapi([tokenize(m.text) for m in self.memories], k1=self.k1, b=self.b)

    def scores(self, query: str) -> np.ndarray:
        return np.asarray(self._bm.get_scores(tokenize(query)), dtype=np.float64)

    def retrieve(self, query: str, k: int = 5) -> RetrievalResult:
        t0 = time.perf_counter()
        s = self.scores(query)
        order = np.argsort(-s)[:k]
        return RetrievalResult(
            ids=[self.memories[i].id for i in order],
            scores=[float(s[i]) for i in order],
            latency_ms=(time.perf_counter() - t0) * 1e3,
        )


class HybridRRF(object):
    """Reciprocal-rank fusion of BM25 and dense cosine.

    RRF is the standard hybrid and it is a genuinely strong baseline -- it is what a
    practitioner would actually deploy. Beating dense-only while losing to RRF is not a
    result worth reporting as a win.
    """

    def __init__(self, k_rrf: int = 60):
        self.k_rrf = k_rrf
        self.memories: list[Memory] = []
        self.bm = BM25()
        self.X: np.ndarray | None = None

    def index(self, memories: Sequence[Memory]) -> None:
        self.memories = list(memories)
        self.bm.index(memories)
        self.X = l2_normalize(np.stack([m.embedding for m in memories]), axis=1).T

    def retrieve(self, query: str, q_vec: np.ndarray, k: int = 5) -> RetrievalResult:
        t0 = time.perf_counter()
        lex = self.bm.scores(query)
        den = self.X.T @ l2_normalize(q_vec)
        r_lex = np.empty(len(lex)); r_lex[np.argsort(-lex)] = np.arange(len(lex))
        r_den = np.empty(len(den)); r_den[np.argsort(-den)] = np.arange(len(den))
        fused = 1.0 / (self.k_rrf + 1 + r_lex) + 1.0 / (self.k_rrf + 1 + r_den)
        order = np.argsort(-fused)[:k]
        return RetrievalResult(
            ids=[self.memories[i].id for i in order],
            scores=[float(fused[i]) for i in order],
            latency_ms=(time.perf_counter() - t0) * 1e3,
        )
