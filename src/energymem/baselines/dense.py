"""Baselines.

Read docs/06_RISK_REGISTER.md R-7 before touching this file. A weak baseline is the
single most common reason a paper of this shape gets rejected: reviewers assume you
tuned yours and not theirs. So tune theirs. Give every baseline the same
hyperparameter search budget as EnergyMem and report that budget in the paper.

Baselines to implement (Rajit, Week 1):
    [x] DenseTopK       -- exact cosine, the provable special case
    [ ] BM25            -- lexical; surprisingly strong on entity-heavy queries
    [ ] HybridRRF       -- reciprocal rank fusion of dense + BM25
    [ ] RerankedTopK    -- cosine shortlist + cross-encoder. THE baseline to beat.
    [ ] PAMBaseline     -- arXiv:2602.11322, MIT-licensed runnable code, see citation audit
    [ ] HeLaMemBaseline -- ACL 2026; check whether their GitHub code runs
    [ ] HippoRAG / A-MEM / MemoryBank -- whichever have maintained code
"""

from __future__ import annotations

import time
from typing import Sequence

import numpy as np

from ..base import Memory, RetrievalResult, Retriever, l2_normalize


class DenseTopK(Retriever):
    """Exact cosine top-k.

    Kept as a standalone implementation *in addition to*
    ``energy.dense_topk_config()`` so that the reduction test has an independent
    reference to compare against. If the two ever disagree, that is a real bug.
    """

    def __init__(self, encoder=None) -> None:
        self.encoder = encoder
        self.memories: list[Memory] = []
        self.X: np.ndarray | None = None

    def index(self, memories: Sequence[Memory]) -> None:
        self.memories = list(memories)
        self.X = l2_normalize(
            np.stack([m.embedding for m in memories]), axis=1
        ).T.astype(np.float64)

    def retrieve(self, query: str | np.ndarray, k: int = 5) -> RetrievalResult:
        if self.X is None:
            raise RuntimeError("index() before retrieve()")
        t0 = time.perf_counter()
        q = (
            query.astype(np.float64)
            if isinstance(query, np.ndarray)
            else np.asarray(self.encoder.encode(query), dtype=np.float64)
        )
        sims = self.X.T @ l2_normalize(q)
        order = np.argsort(-sims)[:k]
        return RetrievalResult(
            ids=[self.memories[i].id for i in order],
            scores=[float(sims[i]) for i in order],
            latency_ms=(time.perf_counter() - t0) * 1e3,
        )
