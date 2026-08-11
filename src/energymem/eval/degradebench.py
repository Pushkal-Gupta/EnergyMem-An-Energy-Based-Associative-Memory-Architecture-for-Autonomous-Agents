"""DegradeBench -- controlled query degradation (contribution C4).

Five corruption families x five severities. This is a released artifact: it gets cited
even if EnergyMem loses, so build it to be reusable by people who do not care about
energy-based memory.

    D1 masking          drop tokens from the query
    D2 paraphrase       LLM rewrite at increasing semantic distance
    D3 embedding noise  q <- normalize(q + sigma * eps)
    D4 distractors      inject near-duplicates differing in one key fact
    D5 composite cue    query needs 2-4 memories jointly  <-- the G1 discriminator

D5 is the family that one-shot top-k structurally cannot solve: there is no mechanism
for combining evidence across stored items. D4 is the G2 discriminator: usage history
is the only signal separating near-duplicates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from ..base import l2_normalize

MASK_RATES = [0.0, 0.15, 0.30, 0.50, 0.70]
NOISE_SIGMAS = [0.0, 0.05, 0.10, 0.20, 0.40]
DISTRACTOR_COUNTS = [0, 1, 5, 20, 100]
COMPOSITE_TARGETS = [1, 2, 3, 4]


@dataclass
class DegradedQuery:
    query: str | np.ndarray
    gold_ids: list[str]
    family: str
    severity: int


def mask_tokens(text: str, rate: float, rng: np.random.Generator) -> str:
    """D1. Drop a fraction of tokens, preserving order."""
    toks = text.split()
    if not toks or rate <= 0:
        return text
    keep = rng.random(len(toks)) >= rate
    if not keep.any():          # never return an empty cue
        keep[rng.integers(len(toks))] = True
    return " ".join(t for t, k in zip(toks, keep) if k)


def add_embedding_noise(
    q: np.ndarray, sigma: float, rng: np.random.Generator
) -> np.ndarray:
    """D3. Isotropic Gaussian perturbation, renormalized."""
    if sigma <= 0:
        return q
    return l2_normalize(q + sigma * rng.normal(size=q.shape))


# --- TODO (Rajit, Week 2) ---------------------------------------------------
# D2 paraphrase: needs an LLM. Keep a fully-offline fallback (local model) so that
#     no experiment is blocked on API credits -- see risk register R-9.
# D4 distractors: generate near-duplicates that differ in exactly one factual slot
#     (date, name, quantity). The point is that cosine cannot separate them and
#     usage history can.
# D5 composite: sample k memories sharing an entity, build a query whose answer needs
#     all k. Gold is the full set; score with Recall@k over the set, not top-1.
# ---------------------------------------------------------------------------


def build_suite(
    memories,
    encoder: Callable | None = None,
    seed: int = 0,
) -> list[DegradedQuery]:
    """Generate the full suite. Commit both this generator AND the generated split --
    reviewers must be able to reproduce the exact queries, not just the process."""
    raise NotImplementedError(
        "Week 2 deliverable (Rajit). D1/D3 helpers above are ready to wire up."
    )
