"""Association probes: retrieval targets that are TEMPORALLY adjacent, not semantically
similar.

The whole Phase-3 premise rests on there being queries that similarity search cannot
answer. The canonical case is "what happened next" -- the correct memory follows the cue
in time, and there is no guarantee it looks like it. "Viewed the flat on Warren Street"
is followed by "The broker wants a two-month deposit", which shares almost nothing
lexically and is not the nearest neighbour of the cue.

X16 GATES EVERYTHING: if cosine already solves these, the probe is invalid and the
world-model claim has nothing to stand on. Measure before building.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AssocProbe:
    cue_idx: int          # the episode used as the cue
    gold_idx: int         # the episode that FOLLOWS it in its thread
    thread: str
    hop: int = 1          # 1 = next episode, 2 = two ahead, ...


def build_probes(episodes, hop: int = 1) -> list[AssocProbe]:
    """One probe per consecutive (or hop-distant) pair inside each thread."""
    by_thread: dict[str, list[int]] = {}
    for i, e in enumerate(episodes):
        if e.thread in ("filler",) or e.thread.endswith("_nearmiss"):
            continue
        by_thread.setdefault(e.thread, []).append(i)

    probes: list[AssocProbe] = []
    for th, idxs in by_thread.items():
        idxs = sorted(idxs, key=lambda i: episodes[i].t)
        for a in range(len(idxs) - hop):
            probes.append(AssocProbe(cue_idx=idxs[a], gold_idx=idxs[a + hop],
                                     thread=th, hop=hop))
    return probes
