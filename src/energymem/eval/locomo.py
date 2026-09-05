"""LoCoMo loader (Maharana et al., ACL 2024) -- real long-conversation agent memory.

Each conversation becomes one memory bank: every dialogue turn is a Memory, keyed by
its ``dia_id`` (e.g. "D1:3"). Each QA item becomes a query whose gold set is the
``evidence`` turn ids. That makes LoCoMo a *retrieval* benchmark with ground truth,
which is what we need -- we are evaluating the memory, not an answer-generating reader.

Categories (as distributed): 1 = multi-hop, 2 = temporal, 3 = open-domain,
4 = single-hop, 5 = adversarial. Category 1 averages ~3.1 gold turns per question and
is genuinely multi-target; category 4 averages ~1.07 and is single-target. We report
them separately because the two regimes ask different things of a retriever.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ..base import Memory


@dataclass
class LocomoQuery:
    question: str
    gold_ids: list[str]
    category: int
    answer: str


@dataclass
class LocomoConversation:
    sample_id: str
    memories: list[Memory]
    queries: list[LocomoQuery]


def load_locomo(path: str | Path) -> list[LocomoConversation]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    out: list[LocomoConversation] = []

    for conv_idx, item in enumerate(raw):
        conv = item["conversation"]
        session_keys = [
            k for k in conv
            if k.startswith("session_") and not k.endswith("date_time")
            and isinstance(conv[k], list)
        ]

        # dia_ids are unique only WITHIN a conversation: across the 10 conversations the
        # 5882 turns carry just 1033 distinct ids. Pooling banks without namespacing lets
        # an unrelated turn from another conversation satisfy a gold id and score as a
        # hit. Namespace every id with sample_id; `local_id` keeps the raw id for
        # matching against that conversation's own `evidence` field.
        sample_id = str(item.get("sample_id", f"conv{conv_idx}"))

        memories: list[Memory] = []
        seen: set[str] = set()
        for order, skey in enumerate(sorted(session_keys, key=_session_index)):
            for turn in conv[skey]:
                dia_id = turn.get("dia_id")
                text = turn.get("text")
                if not dia_id or not text or dia_id in seen:
                    continue
                seen.add(dia_id)
                memories.append(
                    Memory(
                        id=f"{sample_id}/{dia_id}",
                        text=f"{turn.get('speaker', '?')}: {text}",
                        timestamp=float(len(memories)),
                        source=skey,
                        metadata={
                            "session": skey,
                            "speaker": turn.get("speaker"),
                            "local_id": dia_id,
                            "sample_id": sample_id,
                        },
                    )
                )

        queries: list[LocomoQuery] = []
        for qa in item.get("qa", []):
            question = qa.get("question")
            gold = [f"{sample_id}/{g}" for g in (qa.get("evidence") or []) if g in seen]
            if not question or not gold:
                continue  # unanswerable-by-retrieval; excluded and counted by the caller
            queries.append(
                LocomoQuery(
                    question=question,
                    gold_ids=gold,
                    category=int(qa.get("category", -1)),
                    answer=str(qa.get("answer", "")),
                )
            )

        out.append(
            LocomoConversation(
                sample_id=str(item.get("sample_id", f"conv{conv_idx}")),
                memories=memories,
                queries=queries,
            )
        )
    return out


def _session_index(key: str) -> int:
    try:
        return int(key.split("_")[1])
    except (IndexError, ValueError):
        return 0
