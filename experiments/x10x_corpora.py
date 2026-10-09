"""X100, X102, X103, X104, X105 -- family H, the second corpus (and the third, fourth...).

Every method, grid and null setting is the LoCoMo one from experiments/locomo_qa.py,
run unchanged on public corpora through experiments/corpus_qa.py. Predictions are in
experiment-backlog/PREREG_HI.md, which was committed alone before this driver existed.

  X100  LongMemEval-S          each question's own ~490-turn haystack; m axis = multi-
                               session vs single-session question types
  X102  HotpotQA, 2WikiMultihop paragraphs of 100 questions pooled per bank; the P0, and
                               2Wiki's bridge_comparison (m=4) vs m=2 types is the
                               within-corpus cardinality axis
  X103  QASPER                 substitute for QuALITY/NarrativeQA (no evidence labels)
  X104  BEIR SciFact, NFCorpus whole corpus is the bank
  X105  transfer               each method's LoCoMo-selected config (the modal pick in
                               x29_baselines.json / x31_expansion.json), applied with no
                               retuning, reported next to the in-corpus selection

X101 (MSC) is the X26 association probe, not a QA task, and lives in x101_msc.py.

    python experiments/x10x_corpora.py hotpot 2wiki longmemeval qasper scifact nfcorpus
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments import corpus_qa as C                            # noqa: E402

RES = C.RESULTS


def locomo_transfer() -> dict:
    """Modal LoCoMo fold pick per method on the all-category stratum."""
    out = {}
    for f in ("x29_baselines.json", "x31_expansion.json"):
        d = json.loads((RES / f).read_text())["results"]["all"]
        for m, v in d.items():
            if isinstance(v, dict) and "picks" in v and m not in out:
                out[m] = collections.Counter(p["picked"] for p in v["picks"]).most_common(1)[0][0]
    return out


def corpus(name):
    """-> (specs, catnames, cluster_level, strata, interactions, group_fn)"""
    if name == "hotpot":
        specs, cn = C.load_hotpot()
        return specs, cn, "bank", {"bridge": (1,), "comparison": (2,), "all": None}, \
            [("bridge", "comparison")]
    if name == "2wiki":
        specs, cn = C.load_2wiki()
        return specs, cn, "bank", {
            "compositional": (1,), "inference": (2,), "comparison": (3,),
            "bridge_comparison (m=4)": (4,), "m=2 types": (1, 2, 3), "all": None}, \
            [("bridge_comparison (m=4)", "m=2 types")]
    if name == "longmemeval":
        specs, cn = C.load_longmemeval()
        return specs, cn, "question", {
            "multi-session": (1,), "temporal-reasoning": (2,), "knowledge-update": (3,),
            "single-session (all 3)": (4, 5, 6), "all": None}, \
            [("multi-session", "single-session (all 3)")]
    if name == "qasper":
        specs, cn = C.load_qasper()
        return specs, cn, "bank", {"single-evidence": (1,), "multi-evidence": (2,), "all": None}, \
            [("multi-evidence", "single-evidence")]
    if name in ("scifact", "nfcorpus"):
        specs, cn = C.load_beir(name)
        return specs, cn, "question", {"single-relevant": (1,), "multi-relevant": (2,),
                                       "all": None}, [("multi-relevant", "single-relevant")]
    raise SystemExit(f"unknown corpus {name}")


IDS = {"hotpot": "X102", "2wiki": "X102", "longmemeval": "X100", "qasper": "X103",
       "scifact": "X104", "nfcorpus": "X104"}


def main(names):
    transfer = locomo_transfer()
    print("X105 transfer configs (LoCoMo modal picks):", transfer)
    for name in names:
        specs, cn, level, strata, inter = corpus(name)
        bks = C.make_banks(specs)
        # banks are clusters for pooled corpora; for per-question corpora the bank count is
        # large (LongMemEval) or the banks are the groups already (BEIR)
        groups = C.groups_by_bank(bks) if len(bks) > C.N_GROUPS else list(range(len(bks)))
        res = C.run_corpus(name, bks, groups, level, cn, strata, interactions=inter,
                           transfer=transfer)
        out = C.stamp(f"{IDS[name]}+X105", encoder="all-mpnet-base-v2",
                      selection=f"leave-one-group-out over {len(set(groups))} groups, train side only",
                      primary=C.PRIMARY, transfer_configs=transfer, **res)
        C.write(f"x10x_{name}.json", out)


if __name__ == "__main__":
    main(sys.argv[1:] or ["hotpot", "2wiki", "longmemeval", "qasper", "scifact", "nfcorpus"])
