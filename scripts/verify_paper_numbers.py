r"""Check numbers TYPED into paper prose against experiments/results/*.json.

    python scripts/verify_paper_numbers.py          # exits 1 on any mismatch

WHY THIS EXISTS. `make_tables.py` enforces generate-never-type for tables, and nothing
enforced it for prose. Prose is where the damage is worse: a table a reader distrusts can be
checked against its generator, but a sentence saying "+0.023 on bridge questions" is
unfalsifiable without the JSON open alongside. Every claim below is asserted against the
file that produced it, so a re-run that changes a number fails this script instead of
silently disagreeing with the paper.

WHAT IT DOES NOT DO. It does not find numbers in the paper and look them up -- it asserts a
hand-written list of (claim, json path, expected value) triples. A number typed into the
paper and omitted from this list is unchecked, and that gap is real. The list is ordered by
section so the omissions are visible.

Tolerance is 5e-4 absolute, i.e. agreement at the three decimals the paper prints.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

RES = Path(__file__).resolve().parents[1] / "experiments" / "results"
TOL = 5e-4

_cache: dict[str, dict] = {}


def J(name: str) -> dict:
    if name not in _cache:
        _cache[name] = json.loads((RES / f"{name}.json").read_text())
    return _cache[name]


def get(name: str, path: str):
    """Walk a '/'-separated path through a JSON document."""
    node = J(name)
    for part in path.split("/"):
        node = node[part]
    return node


# (section, claim as printed, file, json path, expected)
CHECKS = [
    # -- sec:composition, MSC (X201) -----------------------------------------
    ("composition", "MSC hop1 cosine R@1 = 0.049", "x101_msc",
     "hop1/methods/cosine/R@1", 0.049),
    ("composition", "MSC hop1 predictor R@1 = 0.065", "x101_msc",
     "hop1/methods/averaged/R@1", 0.065),
    ("composition", "MSC hop1 delta = +0.016", "x101_msc",
     "hop1/methods/averaged/dR1", 0.016),
    ("composition", "MSC hop1 CI lo = +0.014", "x101_msc",
     "hop1/methods/averaged/lo", 0.014),
    ("composition", "MSC hop1 CI hi = +0.018", "x101_msc",
     "hop1/methods/averaged/hi", 0.018),
    ("composition", "MSC hop1 R@5 delta = +0.028", "x101_msc",
     "hop1/methods/averaged/dR5", 0.028),
    ("composition", "MSC hop1 n = 27815", "x101_msc", "hop1/n", 27815),
    ("composition", "MSC bank = 30320 turns", "x101_msc", "bank", 30320),
    ("composition", "MSC sessions = 2505", "x101_msc", "sessions", 2505),
    ("composition", "MSC mixture below averaging: +0.006", "x101_msc",
     "hop1/methods/mixture/dR1", 0.006),
    ("composition", "MSC hop2 delta = +0.005", "x101_msc",
     "hop2/methods/averaged/dR1", 0.005),
    ("composition", "MSC hop2 R@5 delta = +0.010", "x101_msc",
     "hop2/methods/averaged/dR5", 0.010),
    ("composition", "MSC hop2 n = 25310", "x101_msc", "hop2/n", 25310),

    # -- sec:corpora, the expansion positive (X202) ---------------------------
    ("corpora", "HotpotQA bridge delta = +0.023", "x10x_hotpot",
     "strata/bridge/mixture/vs_cosine/delta", 0.023),
    ("corpora", "HotpotQA bridge CI lo = +0.010", "x10x_hotpot",
     "strata/bridge/mixture/vs_cosine/lo", 0.010),
    ("corpora", "HotpotQA bridge CI hi = +0.037", "x10x_hotpot",
     "strata/bridge/mixture/vs_cosine/hi", 0.037),
    ("corpora", "HotpotQA bridge selects non-null in 10/10 folds", "x10x_hotpot",
     "strata/bridge/mixture/folds_backed_off_to_null", 0),
    ("corpora", "HotpotQA comparison self-nulls 10/10", "x10x_hotpot",
     "strata/comparison/mixture/folds_backed_off_to_null", 10),
    ("corpora", "HotpotQA pooled mixture - RRF = -0.017", "x10x_hotpot",
     "strata/all/mixture_vs_rrf/delta", -0.017),
    ("corpora", "HotpotQA pooled mix-RRF CI lo = -0.030", "x10x_hotpot",
     "strata/all/mixture_vs_rrf/lo", -0.030),
    ("corpora", "HotpotQA pooled mix-RRF CI hi = -0.005", "x10x_hotpot",
     "strata/all/mixture_vs_rrf/hi", -0.005),
    ("corpora", "HotpotQA I4 pooled = +0.009", "x10x_hotpot",
     "strata/all/I4_contrast/delta", 0.009),
    ("corpora", "2Wiki inference delta = +0.171", "x10x_2wiki",
     "strata/inference/mixture/vs_cosine/delta", 0.171),
    ("corpora", "2Wiki inference CI lo = +0.118", "x10x_2wiki",
     "strata/inference/mixture/vs_cosine/lo", 0.118),
    ("corpora", "2Wiki inference CI hi = +0.219", "x10x_2wiki",
     "strata/inference/mixture/vs_cosine/hi", 0.219),
    ("corpora", "2Wiki inference beats RRF by +0.093", "x10x_2wiki",
     "strata/inference/mixture_vs_rrf/delta", 0.093),
    ("corpora", "2Wiki compositional delta = +0.067", "x10x_2wiki",
     "strata/compositional/mixture/vs_cosine/delta", 0.067),
    ("corpora", "2Wiki compositional beats RRF by +0.061", "x10x_2wiki",
     "strata/compositional/mixture_vs_rrf/delta", 0.061),
    ("corpora", "2Wiki comparison self-nulls 10/10", "x10x_2wiki",
     "strata/comparison/mixture/folds_backed_off_to_null", 10),
    ("corpora", "2Wiki m=4 stratum is flat: -0.001", "x10x_2wiki",
     "strata/bridge_comparison (m=4)/mixture/vs_cosine/delta", -0.001),
    ("corpora", "QASPER multi-evidence delta = -0.027", "x10x_qasper",
     "strata/multi-evidence/mixture/vs_cosine/delta", -0.027),
    ("corpora", "QASPER multi-evidence CI lo = -0.049", "x10x_qasper",
     "strata/multi-evidence/mixture/vs_cosine/lo", -0.049),
    ("corpora", "QASPER multi-evidence CI hi = -0.010", "x10x_qasper",
     "strata/multi-evidence/mixture/vs_cosine/hi", -0.010),
    ("corpora", "QASPER pooled mixture - RRF = -0.044", "x10x_qasper",
     "strata/all/mixture_vs_rrf/delta", -0.044),
    ("corpora", "NFCorpus multi-relevant delta = +0.017", "x10x_nfcorpus",
     "strata/multi-relevant/mixture/vs_cosine/delta", 0.017),
    ("corpora", "QASPER bank = ~47 paragraphs", "x10x_qasper", "mean_bank", 47.067),

    # -- sec:corpora, LongMemEval (X200) --------------------------------------
    ("corpora", "LongMemEval expansion = -0.007", "x10x_longmemeval",
     "strata/all/mixture/vs_cosine/delta", -0.007),
    ("corpora", "LongMemEval expansion CI lo = -0.032", "x10x_longmemeval",
     "strata/all/mixture/vs_cosine/lo", -0.032),
    ("corpora", "LongMemEval expansion CI hi = +0.018", "x10x_longmemeval",
     "strata/all/mixture/vs_cosine/hi", 0.018),
    ("corpora", "LongMemEval RRF = +0.156", "x10x_longmemeval",
     "strata/all/rrf/vs_cosine/delta", 0.156),
    ("corpora", "LongMemEval RRF CI lo = +0.125", "x10x_longmemeval",
     "strata/all/rrf/vs_cosine/lo", 0.125),
    ("corpora", "LongMemEval RRF CI hi = +0.189", "x10x_longmemeval",
     "strata/all/rrf/vs_cosine/hi", 0.189),
    ("corpora", "LongMemEval RRF temporal = +0.255", "x10x_longmemeval",
     "strata/temporal-reasoning/rrf/vs_cosine/delta", 0.255),
    ("corpora", "LongMemEval knowledge-update = -0.049", "x10x_longmemeval",
     "strata/knowledge-update/mixture/vs_cosine/delta", -0.049),
    ("corpora", "LongMemEval mixture - RRF = -0.163", "x10x_longmemeval",
     "strata/all/mixture_vs_rrf/delta", -0.163),
    ("corpora", "LongMemEval n = 479", "x10x_longmemeval", "n_queries", 479),

    # -- sec:corpora, cardinality interactions (Table 8) ----------------------
    ("cardinality", "2Wiki interaction = -0.023", "x10x_2wiki",
     "interactions/bridge_comparison (m=4) - m=2 types/dod", -0.023),
    ("cardinality", "QASPER interaction = -0.036", "x10x_qasper",
     "interactions/multi-evidence - single-evidence/dod", -0.036),
    ("cardinality", "SciFact interaction = -0.025", "x10x_scifact",
     "interactions/multi-relevant - single-relevant/dod", -0.025),
    ("cardinality", "NFCorpus interaction = +0.011", "x10x_nfcorpus",
     "interactions/multi-relevant - single-relevant/dod", 0.011),
    ("cardinality", "LongMemEval interaction = +0.002", "x10x_longmemeval",
     "interactions/multi-session - single-session (all 3)/dod", 0.002),
    ("cardinality", "Flickr8k interaction = -0.008", "x112_crossmodal",
     "interaction_i2t_minus_t2i/dod", -0.008),
    ("cardinality", "mosaic interaction is exactly 0", "x111_mosaic",
     "interactions/m=4 - m=1/dod", 0.0),
    ("cardinality", "mosaic MMR at m=2 = +0.070", "x111_mosaic",
     "strata/m=2/mmr/vs_cosine/delta", 0.070),
    ("cardinality", "Flickr8k i2t averaging = +0.012", "x112_crossmodal",
     "image_to_text/strata/all/mixture_avg/vs_cosine/delta", 0.012),
    ("cardinality", "Flickr8k i2t mixture n.s. = +0.004", "x112_crossmodal",
     "image_to_text/strata/all/mixture/vs_cosine/delta", 0.004),

    # -- sec:taxonomy, X210 ---------------------------------------------------
    ("taxonomy", "X210 image descent - vote = -0.0313", "x110_occlusion",
     "image/energy-knn/delta", -0.0313),
    ("taxonomy", "X210 image descent - vote CI lo = -0.0397", "x110_occlusion",
     "image/energy-knn/lo", -0.0397),
    ("taxonomy", "X210 image descent - vote CI hi = -0.0227", "x110_occlusion",
     "image/energy-knn/hi", -0.0227),
    ("taxonomy", "X210 text descent - cosine = +0.003", "x110_occlusion",
     "text_delete/energy-cosine/delta", 0.003),
    ("taxonomy", "X210 text vote - cosine = +0.104", "x110_occlusion",
     "text_delete/knn-cosine/delta", 0.104),
    ("taxonomy", "X210 masked text descent = +0.002", "x110_occlusion",
     "text_mask/energy-cosine/delta", 0.002),
    ("taxonomy", "X210 masked text vote = +0.111", "x110_occlusion",
     "text_mask/knn-cosine/delta", 0.111),
    ("taxonomy", "X210 and X113a measure the same descent effect", "x110_occlusion",
     "image/energy-cosine/delta",
     get("x113_class_vote", "X113a/contrasts/descent_minus_cosine/delta")),

    # -- sec:occlusion, X113 (already audited; re-asserted so a re-run is caught)
    ("occlusion", "X113a descent - vote = -0.036", "x113_class_vote",
     "X113a/contrasts/descent_minus_knn/delta", -0.036),
    ("occlusion", "X113b same-image delta = -0.003", "x113_class_vote",
     "X113b/same_image/delta", -0.003),
]


def main() -> int:
    bad, by_section = [], {}
    for section, claim, fname, path, expected in CHECKS:
        by_section.setdefault(section, [0, 0])
        try:
            actual = get(fname, path)
        except (KeyError, FileNotFoundError) as exc:
            bad.append((section, claim, fname, path, expected, f"MISSING: {exc}"))
            by_section[section][1] += 1
            continue
        ok = abs(float(actual) - float(expected)) <= (
            TOL if isinstance(expected, float) else 0)
        if ok:
            by_section[section][0] += 1
        else:
            by_section[section][1] += 1
            bad.append((section, claim, fname, path, expected, actual))

    for section, (good, fail) in by_section.items():
        mark = "ok  " if not fail else "FAIL"
        print(f"  {mark} {section:12} {good}/{good + fail}")

    if bad:
        print(f"\n{len(bad)} MISMATCH(ES) -- the paper disagrees with the results files:")
        for section, claim, fname, path, expected, actual in bad:
            print(f"  [{section}] {claim}")
            print(f"      {fname}.json :: {path}")
            print(f"      paper says {expected!r}, file says {actual!r}")
        return 1

    total = sum(g + f for g, f in by_section.values())
    print(f"\n{total}/{total} typed numbers agree with experiments/results/*.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
