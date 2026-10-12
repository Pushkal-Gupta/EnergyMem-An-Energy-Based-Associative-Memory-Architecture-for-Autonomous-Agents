#!/usr/bin/env python3
"""Preflight: what a fresh checkout can and cannot re-run, and exactly what is missing.

The paper's numbers all live in committed JSON under `experiments/results/`, so the
table generator, the typed-number verifier and the test suite run with no corpora
present at all. Re-running a *driver* is a different matter: `data/` is gitignored and
most corpora are redistributed under their own licences, so the reader has to fetch
them. This script says which, where to put them, and what breaks without them -- so
the reproducibility claim in the appendix can be audited rather than taken on trust.

    python scripts/check_data.py            # report
    python scripts/check_data.py --strict   # exit 1 if any input is missing

Tier 0 inputs need no data. Tier 1 is what this repo's README already documents.
Tier 2 was added by the later corpus work and is the part a fresh environment will
trip over.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RES = ROOT / "experiments" / "results"

# (tier, expected path under data/, dataset, source, drivers that need it)
INPUTS = [
    (1, "locomo10.json", "LoCoMo",
     "raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json",
     ["scripts/exp_locomo_anchored.py", "experiments/locomo_assoc.py"]),
    (1, "imagenette2-160/", "Imagenette-160 (99 MB)",
     "s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz (untar into data/)",
     ["scripts/test_imagenet_hopfield.py", "experiments/x113_class_vote.py",
      "experiments/x11x_modalities.py"]),
    (2, "hotpotqa/validation.parquet", "HotpotQA (distractor, validation)",
     "HF dataset `hotpotqa/hotpot_qa`, config `distractor`",
     ["experiments/x10x_corpora.py", "experiments/x102_checks.py"]),
    (2, "2wiki/validation.parquet", "2WikiMultihopQA (validation)",
     "github.com/Alab-NII/2wikimultihop",
     ["experiments/x10x_corpora.py"]),
    (2, "qasper/qasper-dev-v0.3.json", "QASPER v0.3 (dev)",
     "allenai.org/data/qasper",
     ["experiments/x10x_corpora.py"]),
    (2, "longmemeval/longmemeval_s_cleaned.json", "LongMemEval-S",
     "github.com/xiaowu0162/LongMemEval",
     ["experiments/x10x_corpora.py"]),
    (2, "beir/scifact/corpus.parquet", "BEIR SciFact",
     "HF dataset `BeIR/scifact` (+ `BeIR/scifact-qrels`)",
     ["experiments/x10x_corpora.py", "experiments/x104_beir.py"]),
    (2, "beir/nfcorpus/corpus.parquet", "BEIR NFCorpus",
     "HF dataset `BeIR/nfcorpus` (+ `BeIR/nfcorpus-qrels`)",
     ["experiments/x10x_corpora.py"]),
    (2, "msc/test.parquet", "Multi-Session Chat (test)",
     "ParlAI `msc` task; the run recorded `corpus = \"MSC test\"`",
     ["experiments/x101_msc.py"]),
    (2, "flickr8k/test.parquet", "Flickr8k (test)",
     "HF dataset `jxie/flickr8k`, as recorded in x112_crossmodal.json",
     ["experiments/x11x_modalities.py"]),
    (2, "yahoo/test.parquet", "Yahoo Answers Topics (test)",
     "HF dataset `yahoo_answers_topics`",
     ["experiments/x11x_modalities.py"]),
]

# Things that run with no corpora at all, because they read committed JSON.
TIER0 = [
    ("python -m pytest -q", "31 tests, including the Proposition 1 gate"),
    ("python scripts/make_tables.py", "9 tables from experiments/results/*.json"),
    ("python scripts/verify_paper_numbers.py", "typed prose numbers vs. those JSONs"),
    ("cd paper && tectonic -X compile main.tex", "the PDF"),
]

OPTIONAL_DEPS = [
    ("pyarrow", "every Tier-2 parquet corpus"),
    ("torch, torchvision", "the image arms (Imagenette, mosaics)"),
    ("sentence-transformers", "all text encoders; downloads all-mpnet-base-v2 on first use"),
    ("sentence-transformers + CLIP weights", "clip-ViT-B-32, for the cross-modal arm"),
    ("matplotlib", "figure regeneration"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true",
                    help="exit 1 if any declared input is missing")
    args = ap.parse_args()

    print("Tier 0 -- runs in a fresh checkout with NO corpora present:")
    for cmd, what in TIER0:
        print(f"  {cmd:44} {what}")
    n_res = len(list(RES.glob('*.json'))) if RES.is_dir() else 0
    print(f"  ({n_res} committed result files under experiments/results/)\n")

    missing = []
    for tier in (1, 2):
        label = {1: "Tier 1 -- documented in README.md",
                 2: "Tier 2 -- added by the later corpus work"}[tier]
        print(f"{label}:")
        for t, rel, name, source, drivers in INPUTS:
            if t != tier:
                continue
            path = DATA / rel
            ok = path.exists()
            if not ok:
                missing.append((rel, name, source, drivers))
            print(f"  [{'OK  ' if ok else 'MISS'}] data/{rel}")
            print(f"         {name}")
            if not ok:
                print(f"         fetch: {source}")
                print(f"         blocks: {', '.join(drivers)}")
        print()

    print("Optional dependencies, and what needs them:")
    for dep, what in OPTIONAL_DEPS:
        print(f"  {dep:38} {what}")
    print()

    total = len(INPUTS)
    print(f"{total - len(missing)}/{total} declared data inputs present.")
    if missing:
        print("\nThe commands above that touch a missing input will fail with "
              "FileNotFoundError.\nEvery number in the paper is still backed by its "
              "committed result file; what is\nunavailable without these is "
              "re-running the driver that produced it.")
    return 1 if (missing and args.strict) else 0


if __name__ == "__main__":
    sys.exit(main())
