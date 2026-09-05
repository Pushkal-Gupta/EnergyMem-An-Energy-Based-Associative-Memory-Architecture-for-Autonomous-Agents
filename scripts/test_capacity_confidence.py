"""Two follow-up tests on top of test_imagenet_hopfield.py:

  1. CAPACITY SWEEP -- as the memory bank grows (100 -> 4,000 images), does recall
     hold up for cosine vs. energy descent? Classic Hopfield-network question: every
     associative memory has a capacity limit past which stored patterns interfere.

  2. CONFIDENCE CALIBRATION -- each retrieval already returns a "margin" (how much the
     winning memory beats the runner-up). Does a low margin actually predict a wrong
     answer? If yes, the energy memory can flag its own uncertainty -- something plain
     cosine top-1 does not cleanly give you.

Reuses the encoder/data helpers from test_imagenet_hopfield.py so nothing is duplicated.

Run:
    python scripts/test_capacity_confidence.py
"""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from test_imagenet_hopfield import (  # noqa: E402
    DATA_ROOT, build_encoder, embed_images, occlude, collect_paths,
)
from energymem.base import Memory, l2_normalize          # noqa: E402
from energymem.baselines.dense import DenseTopK          # noqa: E402
from energymem.memory.energy import EnergyMemory         # noqa: E402
from energymem.eval.metrics import bootstrap_ci          # noqa: E402
from PIL import Image                                     # noqa: E402

N_PER_CLASS_VALUES = [10, 20, 40, 80, 150, 250, 400]   # -> totals 100..4000
MAX_N_PER_CLASS = max(N_PER_CLASS_VALUES)
N_QUERY_PER_CLASS = 8
FIXED_OCCLUSION = 0.5
SEEDS = [0, 1, 2]

CONF_N_PER_CLASS = 30          # memory bank for the confidence test
CONF_OCCLUSIONS = [0.0, 0.2, 0.4, 0.6, 0.8]
CONF_SEEDS = [0, 1, 2, 3, 4]


def build_memories(mem_paths, embs):
    return [
        Memory(id=str(p), text=p.parent.name, embedding=embs[i])
        for i, p in enumerate(mem_paths)
    ]


def run_capacity(model, preprocess):
    print("\n" + "=" * 72)
    print("TEST 1 -- CAPACITY SWEEP")
    print("does recall hold up as the memory bank grows?")
    print("=" * 72)

    rng = random.Random(0)
    max_mem_paths = collect_paths("train", MAX_N_PER_CLASS, rng)
    query_paths = collect_paths("val", N_QUERY_PER_CLASS, rng)
    print(f"\nembedding max pool: {len(max_mem_paths)} images (largest N needed)...")
    t0 = time.perf_counter()
    max_embs = embed_images(model, preprocess, max_mem_paths)
    print(f"done in {time.perf_counter() - t0:.1f}s")

    # group by class so we can slice per-class subsets cleanly
    by_class: dict[str, list[int]] = {}
    for i, p in enumerate(max_mem_paths):
        by_class.setdefault(p.parent.name, []).append(i)

    results = {"cosine": {}, "energy T=8 beta=32": {}}

    for n_per_class in N_PER_CLASS_VALUES:
        idx = [i for c in by_class for i in by_class[c][:n_per_class]]
        sub_paths = [max_mem_paths[i] for i in idx]
        sub_embs = max_embs[idx]
        memories = build_memories(sub_paths, sub_embs)

        dense = DenseTopK()
        dense.index(memories)
        energy = EnergyMemory(n_steps=8, beta=32.0)
        energy.index(memories)

        cos_hits, en_hits = [], []
        for seed in SEEDS:
            srng = random.Random(seed)
            corrupted = [occlude(Image.open(p).convert("RGB"), FIXED_OCCLUSION, srng) for p in query_paths]
            with torch.no_grad():
                batch = torch.stack([preprocess(im) for im in corrupted])
                q_embs = l2_normalize(model(batch).numpy().astype(np.float64), axis=1)
            gold_classes = [p.parent.name for p in query_paths]

            for q, gold in zip(q_embs, gold_classes):
                c_res = dense.retrieve(q, k=1)
                e_res = energy.retrieve(q, k=1)
                c_pred = next(m.text for m in memories if m.id == c_res.ids[0])
                e_pred = next(m.text for m in memories if m.id == e_res.ids[0])
                cos_hits.append(float(c_pred == gold))
                en_hits.append(float(e_pred == gold))

        total_n = len(memories)
        c_mean, c_lo, c_hi = bootstrap_ci(cos_hits)
        e_mean, e_lo, e_hi = bootstrap_ci(en_hits)
        results["cosine"][total_n] = (c_mean, c_lo, c_hi)
        results["energy T=8 beta=32"][total_n] = (e_mean, e_lo, e_hi)
        print(f"  N={total_n:>5}  cosine={c_mean:.3f} [{c_lo:.3f},{c_hi:.3f}]   "
              f"energy={e_mean:.3f} [{e_lo:.3f},{e_hi:.3f}]")

    return results


def run_confidence(model, preprocess):
    print("\n" + "=" * 72)
    print("TEST 2 -- CONFIDENCE CALIBRATION")
    print("does a low margin (top-1 vs top-2 score) predict a wrong answer?")
    print("=" * 72)

    rng = random.Random(0)
    mem_paths = collect_paths("train", CONF_N_PER_CLASS, rng)
    query_paths = collect_paths("val", 15, rng)
    print(f"\nmemory bank: {len(mem_paths)} images; query pool: {len(query_paths)} images")
    mem_embs = embed_images(model, preprocess, mem_paths)
    memories = build_memories(mem_paths, mem_embs)

    dense = DenseTopK()
    dense.index(memories)
    energy = EnergyMemory(n_steps=8, beta=32.0)
    energy.index(memories)

    records = {"cosine": [], "energy T=8 beta=32": []}   # each: (margin, correct)

    for seed in CONF_SEEDS:
        srng = random.Random(seed)
        for lvl in CONF_OCCLUSIONS:
            corrupted = [occlude(Image.open(p).convert("RGB"), lvl, srng) for p in query_paths]
            with torch.no_grad():
                batch = torch.stack([preprocess(im) for im in corrupted])
                q_embs = l2_normalize(model(batch).numpy().astype(np.float64), axis=1)
            gold_classes = [p.parent.name for p in query_paths]

            for q, gold in zip(q_embs, gold_classes):
                for name, retr in [("cosine", dense), ("energy T=8 beta=32", energy)]:
                    res = retr.retrieve(q, k=2)
                    pred = next(m.text for m in memories if m.id == res.ids[0])
                    correct = float(pred == gold)
                    margin = res.scores[0] - res.scores[1] if len(res.scores) > 1 else 1.0
                    records[name].append((margin, correct))

    print()
    for name in records:
        margins = np.array([m for m, _ in records[name]])
        correct = np.array([c for _, c in records[name]])
        # bucket into quartiles of margin, report accuracy per bucket
        qs = np.quantile(margins, [0.0, 0.25, 0.5, 0.75, 1.0])
        print(f"  {name}:")
        for i in range(4):
            lo, hi = qs[i], qs[i + 1]
            mask = (margins >= lo) & (margins <= hi if i == 3 else margins < hi)
            if mask.sum() == 0:
                continue
            acc = correct[mask].mean()
            print(f"    margin quartile {i+1} [{lo:.3f},{hi:.3f}]: "
                  f"accuracy={acc:.3f}  (n={mask.sum()})")
        corr = np.corrcoef(margins, correct)[0, 1]
        print(f"    correlation(margin, correct) = {corr:.3f}\n")

    return records


def plot_capacity(results, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    for name, series in results.items():
        ns = sorted(series.keys())
        means = [series[n][0] for n in ns]
        los = [series[n][1] for n in ns]
        his = [series[n][2] for n in ns]
        ax.plot(ns, means, marker="o", label=name)
        ax.fill_between(ns, los, his, alpha=0.15)
    ax.set_xscale("log")
    ax.set_xlabel("memory bank size (N, log scale)")
    ax.set_ylabel("class-level Recall@1 (50% occlusion)")
    ax.set_title("Capacity sweep: recall vs. memory bank size")
    ax.legend()
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"capacity plot saved to {out_path}")


def plot_confidence(records, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    for name in records:
        margins = np.array([m for m, _ in records[name]])
        correct = np.array([c for _, c in records[name]])
        qs = np.quantile(margins, [0.0, 0.25, 0.5, 0.75, 1.0])
        xs, ys = [], []
        for i in range(4):
            lo, hi = qs[i], qs[i + 1]
            mask = (margins >= lo) & (margins <= hi if i == 3 else margins < hi)
            if mask.sum() == 0:
                continue
            xs.append(i + 1)
            ys.append(correct[mask].mean())
        ax.plot(xs, ys, marker="o", label=name)
    ax.set_xlabel("margin quartile (1=least confident, 4=most confident)")
    ax.set_ylabel("accuracy")
    ax.set_title("Confidence calibration:\ndoes a bigger margin mean a more trustworthy answer?")
    ax.set_xticks([1, 2, 3, 4])
    ax.legend()
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"confidence plot saved to {out_path}")


def main() -> int:
    if not DATA_ROOT.exists():
        print(f"Missing dataset at {DATA_ROOT}.")
        return 1

    print("building encoder...")
    model, preprocess = build_encoder()

    cap_results = run_capacity(model, preprocess)
    conf_records = run_confidence(model, preprocess)

    out_dir = Path(__file__).resolve().parents[1] / "experiments" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_capacity(cap_results, out_dir / "capacity_sweep.png")
    plot_confidence(conf_records, out_dir / "confidence_calibration.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
