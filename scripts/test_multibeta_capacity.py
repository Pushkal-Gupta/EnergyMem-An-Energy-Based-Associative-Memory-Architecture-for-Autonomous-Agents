"""Capacity sweep, take 2: does a per-memory beta (sculpted from local crowding,
self-calibrated) survive where a single global beta=32 broke down?

Reruns the exact same setup as test_capacity_confidence.py's capacity sweep
(N=100..4000, 50% occlusion, Imagenette-160 + ResNet18), adding a third condition:
MultiBetaEnergyMemory (per-memory beta, see src/energymem/memory/multibeta.py).

Run:
    python scripts/test_multibeta_capacity.py
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
from energymem.base import Memory, l2_normalize                  # noqa: E402
from energymem.baselines.dense import DenseTopK                  # noqa: E402
from energymem.memory.energy import EnergyMemory                 # noqa: E402
from energymem.memory.multibeta import MultiBetaEnergyMemory     # noqa: E402
from energymem.eval.metrics import bootstrap_ci                  # noqa: E402
from PIL import Image                                              # noqa: E402

N_PER_CLASS_VALUES = [10, 20, 40, 80, 150, 250, 400]   # -> totals 100..4000
MAX_N_PER_CLASS = max(N_PER_CLASS_VALUES)
N_QUERY_PER_CLASS = 8
FIXED_OCCLUSION = 0.5
SEEDS = [0, 1, 2]


def build_memories(mem_paths, embs):
    return [
        Memory(id=str(p), text=p.parent.name, embedding=embs[i])
        for i, p in enumerate(mem_paths)
    ]


def main() -> int:
    if not DATA_ROOT.exists():
        print(f"Missing dataset at {DATA_ROOT}.")
        return 1

    print("building encoder...")
    model, preprocess = build_encoder()

    print("\n" + "=" * 72)
    print("CAPACITY SWEEP v2 -- cosine vs fixed-beta=32 vs per-memory multi-beta")
    print("=" * 72)

    rng = random.Random(0)
    max_mem_paths = collect_paths("train", MAX_N_PER_CLASS, rng)
    query_paths = collect_paths("val", N_QUERY_PER_CLASS, rng)
    print(f"\nembedding max pool: {len(max_mem_paths)} images...")
    t0 = time.perf_counter()
    max_embs = embed_images(model, preprocess, max_mem_paths)
    print(f"done in {time.perf_counter() - t0:.1f}s")

    by_class: dict[str, list[int]] = {}
    for i, p in enumerate(max_mem_paths):
        by_class.setdefault(p.parent.name, []).append(i)

    results = {"cosine": {}, "energy fixed beta=32": {}, "energy multi-beta (self-calibrated)": {}}
    gains_used = {}

    for n_per_class in N_PER_CLASS_VALUES:
        idx = [i for c in by_class for i in by_class[c][:n_per_class]]
        sub_paths = [max_mem_paths[i] for i in idx]
        sub_embs = max_embs[idx]
        memories = build_memories(sub_paths, sub_embs)
        total_n = len(memories)

        dense = DenseTopK()
        dense.index(memories)
        energy_fixed = EnergyMemory(n_steps=8, beta=32.0)
        energy_fixed.index(memories)
        t0 = time.perf_counter()
        energy_multi = MultiBetaEnergyMemory(n_steps=8)
        energy_multi.index(memories)
        calib_time = time.perf_counter() - t0
        gains_used[total_n] = energy_multi.gain_used

        hits = {name: [] for name in results}
        for seed in SEEDS:
            srng = random.Random(seed)
            corrupted = [occlude(Image.open(p).convert("RGB"), FIXED_OCCLUSION, srng) for p in query_paths]
            with torch.no_grad():
                batch = torch.stack([preprocess(im) for im in corrupted])
                q_embs = l2_normalize(model(batch).numpy().astype(np.float64), axis=1)
            gold_classes = [p.parent.name for p in query_paths]

            for q, gold in zip(q_embs, gold_classes):
                for name, retr in [("cosine", dense), ("energy fixed beta=32", energy_fixed),
                                    ("energy multi-beta (self-calibrated)", energy_multi)]:
                    res = retr.retrieve(q, k=1)
                    pred = next(m.text for m in memories if m.id == res.ids[0])
                    hits[name].append(float(pred == gold))

        cis = {name: bootstrap_ci(hits[name]) for name in results}
        for name in results:
            results[name][total_n] = cis[name]
        print(f"\n  N={total_n:>5}  (multi-beta calibration took {calib_time:.1f}s, chose gain={gains_used[total_n]})")
        for name in results:
            m, lo, hi = cis[name]
            print(f"    {name:<38} {m:.3f} [{lo:.3f},{hi:.3f}]")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for name, series in results.items():
        ns = sorted(series.keys())
        means = [series[n][0] for n in ns]
        los = [series[n][1] for n in ns]
        his = [series[n][2] for n in ns]
        ax.plot(ns, means, marker="o", label=name)
        ax.fill_between(ns, los, his, alpha=0.12)
    ax.set_xscale("log")
    ax.set_xlabel("memory bank size (N, log scale)")
    ax.set_ylabel("class-level Recall@1 (50% occlusion)")
    ax.set_title("Capacity sweep: does per-memory beta rescue the\ncapacity limit found with a fixed global beta?")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    out_path = Path(__file__).resolve().parents[1] / "experiments" / "results" / "capacity_sweep_multibeta.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nplot saved to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
