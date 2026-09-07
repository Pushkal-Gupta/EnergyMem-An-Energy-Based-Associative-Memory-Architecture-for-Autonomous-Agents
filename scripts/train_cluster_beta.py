"""Gradient-trained per-cluster beta vs. fixed beta vs. cosine.

This is the trained version of the per-pattern-beta idea. Regions = k-means clusters
(the high-dimensional stand-in for "fixed-size regions" -- see beta_training.py
docstring). Beta per region is TRAINED by gradient descent on a self-supervised
recall loss, not hand-derived from a formula.

Two evaluations, deliberately different from training:
  1. Held-out validation corruption (same family as training: embedding dropout) --
     sanity that training converged at all.
  2. The REAL test pipeline used throughout this project: pixel occlusion on actual
     held-out images, re-embedded through ResNet18. This corruption family is NEVER
     seen during training. If the trained beta only helps on (1) and not (2), that is
     itself the finding -- it means the model learned to exploit the synthetic
     corruption rather than anything about the landscape's real structure.

Also reports: did training discover higher beta for crowded clusters on its own
(without being told the crowding heuristic), and does the reduction check still hold
before training starts (uniform init must match EnergyMemory exactly).

Run:
    python scripts/train_cluster_beta.py
"""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from test_imagenet_hopfield import (  # noqa: E402
    DATA_ROOT, build_encoder, embed_images, occlude, collect_paths,
)
from energymem.base import Memory, l2_normalize                          # noqa: E402
from energymem.baselines.dense import DenseTopK                          # noqa: E402
from energymem.memory.energy import EnergyMemory                         # noqa: E402
from energymem.memory.multibeta import MultiBetaEnergyMemory, crowding_z # noqa: E402
from energymem.memory.beta_training import kmeans_regions, train_cluster_beta  # noqa: E402
from energymem.eval.metrics import bootstrap_ci                          # noqa: E402

N_PER_CLASS = 40          # -> 400-image bank, matches the earlier capacity-sweep point
N_QUERY_PER_CLASS = 8
OCCLUSION_LEVELS = [0.0, 0.3, 0.5, 0.7]
SEEDS = [0, 1, 2]
N_CLUSTERS = 16
EPOCHS = 400


def build_memories(paths, embs):
    return [Memory(id=str(p), text=p.parent.name, embedding=embs[i])
            for i, p in enumerate(paths)]


def eval_condition(retr, id_to_class, q_sets, gold) -> float:
    hits = []
    for qs in q_sets:
        for q, g in zip(qs, gold):
            pred_id = retr.retrieve(q, k=1).ids[0]
            hits.append(float(id_to_class[pred_id] == g))
    return float(np.mean(hits))


def main() -> int:
    if not DATA_ROOT.exists():
        print(f"missing dataset at {DATA_ROOT}")
        return 1

    print("building encoder...")
    model, preprocess = build_encoder()

    rng = random.Random(0)
    mem_paths = collect_paths("train", N_PER_CLASS, rng)
    query_paths = collect_paths("val", N_QUERY_PER_CLASS, rng)
    print(f"memory bank: {len(mem_paths)} images; queries: {len(query_paths)} images")
    mem_embs = embed_images(model, preprocess, mem_paths)
    memories = build_memories(mem_paths, mem_embs)
    id_to_class = {m.id: m.text for m in memories}
    gold = [p.parent.name for p in query_paths]

    X = l2_normalize(np.stack(mem_embs), axis=1).T  # (d, N)

    # ---------------------------------------------------------------- reduction check
    print("\n[reduction] uniform init (before any training) must equal EnergyMemory(beta)")
    probes = l2_normalize(np.random.default_rng(1).normal(size=(40, X.shape[0])), axis=1)
    ok = True
    for beta in (16.0, 32.0):
        ref = EnergyMemory(n_steps=8, beta=beta); ref.index(memories)
        mb = MultiBetaEnergyMemory(n_steps=8, beta_base=beta, lam=0.0); mb.index(memories)
        agree = sum(ref.retrieve(q, 5).ids == mb.retrieve(q, 5).ids for q in probes)
        good = agree == len(probes)
        ok &= good
        print(f"    beta={beta}: {agree}/{len(probes)} identical  {'OK' if good else 'FAIL'}")
    if not ok:
        print("    => reduction FAILED; stopping.")
        return 1

    # ---------------------------------------------------------------- cluster + train
    print(f"\nclustering {X.shape[1]} memories into {N_CLUSTERS} regions (k-means)...")
    cluster_id = kmeans_regions(X, N_CLUSTERS, seed=0)
    sizes = np.bincount(cluster_id)
    print(f"  region sizes: min={sizes.min()} max={sizes.max()} mean={sizes.mean():.1f}")

    print(f"\ntraining beta per region for {EPOCHS} epochs (gradient descent, Adam)...")
    t0 = time.perf_counter()
    beta_vec, hist = train_cluster_beta(X, cluster_id, n_steps=8, epochs=EPOCHS, beta_init=32.0)
    print(f"  done in {time.perf_counter() - t0:.1f}s")
    print(f"  val loss: {hist.val_loss[0]:.3f} -> {hist.val_loss[-1]:.3f}   "
          f"val acc: {hist.val_acc[0]:.3f} -> {hist.val_acc[-1]:.3f}")

    beta_per_cluster = beta_vec[np.unique(cluster_id, return_index=True)[1]]
    # recompute properly: one representative beta per cluster id in order
    beta_per_cluster = np.array([beta_vec[cluster_id == c][0] for c in range(N_CLUSTERS)])
    print(f"\n  learned beta per region: min={beta_per_cluster.min():.1f} "
          f"max={beta_per_cluster.max():.1f} spread={beta_per_cluster.max()-beta_per_cluster.min():.1f}")

    # does the model rediscover "crowded regions need higher beta" on its own?
    z = crowding_z(X)
    region_crowding = np.array([z[cluster_id == c].mean() for c in range(N_CLUSTERS)])
    corr = np.corrcoef(region_crowding, beta_per_cluster)[0, 1]
    print(f"  correlation(region crowding, learned beta) = {corr:.3f}  "
          f"({'crowded regions got HIGHER beta, as hypothesized' if corr > 0.2 else 'no clear relationship' if abs(corr) <= 0.2 else 'crowded regions got LOWER beta -- opposite of hypothesis'})")

    # ---------------------------------------------------------------- build conditions
    dense = DenseTopK(); dense.index(memories)
    fixed = EnergyMemory(n_steps=8, beta=32.0); fixed.index(memories)
    trained = MultiBetaEnergyMemory(n_steps=8, beta_base=32.0, lam=0.0)
    trained.index(memories)
    trained.beta_vec = beta_vec  # drop in the TRAINED per-region values

    # ---------------------------------------------------------------- eval 1: same-family (dropout)
    print("\n" + "=" * 74)
    print("EVAL 1 -- held-out embedding-dropout corruption (SAME family as training)")
    print("=" * 74)
    val_rng = np.random.default_rng(999)
    for rate in [0.3, 0.5, 0.7]:
        results = {"cosine": [], "energy fixed b=32": [], "trained per-region beta": []}
        for i, m in enumerate(memories):
            keep = val_rng.random(X.shape[0]) >= rate
            if not keep.any():
                keep[val_rng.integers(X.shape[0])] = True
            q = l2_normalize(X[:, i] * keep)
            for name, retr in [("cosine", dense), ("energy fixed b=32", fixed),
                                ("trained per-region beta", trained)]:
                pred = retr.retrieve(q, 1).ids[0]
                results[name].append(float(pred == m.id))
        print(f"\n  dropout={rate:.0%}")
        for name, hits in results.items():
            m_, lo, hi = bootstrap_ci(hits)
            print(f"    {name:<26} {m_:.3f} [{lo:.3f},{hi:.3f}]")

    # ---------------------------------------------------------------- eval 2: real pixel occlusion
    print("\n" + "=" * 74)
    print("EVAL 2 -- REAL pixel occlusion on held-out images (NEVER seen during training)")
    print("=" * 74)
    q_sets_by_level = {}
    for lvl in OCCLUSION_LEVELS:
        sets = []
        for seed in SEEDS:
            srng = random.Random(seed)
            imgs = [occlude(Image.open(p).convert("RGB"), lvl, srng) for p in query_paths]
            with torch.no_grad():
                batch = torch.stack([preprocess(im) for im in imgs])
                sets.append(l2_normalize(model(batch).numpy().astype(np.float64), axis=1))
        q_sets_by_level[lvl] = sets

    print(f"\n  {'occlusion':>10} | {'cosine':>10} | {'energy b=32':>12} | {'trained region-beta':>20}")
    print("  " + "-" * 60)
    for lvl in OCCLUSION_LEVELS:
        row = {}
        for name, retr in [("cosine", dense), ("energy b=32", fixed), ("trained", trained)]:
            row[name] = eval_condition(retr, id_to_class, q_sets_by_level[lvl], gold)
        print(f"  {lvl:>9.0%} | {row['cosine']:>10.3f} | {row['energy b=32']:>12.3f} | {row['trained']:>20.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
