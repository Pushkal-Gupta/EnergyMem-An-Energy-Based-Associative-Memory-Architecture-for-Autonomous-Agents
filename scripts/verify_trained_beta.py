"""Verification and controls for the gradient-trained per-region beta.

The training run reported val_acc 1.000 -> 1.000 and val_loss 1.515 -> 1.510: the
objective was already saturated at initialization, so there was almost no gradient
telling the model how to retrieve *better*. Cross-entropy still has a gradient at 100%
accuracy -- it pushes for higher confidence -- and raising beta raises confidence. That
predicts every beta drifts upward for reasons unrelated to retrieval quality, which is
exactly what happened (32 -> 52..88).

So the headline comparison (trained heterogeneous beta vs fixed beta=32) is confounded:
any apparent gain could come purely from the higher AVERAGE beta, with the
region-to-region variation contributing nothing. Three controls settle it:

  C1  fixed beta = mean(learned betas)   -- isolates "higher beta" from "varying beta"
  C2  best fixed beta from a sweep       -- the honest baseline; beta=32 was never tuned
  C3  shuffled assignment                -- same learned beta VALUES, randomly reassigned
                                            to regions. If this matches the trained model,
                                            the learned assignment carries no information
                                            and only the magnitudes matter.

All conditions are scored on the same real pixel-occlusion queries with bootstrap CIs.

Run:
    python scripts/verify_trained_beta.py
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

N_PER_CLASS = 40
N_QUERY_PER_CLASS = 8
OCCLUSION_LEVELS = [0.3, 0.5, 0.7]
SEEDS = [0, 1, 2]
N_CLUSTERS = 16
EPOCHS = 400
FIXED_SWEEP = [8, 16, 32, 48, 64, 80, 96, 128]


def build_memories(paths, embs):
    return [Memory(id=str(p), text=p.parent.name, embedding=embs[i])
            for i, p in enumerate(paths)]


def make_beta_model(memories, beta_vec):
    m = MultiBetaEnergyMemory(n_steps=8, beta_base=32.0, lam=0.0)
    m.index(memories)
    m.beta_vec = np.asarray(beta_vec, dtype=np.float64)
    return m


def score(retr, id_to_class, q_sets_by_level, gold) -> tuple[list[float], dict]:
    """Returns (pooled per-query hits across all levels, per-level means)."""
    pooled, per_level = [], {}
    for lvl, sets in q_sets_by_level.items():
        hits = []
        for qs in sets:
            for q, g in zip(qs, gold):
                hits.append(float(id_to_class[retr.retrieve(q, 1).ids[0]] == g))
        per_level[lvl] = float(np.mean(hits))
        pooled.extend(hits)
    return pooled, per_level


def main() -> int:
    if not DATA_ROOT.exists():
        print(f"missing dataset at {DATA_ROOT}")
        return 1

    print("building encoder...")
    model, preprocess = build_encoder()

    rng = random.Random(0)
    mem_paths = collect_paths("train", N_PER_CLASS, rng)
    query_paths = collect_paths("val", N_QUERY_PER_CLASS, rng)
    mem_embs = embed_images(model, preprocess, mem_paths)
    memories = build_memories(mem_paths, mem_embs)
    id_to_class = {m.id: m.text for m in memories}
    gold = [p.parent.name for p in query_paths]
    X = l2_normalize(np.stack(mem_embs), axis=1).T
    print(f"bank {len(memories)} images, {len(query_paths)} queries")

    # ------------------------------------------------------------------ train
    cluster_id = kmeans_regions(X, N_CLUSTERS, seed=0)
    print(f"\ntraining {N_CLUSTERS} region betas for {EPOCHS} epochs...")
    t0 = time.perf_counter()
    beta_vec, hist = train_cluster_beta(X, cluster_id, n_steps=8, epochs=EPOCHS, beta_init=32.0)
    print(f"  {time.perf_counter() - t0:.1f}s   val_loss {hist.val_loss[0]:.4f} -> "
          f"{hist.val_loss[-1]:.4f}   val_acc {hist.val_acc[0]:.3f} -> {hist.val_acc[-1]:.3f}")
    if hist.val_acc[0] >= 0.999:
        print("  ** WARNING: val accuracy was already saturated at init. The objective")
        print("     could only push confidence, not correctness -- treat learned betas as")
        print("     confidence-driven drift, not evidence that heterogeneity helps. **")

    per_cluster = np.array([beta_vec[cluster_id == c][0] for c in range(N_CLUSTERS)])
    order = np.argsort(-per_cluster)
    z = crowding_z(X)
    region_crowd = np.array([z[cluster_id == c].mean() for c in range(N_CLUSTERS)])
    sizes = np.bincount(cluster_id, minlength=N_CLUSTERS)

    print(f"\n  learned beta per region (init was {32.0} for every region):")
    print(f"    {'region':>7} {'beta':>8} {'size':>6} {'crowding z':>12}")
    for c in order:
        print(f"    {c:>7} {per_cluster[c]:>8.2f} {sizes[c]:>6} {region_crowd[c]:>12.2f}")
    mean_beta = float(per_cluster.mean())
    print(f"\n    mean   = {mean_beta:.2f}   (fixed baseline uses 32.00)")
    print(f"    min    = {per_cluster.min():.2f}")
    print(f"    max    = {per_cluster.max():.2f}")
    print(f"    std    = {per_cluster.std():.2f}   "
          f"(spread is {per_cluster.std() / mean_beta:.1%} of the mean)")
    print(f"    corr(crowding, beta) = {np.corrcoef(region_crowd, per_cluster)[0, 1]:.3f}")

    # ------------------------------------------------------------------ queries
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

    # ------------------------------------------------------------------ conditions
    conds: dict[str, object] = {}
    dense = DenseTopK(); dense.index(memories)
    conds["cosine"] = dense

    print(f"\n[C2] sweeping fixed beta to find the honest baseline...")
    best_fixed, best_acc = None, -1.0
    fixed_rows = []
    for b in FIXED_SWEEP:
        m = EnergyMemory(n_steps=8, beta=float(b)); m.index(memories)
        pooled, _ = score(m, id_to_class, q_sets_by_level, gold)
        acc = float(np.mean(pooled))
        fixed_rows.append((b, acc))
        if acc > best_acc:
            best_acc, best_fixed = acc, b
    for b, acc in fixed_rows:
        mark = "  <-- best" if b == best_fixed else ""
        print(f"    fixed beta={b:>4}: {acc:.3f}{mark}")

    conds["fixed beta=32 (original)"] = make_beta_model(memories, np.full(len(memories), 32.0))
    conds[f"[C1] fixed beta={mean_beta:.1f} (= mean of learned)"] = make_beta_model(
        memories, np.full(len(memories), mean_beta))
    conds[f"[C2] fixed beta={best_fixed} (best of sweep)"] = make_beta_model(
        memories, np.full(len(memories), float(best_fixed)))
    conds["TRAINED per-region beta"] = make_beta_model(memories, beta_vec)

    shuf_rng = np.random.default_rng(7)
    shuffled = per_cluster.copy()
    shuf_rng.shuffle(shuffled)
    conds["[C3] shuffled region assignment"] = make_beta_model(memories, shuffled[cluster_id])

    # ------------------------------------------------------------------ evaluate
    print("\n" + "=" * 84)
    print("CONTROLS -- real pixel occlusion (30/50/70% pooled), 3 seeds, bootstrap 95% CI")
    print("=" * 84)
    print(f"\n  {'condition':<42}{'pooled':>22}" +
          "".join(f"{str(int(l * 100)) + '%':>8}" for l in OCCLUSION_LEVELS))
    print("  " + "-" * 80)
    results = {}
    for name, retr in conds.items():
        pooled, per_level = score(retr, id_to_class, q_sets_by_level, gold)
        m_, lo, hi = bootstrap_ci(pooled)
        results[name] = (m_, lo, hi)
        print(f"  {name:<42}{m_:>7.3f} [{lo:.3f},{hi:.3f}]" +
              "".join(f"{per_level[l]:>8.3f}" for l in OCCLUSION_LEVELS))

    # ------------------------------------------------------------------ verdict
    print("\n" + "=" * 84)
    print("VERDICT")
    print("=" * 84)
    tr = results["TRAINED per-region beta"]
    c1 = results[f"[C1] fixed beta={mean_beta:.1f} (= mean of learned)"]
    c2 = results[f"[C2] fixed beta={best_fixed} (best of sweep)"]
    c3 = results["[C3] shuffled region assignment"]
    cos = results["cosine"]

    print(f"\n  trained heterogeneous : {tr[0]:.3f} [{tr[1]:.3f},{tr[2]:.3f}]")
    print(f"  C1 uniform at mean    : {c1[0]:.3f} [{c1[1]:.3f},{c1[2]:.3f}]   "
          f"delta vs trained = {tr[0] - c1[0]:+.3f}")
    print(f"  C3 shuffled regions   : {c3[0]:.3f} [{c3[1]:.3f},{c3[2]:.3f}]   "
          f"delta vs trained = {tr[0] - c3[0]:+.3f}")
    print(f"  C2 best fixed beta    : {c2[0]:.3f} [{c2[1]:.3f},{c2[2]:.3f}]   "
          f"delta vs trained = {tr[0] - c2[0]:+.3f}")
    print(f"  cosine                : {cos[0]:.3f} [{cos[1]:.3f},{cos[2]:.3f}]   "
          f"delta vs trained = {tr[0] - cos[0]:+.3f}")

    print()
    if abs(tr[0] - c1[0]) < 0.01:
        print("  => Heterogeneity contributes ~nothing: a single uniform beta at the same")
        print("     MEAN performs the same. The learned variation across regions is inert;")
        print("     all of the movement came from raising the average beta.")
    if abs(tr[0] - c3[0]) < 0.01:
        print("  => Shuffling which region gets which beta changes nothing, so the learned")
        print("     ASSIGNMENT carries no information -- only the magnitudes matter.")
    if tr[0] <= c2[0] + 0.005:
        print("  => The trained model does not beat a plain tuned scalar beta. Any gain over")
        print("     beta=32 is a tuning artifact: beta=32 was never tuned in the first place.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
