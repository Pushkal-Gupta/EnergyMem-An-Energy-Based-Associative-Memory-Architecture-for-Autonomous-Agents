"""Fine-grained beta sweep through the TRANSITION ZONE.

Why this exists. The earlier coarse sweep sampled beta at 8, 16, 32, 48, 64, ... and
reported a "plateau" above 32. That reading was wrong in an important way:

    high beta -> softmax(beta*s) becomes one-hot -> the update returns the top-1 memory
                 -> step 2 is a fixed point. The method DEGENERATES INTO COSINE.
                 (This is Proposition 1: beta -> inf gives exact top-1 cosine.)
    low beta  -> softmax is flat -> the state collapses to the corpus centroid -> chance.

So the "plateau" is not a plateau of a working method; it is the method having collapsed
into the baseline. Genuine pattern completion -- blending several memories to reconstruct
a degraded cue -- can only occur in the BAND BETWEEN the two failure modes, and the old
sweep jumped 16 -> 32 without sampling it at all.

This script samples that band densely, per occlusion level, with bootstrap CIs, and
reports whether a non-degenerate optimum exists strictly between "centroid collapse" and
"is just cosine". Everything downstream (per-region beta, query-conditional beta) depends
on the answer: if the useful band exists and is narrow, heterogeneous beta must be
initialized INSIDE it, which no previous run did.

Embeddings are cached to experiments/cache/ so follow-up experiments are fast.

Run:
    python scripts/sweep_beta_transition.py
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
from energymem.base import Memory, l2_normalize          # noqa: E402
from energymem.baselines.dense import DenseTopK          # noqa: E402
from energymem.memory.energy import EnergyMemory         # noqa: E402
from energymem.eval.metrics import bootstrap_ci          # noqa: E402

CACHE = Path(__file__).resolve().parents[1] / "experiments" / "cache"
N_PER_CLASS = 40
N_QUERY_PER_CLASS = 8
OCCLUSION_LEVELS = [0.3, 0.5, 0.7]
SEEDS = [0, 1, 2]

# dense through the transition, coarse outside it
BETAS = [4, 8, 12, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 30,
         32, 36, 40, 48, 64, 128, 512]


def load_or_build_cache():
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"img_n{N_PER_CLASS}_q{N_QUERY_PER_CLASS}.npz"
    if f.exists():
        print(f"loading cached embeddings from {f.name}")
        d = np.load(f, allow_pickle=True)
        return (d["mem_emb"], list(d["mem_cls"]), d["q_emb"], list(d["q_cls"]),
                list(d["levels"]), list(d["seeds"]))

    print("building encoder + embedding (first run only; results cached)...")
    model, preprocess = build_encoder()
    rng = random.Random(0)
    mem_paths = collect_paths("train", N_PER_CLASS, rng)
    query_paths = collect_paths("val", N_QUERY_PER_CLASS, rng)
    mem_emb = embed_images(model, preprocess, mem_paths)
    mem_cls = [p.parent.name for p in mem_paths]
    q_cls = [p.parent.name for p in query_paths]

    # q_emb[level_idx, seed_idx] -> (Q, d)
    q_emb = np.zeros((len(OCCLUSION_LEVELS), len(SEEDS), len(query_paths), mem_emb.shape[1]))
    for li, lvl in enumerate(OCCLUSION_LEVELS):
        for si, seed in enumerate(SEEDS):
            srng = random.Random(seed)
            imgs = [occlude(Image.open(p).convert("RGB"), lvl, srng) for p in query_paths]
            with torch.no_grad():
                batch = torch.stack([preprocess(im) for im in imgs])
                q_emb[li, si] = l2_normalize(model(batch).numpy().astype(np.float64), axis=1)

    np.savez_compressed(f, mem_emb=mem_emb, mem_cls=np.array(mem_cls), q_emb=q_emb,
                        q_cls=np.array(q_cls), levels=np.array(OCCLUSION_LEVELS),
                        seeds=np.array(SEEDS))
    print(f"  cached to {f.name}")
    return mem_emb, mem_cls, q_emb, q_cls, OCCLUSION_LEVELS, SEEDS


def main() -> int:
    if not DATA_ROOT.exists() and not (CACHE / f"img_n{N_PER_CLASS}_q{N_QUERY_PER_CLASS}.npz").exists():
        print("need either the dataset or a cache")
        return 1

    mem_emb, mem_cls, q_emb, q_cls, levels, seeds = load_or_build_cache()
    memories = [Memory(id=f"m{i}", text=mem_cls[i], embedding=mem_emb[i])
                for i in range(len(mem_cls))]
    id_to_class = {m.id: m.text for m in memories}
    print(f"bank {len(memories)} memories, {q_emb.shape[2]} queries x "
          f"{len(levels)} levels x {len(seeds)} seeds")

    dense = DenseTopK(); dense.index(memories)
    cos_by_level, cos_pooled = {}, []
    for li, lvl in enumerate(levels):
        hits = [float(id_to_class[dense.retrieve(q, 1).ids[0]] == g)
                for si in range(len(seeds)) for q, g in zip(q_emb[li, si], q_cls)]
        cos_by_level[float(lvl)] = float(np.mean(hits))
        cos_pooled.extend(hits)
    cm, clo, chi = bootstrap_ci(cos_pooled)
    print(f"\ncosine baseline: pooled {cm:.3f} [{clo:.3f},{chi:.3f}]   " +
          "  ".join(f"{int(l*100)}%={cos_by_level[float(l)]:.3f}" for l in levels))

    print("\n" + "=" * 78)
    print("BETA SWEEP -- dense through the transition zone")
    print("=" * 78)
    print(f"\n  {'beta':>6} | {'pooled':>21} | " +
          " ".join(f"{str(int(l*100)) + '%':>7}" for l in levels) + " | conv")
    print("  " + "-" * 74)

    rows = []
    t0 = time.perf_counter()
    for beta in BETAS:
        m = EnergyMemory(n_steps=8, beta=float(beta)); m.index(memories)
        pooled, per_level, steps = [], {}, []
        for li, lvl in enumerate(levels):
            hits = []
            for si in range(len(seeds)):
                for q, g in zip(q_emb[li, si], q_cls):
                    r = m.retrieve(q, 1)
                    hits.append(float(id_to_class[r.ids[0]] == g))
                    steps.append(r.n_steps)
            per_level[float(lvl)] = float(np.mean(hits))
            pooled.extend(hits)
        mu, lo, hi = bootstrap_ci(pooled)
        rows.append((beta, mu, lo, hi, per_level))
        print(f"  {beta:>6} | {mu:>7.3f} [{lo:.3f},{hi:.3f}] | " +
              " ".join(f"{per_level[float(l)]:>7.3f}" for l in levels) +
              f" | {np.mean(steps):.1f}")
    print(f"\n  ({time.perf_counter() - t0:.1f}s)")

    # ------------------------------------------------------------------ analysis
    best = max(rows, key=lambda r: r[1])
    print("\n" + "=" * 78)
    print("ANALYSIS")
    print("=" * 78)
    print(f"\n  best beta = {best[0]}  ->  {best[1]:.3f} [{best[2]:.3f},{best[3]:.3f}]")
    print(f"  cosine                 {cm:.3f} [{clo:.3f},{chi:.3f}]")
    print(f"  delta over cosine      {best[1] - cm:+.3f}")

    hi_beta = [r for r in rows if r[0] >= 128][0]
    print(f"\n  beta=512 (should be ~cosine by Proposition 1): {hi_beta[1]:.3f}  "
          f"vs cosine {cm:.3f}  -> diff {hi_beta[1] - cm:+.3f}")

    # is the optimum interior (a real band) or at the boundary (degenerate)?
    interior = best[0] not in (BETAS[0], BETAS[-1]) and best[0] < 64
    if interior and best[1] - cm > 0.01:
        print(f"\n  => INTERIOR OPTIMUM at beta={best[0]}, above cosine. A genuine")
        print("     completion band exists between centroid-collapse and cosine-degeneracy.")
        print("     Per-region / query-conditional beta must be searched INSIDE this band.")
    elif interior:
        print(f"\n  => interior optimum at beta={best[0]} but within noise of cosine")
        print("     (delta {:+.3f}). Band exists but its benefit is not established here."
              .format(best[1] - cm))
    else:
        print("\n  => optimum sits at the high-beta boundary, i.e. the method is best when")
        print("     it has fully degenerated into cosine. No useful completion band here.")

    # per-level: does the band help more at heavy corruption?
    print("\n  best beta per occlusion level (does heavier damage want a softer beta?):")
    for l in levels:
        bl = max(rows, key=lambda r: r[4][float(l)])
        print(f"    occlusion {int(l*100):>2}%: beta={bl[0]:>4}  acc={bl[4][float(l)]:.3f}   "
              f"(cosine {cos_by_level[float(l)]:.3f}, delta {bl[4][float(l)] - cos_by_level[float(l)]:+.3f})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
