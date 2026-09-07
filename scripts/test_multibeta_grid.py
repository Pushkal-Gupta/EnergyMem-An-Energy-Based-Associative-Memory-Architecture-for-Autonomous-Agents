"""Per-pattern beta, redone with a search space that contains the baseline.

The earlier run (test_multibeta_capacity.py) is void: its parameterization could only
express a uniform beta of 4.0, which F2 already showed collapses to the corpus centroid.
Uniform beta=32 -- the very baseline it was meant to generalize -- was unreachable, so
"multi-beta loses to fixed beta" measured a broken search, not the idea.

Here beta_i = beta_base * (1 + lam * z_i) with z the standardized crowding score, so
lam=0 reproduces fixed beta exactly for ANY beta_base, and (beta_base, lam) are searched
jointly. Step 1 asserts that reduction numerically; if it fails nothing below counts.

The question this answers: at the joint optimum, is lam > 0?

Run:
    python scripts/test_multibeta_grid.py
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
from energymem.base import Memory, l2_normalize                     # noqa: E402
from energymem.baselines.dense import DenseTopK                     # noqa: E402
from energymem.memory.energy import EnergyMemory                    # noqa: E402
from energymem.memory.multibeta import MultiBetaEnergyMemory, crowding_z  # noqa: E402
from energymem.eval.metrics import bootstrap_ci                     # noqa: E402

BETA_BASES = [8.0, 16.0, 32.0, 64.0, 128.0]
LAMS = [0.0, 0.1, 0.25, 0.5, 1.0]
BANK_SIZES = [(40, "N=400"), (150, "N=1500"), (400, "N=4000")]
N_QUERY_PER_CLASS = 8
OCCLUSION = 0.5
SEEDS = [0, 1, 2]


def build_memories(paths, embs):
    return [Memory(id=str(p), text=p.parent.name, embedding=embs[i])
            for i, p in enumerate(paths)]


def reduction_check(memories) -> bool:
    """lam=0 must be bit-comparable to EnergyMemory at the same beta."""
    print("\n[reduction] lam=0 must equal EnergyMemory(beta=beta_base) exactly")
    rng = np.random.default_rng(0)
    probes = l2_normalize(rng.normal(size=(60, memories[0].embedding.shape[0])), axis=1)
    ok = True
    for beta in (8.0, 32.0, 128.0):
        ref = EnergyMemory(n_steps=8, beta=beta); ref.index(memories)
        mb = MultiBetaEnergyMemory(n_steps=8, beta_base=beta, lam=0.0); mb.index(memories)
        agree = sum(ref.retrieve(q, k=5).ids == mb.retrieve(q, k=5).ids for q in probes)
        good = agree == len(probes)
        ok &= good
        print(f"    beta={beta:<6} identical top-5: {agree}/{len(probes)}  "
              f"{'OK' if good else 'FAIL'}")
    return ok


def main() -> int:
    if not DATA_ROOT.exists():
        print(f"missing dataset at {DATA_ROOT}")
        return 1

    print("building encoder...")
    model, preprocess = build_encoder()

    rng = random.Random(0)
    max_paths = collect_paths("train", max(n for n, _ in BANK_SIZES), rng)
    query_paths = collect_paths("val", N_QUERY_PER_CLASS, rng)
    print(f"embedding pool of {len(max_paths)} images...")
    t0 = time.perf_counter()
    max_embs = embed_images(model, preprocess, max_paths)
    print(f"  done in {time.perf_counter() - t0:.1f}s")

    by_class: dict[str, list[int]] = {}
    for i, p in enumerate(max_paths):
        by_class.setdefault(p.parent.name, []).append(i)

    # occluded queries, embedded once and reused for every configuration
    q_sets = []
    for seed in SEEDS:
        srng = random.Random(seed)
        imgs = [occlude(Image.open(p).convert("RGB"), OCCLUSION, srng) for p in query_paths]
        with torch.no_grad():
            batch = torch.stack([preprocess(im) for im in imgs])
            q_sets.append(l2_normalize(model(batch).numpy().astype(np.float64), axis=1))
    gold = [p.parent.name for p in query_paths]

    first = True
    for n_per_class, label in BANK_SIZES:
        idx = [i for c in by_class for i in by_class[c][:n_per_class]]
        memories = build_memories([max_paths[i] for i in idx], max_embs[idx])

        if first:
            if not reduction_check(memories):
                print("    => reduction FAILED; implementation is wrong. Stopping.")
                return 1
            first = False

        print("\n" + "=" * 74)
        print(f"{label}  ({len(memories)} images)   Recall@1, 50% occlusion, 3 seeds")
        print("=" * 74)

        dense = DenseTopK(); dense.index(memories)
        cos_hits = [float(next(m.text for m in memories if m.id == dense.retrieve(q, 1).ids[0]) == g)
                    for qs in q_sets for q, g in zip(qs, gold)]
        cm, clo, chi = bootstrap_ci(cos_hits)
        print(f"\n  cosine baseline: {cm:.3f} [{clo:.3f},{chi:.3f}]")

        # crowding depends only on the bank -- compute once, reuse across the grid
        X = l2_normalize(np.stack([m.embedding for m in memories]), axis=1).T
        t0 = time.perf_counter()
        z = crowding_z(X)
        print(f"  crowding score computed in {time.perf_counter() - t0:.1f}s "
              f"(z range {z.min():.2f}..{z.max():.2f})")

        print(f"\n  {'beta_base':>10} | " + " ".join(f"{'lam=' + str(l):>12}" for l in LAMS))
        print("  " + "-" * (12 + 13 * len(LAMS)))

        best = (-1.0, None)
        for beta_base in BETA_BASES:
            cells = []
            for lam in LAMS:
                mb = MultiBetaEnergyMemory(n_steps=8, beta_base=beta_base, lam=lam)
                mb.index(memories, z=z)
                hits = [
                    float(next(m.text for m in memories if m.id == mb.retrieve(q, 1).ids[0]) == g)
                    for qs in q_sets for q, g in zip(qs, gold)
                ]
                acc = float(np.mean(hits))
                cells.append(acc)
                if acc > best[0]:
                    best = (acc, (beta_base, lam))
            print(f"  {beta_base:>10.0f} | " + " ".join(f"{c:>12.3f}" for c in cells))

        acc, (bb, bl) = best
        base_col = None
        mb0 = MultiBetaEnergyMemory(n_steps=8, beta_base=bb, lam=0.0)
        mb0.index(memories, z=z)
        base_hits = [
            float(next(m.text for m in memories if m.id == mb0.retrieve(q, 1).ids[0]) == g)
            for qs in q_sets for q, g in zip(qs, gold)
        ]
        base_col = float(np.mean(base_hits))
        print(f"\n  best: beta_base={bb:g}, lam={bl:g} -> {acc:.3f}")
        print(f"  same beta_base with lam=0 (i.e. fixed beta) -> {base_col:.3f}")
        if bl == 0.0:
            print("  => optimal lam is ZERO: per-pattern sharpness does not earn its place here.")
        else:
            print(f"  => optimal lam is NONZERO: heterogeneity adds {acc - base_col:+.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
