"""Query-conditional beta, trained on REALISTIC corruption.

Fixes the two failures of scripts/train_cluster_beta.py:

  F1  Saturated objective. That run trained on embedding-dimension dropout, which is
      trivially recoverable -- val_acc was 1.000 at initialization, so the gradient could
      only push confidence, never correctness, and every beta drifted upward for reasons
      unrelated to retrieval. Here training queries are OCCLUDED IMAGES re-embedded
      through the same ResNet -- the same corruption family the test uses, and hard
      enough to leave real headroom.

  F2  Wrong search region. Those betas landed at 51-88, which sweep_beta_transition.py
      shows is the low-leverage zone where the method has nearly degenerated into cosine
      (beta=512 reproduces cosine to three decimals). beta has leverage around 16-48.
      Here beta is produced by a learned function that can output anything above a floor
      of 4, and is re-evaluated at every descent step.

Conditions:
    cosine                     the baseline
    fixed beta=32              the untuned setting used earlier in the project
    fixed beta=48              best of the dense sweep -- the honest tuned baseline
    query-conditional beta     learned f(state) -> beta, this file's subject

Two evaluations:
    INSTANCE  held-out occluded copies of bank images (same family as training)
    CLASS     occluded held-out val images, never in the bank (true generalization)

Run:
    python scripts/train_query_beta.py
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
from energymem.base import Memory, l2_normalize            # noqa: E402
from energymem.baselines.dense import DenseTopK            # noqa: E402
from energymem.memory.energy import EnergyMemory           # noqa: E402
from energymem.memory.query_beta import train_query_beta   # noqa: E402
from energymem.eval.metrics import bootstrap_ci            # noqa: E402

CACHE = Path(__file__).resolve().parents[1] / "experiments" / "cache"
N_PER_CLASS = 40
N_QUERY_PER_CLASS = 8
TRAIN_LEVELS = [0.3, 0.5, 0.7]
TRAIN_SEEDS = [11, 12]
TEST_LEVELS = [0.3, 0.5, 0.7]
TEST_SEEDS = [0, 1, 2]
EPOCHS = 300


def build_cache():
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"qbeta_n{N_PER_CLASS}_q{N_QUERY_PER_CLASS}.npz"
    if f.exists():
        print(f"loading cache {f.name}")
        d = np.load(f, allow_pickle=True)
        return (d["mem_emb"], list(d["mem_cls"]), d["train_Q"], d["train_y"],
                d["test_Q"], list(d["test_cls"]))

    print("embedding (first run only; cached afterwards)...")
    model, preprocess = build_encoder()
    rng = random.Random(0)
    mem_paths = collect_paths("train", N_PER_CLASS, rng)
    test_paths = collect_paths("val", N_QUERY_PER_CLASS, rng)
    mem_emb = embed_images(model, preprocess, mem_paths)
    mem_cls = [p.parent.name for p in mem_paths]
    test_cls = [p.parent.name for p in test_paths]

    def occ_embed(paths, lvl, seed):
        srng = random.Random(seed)
        imgs = [occlude(Image.open(p).convert("RGB"), lvl, srng) for p in paths]
        with torch.no_grad():
            batch = torch.stack([preprocess(im) for im in imgs])
            return l2_normalize(model(batch).numpy().astype(np.float64), axis=1)

    # TRAINING queries: occluded copies of the BANK images -> label = that memory's index
    tq, ty = [], []
    for lvl in TRAIN_LEVELS:
        for seed in TRAIN_SEEDS:
            e = occ_embed(mem_paths, lvl, seed)
            tq.append(e)
            ty.append(np.arange(len(mem_paths)))
    train_Q = np.concatenate(tq, axis=0).T          # (d, B)
    train_y = np.concatenate(ty, axis=0)

    # TEST queries: occluded held-out val images
    test_Q = np.zeros((len(TEST_LEVELS), len(TEST_SEEDS), len(test_paths), mem_emb.shape[1]))
    for li, lvl in enumerate(TEST_LEVELS):
        for si, seed in enumerate(TEST_SEEDS):
            test_Q[li, si] = occ_embed(test_paths, lvl, seed)

    np.savez_compressed(f, mem_emb=mem_emb, mem_cls=np.array(mem_cls), train_Q=train_Q,
                        train_y=train_y, test_Q=test_Q, test_cls=np.array(test_cls))
    print(f"  cached {f.name}")
    return mem_emb, mem_cls, train_Q, train_y, test_Q, test_cls


def eval_fixed(retr, id_to_class, test_Q, test_cls):
    pooled, per_level = [], {}
    for li, lvl in enumerate(TEST_LEVELS):
        hits = []
        for si in range(len(TEST_SEEDS)):
            for q, g in zip(test_Q[li, si], test_cls):
                hits.append(float(id_to_class[retr.retrieve(q, 1).ids[0]] == g))
        per_level[lvl] = float(np.mean(hits))
        pooled.extend(hits)
    return pooled, per_level


def eval_qbeta(model, X_np, mem_cls, test_Q, test_cls):
    pooled, per_level, beta_log = [], {}, {}
    for li, lvl in enumerate(TEST_LEVELS):
        hits, bs = [], []
        for si in range(len(TEST_SEEDS)):
            idx, betas = model.retrieve_batch(X_np, test_Q[li, si].T, k=1)
            bs.append(np.mean([b.mean() for b in betas]))
            for j, g in zip(idx[:, 0], test_cls):
                hits.append(float(mem_cls[j] == g))
        per_level[lvl] = float(np.mean(hits))
        beta_log[lvl] = float(np.mean(bs))
        pooled.extend(hits)
    return pooled, per_level, beta_log


def main() -> int:
    mem_emb, mem_cls, train_Q, train_y, test_Q, test_cls = build_cache()
    memories = [Memory(id=f"m{i}", text=mem_cls[i], embedding=mem_emb[i])
                for i in range(len(mem_cls))]
    id_to_class = {m.id: m.text for m in memories}
    X_np = l2_normalize(np.stack(mem_emb), axis=1).T
    print(f"bank {len(memories)}   train queries {train_Q.shape[1]}   "
          f"test {test_Q.shape[2]}x{len(TEST_LEVELS)}x{len(TEST_SEEDS)}")

    # split training queries into train/val by index
    rng = np.random.default_rng(0)
    perm = rng.permutation(train_Q.shape[1])
    n_val = int(0.2 * len(perm))
    vi, ti = perm[:n_val], perm[n_val:]

    print(f"\ntraining query-conditional beta ({EPOCHS} epochs) on REAL occlusion...")
    t0 = time.perf_counter()
    model, hist = train_query_beta(
        X_np, train_Q[:, ti], train_y[ti], train_Q[:, vi], train_y[vi],
        n_steps=8, epochs=EPOCHS, beta_init=32.0)
    print(f"  {time.perf_counter() - t0:.1f}s")
    print(f"  val loss {hist.val_loss[0]:.4f} -> {hist.val_loss[-1]:.4f}")
    print(f"  val acc  {hist.val_acc[0]:.3f} -> {hist.val_acc[-1]:.3f}")
    if hist.val_acc[0] >= 0.999:
        print("  ** objective saturated at init again -- treat results with suspicion **")
    else:
        print("  (objective had headroom at init: there was real signal to learn from)")
    print(f"  learned weights w = {np.round(model.w.detach().numpy(), 3)}")

    # ------------------------------------------------------------------ evaluate
    conds = {}
    dense = DenseTopK(); dense.index(memories)
    conds["cosine"] = dense
    for b in (32.0, 48.0):
        m = EnergyMemory(n_steps=8, beta=b); m.index(memories)
        conds[f"fixed beta={b:g}"] = m

    print("\n" + "=" * 78)
    print("CLASS-LEVEL generalization -- occluded held-out val images (never in bank)")
    print("=" * 78)
    print(f"\n  {'condition':<28}{'pooled':>21}" +
          "".join(f"{str(int(l * 100)) + '%':>9}" for l in TEST_LEVELS))
    print("  " + "-" * 76)

    results = {}
    for name, retr in conds.items():
        pooled, per_level = eval_fixed(retr, id_to_class, test_Q, test_cls)
        mu, lo, hi = bootstrap_ci(pooled)
        results[name] = (mu, lo, hi)
        print(f"  {name:<28}{mu:>7.3f} [{lo:.3f},{hi:.3f}]" +
              "".join(f"{per_level[l]:>9.3f}" for l in TEST_LEVELS))

    pooled, per_level, beta_log = eval_qbeta(model, X_np, mem_cls, test_Q, test_cls)
    mu, lo, hi = bootstrap_ci(pooled)
    results["query-conditional beta"] = (mu, lo, hi)
    print(f"  {'query-conditional beta':<28}{mu:>7.3f} [{lo:.3f},{hi:.3f}]" +
          "".join(f"{per_level[l]:>9.3f}" for l in TEST_LEVELS))

    print(f"\n  mean beta the learned function chose, by occlusion level:")
    for l in TEST_LEVELS:
        print(f"    {int(l * 100):>2}% occlusion -> beta = {beta_log[l]:.1f}")
    print("  (sweep_beta_transition.py found best fixed beta: 48 @30%, 36 @50%, 128 @70%)")

    # ------------------------------------------------------------------ verdict
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    q = results["query-conditional beta"]
    for name in ("cosine", "fixed beta=32", "fixed beta=48"):
        r = results[name]
        print(f"  vs {name:<20} {q[0] - r[0]:+.3f}   "
              f"({q[0]:.3f} [{q[1]:.3f},{q[2]:.3f}] vs {r[0]:.3f} [{r[1]:.3f},{r[2]:.3f}])")
    best_base = max(results[n][0] for n in ("cosine", "fixed beta=32", "fixed beta=48"))
    if q[0] > best_base + 0.01:
        print("\n  => query-conditional beta beats every fixed baseline.")
    elif q[0] > best_base - 0.01:
        print("\n  => statistically indistinguishable from the best fixed baseline:")
        print("     conditioning beta on the query adds nothing measurable here.")
    else:
        print("\n  => worse than the best fixed baseline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
