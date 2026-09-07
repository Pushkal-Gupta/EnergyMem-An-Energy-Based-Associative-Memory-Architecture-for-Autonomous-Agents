"""Adequately-powered, PAIRED comparison of beta strategies.

Every earlier comparison in this project was underpowered and used the wrong test:

  P1  Too few distinct queries. 80 val images were reused across 3 occlusion levels and
      3 seeds to produce "720 queries". Those are correlated -- effective n is ~80, not
      720 -- so the bootstrap CIs came out ~0.07 wide while the differences under test
      were ~0.01. Nothing could have been resolved at that scale, in either direction.
      Imagenette's val split holds ~3900 images; this uses many more of them.

  P2  Independent CIs are the wrong statistic. All methods are scored on the SAME
      queries and agree on most of them, so the shared difficulty is common variance
      that cancels in a paired analysis. Comparing two overlapping marginal CIs can hide
      a difference that a paired test resolves cleanly. Here we bootstrap the per-query
      DIFFERENCE against cosine, and also report McNemar-style win/loss counts on the
      queries where the two methods actually disagree.

Interpretation rule fixed in advance: a method beats cosine only if the paired 95% CI of
the difference excludes zero. Marginal CIs are reported for context, not for inference.

Run:
    python scripts/paired_beta_showdown.py
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

CACHE = Path(__file__).resolve().parents[1] / "experiments" / "cache"
N_PER_CLASS = 40          # bank: 400 images
N_TEST_PER_CLASS = 45     # test: 450 DISTINCT val images (was 8 -> 80)
TRAIN_LEVELS = [0.3, 0.5, 0.7]
TRAIN_SEEDS = [11, 12]
TEST_LEVELS = [0.3, 0.5, 0.7]
TEST_SEED = 0             # one seed per level; power comes from distinct images now
EPOCHS = 300
BETAS = [24, 28, 32, 36, 40, 48, 64]


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_resamples: int = 10000, seed: int = 0):
    """Bootstrap the mean of per-query differences (a - b). Returns (mean, lo, hi)."""
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_resamples, d.size))
    means = d[idx].mean(axis=1)
    return float(d.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def build_cache():
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"showdown_n{N_PER_CLASS}_t{N_TEST_PER_CLASS}.npz"
    if f.exists():
        print(f"loading cache {f.name}")
        d = np.load(f, allow_pickle=True)
        return (d["mem_emb"], list(d["mem_cls"]), d["train_Q"], d["train_y"],
                d["test_Q"], list(d["test_cls"]))

    print("embedding (first run only; cached afterwards)...")
    model, preprocess = build_encoder()
    rng = random.Random(0)
    mem_paths = collect_paths("train", N_PER_CLASS, rng)
    test_paths = collect_paths("val", N_TEST_PER_CLASS, rng)
    print(f"  bank {len(mem_paths)} images, test {len(test_paths)} DISTINCT images")
    mem_emb = embed_images(model, preprocess, mem_paths)

    def occ_embed(paths, lvl, seed, chunk=256):
        srng = random.Random(seed)
        out = []
        for i in range(0, len(paths), chunk):
            imgs = [occlude(Image.open(p).convert("RGB"), lvl, srng) for p in paths[i:i + chunk]]
            with torch.no_grad():
                batch = torch.stack([preprocess(im) for im in imgs])
                out.append(l2_normalize(model(batch).numpy().astype(np.float64), axis=1))
        return np.concatenate(out, axis=0)

    tq, ty = [], []
    for lvl in TRAIN_LEVELS:
        for seed in TRAIN_SEEDS:
            tq.append(occ_embed(mem_paths, lvl, seed))
            ty.append(np.arange(len(mem_paths)))
    train_Q = np.concatenate(tq, axis=0).T
    train_y = np.concatenate(ty, axis=0)

    test_Q = np.zeros((len(TEST_LEVELS), len(test_paths), mem_emb.shape[1]))
    for li, lvl in enumerate(TEST_LEVELS):
        test_Q[li] = occ_embed(test_paths, lvl, TEST_SEED)

    np.savez_compressed(f, mem_emb=mem_emb, mem_cls=np.array([p.parent.name for p in mem_paths]),
                        train_Q=train_Q, train_y=train_y, test_Q=test_Q,
                        test_cls=np.array([p.parent.name for p in test_paths]))
    print(f"  cached {f.name}")
    d = np.load(f, allow_pickle=True)
    return (d["mem_emb"], list(d["mem_cls"]), d["train_Q"], d["train_y"],
            d["test_Q"], list(d["test_cls"]))


def main() -> int:
    mem_emb, mem_cls, train_Q, train_y, test_Q, test_cls = build_cache()
    memories = [Memory(id=f"m{i}", text=mem_cls[i], embedding=mem_emb[i])
                for i in range(len(mem_cls))]
    id_to_class = {m.id: m.text for m in memories}
    X_np = l2_normalize(np.stack(mem_emb), axis=1).T
    n_test = test_Q.shape[1]
    print(f"\nbank {len(memories)}   distinct test images {n_test}   "
          f"levels {TEST_LEVELS}   -> {n_test * len(TEST_LEVELS)} query instances")

    # ------------------------------------------------------------------ train qbeta
    rng = np.random.default_rng(0)
    perm = rng.permutation(train_Q.shape[1])
    n_val = int(0.2 * len(perm))
    vi, ti = perm[:n_val], perm[n_val:]
    print(f"\ntraining query-conditional beta ({EPOCHS} epochs)...")
    t0 = time.perf_counter()
    qmodel, hist = train_query_beta(X_np, train_Q[:, ti], train_y[ti],
                                    train_Q[:, vi], train_y[vi],
                                    n_steps=8, epochs=EPOCHS, beta_init=32.0)
    print(f"  {time.perf_counter() - t0:.1f}s   val acc {hist.val_acc[0]:.3f} -> "
          f"{hist.val_acc[-1]:.3f}   val loss {hist.val_loss[0]:.3f} -> {hist.val_loss[-1]:.3f}")

    # ------------------------------------------------------------------ score everything
    def hits_fixed(retr):
        out = []
        for li in range(len(TEST_LEVELS)):
            for q, g in zip(test_Q[li], test_cls):
                out.append(float(id_to_class[retr.retrieve(q, 1).ids[0]] == g))
        return np.array(out)

    def hits_qbeta():
        out = []
        for li in range(len(TEST_LEVELS)):
            idx, _ = qmodel.retrieve_batch(X_np, test_Q[li].T, k=1)
            for j, g in zip(idx[:, 0], test_cls):
                out.append(float(mem_cls[j] == g))
        return np.array(out)

    print("\nscoring...")
    dense = DenseTopK(); dense.index(memories)
    scores = {"cosine": hits_fixed(dense)}
    for b in BETAS:
        m = EnergyMemory(n_steps=8, beta=float(b)); m.index(memories)
        scores[f"fixed beta={b}"] = hits_fixed(m)
    scores["query-conditional beta"] = hits_qbeta()

    base = scores["cosine"]
    print("\n" + "=" * 88)
    print("PAIRED COMPARISON vs COSINE  (same queries; CI is on the DIFFERENCE)")
    print("=" * 88)
    print(f"\n  n = {len(base)} query instances from {n_test} distinct images\n")
    print(f"  {'method':<26}{'acc':>7}{'delta':>9}{'95% CI of delta':>22}"
          f"{'win/loss':>12}   verdict")
    print("  " + "-" * 86)

    for name, s in scores.items():
        if name == "cosine":
            print(f"  {name:<26}{s.mean():>7.3f}{'--':>9}{'--':>22}{'--':>12}   baseline")
            continue
        mu, lo, hi = paired_bootstrap(s, base)
        wins = int(((s == 1) & (base == 0)).sum())
        losses = int(((s == 0) & (base == 1)).sum())
        sig = "SIGNIFICANT" if lo > 0 else ("worse" if hi < 0 else "n.s.")
        print(f"  {name:<26}{s.mean():>7.3f}{mu:>+9.3f}"
              f"{f'[{lo:+.3f}, {hi:+.3f}]':>22}{f'{wins}/{losses}':>12}   {sig}")

    # per-level for the best performer
    best_name = max((n for n in scores if n != "cosine"), key=lambda n: scores[n].mean())
    print(f"\n  per-level breakdown for {best_name} vs cosine:")
    for li, lvl in enumerate(TEST_LEVELS):
        sl = slice(li * n_test, (li + 1) * n_test)
        mu, lo, hi = paired_bootstrap(scores[best_name][sl], base[sl])
        sig = "SIGNIFICANT" if lo > 0 else ("worse" if hi < 0 else "n.s.")
        print(f"    occlusion {int(lvl * 100):>2}%: {scores[best_name][sl].mean():.3f} vs "
              f"{base[sl].mean():.3f}   delta {mu:+.3f} [{lo:+.3f},{hi:+.3f}]  {sig}")

    print("\n" + "=" * 88)
    print("VERDICT")
    print("=" * 88)
    mu, lo, hi = paired_bootstrap(scores[best_name], base)
    print(f"\n  best method: {best_name}")
    print(f"  vs cosine:   {mu:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]")
    if lo > 0:
        print("\n  => Energy descent beats cosine, significantly, under a paired test.")
        print("     The earlier 'tie' verdicts were an artifact of an underpowered,")
        print("     unpaired comparison -- not a property of the method.")
    else:
        print("\n  => Still not significant even with adequate power and a paired test.")
        print("     This is now a properly-supported negative result rather than a")
        print("     failure to measure.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
