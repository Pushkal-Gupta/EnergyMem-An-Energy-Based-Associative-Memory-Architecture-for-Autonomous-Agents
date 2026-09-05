"""Confirmatory test with an honest tune/test split.

paired_beta_showdown.py found energy descent at beta=48 beating cosine (+0.018, paired
95% CI [+0.005, +0.030]). Two threats to that result:

  T1  SELECTION BIAS. beta=48 was chosen as the winner of a sweep run on 80 images, and
      those 80 are a subset of the 450 it was then evaluated on. The winner of a sweep
      evaluated on data it was selected on is biased upward.
  T2  MULTIPLE COMPARISONS. Seven beta values were tested and the best reported. At
      alpha=0.05 that inflates the false-positive rate.

Both are closed the same way: split the query images into disjoint TUNE and TEST halves,
select beta using ONLY the tune half, then report a single pre-committed comparison on
the test half. One comparison, on data never used for selection.

Scaled up as well: ~150 images/class instead of 45, for a tighter interval.

Run:
    python scripts/confirm_holdout.py
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

CACHE = Path(__file__).resolve().parents[1] / "experiments" / "cache"
N_PER_CLASS = 40
N_TEST_PER_CLASS = 400     # takes the whole val split (~390/class, ~3900 images).
# Powered to resolve the ~+0.011 effect the 750-image run left at [-0.000, +0.022]:
# n rises ~2.6x, so the paired CI half-width falls from ~0.011 to ~0.007.
LEVELS = [0.3, 0.5, 0.7]
SEED = 0
BETAS = [24, 28, 32, 36, 40, 44, 48, 56, 64, 80]


def paired_bootstrap(a, b, n_resamples=10000, seed=0):
    d = np.asarray(a, np.float64) - np.asarray(b, np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_resamples, d.size))
    m = d[idx].mean(axis=1)
    return float(d.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def build_cache():
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"holdout_n{N_PER_CLASS}_t{N_TEST_PER_CLASS}.npz"
    if f.exists():
        print(f"loading cache {f.name}")
        d = np.load(f, allow_pickle=True)
        return d["mem_emb"], list(d["mem_cls"]), d["test_Q"], list(d["test_cls"])

    print("embedding (first run only)...")
    model, preprocess = build_encoder()
    rng = random.Random(0)
    mem_paths = collect_paths("train", N_PER_CLASS, rng)
    test_paths = collect_paths("val", N_TEST_PER_CLASS, rng)
    print(f"  bank {len(mem_paths)}, test {len(test_paths)} distinct images")
    mem_emb = embed_images(model, preprocess, mem_paths)

    def occ_embed(paths, lvl, seed, chunk=256):
        srng = random.Random(seed)
        out = []
        for i in range(0, len(paths), chunk):
            imgs = [occlude(Image.open(p).convert("RGB"), lvl, srng)
                    for p in paths[i:i + chunk]]
            with torch.no_grad():
                b = torch.stack([preprocess(im) for im in imgs])
                out.append(l2_normalize(model(b).numpy().astype(np.float64), axis=1))
        return np.concatenate(out, axis=0)

    test_Q = np.zeros((len(LEVELS), len(test_paths), mem_emb.shape[1]))
    for li, lvl in enumerate(LEVELS):
        test_Q[li] = occ_embed(test_paths, lvl, SEED)
        print(f"  embedded occlusion {int(lvl*100)}%")

    np.savez_compressed(f, mem_emb=mem_emb,
                        mem_cls=np.array([p.parent.name for p in mem_paths]),
                        test_Q=test_Q,
                        test_cls=np.array([p.parent.name for p in test_paths]))
    d = np.load(f, allow_pickle=True)
    return d["mem_emb"], list(d["mem_cls"]), d["test_Q"], list(d["test_cls"])


def main() -> int:
    mem_emb, mem_cls, test_Q, test_cls = build_cache()
    memories = [Memory(id=f"m{i}", text=mem_cls[i], embedding=mem_emb[i])
                for i in range(len(mem_cls))]
    id_to_class = {m.id: m.text for m in memories}
    n_img = test_Q.shape[1]

    # disjoint split by IMAGE (not by query instance), so no image appears in both halves
    rng = np.random.default_rng(42)
    perm = rng.permutation(n_img)
    tune_img, test_img = perm[:n_img // 2], perm[n_img // 2:]
    print(f"\n{n_img} distinct images -> tune {len(tune_img)} / test {len(test_img)} "
          f"(disjoint), {len(LEVELS)} occlusion levels each")

    def hits(retr, img_idx):
        out = []
        for li in range(len(LEVELS)):
            for j in img_idx:
                q = test_Q[li, j]
                out.append(float(id_to_class[retr.retrieve(q, 1).ids[0]] == test_cls[j]))
        return np.array(out)

    dense = DenseTopK(); dense.index(memories)
    models = {b: EnergyMemory(n_steps=8, beta=float(b)) for b in BETAS}
    for m in models.values():
        m.index(memories)

    # ---------------------------------------------------------------- TUNE half
    print("\n" + "=" * 78)
    print("STAGE 1 -- select beta on the TUNE half only")
    print("=" * 78)
    t0 = time.perf_counter()
    cos_tune = hits(dense, tune_img)
    print(f"\n  cosine (tune) = {cos_tune.mean():.3f}\n")
    best_b, best_d = None, -np.inf
    for b in BETAS:
        h = hits(models[b], tune_img)
        d = h.mean() - cos_tune.mean()
        print(f"    beta={b:>3}: acc={h.mean():.3f}  delta={d:+.3f}")
        if d > best_d:
            best_d, best_b = d, b
    print(f"\n  selected beta = {best_b} (delta {best_d:+.3f} on tune half)")
    print(f"  [{time.perf_counter() - t0:.0f}s]")

    # ---------------------------------------------------------------- TEST half
    print("\n" + "=" * 78)
    print(f"STAGE 2 -- ONE pre-committed comparison on the HELD-OUT half: beta={best_b} vs cosine")
    print("=" * 78)
    cos_test = hits(dense, test_img)
    en_test = hits(models[best_b], test_img)
    mu, lo, hi = paired_bootstrap(en_test, cos_test)
    wins = int(((en_test == 1) & (cos_test == 0)).sum())
    losses = int(((en_test == 0) & (cos_test == 1)).sum())

    print(f"\n  n = {len(cos_test)} query instances ({len(test_img)} distinct images)")
    print(f"  cosine          {cos_test.mean():.3f}")
    print(f"  energy beta={best_b:<3} {en_test.mean():.3f}")
    print(f"  paired delta    {mu:+.3f}   95% CI [{lo:+.3f}, {hi:+.3f}]")
    print(f"  disagreements   energy wins {wins}, loses {losses}")

    print(f"\n  per occlusion level:")
    for li, lvl in enumerate(LEVELS):
        sl = slice(li * len(test_img), (li + 1) * len(test_img))
        m_, l_, h_ = paired_bootstrap(en_test[sl], cos_test[sl])
        tag = "SIG" if l_ > 0 else ("worse" if h_ < 0 else "n.s.")
        print(f"    {int(lvl*100):>2}%: energy {en_test[sl].mean():.3f} vs cosine "
              f"{cos_test[sl].mean():.3f}   delta {m_:+.3f} [{l_:+.3f},{h_:+.3f}]  {tag}")

    print("\n" + "=" * 78)
    if lo > 0:
        print("CONFIRMED: energy descent beats cosine on held-out data, with beta chosen")
        print("on a disjoint tuning split. Selection bias and multiple comparisons are")
        print("both ruled out -- this is a single pre-committed test.")
    elif hi < 0:
        print("REVERSED: energy descent is significantly WORSE on held-out data.")
    else:
        print("NOT CONFIRMED: the effect does not survive an honest tune/test split.")
        print("The earlier significant result was selection bias -- beta was tuned on")
        print("the same images it was scored on.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
