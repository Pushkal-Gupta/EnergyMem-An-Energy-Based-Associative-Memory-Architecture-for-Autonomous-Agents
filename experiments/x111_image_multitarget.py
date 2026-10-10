"""X111 -- the image analogue of LoCoMo cat 1: ONE cue, TWO stored targets.

WHY THIS CELL IS THE ONE WORTH RUNNING. Invariant I4 -- "never average modes" -- has now
been falsified on three text corpora, and the honest objection to all three is that the
text regimes were ones where descent loses anyway (X110 has since shown descent loses on
text even when the cue IS a degraded copy). I4 has never been tested in the modality where
descent WINS, with the multi-target structure that makes the invariant meaningful.

THE CUE. Splice two stored images: left half of A, right half of B. The cue is then a
degraded copy of A *and* a degraded copy of B simultaneously -- Claim 1's premise holds for
both -- and the gold set is {A, B}, two images from different classes, so the targets are
much closer to orthogonal than LoCoMo's eps = 0.527 gold turns. This is exactly the
geometry a multi-mode readout is supposed to need.

If logsumexp beats averaging anywhere, it is here. If it does not beat averaging here,
I4 is not a property of the architecture in any regime we can construct.

Protocol follows the image study it extends: resnet18 penultimate features (512-d, the
same encoder as `scripts/test_imagenet_hopfield.py`), a memory bank from train/, cues
built from held-out val/ images of the SAME stored items, beta selected on a disjoint tune
half of the cues, paired bootstrap resampling the cue.
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from test_imagenet_hopfield import (                                  # noqa: E402
    DATA_ROOT, build_encoder, collect_paths, embed_images,
)
from energymem.base import l2_normalize                               # noqa: E402
from energymem.readout import averaged_scores, mixture_scores         # noqa: E402
from energymem.eval.setmetrics import (                               # noqa: E402
    full_set_recall_at_k, mean_gold_recall_at_k,
)
from experiments.x6x_energy import descend                            # noqa: E402

OUT = ROOT / "experiments" / "results" / "x111_image_multitarget.json"
N_PER_CLASS_MEM = 40
N_CUES = 600
BETAS = (8.0, 16.0, 32.0, 48.0, 64.0, 128.0, 256.0)
K0S = (1, 2, 3)          # K0 < k = 5, the gate from the seventh protocol error
SEED = 12345
K_CUTOFF = 5
assert max(K0S) < K_CUTOFF


def hr(t):
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


def splice(a: Image.Image, b: Image.Image, size=160) -> Image.Image:
    """Left half of a, right half of b. A degraded copy of BOTH, by construction."""
    a = a.convert("RGB").resize((size, size))
    b = b.convert("RGB").resize((size, size))
    out = a.copy()
    out.paste(b.crop((size // 2, 0, size, size)), (size // 2, 0))
    return out


def boot(d, n=20_000, seed=0):
    d = np.asarray(d, float)
    rng = np.random.default_rng(seed)
    m = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)])
    return (float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)),
            float(min(1.0, 2 * min((m <= 0).mean(), (m >= 0).mean()))))


def main():
    t0 = time.time()
    if not DATA_ROOT.exists():
        print(f"missing {DATA_ROOT}"); return 1
    rng = random.Random(SEED)
    model, prep = build_encoder()

    mem_paths = collect_paths("train", N_PER_CLASS_MEM, rng)
    print(f"bank: {len(mem_paths)} images, resnet18 penultimate (512-d)")
    Z = np.concatenate([embed_images(model, prep, mem_paths[i:i + 64])
                        for i in range(0, len(mem_paths), 64)])
    cls = [p.parent.name for p in mem_paths]

    # cues splice two BANK images from DIFFERENT classes; gold = those two bank rows
    pairs = []
    while len(pairs) < N_CUES:
        i, j = rng.randrange(len(mem_paths)), rng.randrange(len(mem_paths))
        if cls[i] != cls[j]:
            pairs.append((i, j))
    print(f"cues: {len(pairs)} spliced pairs, always cross-class")
    imgs = [splice(Image.open(mem_paths[i]), Image.open(mem_paths[j])) for i, j in pairs]

    import torch
    with torch.no_grad():
        Q = np.concatenate([
            l2_normalize(model(torch.stack([prep(im) for im in imgs[i:i + 64]]))
                         .numpy().astype(np.float64), axis=1)
            for i in range(0, len(imgs), 64)])
    gold = [[i, j] for i, j in pairs]

    eps = float(np.mean([float(Z[i] @ Z[j]) for i, j in pairs]))
    print(f"\n  gold-pair inner product eps = {eps:.4f}   "
          f"(LoCoMo cat 1: 0.5270)")

    perm = np.random.default_rng(SEED).permutation(len(pairs))
    tune, test = sorted(perm[:len(pairs) // 2]), sorted(perm[len(pairs) // 2:])

    def metrics(S, idxs):
        g5, f5, g2 = [], [], []
        for t in idxs:
            o = np.argsort(-S[t])
            g5.append(mean_gold_recall_at_k(o, gold[t], 5))
            f5.append(full_set_recall_at_k(o, gold[t], 5))
            g2.append(mean_gold_recall_at_k(o, gold[t], 2))
        return np.asarray(g5), np.asarray(f5), np.asarray(g2)

    S_cos = Q @ Z.T
    hr("X111a -- does the premise hold? cosine vs descent on a two-target cue")
    c5, cf5, c2 = metrics(S_cos, test)
    print(f"  cosine        gold@5 {c5.mean():.4f}  full@5 {cf5.mean():.4f}  "
          f"gold@2 {c2.mean():.4f}")
    best_b, best_d = None, -np.inf
    for beta in BETAS:
        S = np.stack([Z @ descend(Z, Q[t], beta, 8) for t in range(len(pairs))])
        d = metrics(S, tune)[0].mean() - metrics(S_cos, tune)[0].mean()
        print(f"    tune beta={beta:>6g}  delta gold@5 = {d:+.4f}")
        if d > best_d:
            best_b, best_d = beta, d
    S_desc = np.stack([Z @ descend(Z, Q[t], best_b, 8) for t in range(len(pairs))])
    d5, df5, d2 = metrics(S_desc, test)
    e5 = boot(d5 - c5); e2 = boot(d2 - c2)
    print(f"\n  selected beta = {best_b:g} (tune {best_d:+.4f}), held-out:")
    print(f"  descent T=8   gold@5 {d5.mean():.4f}  delta {e5[0]:+.4f} "
          f"[{e5[1]:+.4f},{e5[2]:+.4f}] p={e5[3]:.4f}")
    print(f"                gold@2 {d2.mean():.4f}  delta {e2[0]:+.4f} "
          f"[{e2[1]:+.4f},{e2[2]:+.4f}] p={e2[3]:.4f}")
    res = {"eps_gold_pair": eps, "n_cues": len(pairs), "bank_n": len(mem_paths),
           "X111a": dict(cosine_gold5=float(c5.mean()), cosine_full5=float(cf5.mean()),
                         selected_beta=best_b, tune_delta=best_d,
                         descent_gold5=float(d5.mean()),
                         delta_gold5=e5[0], lo=e5[1], hi=e5[2], p=e5[3],
                         delta_gold2=e2[0], lo2=e2[1], hi2=e2[2], p2=e2[3])}

    hr("X111b -- I4 with matched modes, in the modality where descent wins")
    print("  Same hypotheses, same weights; only logsumexp vs averaging differs.\n")
    print(f"  {'K0':>3}{'beta':>7}{'mixture':>10}{'averaged':>10}{'delta':>10}"
          f"{'95% CI':>22}{'p':>9}{'vs cosine':>11}")
    rows, best = {}, (None, -np.inf)
    for K0 in K0S:
        for beta in BETAS:
            Sm, Sa = [], []
            for t in range(len(pairs)):
                top = np.argsort(-S_cos[t])[:K0]
                w = np.exp(beta * (Z[top] @ Q[t])); w = w / w.sum()
                h = [(Q[t], 1.0)] + [(Z[j], float(wj) + 1e-12) for j, wj in zip(top, w)]
                Sm.append(mixture_scores(h, Z, beta))
                Sa.append(averaged_scores(h, Z))
            Sm, Sa = np.stack(Sm), np.stack(Sa)
            m5 = metrics(Sm, test)[0]; a5 = metrics(Sa, test)[0]
            e = boot(m5 - a5)
            tune_gap = metrics(Sm, tune)[0].mean() - metrics(Sa, tune)[0].mean()
            rows[f"K0={K0},beta={beta:g}"] = dict(
                mixture=float(m5.mean()), averaged=float(a5.mean()), delta=e[0],
                lo=e[1], hi=e[2], p=e[3], tune_gap=float(tune_gap),
                mixture_vs_cosine=float(m5.mean() - c5.mean()))
            print(f"  {K0:>3}{beta:>7g}{m5.mean():>10.4f}{a5.mean():>10.4f}{e[0]:>+10.4f}"
                  f"   [{e[1]:>+.4f},{e[2]:>+.4f}]{e[3]:>9.4f}"
                  f"{m5.mean() - c5.mean():>+11.4f}")
            if tune_gap > best[1]:
                best = (f"K0={K0},beta={beta:g}", tune_gap)
    sel = rows[best[0]]
    print(f"\n  cell selected on the TUNE half: {best[0]} (tune gap {best[1]:+.4f})")
    print(f"  held-out there: mixture {sel['mixture']:.4f} vs averaged "
          f"{sel['averaged']:.4f}, delta {sel['delta']:+.4f} "
          f"[{sel['lo']:+.4f},{sel['hi']:+.4f}] p={sel['p']:.4f}")
    npos = sum(1 for v in rows.values() if v["lo"] > 0)
    print(f"  cells where logsumexp beats averaging with the interval excluding 0: "
          f"{npos}/{len(rows)}")
    print(f"  cells where the mixture also beats plain cosine: "
          f"{sum(1 for v in rows.values() if v['mixture_vs_cosine'] > 0)}/{len(rows)}")
    res["X111b_I4"] = dict(rows=rows, selected_cell=best[0], selected=sel,
                           n_cells_mixture_beats_avg=npos, n_cells=len(rows))
    res["_meta"] = dict(seconds=round(time.time() - t0, 1), betas=list(BETAS),
                        K0s=list(K0S), k_cutoff=K_CUTOFF, T=8,
                        encoder="resnet18 penultimate 512-d",
                        cue="left half of A spliced with right half of B, cross-class")
    OUT.write_text(json.dumps(res, indent=2, default=str))
    print(f"\nwrote {OUT}  ({res['_meta']['seconds']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
