"""Does mixing multiple beta SCALES inside Hopfield settling beat the best single beta,
beyond what independent Hopfield runs already give?

THREE MECHANISMS (see module math below; all use T steps, unit-norm X (d,N), cue xi0).

  (1) SINGLE BETA
        p_t   = softmax(beta * X^T xi_t)
        xi_t+1 = normalize(X p_t)

  (2) INTEGRATED MIXTURE -- mixed INSIDE each iteration, ONE coupled trajectory
        p_t   = sum_j w_j softmax(beta_j * X^T xi_t)
        xi_t+1 = normalize(X p_t)
      All scales read the SAME state each step and produce ONE next state, so the scales
      interact at every step. p_t is a mixture of softmaxes -- provably not expressible as
      softmax(beta* s) for any single beta* (the softmax-bottleneck / MoS rank argument,
      Yang et al. ICLR 2018).

  (3) NAIVE ENSEMBLE -- m INDEPENDENT trajectories, combined once at readout
        xi^j_t+1 = normalize(X softmax(beta_j * X^T xi^j_t))     (separate per j)
        sbar     = sum_j w_j (X^T xi^j_T)
      No interaction between scales until the very end.

(2) != (3) because composition and mixing do not commute: G^T != combine o (F_bj^T)_j for
T > 1; they agree only at T = 1. The experiment checks this empirically rather than
assuming it -- if (2) and (3) score identically at T=8 the coupling is inert.

COST (why a tie still favours (2)): per step (2) computes s once, m cheap softmaxes
O(mN), then ONE matmul X p_mix -> ~2TNd + TmN. (3) runs m full trajectories -> ~2mTNd.

PROTOCOL. Same discipline as confirm_holdout.py, which is what finally produced a
trustworthy result here: beta / mixture set chosen on a disjoint TUNE half of images, one
pre-committed comparison on the HELD-OUT half. Bootstrap resamples IMAGES (not query
instances), so the correlation between the 3 occlusion levels x 3 seeds of the same image
is handled correctly.

Run:
    python scripts/exp_multibeta_scales.py
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
from energymem.base import l2_normalize  # noqa: E402

CACHE = Path(__file__).resolve().parents[1] / "experiments" / "cache"
N_PER_CLASS = 40
N_TEST_PER_CLASS = 130
LEVELS = [0.3, 0.5, 0.7]
SEEDS = [0, 1, 2]
T_STEPS = 8

SINGLE_GRID = [24, 28, 32, 36, 40, 44, 48, 56, 64]
# candidate mixture sets, all inside the productive band measured by sweep_beta_transition
MIX_SETS = {
    "[16,32,48]": [16.0, 32.0, 48.0],
    "[24,40,56]": [24.0, 40.0, 56.0],
    "[32,40,48]": [32.0, 40.0, 48.0],
    "[16,28,40,52]": [16.0, 28.0, 40.0, 52.0],
}
CONTROL_SET = [1.0, 5.0, 20.0]   # predicted to fail: 1 and 5 are below the collapse cliff


# ----------------------------------------------------------------- mechanisms

def _softmax_cols(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=0, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=0, keepdims=True)


def descend_single(X, Q, beta, T=T_STEPS):
    """(1) p_t = softmax(beta s); xi <- normalize(X p_t).  Q is (d,B). Returns sims (N,B)."""
    xi = Q / (np.linalg.norm(Q, axis=0, keepdims=True) + 1e-12)
    for _ in range(T):
        p = _softmax_cols(beta * (X.T @ xi))
        nxt = X @ p
        xi = nxt / (np.linalg.norm(nxt, axis=0, keepdims=True) + 1e-12)
    return X.T @ xi


def descend_mixture(X, Q, betas, weights=None, T=T_STEPS):
    """(2) p_t = sum_j w_j softmax(beta_j s); ONE coupled trajectory."""
    betas = np.asarray(betas, dtype=np.float64)
    w = np.full(len(betas), 1.0 / len(betas)) if weights is None else np.asarray(weights)
    xi = Q / (np.linalg.norm(Q, axis=0, keepdims=True) + 1e-12)
    for _ in range(T):
        s = X.T @ xi                                   # (N,B) -- computed ONCE per step
        p = np.zeros_like(s)
        for wj, bj in zip(w, betas):
            p += wj * _softmax_cols(bj * s)            # cheap: O(mN)
        nxt = X @ p                                    # ONE matmul, not m
        xi = nxt / (np.linalg.norm(nxt, axis=0, keepdims=True) + 1e-12)
    return X.T @ xi


def ensemble_scores(X, Q, betas, weights=None, T=T_STEPS):
    """(3) m independent trajectories; fuse final score vectors once."""
    betas = np.asarray(betas, dtype=np.float64)
    w = np.full(len(betas), 1.0 / len(betas)) if weights is None else np.asarray(weights)
    out = None
    for wj, bj in zip(w, betas):
        s = descend_single(X, Q, bj, T=T)              # full separate trajectory
        out = wj * s if out is None else out + wj * s
    return out


def cosine_scores(X, Q):
    return X.T @ (Q / (np.linalg.norm(Q, axis=0, keepdims=True) + 1e-12))


# ----------------------------------------------------------------- metrics

def metrics_from_scores(sims, mem_cls_idx, gold_idx, k=5):
    """sims (N,B); returns per-query recall@1, recall@k, reciprocal rank (class-level)."""
    order = np.argsort(-sims, axis=0)                  # (N,B)
    top = mem_cls_idx[order]                           # class of each ranked memory
    hit = (top == gold_idx[None, :])
    r1 = hit[0].astype(np.float64)
    rk = hit[:k].any(axis=0).astype(np.float64)
    first = np.argmax(hit, axis=0)
    has = hit.any(axis=0)
    rr = np.where(has, 1.0 / (first + 1.0), 0.0)
    return r1, rk, rr


def paired_bootstrap_by_image(a, b, img_of_instance, n_img, n_resamples=10000, seed=0):
    """Bootstrap the mean paired difference, resampling IMAGES to respect the fact that
    each image contributes several query instances (levels x seeds)."""
    d = np.asarray(a, np.float64) - np.asarray(b, np.float64)
    by_img = [d[img_of_instance == i] for i in range(n_img)]
    sums = np.array([x.sum() for x in by_img])
    cnts = np.array([x.size for x in by_img])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_img, size=(n_resamples, n_img))
    means = sums[idx].sum(axis=1) / np.maximum(cnts[idx].sum(axis=1), 1)
    return float(d.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


# ----------------------------------------------------------------- data

def build_cache():
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"scales_n{N_PER_CLASS}_t{N_TEST_PER_CLASS}_s{len(SEEDS)}.npz"
    if f.exists():
        print(f"loading cache {f.name}")
        d = np.load(f, allow_pickle=True)
        return d["mem_emb"], list(d["mem_cls"]), d["test_Q"], list(d["test_cls"])

    print("embedding (first run only; cached afterwards)...")
    model, preprocess = build_encoder()
    rng = random.Random(0)
    mem_paths = collect_paths("train", N_PER_CLASS, rng)
    test_paths = collect_paths("val", N_TEST_PER_CLASS, rng)
    print(f"  bank {len(mem_paths)}, test {len(test_paths)} images "
          f"x {len(LEVELS)} levels x {len(SEEDS)} seeds")
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

    test_Q = np.zeros((len(LEVELS), len(SEEDS), len(test_paths), mem_emb.shape[1]))
    for li, lvl in enumerate(LEVELS):
        for si, sd in enumerate(SEEDS):
            test_Q[li, si] = occ_embed(test_paths, lvl, sd)
            print(f"  embedded level {int(lvl*100)}% seed {sd}")

    np.savez_compressed(f, mem_emb=mem_emb,
                        mem_cls=np.array([p.parent.name for p in mem_paths]),
                        test_Q=test_Q,
                        test_cls=np.array([p.parent.name for p in test_paths]))
    d = np.load(f, allow_pickle=True)
    return d["mem_emb"], list(d["mem_cls"]), d["test_Q"], list(d["test_cls"])


def main() -> int:
    mem_emb, mem_cls, test_Q, test_cls = build_cache()
    X = l2_normalize(np.stack(mem_emb), axis=1).T                    # (d,N)
    classes = sorted(set(mem_cls))
    cidx = {c: i for i, c in enumerate(classes)}
    mem_cls_idx = np.array([cidx[c] for c in mem_cls])
    n_img = test_Q.shape[2]

    # flatten (level, seed, image) -> instance, remembering which image each came from
    Qs, golds, img_of, lvl_of, seed_of = [], [], [], [], []
    for li in range(len(LEVELS)):
        for si in range(len(SEEDS)):
            Qs.append(test_Q[li, si])
            golds.append(np.array([cidx[c] for c in test_cls]))
            img_of.append(np.arange(n_img))
            lvl_of.append(np.full(n_img, li))
            seed_of.append(np.full(n_img, si))
    Q_all = np.concatenate(Qs, axis=0).T                              # (d, B)
    gold_all = np.concatenate(golds)
    img_of = np.concatenate(img_of)
    lvl_of = np.concatenate(lvl_of)
    seed_of = np.concatenate(seed_of)
    print(f"\nbank N={X.shape[1]}, d={X.shape[0]};  {n_img} images -> {Q_all.shape[1]} instances")

    # ---------------------------------------------------------- sanity checks
    print("\n" + "=" * 80)
    print("MECHANISM CHECKS")
    print("=" * 80)
    probe = Q_all[:, :200]

    s_single = descend_single(X, probe, 40.0)
    s_mixdup = descend_mixture(X, probe, [40.0, 40.0, 40.0])
    err = float(np.abs(s_single - s_mixdup).max())
    print(f"\n  [reduction] mixture with all betas equal == single beta:  max|diff| = {err:.2e}"
          f"   {'OK' if err < 1e-9 else 'FAIL'}")
    if err >= 1e-9:
        print("    => implementation is wrong; stopping.")
        return 1

    b_test = [16.0, 32.0, 48.0]
    s_mix1 = descend_mixture(X, probe, b_test, T=1)
    s_ens1 = ensemble_scores(X, probe, b_test, T=1)
    s_mix8 = descend_mixture(X, probe, b_test, T=8)
    s_ens8 = ensemble_scores(X, probe, b_test, T=8)
    d1 = float(np.abs(s_mix1 - s_ens1).mean())
    d8 = float(np.abs(s_mix8 - s_ens8).mean())
    agree1 = float((np.argmax(s_mix1, 0) == np.argmax(s_ens1, 0)).mean())
    agree8 = float((np.argmax(s_mix8, 0) == np.argmax(s_ens8, 0)).mean())
    print(f"\n  [divergence] integrated (2) vs ensemble (3), same betas {b_test}:")
    print(f"    T=1: mean|score diff| = {d1:.4f}   top-1 agreement = {agree1:.3f}")
    print(f"    T=8: mean|score diff| = {d8:.4f}   top-1 agreement = {agree8:.3f}")
    print("    (they must diverge as T grows -- composition and mixing do not commute)")

    # ---------------------------------------------------------- split
    rng = np.random.default_rng(42)
    perm = rng.permutation(n_img)
    tune_img = set(perm[:n_img // 2].tolist())
    tune_mask = np.array([i in tune_img for i in img_of])
    test_mask = ~tune_mask
    print(f"\n  tune {tune_mask.sum()} instances / test {test_mask.sum()} instances "
          f"({n_img//2} vs {n_img - n_img//2} disjoint images)")

    def r1_of(sims, mask):
        r1, _, _ = metrics_from_scores(sims[:, mask], mem_cls_idx, gold_all[mask])
        return r1

    # ---------------------------------------------------------- STAGE 1: tune
    print("\n" + "=" * 80)
    print("STAGE 1 -- select on the TUNE half only")
    print("=" * 80)
    print("\n  single beta:")
    best_b, best_acc = None, -1.0
    for b in SINGLE_GRID:
        acc = r1_of(descend_single(X, Q_all, float(b)), tune_mask).mean()
        print(f"    beta={b:>3}: {acc:.4f}")
        if acc > best_acc:
            best_acc, best_b = acc, float(b)
    print(f"  -> best single beta = {best_b:g} ({best_acc:.4f})")

    print("\n  integrated mixture sets:")
    best_set, best_set_acc = None, -1.0
    for name, bs in MIX_SETS.items():
        acc = r1_of(descend_mixture(X, Q_all, bs), tune_mask).mean()
        print(f"    {name:<14}: {acc:.4f}")
        if acc > best_set_acc:
            best_set_acc, best_set = acc, (name, bs)
    print(f"  -> best mixture = {best_set[0]} ({best_set_acc:.4f})")

    # ---------------------------------------------------------- STAGE 2: held out
    print("\n" + "=" * 80)
    print(f"STAGE 2 -- pre-committed evaluation on HELD-OUT half")
    print(f"  single beta={best_b:g}   mixture={best_set[0]}   ensemble uses the same betas")
    print("=" * 80)

    conds = {}
    t0 = time.perf_counter(); conds["cosine"] = cosine_scores(X, Q_all); t_cos = time.perf_counter() - t0
    t0 = time.perf_counter(); conds[f"single beta={best_b:g}"] = descend_single(X, Q_all, best_b); t_sgl = time.perf_counter() - t0
    t0 = time.perf_counter(); conds[f"INTEGRATED mix {best_set[0]}"] = descend_mixture(X, Q_all, best_set[1]); t_mix = time.perf_counter() - t0
    t0 = time.perf_counter(); conds[f"ensemble {best_set[0]}"] = ensemble_scores(X, Q_all, best_set[1]); t_ens = time.perf_counter() - t0
    conds["CONTROL integrated [1,5,20]"] = descend_mixture(X, Q_all, CONTROL_SET)
    conds["CONTROL ensemble [1,5,20]"] = ensemble_scores(X, Q_all, CONTROL_SET)

    m = len(best_set[1])
    timings = {"cosine": t_cos, f"single beta={best_b:g}": t_sgl,
               f"INTEGRATED mix {best_set[0]}": t_mix, f"ensemble {best_set[0]}": t_ens}
    matmuls = {"cosine": 1, f"single beta={best_b:g}": 2 * T_STEPS,
               f"INTEGRATED mix {best_set[0]}": 2 * T_STEPS,
               f"ensemble {best_set[0]}": 2 * T_STEPS * m}

    print(f"\n  {'condition':<30}{'R@1':>8}{'R@5':>8}{'MRR':>8}"
          f"{'30%':>8}{'50%':>8}{'70%':>8}{'ms/q':>9}{'matmul':>8}")
    print("  " + "-" * 95)

    res = {}
    for name, sims in conds.items():
        r1, r5, rr = metrics_from_scores(sims[:, test_mask], mem_cls_idx, gold_all[test_mask])
        res[name] = r1
        per_lvl = [r1[lvl_of[test_mask] == li].mean() for li in range(len(LEVELS))]
        tms = timings.get(name)
        mm = matmuls.get(name)
        tstr = f"{tms / Q_all.shape[1] * 1e3:>9.3f}" if tms else f"{'-':>9}"
        mstr = f"{mm:>8}" if mm else f"{'-':>8}"
        print(f"  {name:<30}{r1.mean():>8.4f}{r5.mean():>8.4f}{rr.mean():>8.4f}"
              + "".join(f"{v:>8.4f}" for v in per_lvl) + tstr + mstr)

    # --- emit JSON so paper Table 5 is GENERATED, not typed (repo policy) ---
    _rows = {}
    for name, sims in conds.items():
        r1, r5, rr = metrics_from_scores(sims[:, test_mask], mem_cls_idx, gold_all[test_mask])
        _rows[name] = {
            "R@1": float(r1.mean()), "R@5": float(r5.mean()), "MRR": float(rr.mean()),
            "matmuls": matmuls.get(name),
            "ms_per_query": (timings[name] / Q_all.shape[1] * 1e3) if name in timings else None,
            "per_level": {str(int(LEVELS[li] * 100)): float(r1[lvl_of[test_mask] == li].mean())
                          for li in range(len(LEVELS))},
            "is_control": name.startswith("CONTROL"),
        }
    _mb = {"rows": _rows, "best_beta": float(best_b), "best_set": list(best_set[1]),
           "n_test_instances": int(test_mask.sum()), "T_steps": int(T_STEPS)}

    # ---------------------------------------------------------- seed variance
    print(f"\n  variance across occlusion seeds (Recall@1, held-out half):")
    for name in conds:
        per_seed = [res[name][seed_of[test_mask] == si].mean() for si in range(len(SEEDS))]
        print(f"    {name:<30}" + "  ".join(f"{v:.4f}" for v in per_seed) +
              f"   mean {np.mean(per_seed):.4f} sd {np.std(per_seed):.4f}")

    # ---------------------------------------------------------- significance
    img_te = img_of[test_mask]
    remap = {v: i for i, v in enumerate(sorted(set(img_te.tolist())))}
    img_te_c = np.array([remap[v] for v in img_te])
    n_img_te = len(remap)

    print("\n" + "=" * 80)
    print("THE SCIENTIFIC QUESTION")
    print("=" * 80)
    mix_name = f"INTEGRATED mix {best_set[0]}"
    ens_name = f"ensemble {best_set[0]}"
    sgl_name = f"single beta={best_b:g}"

    print("\n  paired differences (bootstrap resamples IMAGES; 95% CI):\n")
    for a, b in [(mix_name, sgl_name), (mix_name, ens_name), (ens_name, sgl_name),
                 (sgl_name, "cosine"), (mix_name, "cosine")]:
        mu, lo, hi = paired_bootstrap_by_image(res[a], res[b], img_te_c, n_img_te)
        verdict = "SIGNIFICANT" if lo > 0 else ("worse" if hi < 0 else "n.s.")
        print(f"    {a:<30} - {b:<26} {mu:>+8.4f}  [{lo:+.4f},{hi:+.4f}]  {verdict}")

    _mb["n_test_images"] = int(n_img_te)
    _mb["contrasts"] = {}
    for _a, _b in [(mix_name, sgl_name), (mix_name, ens_name), (ens_name, sgl_name),
                   (sgl_name, "cosine"), (mix_name, "cosine")]:
        _m, _l, _h = paired_bootstrap_by_image(res[_a], res[_b], img_te_c, n_img_te)
        _mb["contrasts"][f"{_a} - {_b}"] = {"delta": float(_m), "lo": float(_l),
                                            "hi": float(_h), "sig": bool(_l > 0 or _h < 0)}
    import json as _json
    _dst = Path(__file__).resolve().parents[1] / "experiments" / "results" / "multibeta.json"
    _dst.parent.mkdir(parents=True, exist_ok=True)
    _dst.write_text(_json.dumps(_mb, indent=2))
    print(f"\n  wrote {_dst}")

    mu_ms, lo_ms, _ = paired_bootstrap_by_image(res[mix_name], res[sgl_name], img_te_c, n_img_te)
    mu_me, lo_me, hi_me = paired_bootstrap_by_image(res[mix_name], res[ens_name], img_te_c, n_img_te)
    print("\n  ANSWER: does mixing scales DURING settling beat the best single beta,")
    print("          beyond what independent runs already give?")
    if lo_ms > 0 and lo_me > 0:
        print("    => YES. Integrated mixture beats both the best single beta AND the")
        print("       ensemble. The per-step coupling is doing real work; investigate why.")
    elif lo_ms > 0:
        print("    => PARTIAL. Integrated beats single beta, but NOT the ensemble --")
        print("       the gain is explained by having several scales at all, not by")
        print(f"       coupling them during settling (and it costs {2*T_STEPS} matmuls vs "
              f"{2*T_STEPS*m} for the ensemble, so it is the cheaper way to get it).")
    else:
        print("    => NO. Mixing scales during settling does not beat the best single beta.")
        print("       Reported as a negative result; the single-beta configuration stands.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
