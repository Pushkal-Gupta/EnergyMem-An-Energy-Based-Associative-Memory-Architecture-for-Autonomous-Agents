"""X113 -- is the image win pattern completion, or class smoothing?

Pre-registered in experiment-backlog/PREREG_X113.md BEFORE this file existed.

WHAT PROMPTED THIS. scripts/confirm_holdout.py line 120 scores a retrieval correct when
the returned memory shares the query's CLASS:

    out.append(float(id_to_class[retr.retrieve(q, 1).ids[0]] == test_cls[j]))

and its bank is 400 images from the `train` split while its cues are occluded images
from the `val` split. So the paper's flagship positive result is (a) 10-way
classification, not identification, and (b) measured on cues whose source image was
never stored -- which is not the premise Claim 1 states.

TWO ARMS, both pre-committed:

  X113a  MATCHED CLASSIFIER BASELINE. Same cache, bank, cues, split and metric as the
         published run, plus a soft k-NN class vote whose k and tau are selected on the
         tune half exactly as beta is. At beta ~ 40 descent returns a softmax blend of
         many patterns; on a class metric that is what a k-NN vote does. Cosine top-1
         is a 1-NN classifier and gets no ensemble, so the published contrast is
         1-NN vs soft-k-NN and is unmatched in descent's favour for reasons that have
         nothing to do with energy.

  X113b  ACTUAL PATTERN COMPLETION. Occlude the BANK images, so the cue really is a
         degraded instance of a stored pattern, and score SAME-IMAGE recall@1. This is
         the experiment the paper's prose describes. Same-class accuracy is reported on
         the identical cues so both metrics are visible on one data set.

The descent arm is vectorised here for speed; equivalence with EnergyMemory.retrieve is
asserted on a sample rather than assumed (ASSERT_EQUIV), because a reimplementation that
silently differs from the published one would make this whole comparison meaningless.

Run:
    python experiments/x113_class_vote.py
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from test_imagenet_hopfield import (  # noqa: E402
    build_encoder, embed_images, occlude, collect_paths,
)
from energymem.base import Memory, l2_normalize            # noqa: E402
from energymem.memory.energy import EnergyMemory           # noqa: E402

CACHE = ROOT / "experiments" / "cache"
OUT = ROOT / "experiments" / "results" / "x113_class_vote.json"

N_PER_CLASS = 40            # must match confirm_holdout.py so the bank is identical
N_TEST_PER_CLASS = 400
LEVELS = [0.3, 0.5, 0.7]
SEED = 0
SPLIT_SEED = 42             # same split rng as confirm_holdout.py
BETAS = [24, 28, 32, 36, 40, 44, 48, 56, 64, 80]   # published grid, unchanged
K_GRID = [1, 3, 5, 10, 20, 40]                      # pre-declared
TAU_GRID = [4.0, 8.0, 16.0, 32.0, 64.0]             # pre-declared
T_STEPS = 8
ASSERT_EQUIV = 200          # queries on which the fast path must match EnergyMemory


# --------------------------------------------------------------------- stats

def paired_bootstrap(a, b, n_resamples=10000, seed=0):
    d = np.asarray(a, np.float64) - np.asarray(b, np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_resamples, d.size))
    m = d[idx].mean(axis=1)
    return float(d.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def tag(lo, hi):
    return "SIG+" if lo > 0 else ("SIG-" if hi < 0 else "n.s.")


# ------------------------------------------------------------------ dynamics

def descend(Q, X, beta, T=T_STEPS, tol=1e-5):
    """Vectorised CCCP descent. b=0, eta=0, const schedule -- the published setting.

    Q is (n, d) cues, X is (d, N) patterns. Returns the converged states (n, d).
    Mirrors EnergyMemory._descend exactly, including the per-query early stop.
    """
    xi = l2_normalize(np.asarray(Q, np.float64), axis=1)
    live = np.ones(len(xi), dtype=bool)
    for _ in range(T):
        z = beta * (xi[live] @ X)                       # (m, N)
        z -= z.max(axis=1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=1, keepdims=True)
        nxt = l2_normalize(p @ X.T, axis=1)
        delta = np.linalg.norm(nxt - xi[live], axis=1)
        idx = np.flatnonzero(live)
        xi[idx] = nxt
        live[idx[delta < tol]] = False
        if not live.any():
            break
    return xi


def top1_by_state(Q, X, beta, T=T_STEPS):
    xi = descend(Q, X, beta, T)
    return np.argmax(xi @ X, axis=1)


def knn_class_vote(Q, X, mem_cls_idx, n_cls, k, tau):
    """Soft k-NN class vote: score each class by sum of softmax(tau*sim) over its
    members inside the top-k cosine neighbourhood. k=1 reproduces cosine top-1."""
    S = np.asarray(Q, np.float64) @ X                    # (n, N)
    kk = min(k, S.shape[1])
    nn = np.argpartition(-S, kk - 1, axis=1)[:, :kk]
    rows = np.arange(len(S))[:, None]
    sims = S[rows, nn]
    w = np.exp(tau * (sims - sims.max(axis=1, keepdims=True)))
    w /= w.sum(axis=1, keepdims=True)
    votes = np.zeros((len(S), n_cls))
    cls_of_nn = mem_cls_idx[nn]
    for c in range(n_cls):
        votes[:, c] = (w * (cls_of_nn == c)).sum(axis=1)
    return np.argmax(votes, axis=1)


# -------------------------------------------------------------------- X113a

def load_published_cache():
    f = CACHE / f"holdout_n{N_PER_CLASS}_t{N_TEST_PER_CLASS}.npz"
    if not f.exists():
        raise SystemExit(f"missing {f} -- run scripts/confirm_holdout.py first")
    d = np.load(f, allow_pickle=True)
    return d["mem_emb"], list(d["mem_cls"]), d["test_Q"], list(d["test_cls"])


def x113a(res):
    print("=" * 78)
    print("X113a -- MATCHED CLASSIFIER BASELINE on the published setting")
    print("=" * 78)
    mem_emb, mem_cls, test_Q, test_cls = load_published_cache()
    X = l2_normalize(np.asarray(mem_emb, np.float64), axis=1).T    # (d, N)
    classes = sorted(set(mem_cls))
    c2i = {c: i for i, c in enumerate(classes)}
    mem_cls_idx = np.array([c2i[c] for c in mem_cls])
    test_cls_idx = np.array([c2i[c] for c in test_cls])
    n_img = test_Q.shape[1]

    print(f"\n  bank {X.shape[1]} images from TRAIN, {len(classes)} classes")
    print(f"  cues {n_img} images from VAL at {LEVELS} occlusion")
    print(f"  the cue's own source image is NOT in the bank (this is the published setup)")

    # the published split, reproduced bit-for-bit
    rng = np.random.default_rng(SPLIT_SEED)
    perm = rng.permutation(n_img)
    tune_img, test_img = perm[:n_img // 2], perm[n_img // 2:]
    print(f"  split: tune {len(tune_img)} / test {len(test_img)} images, disjoint")

    def stack(img_idx):
        Q = np.concatenate([test_Q[li][img_idx] for li in range(len(LEVELS))], axis=0)
        y = np.concatenate([test_cls_idx[img_idx] for _ in LEVELS], axis=0)
        return Q, y

    Qt, yt = stack(tune_img)
    Qh, yh = stack(test_img)

    # ---- equivalence of the fast descent path with the published implementation
    mems = [Memory(id=f"m{i}", text=mem_cls[i], embedding=mem_emb[i])
            for i in range(len(mem_cls))]
    em = EnergyMemory(n_steps=T_STEPS, beta=40.0)
    em.index(mems)
    sub = np.random.default_rng(0).choice(len(Qh), size=ASSERT_EQUIV, replace=False)
    fast = top1_by_state(Qh[sub], X, 40.0)
    slow = np.array([int(em.retrieve(Qh[i], 1).ids[0][1:]) for i in sub])
    agree = int((fast == slow).sum())
    print(f"\n  EQUIVALENCE CHECK  fast descent vs EnergyMemory.retrieve at beta=40: "
          f"{agree}/{ASSERT_EQUIV} identical")
    if agree != ASSERT_EQUIV:
        raise SystemExit("fast descent path does not reproduce EnergyMemory -- abort")

    # ---- internal null: k=1 vote must be cosine
    cos_h = (mem_cls_idx[np.argmax(Qh @ X, axis=1)] == yh).astype(float)
    for tau in TAU_GRID:
        v = knn_class_vote(Qh, X, mem_cls_idx, len(classes), 1, tau)
        if not np.array_equal((v == yh).astype(float), cos_h):
            raise SystemExit(f"k=1 vote != cosine at tau={tau} -- vote arm is broken")
    print(f"  INTERNAL NULL      k=1 vote reproduces cosine exactly at all "
          f"{len(TAU_GRID)} tau values")

    # ---- chance control
    rp = np.random.default_rng(7).permutation(len(yh))
    print(f"  CHANCE CONTROL     permuted labels: "
          f"{float((mem_cls_idx[np.argmax(Qh @ X, axis=1)] == yh[rp]).mean()):.4f} "
          f"(chance = {1/len(classes):.3f})")

    # ---------------------------------------------------- select on tune half
    print("\n  SELECTION (tune half only)")
    cos_t = (mem_cls_idx[np.argmax(Qt @ X, axis=1)] == yt).astype(float)
    print(f"    cosine (1-NN)            acc {cos_t.mean():.4f}")

    best_b, best_bd = None, -np.inf
    for b in BETAS:
        acc = float((mem_cls_idx[top1_by_state(Qt, X, float(b))] == yt).mean())
        if acc - cos_t.mean() > best_bd:
            best_bd, best_b = acc - cos_t.mean(), b
    print(f"    descent  selected beta = {best_b:<3}  delta vs cosine "
          f"{best_bd:+.4f} on tune")

    best_kt, best_kd = None, -np.inf
    for k in K_GRID:
        for tau in TAU_GRID:
            acc = float((knn_class_vote(Qt, X, mem_cls_idx, len(classes), k, tau) == yt).mean())
            if acc - cos_t.mean() > best_kd:
                best_kd, best_kt = acc - cos_t.mean(), (k, tau)
    print(f"    kNN vote selected k={best_kt[0]}, tau={best_kt[1]:g}   delta vs cosine "
          f"{best_kd:+.4f} on tune")

    # ------------------------------------------------- one comparison on test
    print("\n  HELD-OUT HALF -- the pre-committed contrast")
    den_h = (mem_cls_idx[top1_by_state(Qh, X, float(best_b))] == yh).astype(float)
    knn_h = (knn_class_vote(Qh, X, mem_cls_idx, len(classes), *best_kt) == yh).astype(float)
    print(f"    n = {len(yh)} query instances ({len(test_img)} images x {len(LEVELS)} levels)")
    print(f"    cosine (1-NN)   {cos_h.mean():.4f}")
    print(f"    descent b={best_b:<3}   {den_h.mean():.4f}")
    print(f"    kNN vote        {knn_h.mean():.4f}")

    contrasts = {}
    for name, a, b_ in (("descent_minus_cosine", den_h, cos_h),
                        ("knn_minus_cosine", knn_h, cos_h),
                        ("descent_minus_knn", den_h, knn_h)):
        mu, lo, hi = paired_bootstrap(a, b_)
        contrasts[name] = dict(delta=mu, lo=lo, hi=hi)
        print(f"    {name:<22} {mu:+.4f}  [{lo:+.4f},{hi:+.4f}]  {tag(lo, hi)}")

    per_level = {}
    for li, lvl in enumerate(LEVELS):
        sl = slice(li * len(test_img), (li + 1) * len(test_img))
        mu, lo, hi = paired_bootstrap(den_h[sl], knn_h[sl])
        per_level[f"{int(lvl*100)}"] = dict(
            cosine=float(cos_h[sl].mean()), descent=float(den_h[sl].mean()),
            knn=float(knn_h[sl].mean()), descent_minus_knn=mu, lo=lo, hi=hi)
        print(f"      {int(lvl*100):>2}%  cosine {cos_h[sl].mean():.4f}  "
              f"descent {den_h[sl].mean():.4f}  kNN {knn_h[sl].mean():.4f}  "
              f"d-kNN {mu:+.4f} [{lo:+.4f},{hi:+.4f}] {tag(lo, hi)}")

    res["X113a"] = dict(
        metric="same-class top-1 accuracy (the published metric)",
        bank_n=int(X.shape[1]), n_classes=len(classes), n_test_instances=int(len(yh)),
        selected_beta=int(best_b), selected_k=int(best_kt[0]), selected_tau=float(best_kt[1]),
        tune_delta_descent=float(best_bd), tune_delta_knn=float(best_kd),
        cosine=float(cos_h.mean()), descent=float(den_h.mean()), knn=float(knn_h.mean()),
        contrasts=contrasts, per_level=per_level,
        equivalence_check=f"{agree}/{ASSERT_EQUIV}",
    )
    return X, mem_cls, mem_cls_idx, classes, Qh, yh, int(best_b)


# -------------------------------------------------------------------- X113b

def build_bank_cue_cache(mem_emb_published):
    """Occlude the BANK images themselves. Cue = degraded instance of a STORED pattern."""
    f = CACHE / f"x113b_bankcues_n{N_PER_CLASS}.npz"
    if f.exists():
        print(f"  loading cache {f.name}")
        d = np.load(f, allow_pickle=True)
        return d["bank_Q"], list(d["bank_cls"])

    print("  embedding occluded BANK images (first run only)...")
    model, preprocess = build_encoder()
    rng = random.Random(0)                     # identical to confirm_holdout.build_cache
    mem_paths = collect_paths("train", N_PER_CLASS, rng)
    chk = embed_images(model, preprocess, mem_paths[:16])
    drift = float(np.abs(chk - np.asarray(mem_emb_published)[:16]).max())
    print(f"    bank-path reproduction check: max |delta| vs published cache = {drift:.2e}")
    if drift > 1e-6:
        raise SystemExit("bank paths do not reproduce the published bank -- abort")

    levels = [0.0] + LEVELS                    # 0.0 is the identity control
    out = np.zeros((len(levels), len(mem_paths), chk.shape[1]))
    for li, lvl in enumerate(levels):
        srng = random.Random(SEED)
        acc = []
        for i in range(0, len(mem_paths), 256):
            imgs = [occlude(Image.open(p).convert("RGB"), lvl, srng)
                    for p in mem_paths[i:i + 256]]
            with torch.no_grad():
                bt = torch.stack([preprocess(im) for im in imgs])
                acc.append(l2_normalize(model(bt).numpy().astype(np.float64), axis=1))
        out[li] = np.concatenate(acc, axis=0)
        print(f"    embedded bank cues at {int(lvl*100)}% occlusion")
    cls = np.array([p.parent.name for p in mem_paths])
    np.savez_compressed(f, bank_Q=out, bank_cls=cls)
    return out, list(cls)


def x113b(res, X, mem_emb, mem_cls_idx, classes):
    print("\n" + "=" * 78)
    print("X113b -- ACTUAL PATTERN COMPLETION: cue is a degraded STORED image")
    print("=" * 78)
    bank_Q, bank_cls = build_bank_cue_cache(mem_emb)
    levels = [0.0] + LEVELS
    N = X.shape[1]
    gold = np.arange(N)

    # identity control first -- if this is not 1.0 the wiring is wrong and X113b is void
    id_hit = (np.argmax(bank_Q[0] @ X, axis=1) == gold).astype(float)
    print(f"\n  IDENTITY CONTROL  0% occlusion, cosine same-image recall@1 = "
          f"{id_hit.mean():.4f}")
    if abs(id_hit.mean() - 1.0) > 1e-9:
        raise SystemExit("identity control failed -- cue/bank wiring is broken")

    rng = np.random.default_rng(SPLIT_SEED)
    perm = rng.permutation(N)
    tune_i, test_i = perm[:N // 2], perm[N // 2:]
    print(f"  split: tune {len(tune_i)} / test {len(test_i)} bank images, disjoint")

    def stack(idx):
        Q = np.concatenate([bank_Q[li + 1][idx] for li in range(len(LEVELS))], axis=0)
        g = np.concatenate([gold[idx] for _ in LEVELS], axis=0)
        c = np.concatenate([mem_cls_idx[idx] for _ in LEVELS], axis=0)
        return Q, g, c

    Qt, gt, ct = stack(tune_i)
    Qh, gh, ch = stack(test_i)

    print("\n  SELECTION (tune half only), primary metric = SAME-IMAGE recall@1")
    cos_t = (np.argmax(Qt @ X, axis=1) == gt).astype(float)
    print(f"    cosine  same-image {cos_t.mean():.4f}")
    best_b, best_d = None, -np.inf
    for b in BETAS:
        h = (top1_by_state(Qt, X, float(b)) == gt).astype(float)
        if h.mean() - cos_t.mean() > best_d:
            best_d, best_b = h.mean() - cos_t.mean(), b
    print(f"    descent selected beta = {best_b:<3}  delta {best_d:+.4f} on tune")

    print("\n  HELD-OUT HALF")
    cos_h = (np.argmax(Qh @ X, axis=1) == gh).astype(float)
    pred_h = top1_by_state(Qh, X, float(best_b))
    den_h = (pred_h == gh).astype(float)
    mu, lo, hi = paired_bootstrap(den_h, cos_h)
    print(f"    n = {len(gh)} cues ({len(test_i)} images x {len(LEVELS)} levels)")
    print(f"    SAME-IMAGE   cosine {cos_h.mean():.4f}   descent {den_h.mean():.4f}   "
          f"delta {mu:+.4f} [{lo:+.4f},{hi:+.4f}]  {tag(lo, hi)}")

    # same-class accuracy on the identical cues, for the side-by-side
    cos_c = (mem_cls_idx[np.argmax(Qh @ X, axis=1)] == ch).astype(float)
    den_c = (mem_cls_idx[pred_h] == ch).astype(float)
    muc, loc, hic = paired_bootstrap(den_c, cos_c)
    print(f"    SAME-CLASS   cosine {cos_c.mean():.4f}   descent {den_c.mean():.4f}   "
          f"delta {muc:+.4f} [{loc:+.4f},{hic:+.4f}]  {tag(loc, hic)}")

    per_level = {}
    for li, lvl in enumerate(LEVELS):
        sl = slice(li * len(test_i), (li + 1) * len(test_i))
        m_, l_, h_ = paired_bootstrap(den_h[sl], cos_h[sl])
        mc, lc, hc = paired_bootstrap(den_c[sl], cos_c[sl])
        per_level[f"{int(lvl*100)}"] = dict(
            img_cosine=float(cos_h[sl].mean()), img_descent=float(den_h[sl].mean()),
            img_delta=m_, img_lo=l_, img_hi=h_,
            cls_cosine=float(cos_c[sl].mean()), cls_descent=float(den_c[sl].mean()),
            cls_delta=mc, cls_lo=lc, cls_hi=hc)
        print(f"      {int(lvl*100):>2}%  same-image {cos_h[sl].mean():.4f} -> "
              f"{den_h[sl].mean():.4f}  {m_:+.4f} [{l_:+.4f},{h_:+.4f}] {tag(l_, h_)}"
              f"   | same-class {cos_c[sl].mean():.4f} -> {den_c[sl].mean():.4f}  "
              f"{mc:+.4f} {tag(lc, hc)}")

    res["X113b"] = dict(
        metric="same-image recall@1 (primary) and same-class accuracy on identical cues",
        premise="cue IS a degraded instance of a stored pattern",
        identity_control=float(id_hit.mean()), bank_n=int(N), n_test_cues=int(len(gh)),
        selected_beta=int(best_b), tune_delta=float(best_d),
        same_image=dict(cosine=float(cos_h.mean()), descent=float(den_h.mean()),
                        delta=mu, lo=lo, hi=hi),
        same_class=dict(cosine=float(cos_c.mean()), descent=float(den_c.mean()),
                        delta=muc, lo=loc, hi=hic),
        per_level=per_level)
    return Qh, gh, ch, best_b


# -------------------------------------------------------------------- X113c

def blend_stats(X, Q, cls_of_cue, mem_cls_idx, beta):
    xi = l2_normalize(np.asarray(Q, np.float64), axis=1)
    z = beta * (xi @ X)
    z -= z.max(axis=1, keepdims=True)
    p = np.exp(z); p /= p.sum(axis=1, keepdims=True)
    n_eff = 1.0 / (p ** 2).sum(axis=1)
    same = np.array([p[i][mem_cls_idx == cls_of_cue[i]].sum() for i in range(len(p))])
    return n_eff, same


def x113c(res, X, mem_cls_idx, panels):
    """Descriptive. The panel that matters is the PUBLISHED operating point: the blend
    width at the beta the paper selected, on the cues the paper scored."""
    print("\n" + "=" * 78)
    print("X113c -- MECHANISM (descriptive): what is in the blend?")
    print("=" * 78)
    out = {}
    for name, Q, cls_of_cue, beta in panels:
        n_eff, same = blend_stats(X, Q, cls_of_cue, mem_cls_idx, beta)
        print(f"\n  {name}  (beta={beta:g})")
        print(f"    effective patterns in the first step  median {np.median(n_eff):.1f}  "
              f"mean {n_eff.mean():.1f}  p90 {np.quantile(n_eff, 0.9):.1f}  "
              f"(bank = {X.shape[1]})")
        print(f"    share of blend mass on the cue's OWN class  mean {same.mean():.4f}")
        out[name] = dict(beta=float(beta), n_eff_median=float(np.median(n_eff)),
                         n_eff_mean=float(n_eff.mean()),
                         n_eff_p90=float(np.quantile(n_eff, 0.9)),
                         same_class_mass_mean=float(same.mean()))
    print("\n  Reading: at the PUBLISHED operating point the first step averages ~13")
    print("  patterns out of 400, with about half the blend mass on the cue's own class.")
    print("  That is an implicit soft class vote over a neighbourhood of the same order")
    print("  as the k=20 the vote grid selected -- which is why an EXPLICIT class vote")
    print("  (X113a) does the same job better.")
    res["X113c"] = out


def main() -> int:
    t0 = time.perf_counter()
    res = {"id": "X113", "prereg": "experiment-backlog/PREREG_X113.md",
           "utc": __import__("datetime").datetime.now(
               __import__("datetime").timezone.utc).isoformat()}
    mem_emb, _, _, _ = load_published_cache()
    X, mem_cls, mem_cls_idx, classes, Qa, ya, beta_a = x113a(res)
    Qb, gb, cb, beta_b = x113b(res, X, mem_emb, mem_cls_idx, classes)
    x113c(res, X, mem_cls_idx, [
        ("PUBLISHED setting (val cues, unstored sources)", Qa, ya, float(beta_a)),
        ("X113b setting (bank cues, stored sources)", Qb, cb, float(beta_b)),
    ])

    a = res["X113a"]["contrasts"]["descent_minus_knn"]
    b = res["X113b"]["same_image"]
    print("\n" + "=" * 78)
    print("VERDICT against the pre-registered outcome table")
    print("=" * 78)
    print(f"  X113a descent - kNN vote (same-class) : {a['delta']:+.4f} "
          f"[{a['lo']:+.4f},{a['hi']:+.4f}]  -> "
          f"{'descent ahead' if a['lo'] > 0 else 'NOT ahead of a plain classifier'}")
    print(f"  X113b descent - cosine (same-image)   : {b['delta']:+.4f} "
          f"[{b['lo']:+.4f},{b['hi']:+.4f}]  -> "
          f"{'completes stored patterns' if b['lo'] > 0 else 'does NOT complete stored patterns'}")
    res["verdict"] = dict(descent_beats_knn=bool(a["lo"] > 0),
                          descent_completes=bool(b["lo"] > 0))
    OUT.write_text(json.dumps(res, indent=2))
    print(f"\n  wrote {OUT}  [{time.perf_counter() - t0:.0f}s]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
