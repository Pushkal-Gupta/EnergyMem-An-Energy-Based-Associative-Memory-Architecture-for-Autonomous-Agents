"""X110, X111, X112 -- family I, other modalities. Pre-registered in
experiment-backlog/PREREG_HI.md, committed before this driver existed.

X110  TEXT-OCCLUSION FALSIFICATION, plus a control on BOTH modalities.
      The image result (image_holdout.json: energy descent beats cosine top-1 under
      occlusion, +0.0135 pooled) is re-measured with the protocol of
      scripts/confirm_holdout.py -- beta chosen on a disjoint tune half, ONE comparison on
      the test half -- and the same protocol is applied to text: Yahoo Answers Topics
      (10 classes, as Imagenette), 40 bank texts per class, held-out queries with a
      contiguous span of words deleted (30/50/70% -- the square patch's analogue).
      The CONTROL is a k-NN class vote with k chosen on the tune half. Cosine top-1 IS
      1-NN classification, and the image task scores class match, so "energy beats
      cosine" could be nothing more than "k-NN beats 1-NN". If the vote matches energy,
      the image effect is class smoothing, not memory.
      image_holdout.json is NOT overwritten; this writes x110_occlusion.json.

X111  IMAGE MULTI-TARGET, m SET BY CONSTRUCTION. Bank = Imagenette train images in 10
      disjoint partitions (one bank and one selection group each). Query = a 2x2 mosaic
      of m in {1,2,3,4} distinct bank images from distinct classes (free quadrants cycle
      the query's own images). Gold = the m source images -- instance retrieval of stored
      memories. The X32 interaction with m controlled instead of observed.

X112  CROSS-MODAL, CLIP ViT-B/32 for both modalities. Flickr8k test, 1000 images x 5
      captions, 10 image partitions. text->image has m=1, image->text has m=5.

    python experiments/x11x_modalities.py x110 x111 x112
"""
from __future__ import annotations

import io
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from experiments import corpus_qa as C                     # noqa: E402
from experiments.locomo_qa import Bank                      # noqa: E402
from energymem.base import Memory, l2_normalize             # noqa: E402
from energymem.memory.energy import EnergyMemory            # noqa: E402

CACHE = ROOT / "experiments" / "cache"
LEVELS = [0.3, 0.5, 0.7]
BETAS = [24, 28, 32, 36, 40, 44, 48, 56, 64, 80]          # confirm_holdout.py's grid
KNN_K = [1, 3, 5, 10, 20, 40]
IMG_METHODS = ["cosine", "mmr", "rocchio", "mixture", "mixture_avg"]


# ============================================================== X110 occlusion

def _yahoo(n_bank=40, n_test=400, max_words=60, seed=0):
    import pyarrow.parquet as pq
    rows = pq.read_table(ROOT / "data/yahoo/test.parquet").to_pylist()
    rng = random.Random(seed)
    rng.shuffle(rows)
    by = {}
    for r in rows:
        w = f"{r['question_title']} {r['question_content']} {r['best_answer']}".split()
        if len(w) < 20:
            continue
        by.setdefault(int(r["topic"]), []).append(" ".join(w[:max_words]))
    bank, bcls, test, tcls = [], [], [], []
    for c in sorted(by):
        bank += by[c][:n_bank]; bcls += [c] * n_bank
        test += by[c][n_bank:n_bank + n_test]; tcls += [c] * n_test
    return bank, np.array(bcls), test, np.array(tcls)


def occlude_text(text: str, frac: float, rng: random.Random, mask: str | None = None) -> str:
    """Delete (or replace with `mask`) a contiguous span covering `frac` of the words."""
    w = text.split()
    n = int(round(frac * len(w)))
    if n <= 0:
        return text
    s = rng.randint(0, len(w) - n)
    return " ".join(w[:s] + ([mask] * n if mask else []) + w[s + n:])


def _classify(bank_Z, bank_cls, Q, method, param):
    """Top-1 class for every query row. cosine = 1-NN; energy = EnergyMemory descent."""
    if method == "cosine":
        return bank_cls[np.argmax(Q @ bank_Z.T, axis=1)]
    if method == "knn":
        S = Q @ bank_Z.T
        nn = np.argsort(-S, axis=1)[:, :param]
        out = np.empty(len(Q), dtype=bank_cls.dtype)
        for i in range(len(Q)):
            votes = {}
            for j in nn[i]:                      # ties broken by summed similarity
                c = bank_cls[j]
                n, s = votes.get(c, (0, 0.0))
                votes[c] = (n + 1, s + S[i, j])
            out[i] = max(votes, key=lambda c: votes[c])
        return out
    if method == "energy":
        em = EnergyMemory(n_steps=8, beta=float(param))
        em.index([Memory(id=str(i), text="", embedding=z) for i, z in enumerate(bank_Z)])
        return np.array([bank_cls[int(em.retrieve(q, 1).ids[0])] for q in Q])
    raise ValueError(method)


def _holdout(bank_Z, bank_cls, test_Q, test_cls, label):
    """confirm_holdout.py's protocol, generalized: tune half selects, test half reports.

    test_Q: (L, n, d), one slab per occlusion level. Cluster = the source item (its L
    occluded versions are dependent)."""
    L, n, _ = test_Q.shape
    perm = np.random.default_rng(42).permutation(n)        # confirm_holdout's split seed
    tune, test = perm[:n // 2], perm[n // 2:]

    def hits(method, param, idx):
        return np.concatenate([(_classify(bank_Z, bank_cls, test_Q[li, idx], method, param)
                                == test_cls[idx]).astype(float) for li in range(L)])

    cos_t = hits("cosine", None, tune)
    sel = {}
    for meth, grid in (("energy", BETAS), ("knn", KNN_K)):
        scores = {p: hits(meth, p, tune).mean() for p in grid}
        sel[meth] = max(scores, key=scores.get)
        print(f"  [{label}] tune: cosine {cos_t.mean():.3f}; {meth} "
              + " ".join(f"{p}:{s:.3f}" for p, s in scores.items()) + f"  -> {sel[meth]}")
    clus = np.concatenate([test for _ in range(L)]).astype(str)
    r = {"cosine": hits("cosine", None, test), "energy": hits("energy", sel["energy"], test),
         "knn": hits("knn", sel["knn"], test)}
    lvl = np.repeat(np.arange(L), len(test))
    out = dict(n_items_test=int(len(test)), n_instances=int(len(clus)),
               selected=dict(beta=sel["energy"], k=int(sel["knn"])),
               acc={k: float(v.mean()) for k, v in r.items()}, per_level={})
    for a, b in (("energy", "cosine"), ("knn", "cosine"), ("energy", "knn")):
        out[f"{a}-{b}"] = C.paired(r[a], r[b], clus)
        x = out[f"{a}-{b}"]
        print(f"  [{label}] {a:>6} - {b:<6} {x['delta']:+.4f} [{x['lo']:+.4f}, {x['hi']:+.4f}]"
              f"  p={x['p']:.4g}")
    for li, f in enumerate(LEVELS):
        s = lvl == li
        row = {k: float(v[s].mean()) for k, v in r.items()}
        for a, b in (("energy", "cosine"), ("knn", "cosine"), ("energy", "knn")):
            row[f"{a}-{b}"] = C.paired(r[a][s], r[b][s], clus[s])
        out["per_level"][str(int(f * 100))] = row
        print(f"  [{label}] {int(f * 100)}%: cos {row['cosine']:.3f} energy {row['energy']:.3f} "
              f"knn {row['knn']:.3f} | e-c {row['energy-cosine']['delta']:+.4f} "
              f"[{row['energy-cosine']['lo']:+.4f},{row['energy-cosine']['hi']:+.4f}] "
              f"k-c {row['knn-cosine']['delta']:+.4f} e-k {row['energy-knn']['delta']:+.4f} "
              f"[{row['energy-knn']['lo']:+.4f},{row['energy-knn']['hi']:+.4f}]")
    return out


def x110():
    out = C.stamp("X110", protocol="scripts/confirm_holdout.py generalized; k-NN vote control")
    # ---- image arm, the existing cache builder and occlusion, untouched
    import confirm_holdout as CH
    mem_emb, mem_cls, test_Q, test_cls = CH.build_cache()
    print(f"\nIMAGE  bank {len(mem_cls)}, test {test_Q.shape[1]} images x {test_Q.shape[0]} levels")
    out["image"] = _holdout(np.asarray(mem_emb), np.asarray(mem_cls), test_Q,
                            np.asarray(test_cls), "image")
    out["image"]["encoder"] = "resnet18 IMAGENET1K_V1, penultimate"
    # ---- text arm
    bank, bcls, test, tcls = _yahoo()
    print(f"\nTEXT   bank {len(bank)}, test {len(test)} texts x {len(LEVELS)} levels")
    Zb = C.encode_unique(bank)
    for arm, mask in (("text_delete", None), ("text_mask", "[MASK]")):
        rng = random.Random(0)
        Q = np.stack([C.encode_unique([occlude_text(t, f, rng, mask) for t in test])
                      for f in LEVELS])
        out[arm] = _holdout(Zb, bcls, Q, tcls, arm)
        out[arm]["encoder"] = "all-mpnet-base-v2"
    out["text_corpus"] = "Yahoo Answers Topics test split, title+content+answer, first 60 words"
    C.write("x110_occlusion.json", out)


# ============================================================== X111 mosaics

def _resnet():
    from test_imagenet_hopfield import build_encoder
    return build_encoder()


def x111(n_groups=10, per_class=20, q_per_m=50, seed=0):
    import torch
    from PIL import Image
    from test_imagenet_hopfield import DATA_ROOT
    rng = random.Random(seed)
    classes = sorted(p.name for p in (DATA_ROOT / "train").iterdir() if p.is_dir())
    files = {c: sorted((DATA_ROOT / "train" / c).glob("*.JPEG")) for c in classes}
    for c in classes:
        rng.shuffle(files[c])
    model, prep = _resnet()
    f = CACHE / f"x111_g{n_groups}_pc{per_class}_q{q_per_m}_s{seed}.npz"
    CACHE.mkdir(parents=True, exist_ok=True)

    def emb(imgs):
        with torch.no_grad():
            outs = [model(torch.stack([prep(im) for im in imgs[i:i + 128]])).numpy()
                    for i in range(0, len(imgs), 128)]
        return l2_normalize(np.concatenate(outs).astype(np.float64), axis=1)

    def mosaic(paths):
        tile = [Image.open(p).convert("RGB").resize((80, 80)) for p in paths]
        out = Image.new("RGB", (160, 160))
        for k, (x, y) in enumerate(((0, 0), (80, 0), (0, 80), (80, 80))):
            out.paste(tile[k % len(tile)], (x, y))
        return out

    specs = []
    for g in range(n_groups):
        paths, cls = [], []
        for ci, c in enumerate(classes):
            paths += files[c][g * per_class:(g + 1) * per_class]; cls += [ci] * per_class
        qs, gold, cats = [], [], []
        for m in (1, 2, 3, 4):
            for _ in range(q_per_m):
                cs = rng.sample(range(len(classes)), m)
                gi = [ci * per_class + rng.randrange(per_class) for ci in cs]
                qs.append(gi); gold.append(sorted(gi)); cats.append(m)
        specs.append(dict(paths=paths, qs=qs, gold=gold, cats=cats))
    if f.exists():
        d = np.load(f)
        Zs, Qs = list(d["Z"]), list(d["Q"])
    else:
        Zs, Qs = [], []
        for g, s in enumerate(specs):
            Zs.append(emb([Image.open(p).convert("RGB") for p in s["paths"]]))
            Qs.append(emb([mosaic([s["paths"][i] for i in q]) for q in s["qs"]]))
            print(f"  embedded group {g}", flush=True)
        np.savez_compressed(f, Z=np.stack(Zs), Q=np.stack(Qs))
    bks = [Bank(sample_id=f"g{g}", ids=[str(p) for p in s["paths"]], texts=[""] * len(s["paths"]),
                Z=Zs[g], Q=Qs[g], gold=s["gold"], cats=s["cats"], answers=[""] * len(s["qs"]),
                questions=[""] * len(s["qs"]), sessions=[], speakers=[])
           for g, s in enumerate(specs)]
    strata = {f"m={m}": (m,) for m in (1, 2, 3, 4)} | {"all": None}
    res = C.run_corpus("x111_mosaic", bks, list(range(n_groups)), "bank",
                       {m: f"m={m}" for m in (1, 2, 3, 4)}, strata, methods=IMG_METHODS,
                       interactions=[("m=4", "m=1"), ("m=3", "m=1"), ("m=2", "m=1")])
    C.write("x111_mosaic.json", C.stamp("X111", encoder="resnet18 IMAGENET1K_V1, penultimate",
                                        bank_per_group=len(specs[0]["paths"]),
                                        queries_per_m_per_group=q_per_m, **res))


# ============================================================== X112 cross-modal

def x112(n_groups=10):
    import pyarrow.parquet as pq
    from PIL import Image
    from sentence_transformers import SentenceTransformer
    rows = pq.read_table(ROOT / "data/flickr8k/test.parquet").to_pylist()
    f = CACHE / "x112_flickr8k_clip_b32.npz"
    if f.exists():
        d = np.load(f)
        I, T = d["I"], d["T"]
    else:
        clip = SentenceTransformer("clip-ViT-B-32")
        imgs = [Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB") for r in rows]
        I = clip.encode(imgs, batch_size=64, convert_to_numpy=True, normalize_embeddings=True)
        T = clip.encode([r[f"caption_{k}"] for r in rows for k in range(5)], batch_size=128,
                        convert_to_numpy=True, normalize_embeddings=True)
        CACHE.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(f, I=I, T=T)
    I, T = I.astype(np.float64), T.astype(np.float64).reshape(len(rows), 5, -1)
    caps = [[r[f"caption_{k}"] for k in range(5)] for r in rows]
    perm = np.random.default_rng(C.SPLIT_SEED).permutation(len(rows))
    parts = [np.sort(perm[g::n_groups]) for g in range(n_groups)]
    t2i, i2t = [], []
    for g, idx in enumerate(parts):
        n = len(idx)
        t2i.append(Bank(sample_id=f"g{g}", ids=[str(i) for i in idx], texts=[""] * n,
                        Z=I[idx], Q=T[idx].reshape(n * 5, -1),
                        gold=[[j] for j in range(n) for _ in range(5)], cats=[1] * (5 * n),
                        answers=[""] * (5 * n), questions=[c for i in idx for c in caps[i]],
                        sessions=[], speakers=[]))
        i2t.append(Bank(sample_id=f"g{g}", ids=[f"{i}/{k}" for i in idx for k in range(5)],
                        texts=[c for i in idx for c in caps[i]], Z=T[idx].reshape(n * 5, -1),
                        Q=I[idx], gold=[list(range(5 * j, 5 * j + 5)) for j in range(n)],
                        cats=[5] * n, answers=[""] * n, questions=[""] * n,
                        sessions=[], speakers=[]))
    G = list(range(n_groups))
    out = C.stamp("X112", encoder="clip-ViT-B-32 (sentence-transformers), both modalities",
                  corpus="Flickr8k test (jxie/flickr8k)", n_images=len(rows))
    out["text_to_image"] = C.run_corpus("x112_t2i", t2i, G, "bank", {1: "m=1"},
                                        {"all": None}, methods=IMG_METHODS)
    out["image_to_text"] = C.run_corpus("x112_i2t", i2t, G, "bank", {5: "m=5"},
                                        {"all": None}, methods=IMG_METHODS)
    # interaction: per-query I4 contrast, re-derived per direction under the selection
    deltas = {}
    for key, bks in (("t2i", t2i), ("i2t", i2t)):
        v = {}
        for meth in ("mixture", "mixture_avg"):
            table = C.sweep(bks, C.METHODS[meth], C.GRIDS[meth])
            mets, clus, _ = C.select(bks, G, table, "bank")
            v[meth] = (mets[C.PRIMARY], clus)
        deltas[key] = (v["mixture"][0] - v["mixture_avg"][0], v["mixture"][1])
    out["interaction_i2t_minus_t2i"] = C.dod(deltas["i2t"][0], deltas["i2t"][1],
                                             deltas["t2i"][0], deltas["t2i"][1])
    print("\nINTERACTION Δ[i2t, m=5] - Δ[t2i, m=1]:", out["interaction_i2t_minus_t2i"])
    C.write("x112_crossmodal.json", out)


if __name__ == "__main__":
    for a in sys.argv[1:] or ["x110", "x111", "x112"]:
        {"x110": x110, "x111": x111, "x112": x112}[a]()
