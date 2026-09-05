"""Testing role deliverable -- Hopfield/energy memory as recall over corrupted ImageNet images.

What this shows (Review-2 "50% implementation" claim, image-memory framing):

  Store a bank of real ImageNet (Imagenette-160) images as memories, embedded with a
  pretrained ResNet18 (this is a stand-in for "what an agent would store" -- a feature
  vector, not raw pixels). Query with an OCCLUDED version of a held-out image from the
  same bank, re-embedded through the same encoder. Ask: does energy descent (Hopfield-style
  pattern completion) recover the correct memory better than plain cosine top-1 as
  occlusion gets worse?

Reuses the existing scaffold as-is:
  - energymem.memory.energy.EnergyMemory   -- the Hopfield/CCCP core
  - energymem.baselines.dense.DenseTopK    -- the baseline
  - energymem.eval.metrics                 -- recall@k, bootstrap CI

Run:
    python scripts/test_imagenet_hopfield.py
Needs: torch, torchvision, matplotlib, and data/imagenette2-160/ (see README note below).
"""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchvision
from torchvision import transforms
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from energymem.base import Memory, l2_normalize                      # noqa: E402
from energymem.baselines.dense import DenseTopK                      # noqa: E402
from energymem.memory.energy import EnergyMemory                     # noqa: E402
from energymem.eval.metrics import recall_at_k, bootstrap_ci         # noqa: E402

DATA_ROOT = Path(__file__).resolve().parents[1] / "data" / "imagenette2-160"
N_PER_CLASS_MEM = 30      # memory bank size per class
N_PER_CLASS_QUERY = 8     # held-out query images per class, drawn from val/
SEEDS = [0, 1, 2, 3, 4]
OCCLUSION_LEVELS = [0.0, 0.2, 0.4, 0.6, 0.8]   # fraction of image area blacked out
IMG_SIZE = 160


# --------------------------------------------------------------------------- encoder

def build_encoder():
    weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
    model = torchvision.models.resnet18(weights=weights)
    model.fc = torch.nn.Identity()  # penultimate features, 512-dim
    model.eval()
    preprocess = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return model, preprocess


@torch.no_grad()
def embed_images(model, preprocess, paths: list[Path]) -> np.ndarray:
    batch = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in paths])
    feats = model(batch).numpy().astype(np.float64)
    return l2_normalize(feats, axis=1)


def occlude(img: Image.Image, frac: float, rng: random.Random) -> Image.Image:
    """Black out a square patch covering `frac` of the image area. frac=0 -> no-op."""
    if frac <= 0:
        return img
    w, h = img.size
    side = int((frac ** 0.5) * min(w, h))
    if side <= 0:
        return img
    x0 = rng.randint(0, max(0, w - side))
    y0 = rng.randint(0, max(0, h - side))
    out = img.copy()
    px = out.load()
    for x in range(x0, min(x0 + side, w)):
        for y in range(y0, min(y0 + side, h)):
            px[x, y] = (0, 0, 0)
    return out


# --------------------------------------------------------------------------- data

def collect_paths(split: str, n_per_class: int, rng: random.Random) -> list[Path]:
    classes = sorted(p.name for p in (DATA_ROOT / split).iterdir() if p.is_dir())
    paths = []
    for c in classes:
        files = sorted((DATA_ROOT / split / c).glob("*.JPEG"))
        rng.shuffle(files)
        paths.extend(files[:n_per_class])
    return paths


def main() -> int:
    if not DATA_ROOT.exists():
        print(f"Missing dataset at {DATA_ROOT}. Download+extract imagenette2-160 first.")
        return 1

    print("=" * 72)
    print("EnergyMem -- image-memory test (Imagenette-160, ResNet18 features)")
    print("=" * 72)

    rng = random.Random(0)
    print("\nbuilding encoder (pretrained ResNet18, penultimate layer, 512-dim)...")
    model, preprocess = build_encoder()

    mem_paths = collect_paths("train", N_PER_CLASS_MEM, rng)
    query_paths = collect_paths("val", N_PER_CLASS_QUERY, rng)
    print(f"memory bank: {len(mem_paths)} images across 10 classes")
    print(f"query pool:  {len(query_paths)} held-out images from val/")

    t0 = time.perf_counter()
    mem_embs = embed_images(model, preprocess, mem_paths)
    print(f"embedded memory bank in {time.perf_counter() - t0:.1f}s")

    memories = [
        Memory(id=str(p), text=p.parent.name, embedding=mem_embs[i])
        for i, p in enumerate(mem_paths)
    ]

    dense = DenseTopK()
    dense.index(memories)

    # A few energy configs: T=1 must match dense exactly (Proposition 1 check);
    # T=8 at a couple of beta values is the actual pattern-completion test.
    configs = {
        "cosine (T=1, dense baseline)": DenseTopK(),
        "energy T=8 beta=8": EnergyMemory(n_steps=8, beta=8.0),
        "energy T=8 beta=32": EnergyMemory(n_steps=8, beta=32.0),
    }
    for name, retr in configs.items():
        retr.index(memories)

    # -------------------------------------------------------------- sanity check
    print("\n[sanity] Proposition 1: T=1,beta=1e6 energy config must equal cosine top-1")
    from energymem.memory.energy import dense_topk_config
    em_check = dense_topk_config()
    em_check.index(memories)
    q_check = mem_embs[:20]
    agree = sum(
        em_check.retrieve(q, k=1).ids[0] == dense.retrieve(q, k=1).ids[0] for q in q_check
    )
    print(f"    agreement on 20 uncorrupted queries: {agree}/20  {'OK' if agree == 20 else 'FAIL'}")

    # -------------------------------------------------------------- main sweep
    print("\n[main] Recall@1 recovering the CORRECT SOURCE IMAGE under occlusion")
    print("       (query = corrupted val-set image; gold = its correct class' bank entries")
    print("        -- reported as class-level recall, since exact same-image isn't in the bank)\n")

    results: dict[str, dict[float, list[float]]] = {name: {lvl: [] for lvl in OCCLUSION_LEVELS} for name in configs}
    latencies: dict[str, list[float]] = {name: [] for name in configs}

    for seed in SEEDS:
        srng = random.Random(seed)
        for lvl in OCCLUSION_LEVELS:
            corrupted_imgs = [
                occlude(Image.open(p).convert("RGB"), lvl, srng) for p in query_paths
            ]
            with torch.no_grad():
                batch = torch.stack([preprocess(im) for im in corrupted_imgs])
                q_embs = l2_normalize(model(batch).numpy().astype(np.float64), axis=1)
            gold_classes = [p.parent.name for p in query_paths]

            for name, retr in configs.items():
                for q, gold_cls in zip(q_embs, gold_classes):
                    res = retr.retrieve(q, k=5)
                    latencies[name].append(res.latency_ms)
                    pred_classes = [
                        next(m.text for m in memories if m.id == rid) for rid in res.ids
                    ]
                    hit = recall_at_k(pred_classes, [gold_cls] * 5, k=1)
                    results[name][lvl].append(hit)

    header = f"    {'occlusion':>10} | " + " | ".join(f"{n:>28}" for n in configs)
    print(header)
    print("    " + "-" * (len(header) - 4))
    table_rows = []
    for lvl in OCCLUSION_LEVELS:
        cells = []
        for name in configs:
            mean, lo, hi = bootstrap_ci(results[name][lvl])
            cells.append(f"{mean:.3f} [{lo:.3f},{hi:.3f}]")
        table_rows.append((lvl, cells))
        print(f"    {lvl:>9.0%} | " + " | ".join(f"{c:>28}" for c in cells))

    print("\n[latency] mean ms/query")
    for name in configs:
        print(f"    {name:<32} {np.mean(latencies[name]):.3f} ms")

    # -------------------------------------------------------------- plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    for name in configs:
        means = [bootstrap_ci(results[name][lvl])[0] for lvl in OCCLUSION_LEVELS]
        los = [bootstrap_ci(results[name][lvl])[1] for lvl in OCCLUSION_LEVELS]
        his = [bootstrap_ci(results[name][lvl])[2] for lvl in OCCLUSION_LEVELS]
        ax.plot(OCCLUSION_LEVELS, means, marker="o", label=name)
        ax.fill_between(OCCLUSION_LEVELS, los, his, alpha=0.15)
    ax.set_xlabel("occlusion fraction")
    ax.set_ylabel("class-level Recall@1")
    ax.set_title("Hopfield energy descent vs cosine top-1\nunder image occlusion (Imagenette-160, ResNet18 features)")
    ax.legend()
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    out_path = Path(__file__).resolve().parents[1] / "experiments" / "results" / "image_occlusion_recall.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"\nplot saved to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
