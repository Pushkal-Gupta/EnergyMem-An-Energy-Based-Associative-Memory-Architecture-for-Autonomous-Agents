"""Family F -- cost. What the architecture charges for the accuracy it does not add.

THE SCAR THIS FAMILY EXISTS FOR, restated: the paper once claimed an 8x speedup from
counting matmuls, and the clock said 21x when someone finally ran one. So: clocks only.

WHAT FAMILIES A-D DID TO THE HEADLINE. The backlog asked X90 to aim at "accuracy per
millisecond -- if expansion costs ~2x cosine and matches a cross-encoder at ~100x, that
is the quantifiable-impact number." That headline is dead. At the fold-selected
configuration expansion IS cosine (X31, 10/10 folds), so accuracy per millisecond can
only go down with every millisecond added. The measurement is still worth making, with
the sign reversed: this is the overhead that buys nothing, quantified, and the overhead
RRF charges for the +0.065 it does add.

X90 wall-clock per query     X91 bytes per memory     X92 latency vs bank size
X93 approximate NN is marked dead in the backlog: "does the gain survive HNSW/IVF" is
    gain-conditional, and an ANN index is an engineering choice, not a claim.
"""
from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from experiments.locomo_qa import (                              # noqa: E402
    RESULTS, Bank, banks, s_bm25, s_cosine, s_mixture, s_mixture_avg, s_mmr,
    s_rocchio, s_rrf,
)
from experiments.x6x_energy import descend                       # noqa: E402

OUT = RESULTS / "x90_cost.json"
R: dict = {}


def hr(t: str) -> None:
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


def hardware() -> dict:
    import subprocess
    try:
        cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                             capture_output=True, text=True).stdout.strip()
    except Exception:
        cpu = platform.processor()
    return dict(cpu=cpu, machine=platform.machine(), python=sys.version.split()[0],
                numpy=np.__version__, threads_env={k: v for k, v in
                                                   __import__("os").environ.items()
                                                   if "THREAD" in k.upper()})


def timed(fn, b: Bank, idxs, reps: int = 3, **par) -> tuple[float, float]:
    """(median ms per query, sd across reps). Warm up first -- BM25 builds an index."""
    fn(b, idxs[0], **par)
    per = []
    for _ in range(reps):
        t0 = time.perf_counter()
        for i in idxs:
            fn(b, i, **par)
        per.append((time.perf_counter() - t0) / len(idxs) * 1000.0)
    return float(np.median(per)), float(np.std(per))


# ====================================================================== X90

def x90(bks: list[Bank]) -> None:
    hr("X90  wall-clock per query, one conversation's bank")
    b = bks[0]
    idxs = list(range(0, b.m, 2))
    print(f"  bank n={b.n} turns, d={b.Z.shape[1]}, {len(idxs)} queries x 3 reps\n")
    rows = {}
    spec = [
        ("cosine", s_cosine, {}),
        ("rrf (cosine+bm25, CACHED bm25)", s_rrf, {"k_rrf": 60.0}),
        ("rocchio K0=3", s_rocchio, {"K0": 3, "gamma": 0.5}),
        ("mixture K0=3 (selected: K0=0)", s_mixture, {"K0": 3, "beta": 16.0}),
        ("mixture_avg K0=3", s_mixture_avg, {"K0": 3, "beta": 16.0}),
        ("mmr lam=0.7 pool=50", s_mmr, {"lam": 0.7, "pool": 50}),
    ]
    for name, fn, par in spec:
        ms, sd = timed(fn, b, idxs, **par)
        rows[name] = dict(ms=ms, sd=sd)
    # the energy path, timed on the same bank
    for T in (1, 3, 10):
        t0 = time.perf_counter()
        for i in idxs:
            xi = descend(b.Z, b.Q[i], beta=16.0, T=T)
            _ = b.Z @ xi
        rows[f"energy descent T={T} beta=16"] = dict(
            ms=(time.perf_counter() - t0) / len(idxs) * 1000.0, sd=None)

    # BM25 HONESTLY. `s_bm25` reads a matrix that locomo_qa computes once per bank and
    # caches, so timing it through that path measures an array index and reports 0.000 ms
    # -- a measurement artifact, and exactly the kind of thing the 8x-vs-21x scar was.
    # Build and per-query scoring are therefore timed directly against the index.
    from energymem.baselines.lexical import BM25
    from energymem.base import Memory
    mems = [Memory(id=i, text=t, timestamp=float(j))
            for j, (i, t) in enumerate(zip(b.ids, b.texts))]
    t0 = time.perf_counter()
    idx = BM25(); idx.index(mems)
    build_ms = (time.perf_counter() - t0) * 1000.0
    idx.scores(b.questions[0])
    t0 = time.perf_counter()
    for i in idxs:
        idx.scores(b.questions[i])
    bm25_ms = (time.perf_counter() - t0) / len(idxs) * 1000.0
    rows["bm25 scoring (uncached, per query)"] = dict(ms=bm25_ms, sd=None)
    rows["bm25 index build (one-off)"] = dict(ms=build_ms, sd=None, one_off=True)
    rows[f"bm25 build amortized over {b.m} queries"] = dict(
        ms=build_ms / b.m, sd=None)
    rows["rrf TRUE cost (cosine + uncached bm25 + 2 ranksorts)"] = dict(
        ms=rows["rrf (cosine+bm25, CACHED bm25)"]["ms"] + bm25_ms, sd=None)

    base = rows["cosine"]["ms"]
    for name, v in rows.items():
        print(f"  {name:34s} {v['ms']:8.3f} ms   {v['ms'] / base:6.2f}x cosine")
    R["X90"] = dict(rows=rows, cosine_ms=base, bank_n=b.n,
                    d=int(b.Z.shape[1]), n_queries=len(idxs), hardware=hardware())

    # cross-encoder: COST ONLY. The cached model is an NLI head, not a reranker, so it
    # says nothing about accuracy -- it is here as a same-size forward-pass cost proxy
    # for "rerank the top 50 with a base-sized cross-encoder", and labelled as such.
    try:
        from sentence_transformers import CrossEncoder
        ce = CrossEncoder("cross-encoder/nli-deberta-v3-base")
        top = np.argsort(-s_cosine(b, 0))[:50]
        pairs = [[b.questions[0], b.texts[j]] for j in top]
        ce.predict(pairs[:4])
        t0 = time.perf_counter()
        for i in idxs[:10]:
            t = np.argsort(-s_cosine(b, i))[:50]
            ce.predict([[b.questions[i], b.texts[j]] for j in t], show_progress_bar=False)
        ms = (time.perf_counter() - t0) / 10 * 1000.0
        print(f"\n  cross-encoder rerank top-50 (COST PROXY ONLY, NLI head, no accuracy "
              f"claim): {ms:.1f} ms = {ms / base:.0f}x cosine")
        R["X90"]["cross_encoder_cost_proxy"] = dict(
            ms=ms, ratio=ms / base, model="cross-encoder/nli-deberta-v3-base",
            caveat="NLI head, not a reranker: cost proxy only, no accuracy claim")
    except Exception as e:                                        # pragma: no cover
        print(f"\n  cross-encoder cost not measured: {type(e).__name__}: {e}")
        R["X90"]["cross_encoder_cost_proxy"] = dict(error=f"{type(e).__name__}: {e}")

    print("\n  ACCURACY PER MILLISECOND. At its selected configuration the mixture is")
    print("  cosine (X31, 10/10 folds), so every millisecond above cosine's is spent on")
    print("  an identical ranking. RRF is the only method here that buys anything.")


# ====================================================================== X91

def x91(bks: list[Bank]) -> None:
    hr("X91  bytes per memory")
    b = bks[0]
    d = b.Z.shape[1]
    rows = {
        "dense float64 (as stored in this project)": d * 8,
        "dense float32": d * 4,
        "dense float16": d * 2,
        "dense int8 (scalar-quantized)": d * 1,
    }
    idx = __import__("energymem.baselines.lexical", fromlist=["BM25"]).BM25()
    from energymem.base import Memory
    idx.index([Memory(id=i, text=t, timestamp=float(j))
               for j, (i, t) in enumerate(zip(b.ids, b.texts))])
    import pickle
    rows["BM25 postings (pickled)"] = len(pickle.dumps(idx)) / b.n
    rows["raw turn text (utf-8)"] = sum(len(t.encode()) for t in b.texts) / b.n
    for name, v in rows.items():
        print(f"  {name:44s} {v:9.1f} bytes/memory   "
              f"{v * 1e6 / 2**30:7.3f} GiB per million")
    print("\n  The expansion path adds NO stored state: its modes are bank vectors it")
    print("  already holds. Its cost is entirely time (X90), not memory. Note the")
    print("  project stores float64, which is 2x what any of these numbers need to be.")
    R["X91"] = dict(bytes_per_memory=rows, d=int(d), bank_n=b.n)


# ====================================================================== X92

def x92(bks: list[Bank]) -> None:
    hr("X92  latency vs bank size  (LATENCY ONLY -- see the note)")
    print("  Padding a bank with turns from other conversations changes the task, not")
    print("  just its size: that is X85, which is marked dead as framed. So accuracy is")
    print("  NOT reported against bank size here. Only the clock is.\n")
    big = np.concatenate([b.Z for b in bks])
    texts = [t for b in bks for t in b.texts]
    b0 = bks[0]
    rows = {}
    for n in (500, 1000, 2000, 5000, len(big)):
        n = min(n, len(big))
        sub = Bank(**{**b0.__dict__, "_bm25": None})
        sub.Z = big[:n]
        sub.texts = texts[:n]
        sub.ids = [f"pad/{i}" for i in range(n)]
        idxs = list(range(0, b0.m, 4))
        r = {}
        for name, fn, par in (("cosine", s_cosine, {}),
                              ("rrf", s_rrf, {}),
                              ("mixture K0=3", s_mixture, {"K0": 3}),
                              ("mmr pool=50", s_mmr, {"lam": 0.7, "pool": 50})):
            ms, sd = timed(fn, sub, idxs, reps=5, **par)
            r[name] = ms
            r[name + "/sd"] = sd
        rows[str(n)] = r
        print(f"  n={n:6d}  " + "  ".join(
            f"{k} {r[k]:7.3f}+-{r[k + '/sd']:.3f}ms"
            for k in ("cosine", "rrf", "mixture K0=3", "mmr pool=50")))
    ns = [int(k) for k in rows]
    print("\n  Scaling, smallest to largest bank "
          f"({ns[0]} -> {ns[-1]} turns, {ns[-1] / ns[0]:.1f}x):")
    for name in ("cosine", "rrf", "mixture K0=3", "mmr pool=50"):
        a, bb = rows[str(ns[0])][name], rows[str(ns[-1])][name]
        print(f"    {name:14s} {bb / a:5.1f}x slower  ({a:.3f} -> {bb:.3f} ms)")
    print("\n  READ THESE WITH THE NOISE FLOOR IN MIND. Sub-0.1 ms rows are dominated by")
    print("  Python and BLAS call overhead, not by n, which is why cosine's ratio can")
    print("  exceed the size ratio and why the n=5000 and n=5882 rows need not be")
    print("  ordered. MMR barely scales because its cost is the fixed pool=50 greedy")
    print("  loop, not the bank scan -- the one method here whose cost is in Python.")
    R["X92"] = dict(rows=rows, note="latency only; padding changes the task (see X85). "
                                    "Sub-0.1ms rows are at the call-overhead noise floor.")


if __name__ == "__main__":
    t0 = time.time()
    bks = banks()
    x90(bks)
    x91(bks)
    x92(bks)
    R["_meta"] = dict(seconds=round(time.time() - t0, 1), hardware=hardware(),
                      dead_arms={"X93": "gain-conditional ('does the gain survive "
                                        "HNSW/IVF'); also an engineering choice, not "
                                        "a claim about the architecture"})
    OUT.write_text(json.dumps(R, indent=2, default=str))
    print(f"\nwrote {OUT}  ({R['_meta']['seconds']}s)")
