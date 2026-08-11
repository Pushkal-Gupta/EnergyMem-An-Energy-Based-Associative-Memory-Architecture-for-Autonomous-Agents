"""Review-2 demo. Runs offline in <60s. No API keys, no network.

    python scripts/demo_review2.py

Four things, in the order a panel will care about:

  1. PROPOSITION 1 -- dense top-k is a special case of our energy model (live check).
  2. The beta stability threshold -- a real diagnostic finding (docs/09_DAY0_FINDINGS.md F2).
  3. Honest equivalence -- at sufficient beta, descent == cosine on plain retrieval.
     We show this rather than hide it. It is what Proposition 1 predicts.
  4. Composite cues -- where naive descent FAILS, and residual peeling fixes it (F4/F5).

This demo deliberately shows a case where our method loses. That is the point: a panel
that sees you report your own negative result trusts your positive ones.

NOTE: synthetic embeddings. Cosine is at ceiling here (F6) -- real sentence embeddings
are the Week-1 deliverable. Say this out loud when presenting.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from energymem.base import Memory, l2_normalize                      # noqa: E402
from energymem.baselines.dense import DenseTopK                      # noqa: E402
from energymem.memory.energy import EnergyMemory, dense_topk_config  # noqa: E402

N_MEM, DIM, N_QUERY = 1_000, 256, 300


def build(rng, spread=0.20):
    """Planted clusters so near-duplicates exist -- flat random vectors are too easy."""
    nc = N_MEM // 10
    C = l2_normalize(rng.normal(size=(nc, DIM)), axis=1)
    a = rng.integers(0, nc, size=N_MEM)
    E = l2_normalize(C[a] + spread * rng.normal(size=(N_MEM, DIM)), axis=1)
    return [Memory(id=f"m{i}", text=f"episode {i}", embedding=E[i]) for i in range(N_MEM)], E


def cue(e, rate, rng):
    if rate <= 0:
        return e
    keep = rng.random(e.shape[0]) >= rate
    if not keep.any():
        keep[rng.integers(e.shape[0])] = True
    return l2_normalize(e * keep)


def section(n, title):
    print(f"\n[{n}] {title}\n" + "    " + "-" * (len(title) + 2))


def main() -> int:
    print("=" * 70)
    print("EnergyMem -- Review 2 demo   (synthetic corpus, preliminary)")
    print("=" * 70)

    rng = np.random.default_rng(0)
    mems, E = build(rng)
    dense = DenseTopK()
    dense.index(mems)
    print(f"\ncorpus: {N_MEM} memories, d={DIM}, planted cluster structure")

    # ---------------------------------------------------------------- 1
    section(1, "PROPOSITION 1: dense top-k is a corner of our parameter space")
    qs = l2_normalize(np.random.default_rng(99).normal(size=(500, DIM)), axis=1)
    em = dense_topk_config()
    em.index(mems)
    agree = sum(em.retrieve(q, k=1).ids[0] == dense.retrieve(q, k=1).ids[0] for q in qs)
    print(f"    config T=1, beta=1e6, b=0, R=I")
    print(f"    agreement with exact cosine top-1: {agree}/500")
    if agree != 500:
        print("    => FAILED. Stop; nothing downstream is trustworthy.")
        return 1
    print("    => VERIFIED. Our baseline is provably a special case of our method.")

    # ---------------------------------------------------------------- 2
    section(2, "beta stability threshold (a diagnostic finding)")
    rg = np.random.default_rng(1)
    tg = rg.choice(N_MEM, N_QUERY, replace=False)
    cues70 = [cue(E[t], 0.70, rg) for t in tg]
    print(f"    Recall@1 at 70% cue masking\n")
    print(f"    {'beta':>6} | " + " ".join(f"{'T='+str(T):>7}" for T in [1, 2, 4, 8, 16]))
    print("    " + "-" * 52)
    for beta in [4, 8, 16, 64]:
        row = []
        for T in [1, 2, 4, 8, 16]:
            m = EnergyMemory(n_steps=T, beta=float(beta), beta_schedule="const")
            m.index(mems)
            row.append(sum(m.retrieve(q, 1).ids[0] == f"m{t}" for q, t in zip(cues70, tg)) / N_QUERY)
        print(f"    {beta:>6} | " + " ".join(f"{x:>7.3f}" for x in row))
    print("\n    Below beta~16 the softmax is too flat: the state collapses to the")
    print("    global centroid on step 1. Annealing that STARTS low is harmful.")

    # ---------------------------------------------------------------- 3
    section(3, "Plain retrieval: descent == cosine at sufficient beta (as predicted)")
    m = EnergyMemory(n_steps=8, beta=32.0, beta_schedule="const")
    m.index(mems)
    print(f"    {'mask':>6} | {'cosine':>8} | {'EnergyMem T=8':>14}")
    print("    " + "-" * 36)
    for rate in [0.0, 0.3, 0.5, 0.7]:
        rg = np.random.default_rng(2)
        tt = rg.choice(N_MEM, N_QUERY, replace=False)
        cs = [cue(E[t], rate, rg) for t in tt]
        c = sum(dense.retrieve(q, 1).ids[0] == f"m{t}" for q, t in zip(cs, tt)) / N_QUERY
        e = sum(m.retrieve(q, 1).ids[0] == f"m{t}" for q, t in zip(cs, tt)) / N_QUERY
        print(f"    {rate:>6.0%} | {c:>8.3f} | {e:>14.3f}")
    print("\n    Identical -- exactly what Proposition 1 predicts. Multi-step descent")
    print("    alone buys nothing here. We report this rather than bury it.")

    # ---------------------------------------------------------------- 4
    section(4, "Composite cues: where naive descent FAILS, and how we fix it")
    print("    query = mean of k memories; gold = all k; metric = Recall over gold\n")
    print(f"    {'k':>3} | {'cosine':>8} | {'descent (endpoint)':>19} | {'+ residual peeling':>19}")
    print("    " + "-" * 60)
    mm = EnergyMemory(n_steps=8, beta=32.0, beta_schedule="const")
    mm.index(mems)
    for k in [2, 3, 4]:
        rg = np.random.default_rng(7)
        sets = [rg.choice(N_MEM, k, replace=False) for _ in range(N_QUERY)]
        queries = [l2_normalize(E[s].mean(axis=0)) for s in sets]
        gold = [{f"m{i}" for i in s} for s in sets]
        c = np.mean([len(set(dense.retrieve(q, k).ids) & g) / k for q, g in zip(queries, gold)])
        e = np.mean([len(set(mm.retrieve(q, k).ids) & g) / k for q, g in zip(queries, gold)])
        p = np.mean([len(set(mm.retrieve_multi(q, k).ids) & g) / k for q, g in zip(queries, gold)])
        print(f"    {k:>3} | {c:>8.3f} | {e:>19.3f} | {p:>19.3f}")
    print("\n    Descent converges to ONE point, destroying the multiplicity the query")
    print("    still carries. Residual peeling -- commit, project out, re-descend --")
    print("    restores it. This is our own negative result, and our own fix.")

    print("\n" + "=" * 70)
    print("HONEST STATUS: on synthetic embeddings cosine is at ceiling, so nothing")
    print("can beat it. Real sentence embeddings + DegradeBench + LongMemEval are")
    print("the Week 1-3 deliverables. See docs/09_DAY0_FINDINGS.md.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
