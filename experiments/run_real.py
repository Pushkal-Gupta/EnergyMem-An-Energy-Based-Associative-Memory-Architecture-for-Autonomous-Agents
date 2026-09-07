"""THE REAL-DATA RUN. Every headline comparison, on real sentence embeddings,
5 seeds, bootstrap 95% CIs, with lexical and hybrid baselines.

This is the experiment docs/09_DAY0_FINDINGS.md F6 said had to happen before anything
measured on synthetic vectors could be believed.

    python experiments/run_real.py

Writes experiments/results/real/<timestamp>.json -- full config plus every metric.
"""
from __future__ import annotations

import json, sys, time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from energymem.base import Memory, l2_normalize                       # noqa: E402
from energymem.encoders import SentenceEncoder                        # noqa: E402
from energymem.memory.energy import EnergyMemory                      # noqa: E402
from energymem.baselines.lexical import BM25, HybridRRF               # noqa: E402
from energymem.eval.corpus import build                               # noqa: E402
from energymem.eval.metrics import bootstrap_ci                       # noqa: E402

MODEL = "sentence-transformers/all-mpnet-base-v2"
SEEDS = [0, 1, 2, 3, 4]
N_FILLER = 900
BETA, T = 64.0, 8   # beta chosen from the real-embedding sweep; see F11


# ------------------------------------------------------------------ corruptions
def mask_words(text, rate, rng):
    """D1 -- drop a fraction of the words from the cue."""
    w = text.split()
    if rate <= 0 or len(w) <= 2:
        return text
    keep = [x for x in w if rng.random() >= rate]
    return " ".join(keep) if keep else w[rng.integers(len(w))]


def embed_noise(v, sigma, rng):
    """D3 -- perturb the cue vector. sigma is the perturbation norm as a FRACTION OF
    THE CUE. Dimension-normalized (F10): without the sqrt(d) the same nominal sigma is
    a mild nudge at d=256 and total destruction at d=768."""
    if sigma <= 0:
        return v
    return l2_normalize(v + (sigma / np.sqrt(v.shape[0])) * rng.normal(size=v.shape))


# ------------------------------------------------------------------ readouts
def vote_project(em, q, k, m=32, sigma=0.15, seed=0):
    """F8 -- vote-count readout with residual projection of the cue.

    sigma is the perturbation norm as a fraction of the cue; divided by sqrt(d) so it
    means the same thing at d=256 and d=768 (F10).
    """
    X = em.X
    d = X.shape[0]
    sig_c = sigma / np.sqrt(d)
    rr = np.random.default_rng(seed)
    resid = l2_normalize(q.copy()); banned = set(); found = []
    for _ in range(k):
        Q0 = resid[:, None] + sig_c * rr.normal(size=(d, m))
        Q0 = l2_normalize(Q0, axis=0)
        for _t in range(T):
            Z = BETA * (X.T @ Q0); Z -= Z.max(axis=0, keepdims=True)
            P = np.exp(Z); P /= P.sum(axis=0, keepdims=True)
            Q0 = l2_normalize(X @ P, axis=0)
        S = X.T @ Q0
        if banned: S[list(banned), :] = -9
        L = np.argmax(S, axis=0)
        vals, cnt = np.unique(L, return_counts=True)
        w = int(vals[np.argmax(cnt)]); found.append(w); banned.add(w)
        v = X[:, w]; resid = resid - (resid @ v) * v
        n = np.linalg.norm(resid)
        if n < 1e-9: break
        resid = resid / n
    return found


def main():
    enc = SentenceEncoder(MODEL)
    t_start = time.time()

    CONDITIONS = [
        ("clean",        dict()),
        ("mask-30",      dict(mask=0.3)),
        ("mask-50",      dict(mask=0.5)),
        ("mask-70",      dict(mask=0.7)),
        ("noise-0.5",    dict(noise=0.5)),
        ("noise-1.0",    dict(noise=1.0)),
    ]
    METHODS = ["bm25", "dense", "hybrid_rrf", "energy_1step", "energy_T8",
               "peeling", "repulsion", "repulsion_b16", "vote_proj"]

    # per-condition, per-method: list over (seed, question) of recall
    acc = {c: {m: [] for m in METHODS} for c, _ in CONDITIONS}
    lat = {c: {m: [] for m in METHODS} for c, _ in CONDITIONS}

    for seed in SEEDS:
        eps, qs = build(seed=seed, n_filler=N_FILLER)
        texts = [e.text for e in eps]
        Emb = enc.encode(texts)
        mems = [Memory(id=f"e{i}", text=texts[i], timestamp=float(eps[i].t),
                       embedding=Emb[i]) for i in range(len(eps))]

        dense = EnergyMemory(n_steps=1, beta=1e6); dense.index(mems)     # Prop 1 corner
        e1 = EnergyMemory(n_steps=1, beta=BETA);   e1.index(mems)
        e8 = EnergyMemory(n_steps=T, beta=BETA);   e8.index(mems)
        e16 = EnergyMemory(n_steps=T, beta=16.0);  e16.index(mems)   # repulsion prefers soft beta
        bm = BM25(); bm.index(mems)
        hy = HybridRRF(); hy.index(mems)

        for cname, cfg in CONDITIONS:
            rng = np.random.default_rng(1000 + seed)
            for q in qs:
                k = len(q.gold_idx)
                gold = {f"e{i}" for i in q.gold_idx}
                qtext = mask_words(q.text, cfg.get("mask", 0.0), rng)
                qv = enc.encode(qtext)
                qv = embed_noise(qv, cfg.get("noise", 0.0), rng)

                def rec(ids):
                    return len(gold & set(ids[:k])) / k

                t0 = time.perf_counter(); r = bm.retrieve(qtext, k=k)
                acc[cname]["bm25"].append(rec(r.ids)); lat[cname]["bm25"].append((time.perf_counter()-t0)*1e3)

                t0 = time.perf_counter(); r = dense.retrieve(qv, k=k)
                acc[cname]["dense"].append(rec(r.ids)); lat[cname]["dense"].append((time.perf_counter()-t0)*1e3)

                t0 = time.perf_counter(); r = hy.retrieve(qtext, qv, k=k)
                acc[cname]["hybrid_rrf"].append(rec(r.ids)); lat[cname]["hybrid_rrf"].append((time.perf_counter()-t0)*1e3)

                t0 = time.perf_counter(); r = e1.retrieve(qv, k=k)
                acc[cname]["energy_1step"].append(rec(r.ids)); lat[cname]["energy_1step"].append((time.perf_counter()-t0)*1e3)

                t0 = time.perf_counter(); r = e8.retrieve(qv, k=k)
                acc[cname]["energy_T8"].append(rec(r.ids)); lat[cname]["energy_T8"].append((time.perf_counter()-t0)*1e3)

                t0 = time.perf_counter(); r = e8.retrieve_multi(qv, k=k)
                acc[cname]["peeling"].append(rec(r.ids)); lat[cname]["peeling"].append((time.perf_counter()-t0)*1e3)

                t0 = time.perf_counter(); r = e8.retrieve_multi_parallel(qv, k=k, seed=seed)
                acc[cname]["repulsion"].append(rec(r.ids)); lat[cname]["repulsion"].append((time.perf_counter()-t0)*1e3)

                t0 = time.perf_counter(); r = e16.retrieve_multi_parallel(qv, k=k, seed=seed)
                acc[cname]["repulsion_b16"].append(rec(r.ids)); lat[cname]["repulsion_b16"].append((time.perf_counter()-t0)*1e3)

                t0 = time.perf_counter(); idx = vote_project(e8, qv, k, seed=seed)
                acc[cname]["vote_proj"].append(len(gold & {f"e{i}" for i in idx[:k]})/k)
                lat[cname]["vote_proj"].append((time.perf_counter()-t0)*1e3)

        print(f"  seed {seed} done ({len(eps)} episodes, {len(qs)} questions)", flush=True)

    # ---------------------------------------------------------------- report
    print("\n" + "="*100)
    print(f"REAL SENTENCE EMBEDDINGS -- {MODEL}")
    print(f"{len(SEEDS)} seeds x {len(qs)} questions = {len(acc['clean']['dense'])} measurements per cell")
    print("Recall@|gold|, mean [bootstrap 95% CI]")
    print("="*100)
    hdr = f"{'condition':>10} | " + " ".join(f"{m[:12]:>14}" for m in METHODS)
    print(hdr); print("-"*len(hdr))
    out = {}
    for cname, _ in CONDITIONS:
        row = []
        out[cname] = {}
        for m in METHODS:
            a = np.array(acc[cname][m])
            _mu, lo, hi = bootstrap_ci(a, n_resamples=5000, seed=0)
            out[cname][m] = dict(mean=float(a.mean()), ci=[float(lo), float(hi)],
                                 latency_ms=float(np.mean(lat[cname][m])), n=int(a.size))
            row.append(f"{a.mean():.3f}")
        print(f"{cname:>10} | " + " ".join(f"{v:>14}" for v in row))

    print("\nWith confidence intervals (the numbers that can actually be reported):")
    for cname, _ in CONDITIONS:
        print(f"\n  [{cname}]")
        for m in METHODS:
            d = out[cname][m]
            print(f"    {m:>14}  {d['mean']:.3f}  [{d['ci'][0]:.3f}, {d['ci'][1]:.3f}]"
                  f"   {d['latency_ms']:7.2f} ms")

    rec = Path("experiments/results/real"); rec.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    payload = dict(
        model=MODEL, seeds=SEEDS, n_filler=N_FILLER, beta=BETA, n_steps=T,
        n_episodes=len(eps), n_questions=len(qs), conditions=[c for c, _ in CONDITIONS],
        results=out, wall_seconds=round(time.time()-t_start, 1),
        note="Purpose-built real-text probe (eval/corpus.py), NOT a public benchmark. "
             "LongMemEval / LoCoMo remain the reportable benchmarks (E6).",
    )
    f = rec / f"{stamp}.json"
    f.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
