"""LoCoMo, corrected. Supersedes exp_locomo_adaptive.py, which had three defects:

  D1 (data, invalidating)  dia_ids are unique only WITHIN a conversation -- 5882 turns
      carry 1033 distinct ids, and 132/134 of conversation 0's gold ids also occur in
      other conversations. The old capacity sweep pooled banks without namespacing, so
      unrelated turns satisfied gold ids and scored as hits. Fixed in eval/locomo.py by
      namespacing ids with sample_id.

  D2 (algorithm)  descent with no query anchor replaces the state with a blend of stored
      patterns on step 1; the cue never re-enters, and readout ranks by proximity to the
      blend. Fixed by the `anchor` parameter -- anchor=1.0 reduces to exact cosine, so
      the baseline is now a point in our own parameter space and the sweep measures what
      descent adds or costs.

  D3 (confound)  pp_start=32 forces step 1 to average ~32 memories. docs/09_DAY0_FINDINGS
      F2 already reported that annealing which starts broad is harmful, so "adaptive
      beta" was entangled with a schedule known to hurt. Fixed by sweeping pp_start.

Also: Coverage@k now divides by min(k, |gold|). With k=5 and up to 19 gold turns the old
denominator capped multi-hop coverage at 5/19, deflating absolute numbers.

Run:
    python scripts/exp_locomo_anchored.py
"""

from __future__ import annotations

import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from energymem.baselines.dense import DenseTopK                      # noqa: E402
from energymem.memory.energy import EnergyMemory                     # noqa: E402
from energymem.memory.adaptive import AdaptiveBetaEnergyMemory       # noqa: E402
from energymem.eval.locomo import load_locomo                        # noqa: E402
from energymem.eval.metrics import recall_at_k, bootstrap_ci         # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data" / "locomo10.json"
K = 5
ENCODER_NAME = "sentence-transformers/all-MiniLM-L6-v2"
ANCHORS = [1.0, 0.9, 0.75, 0.5, 0.25, 0.0]
PP_STARTS = [4.0, 32.0]
CAT_NAMES = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}


def coverage_at_k(retrieved, gold, k: int) -> float:
    """|retrieved & gold| / min(k, |gold|) -- cap-aware, so a question with 19 gold
    turns is not scored against an unreachable denominator at k=5."""
    g = set(gold)
    return len(set(retrieved[:k]) & g) / min(k, len(g))


def encode_conv(conv, encoder):
    memb = encoder.encode([m.text for m in conv.memories], batch_size=256,
                          show_progress_bar=False)
    for m, e in zip(conv.memories, memb):
        m.embedding = np.asarray(e, dtype=np.float64)
    qemb = encoder.encode([q.question for q in conv.queries], batch_size=256,
                          show_progress_bar=False)
    return np.asarray(qemb, dtype=np.float64)


def reduction_check(convs, qembs) -> bool:
    """anchor=1.0 must reproduce cosine top-k exactly. If this fails, nothing else counts."""
    print("\n[reduction] anchor=1.0 must equal exact cosine top-5")
    conv, qe = convs[0], qembs[0]
    dense = DenseTopK(); dense.index(conv.memories)
    pinned = AdaptiveBetaEnergyMemory(n_steps=8, anchor=1.0); pinned.index(conv.memories)
    agree = sum(
        dense.retrieve(q, k=K).ids == pinned.retrieve(q, k=K).ids
        for q in qe[:200]
    )
    ok = agree == len(qe[:200])
    print(f"    identical top-5 orderings: {agree}/{len(qe[:200])}  {'OK' if ok else 'FAIL'}")
    return ok


def main() -> int:
    if not DATA.exists():
        print(f"missing {DATA}")
        return 1

    from sentence_transformers import SentenceTransformer
    print(f"loading encoder {ENCODER_NAME} ...")
    encoder = SentenceTransformer(ENCODER_NAME)

    convs = load_locomo(DATA)
    print(f"loaded {len(convs)} conversations, {sum(len(c.memories) for c in convs)} turns, "
          f"{sum(len(c.queries) for c in convs)} retrievable queries")

    t0 = time.perf_counter()
    qembs = [encode_conv(c, encoder) for c in convs]
    print(f"encoded corpus in {time.perf_counter() - t0:.1f}s")

    if not reduction_check(convs, qembs):
        print("    => reduction failed; the implementation is wrong. Stopping.")
        return 1

    # ---------------------------------------------------------------- sweep
    print("\n" + "=" * 78)
    print("ANCHOR x PP_START SWEEP -- does ANY amount of descent beat pure cosine?")
    print("anchor=1.0 IS cosine. Lower anchor = more descent, less query.")
    print("=" * 78)

    cov = defaultdict(list)
    rec = defaultdict(list)
    cats = defaultdict(lambda: defaultdict(list))

    for conv, qe in zip(convs, qembs):
        fixed = EnergyMemory(n_steps=8, beta=32.0); fixed.index(conv.memories)
        models = {}
        for pp in PP_STARTS:
            for a in ANCHORS:
                m = AdaptiveBetaEnergyMemory(n_steps=8, pp_start=pp, pp_end=1.2, anchor=a)
                m.index(conv.memories)
                models[(pp, a)] = m

        for q, qv in zip(conv.queries, qe):
            r = fixed.retrieve(qv, k=K)
            cov["energy fixed b=32 (no anchor)"].append(coverage_at_k(r.ids, q.gold_ids, K))
            rec["energy fixed b=32 (no anchor)"].append(recall_at_k(r.ids, q.gold_ids, K))
            for key, m in models.items():
                res = m.retrieve(qv, k=K)
                name = f"pp0={key[0]:<4g} anchor={key[1]:.2f}"
                c = coverage_at_k(res.ids, q.gold_ids, K)
                cov[name].append(c)
                rec[name].append(recall_at_k(res.ids, q.gold_ids, K))
                cats[name][q.category].append(c)

    print(f"\n  {'condition':<28}{'Coverage@5':>22}{'Recall@5':>12}")
    print("  " + "-" * 62)
    order = sorted(cov, key=lambda n: -np.mean(cov[n]))
    for name in order:
        m, lo, hi = bootstrap_ci(cov[name])
        tag = "  <-- = cosine" if "anchor=1.00" in name else ""
        print(f"  {name:<28}{m:>8.3f} [{lo:.3f},{hi:.3f}]{np.mean(rec[name]):>12.3f}{tag}")

    best = order[0]
    print(f"\n  best configuration: {best}")
    cosine_names = [n for n in cov if "anchor=1.00" in n]
    cos_mean = np.mean(cov[cosine_names[0]])
    print(f"  cosine (anchor=1.00) Coverage@5 = {cos_mean:.3f}")
    if "anchor=1.00" in best:
        print("  => no amount of descent improves on the query itself. Descent is a cost.")
    else:
        delta = np.mean(cov[best]) - cos_mean
        print(f"  => descent helps: +{delta:.3f} Coverage@5 over cosine.")

    # per-category for the best non-cosine config vs cosine
    print(f"\n  per-category Coverage@5 (best descent config vs cosine)")
    nonc = [n for n in order if "anchor=1.00" not in n][0]
    allcats = sorted({c for n in (nonc, cosine_names[0]) for c in cats[n]})
    print(f"    {'condition':<28}" + "".join(f"{CAT_NAMES.get(c, c)[:11]:>13}" for c in allcats))
    for name in (cosine_names[0], nonc):
        print(f"    {name:<28}" + "".join(
            f"{np.mean(cats[name][c]):>13.3f}" if cats[name][c] else f"{'-':>13}"
            for c in allcats))

    capacity(convs, qembs)
    return 0


def capacity(convs, qembs):
    """Corrected capacity sweep. The previous version (exp_locomo_adaptive.py part B) was
    invalid: unnamespaced dia_ids let distractor turns from other conversations satisfy
    gold ids, so scores were inflated by exactly the distractors being added."""
    print("\n" + "=" * 78)
    print("CAPACITY (corrected ids) -- fixed queries, growing distractor bank")
    print("=" * 78)

    target, qe = convs[0], qembs[0]
    others = [m for c in convs[1:] for m in c.memories]
    gold_ids = {g for q in target.queries for g in q.gold_ids}
    leak = sum(1 for m in others if m.id in gold_ids)
    print(f"\n  gold-id collisions among {len(others)} distractors: {leak}  "
          f"({'clean' if leak == 0 else 'STILL LEAKING'})")

    rows = []
    for extra in [0, 500, 1500, 3000, len(others)]:
        memories = list(target.memories) + others[:extra]
        n = len(memories)
        dense = DenseTopK(); dense.index(memories)
        fixed = EnergyMemory(n_steps=8, beta=32.0); fixed.index(memories)
        desc = AdaptiveBetaEnergyMemory(n_steps=8, pp_start=4.0, anchor=0.0)
        desc.index(memories)

        cov = {"cosine": [], "energy fixed b=32": [], "adaptive pp0=4 anchor=0": []}
        betas = []
        for q, qv in zip(target.queries, qe):
            cov["cosine"].append(coverage_at_k(dense.retrieve(qv, k=K).ids, q.gold_ids, K))
            cov["energy fixed b=32"].append(
                coverage_at_k(fixed.retrieve(qv, k=K).ids, q.gold_ids, K))
            r = desc.retrieve(qv, k=K)
            cov["adaptive pp0=4 anchor=0"].append(coverage_at_k(r.ids, q.gold_ids, K))
            betas.append(desc.last_betas[-1])

        row = {"n": n, "beta": float(np.mean(betas))}
        for name in cov:
            row[name] = bootstrap_ci(cov[name])
        rows.append(row)
        print(f"\n  N={n:>5}  (adaptive solved mean final beta = {row['beta']:.1f})")
        for name in cov:
            m, lo, hi = row[name]
            print(f"    {name:<26} Coverage@5 = {m:.3f} [{lo:.3f},{hi:.3f}]")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ns = [r["n"] for r in rows]
    for name in ("cosine", "energy fixed b=32", "adaptive pp0=4 anchor=0"):
        ax.plot(ns, [r[name][0] for r in rows], marker="o", label=name)
        ax.fill_between(ns, [r[name][1] for r in rows], [r[name][2] for r in rows], alpha=0.13)
    ax.set_xlabel("memory bank size (turns)")
    ax.set_ylabel("Coverage@5")
    ax.set_title("LoCoMo capacity, corrected ids\n(fixed queries, growing distractor bank)")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 0.6)
    fig.tight_layout()
    out = Path(__file__).resolve().parents[1] / "experiments" / "results" / "locomo_capacity_corrected.png"
    fig.savefig(out, dpi=150)
    print(f"\n  plot saved to {out}")


if __name__ == "__main__":
    raise SystemExit(main())
