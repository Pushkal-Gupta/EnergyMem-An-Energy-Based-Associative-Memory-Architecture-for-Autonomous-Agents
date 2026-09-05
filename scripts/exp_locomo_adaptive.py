"""SUPERSEDED by exp_locomo_anchored.py -- do not cite results from this file.

Part B (capacity) is INVALID: it pooled memory banks across conversations without
namespacing dia_ids, which are unique only within a conversation. 132 of the 134 gold ids
for conversation 0 also occur in other conversations, so distractor turns satisfied gold
ids and scored as hits -- inflating results in proportion to the distractors being added,
which is exactly the axis under test. Its "adaptive beta improves as the bank grows"
conclusion is an artifact; the corrected sweep shows the opposite.

Part A (per-category retrieval) is unaffected -- each conversation is indexed alone, so
ids are unique within the bank -- but it is confounded: pp_start=32 forces step 1 to
average ~32 memories (the broad-start annealing F2 flagged as harmful), and descent runs
with no query anchor. exp_locomo_anchored.py replaces both with a controlled sweep.

Retained as the record of what was run and why it was wrong.

---

LoCoMo experiment: does query-adaptive, step-dynamic beta beat cosine and fixed beta?

PART A -- retrieval quality per LoCoMo category, one memory bank per conversation.
PART B -- capacity: hold the queries fixed, grow the memory bank with distractor turns
          pooled from other conversations (a realistic "the agent keeps accumulating
          memories" setting), and watch whether each method holds up. Also reports the
          beta the adaptive controller actually solves for at each N -- the scale-free
          claim is that it rises with N on its own.

Conditions:
    cosine                 -- DenseTopK, the baseline that has beaten us so far
    energy fixed beta=32   -- EnergyMemory, the configuration used in the image study
    adaptive beta          -- AdaptiveBetaEnergyMemory (perplexity-targeted)
    adaptive beta + peel   -- same, with residual peeling (multi-target readout)

Metrics (k=5): hit-rate Recall@5, Coverage@5 = |retrieved & gold| / |gold|, nDCG@5.
Coverage is the one that matters for category 1 (avg ~3.1 gold turns per question).

Run:
    python scripts/exp_locomo_adaptive.py
"""

from __future__ import annotations

import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from energymem.base import Memory, l2_normalize                      # noqa: E402
from energymem.baselines.dense import DenseTopK                      # noqa: E402
from energymem.memory.energy import EnergyMemory                     # noqa: E402
from energymem.memory.adaptive import AdaptiveBetaEnergyMemory       # noqa: E402
from energymem.eval.locomo import load_locomo                        # noqa: E402
from energymem.eval.metrics import recall_at_k, ndcg_at_k, bootstrap_ci  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data" / "locomo10.json"
K = 5
CAT_NAMES = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}
ENCODER_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def coverage_at_k(retrieved, gold, k: int) -> float:
    g = set(gold)
    return len(set(retrieved[:k]) & g) / len(g)


def make_conditions(memories):
    dense = DenseTopK()
    dense.index(memories)
    fixed = EnergyMemory(n_steps=8, beta=32.0)
    fixed.index(memories)
    adapt = AdaptiveBetaEnergyMemory(n_steps=8, pp_start=32.0, pp_end=1.2)
    adapt.index(memories)
    return dense, fixed, adapt


def part_a(convs, encoder):
    print("\n" + "=" * 78)
    print("PART A -- LoCoMo retrieval quality (one memory bank per conversation)")
    print("=" * 78)

    # metric -> condition -> category -> list of per-query scores
    store = {m: defaultdict(lambda: defaultdict(list)) for m in ("recall", "coverage", "ndcg")}
    latency = defaultdict(list)
    n_queries = 0

    for ci, conv in enumerate(convs):
        mem_emb = encoder.encode(
            [m.text for m in conv.memories], batch_size=256, show_progress_bar=False
        )
        for m, e in zip(conv.memories, mem_emb):
            m.embedding = np.asarray(e, dtype=np.float64)
        q_emb = encoder.encode(
            [q.question for q in conv.queries], batch_size=256, show_progress_bar=False
        )

        dense, fixed, adapt = make_conditions(conv.memories)
        n_queries += len(conv.queries)

        for q, qe in zip(conv.queries, q_emb):
            qe = np.asarray(qe, dtype=np.float64)
            runs = {
                "cosine": dense.retrieve(qe, k=K),
                "energy fixed beta=32": fixed.retrieve(qe, k=K),
                "adaptive beta": adapt.retrieve(qe, k=K),
                "adaptive beta + peel": adapt.retrieve_multi(qe, k=K),
            }
            for name, res in runs.items():
                latency[name].append(res.latency_ms)
                store["recall"][name][q.category].append(recall_at_k(res.ids, q.gold_ids, K))
                store["coverage"][name][q.category].append(coverage_at_k(res.ids, q.gold_ids, K))
                store["ndcg"][name][q.category].append(ndcg_at_k(res.ids, q.gold_ids, K))

        print(f"  conversation {ci + 1}/{len(convs)}: "
              f"{len(conv.memories)} turns, {len(conv.queries)} queries")

    conditions = ["cosine", "energy fixed beta=32", "adaptive beta", "adaptive beta + peel"]
    cats = sorted({c for name in conditions for c in store["recall"][name]})

    for metric in ("recall", "coverage", "ndcg"):
        label = {"recall": f"Recall@{K} (hit)", "coverage": f"Coverage@{K}",
                 "ndcg": f"nDCG@{K}"}[metric]
        print(f"\n  --- {label} ---")
        head = f"    {'condition':<24}" + "".join(
            f"{CAT_NAMES.get(c, c)[:11]:>13}" for c in cats
        ) + f"{'OVERALL':>13}"
        print(head)
        for name in conditions:
            cells = []
            allv = []
            for c in cats:
                v = store[metric][name][c]
                allv.extend(v)
                cells.append(f"{np.mean(v):>13.3f}" if v else f"{'-':>13}")
            m, lo, hi = bootstrap_ci(allv)
            print(f"    {name:<24}" + "".join(cells) + f"{m:>13.3f}")
        print(f"    (per-category n: " + ", ".join(
            f"{CAT_NAMES.get(c, c)}={len(store[metric]['cosine'][c])}" for c in cats) + ")")

    print(f"\n  --- overall Coverage@{K} with 95% bootstrap CI ---")
    for name in conditions:
        allv = [x for c in cats for x in store["coverage"][name][c]]
        m, lo, hi = bootstrap_ci(allv)
        print(f"    {name:<24} {m:.3f} [{lo:.3f}, {hi:.3f}]   "
              f"latency {np.mean(latency[name]):.2f} ms/query")

    print(f"\n  total queries evaluated: {n_queries}")
    return store


def part_b(convs, encoder):
    print("\n" + "=" * 78)
    print("PART B -- capacity: fixed queries, growing distractor memory bank")
    print("=" * 78)

    target = convs[0]
    others = [m for c in convs[1:] for m in c.memories]

    all_texts = [m.text for m in target.memories] + [m.text for m in others]
    emb = encoder.encode(all_texts, batch_size=256, show_progress_bar=False)
    emb = np.asarray(emb, dtype=np.float64)
    for m, e in zip(target.memories, emb[: len(target.memories)]):
        m.embedding = e
    for m, e in zip(others, emb[len(target.memories):]):
        m.embedding = e

    q_emb = np.asarray(
        encoder.encode([q.question for q in target.queries], batch_size=256,
                       show_progress_bar=False),
        dtype=np.float64,
    )

    base_n = len(target.memories)
    extra_steps = [0, 500, 1500, 3000, len(others)]
    print(f"\n  gold turns always present; base bank = {base_n} turns "
          f"({len(target.queries)} queries), distractors added from other conversations")

    rows = []
    for extra in extra_steps:
        memories = list(target.memories) + others[:extra]
        n = len(memories)
        dense, fixed, adapt = make_conditions(memories)

        cov = {"cosine": [], "energy fixed beta=32": [], "adaptive beta": []}
        solved_betas = []
        for q, qe in zip(target.queries, q_emb):
            cov["cosine"].append(coverage_at_k(dense.retrieve(qe, k=K).ids, q.gold_ids, K))
            cov["energy fixed beta=32"].append(
                coverage_at_k(fixed.retrieve(qe, k=K).ids, q.gold_ids, K))
            r = adapt.retrieve(qe, k=K)
            cov["adaptive beta"].append(coverage_at_k(r.ids, q.gold_ids, K))
            solved_betas.append(adapt.last_betas[-1])

        row = {"n": n, "mean_final_beta": float(np.mean(solved_betas))}
        for name in cov:
            row[name] = bootstrap_ci(cov[name])
        rows.append(row)
        print(f"\n  N={n:>5}   (adaptive controller solved mean final beta = "
              f"{row['mean_final_beta']:.1f})")
        for name in cov:
            m, lo, hi = row[name]
            print(f"    {name:<24} Coverage@{K} = {m:.3f} [{lo:.3f}, {hi:.3f}]")

    return rows


def plot_part_b(rows, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    ns = [r["n"] for r in rows]
    for name in ("cosine", "energy fixed beta=32", "adaptive beta"):
        means = [r[name][0] for r in rows]
        los = [r[name][1] for r in rows]
        his = [r[name][2] for r in rows]
        ax.plot(ns, means, marker="o", label=name)
        ax.fill_between(ns, los, his, alpha=0.13)
    ax.set_xlabel("memory bank size (turns)")
    ax.set_ylabel(f"Coverage@{K}")
    ax.set_title("LoCoMo capacity: fixed queries,\ngrowing distractor bank")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.0)

    ax2.plot(ns, [r["mean_final_beta"] for r in rows], marker="s", color="tab:green")
    ax2.axhline(32.0, ls="--", color="tab:orange", label="fixed beta=32")
    ax2.set_xlabel("memory bank size (turns)")
    ax2.set_ylabel("beta solved by controller (final step)")
    # NOTE: the pre-registered prediction was that beta would RISE with N (scale-free
    # targeting). It does not -- it drifts slightly down. Behaviour is still robust to
    # N (left panel), but not via the mechanism predicted. Title states what happened.
    ax2.set_title("Solved beta drifts slightly DOWN as the bank grows\n"
                  "(predicted rise not observed)")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"\nplot saved to {out_path}")


def main() -> int:
    if not DATA.exists():
        print(f"missing {DATA}")
        return 1

    from sentence_transformers import SentenceTransformer

    print(f"loading encoder {ENCODER_NAME} ...")
    t0 = time.perf_counter()
    encoder = SentenceTransformer(ENCODER_NAME)
    print(f"  ready in {time.perf_counter() - t0:.1f}s")

    convs = load_locomo(DATA)
    print(f"loaded {len(convs)} conversations, "
          f"{sum(len(c.memories) for c in convs)} turns, "
          f"{sum(len(c.queries) for c in convs)} retrievable queries")

    part_a(convs, encoder)
    rows = part_b(convs, encoder)

    out_dir = Path(__file__).resolve().parents[1] / "experiments" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_part_b(rows, out_dir / "locomo_capacity_adaptive.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
