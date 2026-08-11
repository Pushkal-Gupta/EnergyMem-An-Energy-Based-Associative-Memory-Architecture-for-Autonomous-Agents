# Task Allocation

Three people, six weeks. The failure mode for a team this size is **everyone waiting on
the lead**. The split below is designed so the three workstreams touch a shared interface
(`Retriever`) and otherwise never block each other.

## The contract that makes parallel work possible

Everything implements one interface (`src/energymem/base.py`):

```python
class Retriever(Protocol):
    def index(self, memories: list[Memory]) -> None: ...
    def retrieve(self, query: str | np.ndarray, k: int = 5) -> RetrievalResult: ...
```

`RetrievalResult` carries `ids`, `scores`, `latency_ms`, `n_steps`, `converged`,
`final_energy`, `energy_margin`. Baselines fill the energy fields with `None`.

**Agree on this file in the first standup and freeze it.** Every downstream disagreement
in a project like this traces back to an interface that kept moving.

---

## Pushkal (23BCT0253) — Energy core + paper

**Owns:** `src/energymem/memory/`, `paper/`

- Energy functional, CCCP update, β annealing schedule
- Convergence instrumentation: step count, energy margin to runner-up attractor,
  blend/metastable-state detection, non-convergence rate
- Learned routing operator `R` (C3) — the highest-risk, highest-value component
- The reduction proof (C1) and its empirical verification test
- Lead author: Intro, Method, Results, Conclusion

**Why you:** you have the prior IEEE Trans submission and the most reading in this area.
The routing component is the one that decides whether the paper exists.

**Watch out for:** you are on the critical path for both the riskiest code and the writing.
Hand E4 (consolidation) to Anshveer entirely and do not take it back.

## Rajit (23BCE2371) — Baselines, benchmarks, literature

**Owns:** `src/energymem/baselines/`, `DegradeBench` generator, `paper/sections/related_work.tex`

- Dense top-k (FAISS flat + HNSW), BM25, hybrid RRF, cosine + cross-encoder reranker.
  **Make these strong.** A weak baseline is the most common reason a paper like this gets
  rejected — reviewers assume you tuned yours and not theirs. Tune theirs.
- DegradeBench: 5 corruption families × 5 severities, with generation scripts committed
- LongMemEval + LoCoMo integration
- **Citation audit** (`07_CITATION_AUDIT.md`) — you sourced the Gemini/Genspark material,
  so you own verifying it. Every arXiv ID resolved to a real paper or struck from the bib.
- Related Work section

**Why you:** you brought the 30-paper review, so you have the most context on the lit.

**Watch out for:** the audit is boring and load-bearing. A single hallucinated citation
found by a reviewer costs the paper its credibility on everything else.

## Anshveer (23BCE0703) — Lifecycle, systems, figures

**Owns:** `src/energymem/memory/lifecycle.py`, `src/energymem/eval/`, `paper/figures/`

- Memory strength `b_i`: reinforcement, exponential decay, surprise-gated writes
- Consolidation: basin merging / prototype distillation, compression-vs-recall curve (E4)
- Evaluation harness: metrics, seeding, bootstrap CIs, the results-table generator
- Systems metrics: latency/query, throughput, footprint, scaling with N
- **All figures.** Matplotlib, one shared style file, colourblind-safe, vector PDF output

**Why you:** you were the one who flagged "brain word bold claim lag rha hai" during the
title debate — that instinct for overclaiming is exactly what the measurement and figures
workstream needs. You are also the natural person to own the honesty of the numbers.

**Watch out for:** figures always get left to the last week and always take longer than
anyone plans. Build the plotting pipeline in Week 1 against fake data so that in Week 4 it
is a one-command regeneration.

---

## Shared / rotating

| Item | Owner | Cadence |
|---|---|---|
| Attendance sheet + guide meetings | each individually | ≥3×/week, non-negotiable |
| PR review | round-robin, 24h SLA | every PR |
| Standup notes → `docs/standups/` | rotates weekly | Mon + Thu |
| Review-2 / Review-3 decks | Pushkal drafts, all present | per milestone |

## Escalation rule

If any owner is blocked for **>24 h**, it goes in the group chat with the word `BLOCKED`
and gets reassigned at the next standup. The chat history shows a repeated pattern of
meetings being scheduled and quietly not happening. The fix is not more meetings — it is
that a blocked task is visible in writing within a day.
