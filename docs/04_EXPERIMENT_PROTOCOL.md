# Experiment Protocol

Pre-registered. Fill in results; do not change the protocol after seeing them. If you must
change it, log the change and the reason in `experiments/PROTOCOL_CHANGES.md` — reviewers
respect a logged deviation and punish a silent one.

## 0. Fixed experimental substrate

| Element | Choice | Rationale |
|---|---|---|
| Encoder (primary) | `sentence-transformers/all-mpnet-base-v2` | Strong, standard, 768-d, reproducible offline |
| Encoder (robustness) | `BAAI/bge-large-en-v1.5`, `intfloat/e5-large-v2` | Show results are not encoder-specific |
| Normalization | L2, always | Required for the reduction result |
| Seeds | `[0, 1, 2, 3, 4]` | 5 seeds minimum on every headline number |
| CIs | Bootstrap, 10 000 resamples, 95% | Report as `mean [lo, hi]` |
| Hardware | Log GPU/CPU model in every result file | Latency claims are meaningless otherwise |

**Matched-encoder rule:** every method in every table uses the same frozen encoder and the
same corpus. No exceptions. Encoder ablations are a separate table.

## 1. DegradeBench (our released artifact)

Corpus: 10 k synthetic episodic memories (agent-interaction style) + LoCoMo/LongMemEval
sessions. Each memory is a short natural-language episode with `{time, source, salience}`.

Five corruption families × five severity levels (0 = clean → 4 = severe):

| Family | Mechanism | Severity knob |
|---|---|---|
| **D1 Masking** | Drop tokens from the query | 0/15/30/50/70 % dropped |
| **D2 Paraphrase** | LLM rewrite at increasing distance | 5 levels, human-spot-checked |
| **D3 Embedding noise** | `q ← normalize(q + σ·ε)`, `ε~N(0,I)` | σ ∈ {0, .05, .1, .2, .4} |
| **D4 Distractors** | Inject k near-duplicate memories differing in one key fact | k ∈ {0, 1, 5, 20, 100} |
| **D5 Composite cue** | Query references 2–4 memories jointly; correct answer requires all | 2/3/4 targets |

**D5 is the discriminating test for G1** (multi-step descent) — a single top-k lookup has
no mechanism for combining evidence across stored items.
**D4 is the discriminating test for G2** (strength bias) — usage history is the only signal
that separates near-duplicates.

Ship the generator (`src/energymem/eval/degradebench.py`) and the exact generated split.

## 2. Experiments

### E1 — Multi-step descent (RQ1)
- Sweep `T ∈ {1, 2, 4, 8, 16, 32}` × `β ∈ {1, 4, 16, 64, 256}` × β-schedules {const, linear anneal, cosine anneal}
- On D1–D5, all severities
- **Report:** Recall@1/@5, MRR vs severity, one curve per T. Plus steps-to-converge and non-convergence rate.
- **Primary figure:** corruption-severity (x) vs Recall@1 (y), one line per method, error bands.

### E2 — Strength-modulated basins (RQ2)
- `b_i = w_u·log(1+usage_i) + w_r·exp(-Δt_i/τ) + w_s·surprise_i`
- Grid over `(w_u, w_r, w_s, τ)`, plus knockouts: each weight zeroed individually
- **Critical baseline:** cosine top-k with the *same* signals as a linear score prior, and
  cosine top-k + cross-encoder reranker. If either matches EnergyMem, C2 is not a contribution — say so.
- Simulate usage from a Zipf access pattern over 30 simulated days; hold out the final week

### E3 — Learned associative routing (RQ3) ⚠️ **RUN THIS FIRST, WEEK 2**
- Construct pairs `(a, b)` that co-occur temporally but have **cosine(a,b) < 0.2**
- Train `R` on co-occurrence only (contrastive / predictive objective; never on cosine)
- Query with `a`, target is `b`
- **Report:** Recall@1 for cosine top-k (expected: near chance), EnergyMem `R=I` (near chance),
  EnergyMem `R` learned (the whole question), and a graph-retrieval baseline (HippoRAG-style)
- Also report the *cost*: does learned `R` degrade normal same-topic retrieval? Sweep `η`.
- **Gate:** if learned `R` does not clear both cosine and `R=I` by a wide margin, see `06_RISK_REGISTER.md` R-3.

### E4 — Consolidation (RQ4)
- Merge memories whose basins overlap above a threshold into prototypes; keep provenance
- **Report:** compression ratio (x) vs Recall@1 on both specific-episode and general-knowledge
  queries (y). The expected and interesting result is a **crossover**: consolidation helps
  generalization queries and hurts specific-episode queries. Finding and characterizing that
  crossover point is a better contribution than a monotone win.
- Also: interference — recall on old memories after writing 1k/10k new ones

### E5 — Systems (RQ5) — reported in every table, never separately
- Latency/query (p50, p95), steps-to-converge, index build time, RAM footprint
- Scaling: N ∈ {10³, 10⁴, 10⁵, 10⁶}, flat vs HNSW pre-filter (M = 256 shortlist)
- **Explicitly report the accuracy/latency Pareto frontier.** If EnergyMem is 8× slower for
  +2 points Recall@1, the paper says exactly that, in the abstract.

### E6 — Real agent benchmarks (RQ6)
- **LongMemEval** — report the five ability categories separately (single-session,
  multi-session, temporal reasoning, knowledge update, abstention)
- **LoCoMo** — very-long conversational memory
- Baselines: dense top-k, BM25, hybrid, and ≥2 published agent-memory systems
  (MemoryBank / A-MEM / HippoRAG / Mem0 — whichever have runnable code)
- Downstream QA accuracy with a fixed reader LLM; log the exact model + version + temperature

## 3. Ablation table (must appear in the paper)

| Variant | T | b | R | Consolidation | Purpose |
|---|---|---|---|---|---|
| **Cosine top-k** | 1 | 0 | I | ✗ | Baseline = provable special case |
| +Multi-step | 8 | 0 | I | ✗ | Isolates G1 |
| +Strength | 8 | ✓ | I | ✗ | Isolates G2 |
| +Routing | 8 | 0 | ✓ | ✗ | Isolates G3 |
| +Consolidation | 8 | 0 | I | ✓ | Isolates C4 |
| **EnergyMem (full)** | 8 | ✓ | ✓ | ✓ | Combined |
| Full − routing | 8 | ✓ | I | ✓ | Knockout |
| Full − strength | 8 | 0 | ✓ | ✓ | Knockout |

Both a build-up and a knock-out column. Reviewers ask for whichever you omitted.

## 4. Sanity tests that must pass before any result is believed

Live in `tests/`, run in CI:

1. **Reduction test** — `T=1, β=1e6, b=0, R=I` reproduces FAISS top-1 exactly on 1 000 queries.
   *If this fails, every downstream number is meaningless.*
2. **Energy monotonicity** — `E(ξ_{t+1}) ≤ E(ξ_t)` for every step of every query
3. **Capacity test** — store N random orthogonal patterns, verify perfect recall; find where it breaks
4. **Determinism** — same seed ⇒ bitwise-identical results
5. **No leakage** — DegradeBench queries never appear in any training split for `R`

## 5. Results hygiene

- Every run writes `experiments/results/<exp>/<timestamp>_<git-sha>.json` with full config
- Tables and figures are **generated from those files**, never hand-typed
- `make paper` regenerates every number in the PDF from disk
- A number that cannot be regenerated does not go in the paper
