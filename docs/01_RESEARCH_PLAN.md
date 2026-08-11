# Research Plan — the science

This is the technical spine. If you read one doc, read this one.

---

## 1. The problem with the current framing (and how we fix it)

The Review-1 decks say, roughly: *"vector DBs use cosine similarity; we use energy
minimization instead; energy minimization is better because attractor dynamics give
pattern completion."*

A competent reviewer will make the following objection within ninety seconds, and it is
correct:

> The Modern Hopfield update (Ramsauer et al., 2020) is
> `ξ_{t+1} = X softmax(β Xᵀ ξ_t)`.
> For L2-normalized patterns this is *exactly* softmax-weighted cosine retrieval —
> i.e. attention. A single update step with large β returns the top-1 cosine neighbour.
> The authors have therefore reimplemented their own baseline and called it a new
> architecture.

**Any version of this paper that does not confront that objection head-on will be
rejected.** Our response is not to dodge it — it is to make it the paper's opening move.

### The reframe

> Energy-based associative retrieval **strictly generalizes** dense top-k retrieval.
> Top-k cosine search is the `T=1, β→∞, b=0, R=I` corner of our parameter space.
> The research question is not "is energy better than cosine" — it is
> **"which departures from that corner actually pay for themselves?"**

This reframe does three things at once: it disarms the reviewer's strongest attack,
it makes the baseline comparison honest by construction (baseline = special case, same
code path, same encoder), and it turns the paper into a *characterization* study, which
is far more defensible for a first-time team than a "we beat SOTA" study.

## 2. Formal setup

Memories `m_i ∈ ℝ^d`, i = 1..N, L2-normalized, stacked as `X ∈ ℝ^{d×N}`.
Query cue `q ∈ ℝ^d`. Retrieval state `ξ ∈ ℝ^d`, initialized `ξ_0 = q`.

**Energy functional:**

```
E(ξ) = -lse(β, Xᵀξ + b) + ½‖ξ‖² + const
```

where `lse(β, z) = β⁻¹ log Σ_i exp(β z_i)` and `b ∈ ℝ^N` is the **memory-strength bias**
(C2). Concave–convex splitting gives the guaranteed-descent update:

```
ξ_{t+1} = X · softmax(β (Xᵀ ξ_t + b))          (Hopfield / CCCP step)
ξ_{t+1} ← (1-η) ξ_{t+1} + η · R ξ_{t+1}        (associative routing step, C3)
ξ_{t+1} ← ξ_{t+1} / ‖ξ_{t+1}‖
```

`R ∈ ℝ^{d×d}` (or a sparse `N×N` memory-graph operator) is the **learned transition
operator** — trained on temporal co-occurrence, not on embedding similarity.

### The three generalization axes

| Axis | Knob | Baseline value | Why it might help | Can top-k mimic it? |
|---|---|---|---|---|
| **G1 Multi-step descent** | `T` steps, β schedule | `T=1, β→∞` | Composite / partial cues resolve across several stored items before committing | **No** — top-k has no re-query |
| **G2 Strength-modulated basins** | `b_i` from usage/recency/surprise | `b=0` | Unifies reinforce + decay + forget in the objective | **Partially** — a static score prior mimics one step, not the dynamics. *Must be an ablation.* |
| **G3 Learned routing** | `R ≠ I` | `R=I` | Links memories by co-occurrence, so cross-associative recall works when cosine is near zero | **No** — provably outside the span of any reweighting of `Xᵀq` |

**Read the last column carefully.** G3 is where the mathematical departure lives.

> ⚠️ **Revised 2026-08-11 after the citation audit.** G3 as *"association beyond
> similarity"* is **not novel** — PAM (Dury, arXiv:2602.11322) already demonstrates it
> with a JEPA-style predictor, 97 % association P@1 against cosine at chance, with a
> runnable MIT-licensed benchmark. Likewise C2 is adjacent to DMHN (arXiv:2506.01303,
> context-modulated attractor geometry) and C4 is adjacent to HeLa-Mem (ACL 2026,
> Hebbian co-activation + episodic→semantic distillation on LoCoMo).
>
> **The novelty is no longer any single axis. It is the unification** — that G1, G2 and G3
> are terms in *one* energy functional with *one* descent procedure, rather than three
> separate subsystems. PAM's association lives outside its retrieval operator; HeLa-Mem's
> consolidation lives outside its graph traversal; DMHN has no lifecycle at all. In
> EnergyMem, association, strength and consolidation all reshape the same landscape that
> retrieval descends.
>
> This is a **harder** claim to defend and it obligates us to run all three as baselines.
> It is also the only framing that survives contact with the literature. See
> `07_CITATION_AUDIT.md` §"Competitive landscape".

If the experiments show the unified formulation gives no advantage over the three separate
systems, the paper becomes "energy-based unification of agent memory does not pay for
itself, and here is exactly where it breaks" — still publishable, still honest, but you
need to know that by ~mid-September and pivot the framing deliberately.

## 3. Research questions → hypotheses → experiments

| RQ | Hypothesis | Experiment | Kill criterion |
|---|---|---|---|
| **RQ1** | Does multi-step descent beat one-step on degraded cues? | E1 on DegradeBench, sweep `T ∈ {1,2,4,8,16}` | No gain at any corruption level ⇒ drop G1, report as negative |
| **RQ2** | Does strength-modulated basin depth beat a static relevance prior? | E2, ablate `b` vs. score-prior reranker | Reranker matches it ⇒ C2 demoted to engineering note |
| **RQ3** | Does routing *inside the energy functional* match or beat PAM's *external* predictor — and does folding it in additionally buy pattern completion and lifecycle that PAM cannot express? | E3 on low-cosine temporal pairs, **with PAM as the primary baseline** (its benchmark is MIT-licensed and runnable) | Losing to PAM with no compensating advantage ⇒ **central claim fails**; pivot to characterization |
| **RQ3b** | Does the unified energy objective beat the three specialist systems (PAM, HeLa-Mem, DMHN) run separately? | E3b, head-to-head on LoCoMo with matched encoder | No advantage ⇒ report as a negative unification result |
| **RQ4** | Does consolidation reduce footprint without hurting recall? | E4, compression ratio vs. Recall@1 curve | — (this one is descriptive, not pass/fail) |
| **RQ5** | What does it all cost? | E5, latency/step/footprint vs. N | — (always reported) |
| **RQ6** | Does any of this survive on a real agent benchmark? | E6, LongMemEval + LoCoMo | Losing here means the gains were synthetic-only — say so |

**RQ3 is the load-bearing question. Run E3 first (Week 2), not last.** If routing does not
work, you want six weeks to reframe, not six days.

## 4. Why this is Tier-1 material (and where it is fragile)

**Strengths:**
- The reduction theorem (C1) is a genuine, checkable contribution — nobody in the
  agent-memory literature states it cleanly, and it explains *why* the two literatures
  have failed to talk to each other. **This is now the strongest single contribution.**
- The unification claim is real: PAM, HeLa-Mem and DMHN each isolate one mechanism in an
  incompatible formalism. Expressing all three in one energy objective is a legitimate
  architectural contribution *if* it demonstrably buys something.
- DegradeBench is a durable artifact — it gets cited even if EnergyMem loses.
- Matched-encoder protocol is stricter than most agent-memory papers manage. Running PAM,
  HeLa-Mem and DMHN under one encoder and one corpus is, by itself, a service to the field
  that none of those three papers performed.

**Fragilities (be honest with yourselves):**
- **Crowded field, and it got more crowded in 2026.** PAM, HeLa-Mem (ACL 2026) and DMHN all
  landed in the last six months. The window for "energy/associative memory for agents" as a
  novel framing is closing. This argues for the ICLR 2027 September deadline over a slow
  journal path — **speed now has scientific value, not just administrative value.**
- **Baseline burden.** The unification claim obligates three non-trivial reimplementations.
  PAM is MIT-licensed and runnable (good). HeLa-Mem says code is on GitHub (verify in Week 1).
  DMHN may need reimplementation — if so, **descope it to a cited comparison**, do not let it
  eat two weeks.
- **Compute.** Iterative descent over large N. Mitigate with candidate pre-filtering:
  ANN top-M (M≈256) → energy descent within that shortlist. Report as a design choice,
  not an apology; it also makes the latency story survivable.
- **N is small in practice.** Agent memories are 10³–10⁵ items, where flat cosine is
  already fast and accurate. The win must come from *quality on hard queries*, never speed.
- **Hopfield capacity.** Ramsauer's exponential capacity bound assumes well-separated
  patterns. Real embeddings are not well-separated. Expect metastable blends. **Measure
  and report the blend/non-convergence rate — do not hide it.** A characterization of
  when embeddings are too correlated for attractor dynamics is itself a contribution.
- **Team is undergrad, timeline is 6 weeks.** Scope discipline is the single biggest
  predictor of success here.

## 5. What we deliberately are NOT claiming

Carried over from the Review-1 QA slide (slide 17) — these corrections were right, keep them:

- ✗ "guarantees reconstruction" → attractor convergence is not guaranteed for correlated patterns
- ✗ "eliminates catastrophic forgetting" → mitigation hypothesis, needs measurement
- ✗ "infinite planning horizon" → out of scope entirely now
- ✗ "fixed memory footprint" → depends on consolidation, must be measured
- ✗ any claim of biological equivalence → analogy only, and say the word "analogy" in text

## 6. Related-work positioning (the paragraph that must land)

> Agent-memory systems (MemoryBank, MemGPT, A-MEM, HippoRAG, MemOS) improve *what is
> stored and how it is organized*, while leaving the retrieval operator as embedding
> similarity. Associative-memory theory (Hopfield, Krotov & Hopfield, Ramsauer, Millidge)
> studies the *retrieval operator itself*, but almost exclusively on synthetic patterns
> or as an attention-equivalence result. A recent wave of work has begun to close the
> distance from three separate directions: PAM learns association from temporal
> co-occurrence but keeps it in a predictor external to retrieval; HeLa-Mem drives episodic
> memory by Hebbian co-activation and distils it to semantic memory, but over a graph whose
> traversal is still similarity-guided; DMHN modulates attractor geometry by context, but
> without any notion of memory lifetime. **Each isolates one mechanism in a formalism that
> cannot express the other two.** EnergyMem asks whether all three are terms in a single
> energy functional — so that association topology, memory strength and consolidation
> reshape the very landscape that retrieval descends — and measures, under one frozen
> encoder and one corpus, whether that unification buys anything the three specialist
> systems do not deliver separately.

That is the gap. It is narrow, it is defensible, and — critically — it is *stated in terms
of the 2026 literature rather than the 2020 literature*. Do not write a related-work
section that stops at Ramsauer; three reviewers will know PAM and HeLa-Mem by name.
