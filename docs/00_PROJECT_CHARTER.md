# EnergyMem — Project Charter

**Title (locked, VTOP-registered, no colon):**
> EnergyMem An Energy-Based Associative Memory Architecture for Autonomous Agents

**Course:** BCSE497J – Project I, Fall 2026–27, SCOPE, VIT Vellore
**Project ID:** 20113UG01
**Faculty Guide:** Dr. Thangaramya K

| Member | Reg. No. | Primary role |
|---|---|---|
| Pushkal Gupta | 23BCT0253 | Lead author, energy core + retrieval dynamics, paper |
| Rajit Agrawal | 23BCE2371 | Baselines, benchmark harness, lit review + citation audit |
| Anshveer Singh | 23BCE0703 | Lifecycle (decay/reinforce/consolidate), systems metrics, figures |

**Repo:** https://github.com/Pushkal-Gupta/EnergyMem-An-Energy-Based-Associative-Memory-Architecture-for-Autonomous-Agents

---

## 1. The one-sentence claim

> Agent long-term memory retrieval, currently implemented as top-k similarity search
> over a static store, is better modelled as **energy descent over a strength-modulated
> associative field** — and the strict generalizations this buys (multi-step pattern
> completion, usage-modulated basin depth, learned transition topology, consolidation
> by basin merging) yield measurable gains on degraded and cross-associative queries
> at a quantified latency cost.

Everything in this repo exists to support or refute that sentence. If an experiment
does not bear on it, it does not go in the paper.

## 2. Scope decision (IMPORTANT — read before writing anything)

Two scopes were drafted during Review 1:

- **`EnergyMem_Review_1_29_07_2026.pptx` / `_Expanded.pptx`** — narrow: an associative
  retrieval layer benchmarked against vector search.
- **`BEM_Review_1_FINAL_QA_CHECKED.pptx`** — broad: 6 modules, episodic/semantic/
  working/procedural separation, consolidation engine, *and* an active-inference planner.

**We ship the narrow scope.** The broad scope is not one paper; it is a research
programme. A Tier-1 reviewer confronted with six simultaneously-introduced modules and
no ablation budget rejects on "insufficient evidence for any individual claim."

### In scope (paper contributions, C1–C4)

- **C1 — Formalization + reduction.** An energy functional over query/memory states whose
  one-step, zero-temperature limit provably reduces to top-1 cosine retrieval. We state
  this reduction ourselves, up front. It is the paper's spine, not its weakness.
- **C2 — Strength-modulated basins.** Per-memory bias term `b_i` in the energy, driven by
  usage/recency/surprise. Unifies reinforcement, decay and forgetting into one mechanism
  instead of three heuristics bolted onto a vector DB.
- **C3 — Learned associative topology.** A separate routing/transition operator that links
  memories by temporal co-occurrence rather than embedding proximity. **This is the one
  component that provably cannot be expressed as reweighted cosine top-k.**
- **C4 — DegradeBench.** A controlled, released benchmark of query degradation
  (masking / paraphrase / embedding noise / distractor injection / composite multi-cue)
  with a matched-encoder evaluation protocol. Reusable regardless of whether EnergyMem wins.

### Out of scope (explicitly deferred to Future Work, ~1 paragraph in the paper)

- Active inference / expected-free-energy planning
- Full working/episodic/semantic/procedural memory taxonomy
- Multimodal perception, embodied agents, world models
- Any claim of biological equivalence (analogy only — say so in the paper)

**Rationale:** Review-2 needs a *working demo* by mid-August and the Project Outcome
(paper submission) is due **on or before 23-09-2026**. Six weeks buys one sharp claim
evaluated rigorously, or six claims evaluated badly. We take the former.

## 3. Non-negotiables

1. **Matched encoder.** Every baseline and every EnergyMem variant uses the *same*
   frozen embedding model over the *same* corpus. Any result comparing across encoders
   is inadmissible.
2. **Latency is reported, always.** Energy descent is iterative. We never report an
   accuracy win without the wall-clock and step-count cost beside it. Hiding this is
   the fastest route to desk rejection.
3. **Seeds and CIs.** ≥5 seeds, bootstrap 95% CIs on every headline number.
4. **No claim without an ablation.** C2 and C3 each need a knockout run.
5. **Every citation verified against the actual paper.** See `07_CITATION_AUDIT.md`.
   The Review-1 bibliography was machine-generated and is currently untrusted.
6. **Negative results ship.** If energy descent loses to cosine top-k on clean queries,
   that goes in the abstract. A well-run negative result is publishable; a hidden one is
   retracted.

## 4. Definition of done

- [ ] Reproducible prototype, one-command experiment reproduction (`make all`)
- [ ] DegradeBench released with generation scripts
- [ ] Paper submitted to a Tier-1 venue on or before 2026-09-23
- [ ] Review 2 and Review 3 deliverables cleared
- [ ] arXiv preprint posted at submission time (establishes priority)
