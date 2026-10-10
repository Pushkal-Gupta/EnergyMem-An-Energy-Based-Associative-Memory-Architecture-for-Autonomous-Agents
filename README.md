# EnergyMem

**A hybrid predictive–associative memory architecture for autonomous agents**

A research project on how an agent should retrieve its own long-term memory. The working
hypothesis is that retrieval is not a lookup over a static store but **inference in a
learned model of how experience unfolds**, and that dense embedding search is the
degenerate zero-step case of that model.

---

## The question

Agent long-term memory is almost universally retrieved by top-k embedding similarity.
Associative-memory theory offers an alternative — recall as convergence to an attractor in
an energy landscape — but it had never been evaluated as the *persistent memory layer* of
an agent under matched conditions.

We proved that **dense top-k retrieval is a special case of the energy formulation**
(`T=1, β→∞, b=0, R=I`; Proposition 1, verified 1000/1000), then measured which departures
from that corner pay for themselves. Most do not. The one that does is not a better way to
*search* the landscape — it is a **model of what follows what**, run forward.

That result is what the architecture below is built around. Everything in it is forced by a
specific experiment; nothing is there because it sounded good.

---

## The architecture

Five modules. An encoder and a decoder at the boundaries, and between them a latent
controller, a joint-embedding predictor, and a stochastic associative memory. The predictor
supplies *where experience tends to go next*; the memory supplies *what was actually
stored*; the controller decides how much inference to spend; the temperature decides
whether the memory behaves as a Hopfield network (deterministic descent) or as a Boltzmann
machine (sampling from a distribution over futures).

```
   raw experience                                                    ranked memories
   (text / image)                                                    + predicted state
         │                                                                   ▲
         ▼                                                                   │
   ┌───────────┐                                                       ┌───────────┐
   │  E        │   z ∈ S^(d−1)                                         │  D        │
   │  Encoder  │────────────┬────────────────────────────────────────► │  Readout  │
   │  (frozen) │            │                                          │  Decoder  │
   └───────────┘            │                                          └───────────┘
                            ▼                                                ▲
                      ┌───────────┐      control (β, Θ, H, S, λ)              │
                      │  C        │──────────────┬───────────────────┐        │
                      │ Latent    │              │                   │        │
                      │Controller │  context c_t │                   │        │
                      └───────────┘              ▼                   ▼        │
                            │             ┌───────────┐        ┌───────────┐  │
                            └────────────►│  P        │  ẑ_s   │  M        │  │
                                          │ Predictor │───────►│ Stochastic│──┘
                                          │  (JEPA)   │  w_s   │ Assoc.Mem │
                                          └───────────┘        └───────────┘
                                             modes,                 X, b
                                          not a mean            energy landscape
```

Notation: `z` a unit-norm embedding, `X ∈ R^(d×N)` the memory bank, `b ∈ R^N` per-memory
strengths, `β` retrieval sharpness, `Θ` sampling temperature, `H` rollout depth, `S` number
of hypotheses, `λ` predictor–memory coupling.

### E — Encoder (frozen, matched)

`z = E(x) ∈ S^(d−1)`. Text and images both land on the unit sphere. The encoder is frozen
and **identical across every method and every baseline**, so that no comparison can be won
by representation quality. This is a hard rule, not a default: it is what makes the
baseline a point in our own parameter space rather than a separate system.

### C — Latent controller (the reasoning frontend)

Reads the recent trajectory `z_≤t` and emits (i) a context vector `c_t` and (ii) the
inference budget `(β, Θ, H, S, λ)`. It emits **control, not tokens** — this is the
latent-reasoning analogy: the module decides *how much* inference this query deserves, the
way a reasoning model decides how long to think.

It exists because no single fixed configuration wins everywhere. Three regimes have been
measured and they want opposite settings:

| regime | query is… | what wins | budget |
|---|---|---|---|
| **Completion** | a degraded copy of a stored pattern (occluded image) | multi-step descent, β in band | `H=0, T>1, β∈[16,48]` |
| **Composition** | a cue whose answer is *k* steps downstream in a storyline | predictor rollout | `H≥1, S>1, Θ>0` |
| **Semantic match** | a question about stored evidence | plain cosine | `H=0, T=1, β→∞` |

A fixed pipeline must lose two of the three. The minimal controller is a regime classifier
over the query; the strong version learns the budget end-to-end. Either way, when it emits a
constant it reduces exactly to the fixed configuration, which is how it gets tested.

### P — Predictor (joint-embedding, distributional)

Predicts the *representation* of what comes next, never its tokens. Two facts from X17/X17b
constrain its form absolutely:

1. A free-form parametric map does not generalise here. Transitions between consecutive
   episodes are near-orthogonal (mean pairwise cosine 0.006) — there is no global "and then"
   direction to fit — and a rank-2…32 sweep memorised at every capacity (0.784 on seen
   threads, **0.008** on unseen, against a cosine baseline of 0.106).
2. A **nonparametric** predictor that carries the transitions instead of compressing them
   works: locally-weighted displacement transfer, +46% on held-out threads.

So the default `P` is a kernel over observed transitions, and it returns a **weighted set of
modes, not a mean**:

```
P(· | q) = { (ẑ_i, w_i) }_{i∈topK},   ẑ_i = Π[ q + Δ_i ],   w = softmax(τ · Z_c^T q)
```

where `Δ_i = z_next(i) − z_cue(i)` are displacements from transitions the agent actually
experienced, and `Π` is projection back to the sphere. Temporal adjacency is free
supervision — nothing here is labelled.

**Why modes and not a mean.** The conditional next-state distribution is multi-modal: a cue
can legitimately lead to several mutually exclusive futures. A point estimate returns their
average, and in embedding space the average of two distinct memories is a location where
nothing is stored — off-manifold, a ridge between basins rather than a basin. Preliminary
diagnostics support this strongly (predictions land at cosine ≈0.44 from their own nearest
stored neighbour, against ≈0.73 for real episodes) but they are **not yet in the ledger**;
re-running and logging them is the first item in the queue below.

Rollout to `H` hops applies `P` to its own output, expanding the top modes into a weighted
beam of `S` hypotheses. This is where the architecture earns its name: two hops requires
*composing* dynamics the model only ever saw one step at a time.

### M — Stochastic associative memory (Hopfield ↔ Boltzmann)

The memory itself. Energy over the bank:

```
E_M(ξ) = −(1/β)·lse( β (Xᵀξ + b) ) + ½‖ξ‖²
```

with the update, per hypothesis, run in parallel across `s = 1…S`:

```
ξ ← Π[ (1−λ)·X·softmax(β(Xᵀξ + b))  +  λ·ẑ_s  +  √(2Θ)·ε ],    ε ~ N(0, I/d)
```

Three dials, and each one names a known system:

- **`Θ = 0, λ = 0`** — deterministic CCCP descent. Provably monotone, provably halts.
  This is a modern Hopfield network. With `T=1, β→∞` it is *exactly* cosine top-1.
- **`Θ > 0`** — the update becomes a Markov chain that explores rather than commits, with
  an approximate stationary distribution `∝ exp(−E/Θ)`. This is the Boltzmann regime, and
  it is the direct fix for mode-averaging: noise kicks the state off the ridge into one
  basin or the other, so `S` chains **cover** the branching futures instead of collapsing to
  their invalid mean. (The noisy update is not Metropolis-adjusted; an MH correction is
  available behind a flag, and any claim about sampling from the true Gibbs measure must use
  it.)
- **`λ > 0`** — the predictor's output enters the energy as a bias rather than running
  beside it. **This is currently believed to be worth nothing** (X18: interpolating the
  routing operator inside the descent step never exceeded running the predictor externally;
  at `η=1` it merely *recovers* it by replacing the Hopfield step entirely). `λ` is kept as
  an explicit, ablatable term whose null setting reproduces the separated pipeline, precisely
  so that the negative result stays testable instead of being designed away.

Note that `β` and `Θ` are **different temperatures** and conflating them is a bug, not a
simplification. `β` sharpens the retrieval kernel over stored patterns; `Θ` is the thermal
noise of the sampler. `β` has leverage only inside a narrow band (below ≈16 the state
collapses to the corpus centroid on step one; above ≈50 descent degenerates into cosine, so
the apparent plateau is the method dissolving into the baseline).

### D — Readout / decoder

Two jobs.

**Mixture-aware scoring.** Given `S` converged hypotheses with weights, score each stored
memory by

```
score(m) = logsumexp_s [ β·⟨ξ_s, z_m⟩ + log w_s ]
```

and **never** by `⟨mean_s ξ_s, z_m⟩`. Averaging the hypotheses before scoring reintroduces
exactly the failure the sampler was added to fix. The multi-answer case additionally uses
residual peeling — commit to what was found, subtract it from the cue, roll again — which
restores multi-answer recall that single-state convergence structurally destroys.

**Decoding.** Default is non-generative: return the stored source text of the ranked
memories, which keeps evaluation clean and every number attributable to retrieval. A
generative decoder (an LLM conditioned on the retrieved set) sits behind the same interface
for end-task evaluation. A third mode decodes the *predicted* latent `ẑ` that has no stored
counterpart — the model's imagined next state — which is the diagnostic that tells you
whether a rollout went somewhere plausible or off-manifold.

### The reduction ladder

Every rung has a null setting that reproduces the rung below it **exactly**, and each
equivalence is asserted in the test suite. This is the architecture's central discipline:
no module can be credited with an improvement that a coding difference could explain.

| rung | configuration | is |
|---|---|---|
| **R0** | `T=1, β→∞, b=0, H=0, S=1, Θ=0, λ=0` | dense top-k — **Proposition 1**, verified 1000/1000 |
| **R1** | `T>1`, finite β | modern Hopfield descent |
| **R2** | `b ≠ 0` | strength-modulated basin depth (usage/recency/surprise as one physical term) |
| **R3** | `H ≥ 1`, predictor external | world-model rollout |
| **R4** | `Θ > 0, S > 1`, mixture readout | Boltzmann sampling over branching futures |
| **R5** | controller emits per-query budget | latent-reasoning frontend |
| **R6** | `λ > 0` | predictor coupled into the energy — *currently unsupported, kept ablatable* |

**Proposition 1′ (extended reduction).** Dense top-k retrieval is the corner
`(β→∞, T=1, H=0, S=1, Θ=0, λ=0)` of this architecture. The research question is therefore
sharp rather than rhetorical: *which departures from that corner pay for themselves, and in
which regime?*

### Design invariants

1. **Nesting.** Every module has a null setting that reduces to the previous configuration,
   and CI asserts the equivalence numerically.
2. **Matched frozen encoder** across every method, no exceptions.
3. **Dimensionless hyperparameters.** Any scale crossing dimensionalities is normalised
   (`σ_eff = σ/√d`). A missing `√d` once turned a working mechanism into a reported zero.
4. **Never average modes.** Distributions are carried to the readout intact.
5. **Composition over summation.** The predictor runs beside the memory unless `λ>0` is
   shown to help.
6. **Thread-disjoint splits, cluster-bootstrap CIs, held-out hyperparameter selection**, on
   every headline number.

### What forces each module

| module | forced by | the finding |
|---|---|---|
| E (frozen, matched) | X1, X6, X12 | synthetic vectors have no headroom (cosine 1.000 on everything); real anisotropy (unrelated pairs at 0.258) is what creates room to improve |
| M, deterministic | X1, X3, X11 | descent *is* cosine at the β→∞ corner; β has leverage only in a band, per-encoder |
| M, multi-answer readout | X4, X5, X7, X8 | single-state convergence destroys multiplicity; diversity must live in the dynamics, not the readout |
| P, nonparametric | **X17, X17b** | parametric JEPA memorises (0.784 seen → 0.008 unseen); displacement transfer generalises (+46%) |
| P at all | **X16** | "what happened next" is not a similarity relation — cosine R@1 = 0.105, median gold rank 24/1319, errors are lexical collisions across unrelated storylines |
| P **external**, λ=0 default | **X18** | folding the predictor into the energy never beat running it alongside |
| rollout, H ≥ 1 | ~~X19~~ **WITHDRAWN (X26b)** | The 2-hop result this module was designed around is dead: null on our corpus at 4-fold and null on LoCoMo at n=5,013 (Δ −0.002 [−0.008, +0.004]). `H=1` is supported (X26); `H≥2` is not |
| Θ > 0, S > 1, mixture readout | mode-averaging diagnostics *(preliminary, unlogged)* | the conditional next-state distribution is multi-modal; point estimates land off-manifold |
| C (controller) | image vs LoCoMo split | the winning configuration is regime-dependent; a fixed pipeline must lose two of three regimes |

---

## Position in the literature

- **Ramsauer et al., *Hopfield Networks is All You Need*** (arXiv 2020; ICLR 2021) — the
  anchor for Proposition 1. Establishes the equivalence between continuous modern Hopfield
  networks and attention, which is what makes "dense top-k is a corner of our parameter
  space" a theorem rather than an analogy. Also the reason we state the objection ourselves:
  a reviewer will otherwise say we reinvented attention over a database.
- **LeCun, *A Path Towards Autonomous Machine Intelligence*** (2022) — JEPA. Justifies
  predicting latent representations rather than tokens, which is what module `P` does and
  why it survives where a token-level next-episode model would drown in surface detail.
- **Hinton & Sejnowski, Boltzmann machines** (1985); Langevin/SGLD samplers
  (Welling & Teh, 2011) — the `Θ>0` regime of module `M`.
- **Packer et al., *MemGPT*** (2023) — the industry standard being critiqued: sophisticated
  policies for *what gets written and evicted*, sitting on top of an unmodified
  nearest-neighbour lookup. All of the intelligence is in the write path.
- **Gutiérrez et al., *HippoRAG*** (NeurIPS 2024) — the contemporary rival, also moving past
  flat cosine, but via a **spatial** association structure (a knowledge graph). Our claim is
  **temporal**: association learned from the agent's own trajectory, with no graph
  construction step. The contrast belongs in the paper explicitly.
- **Yang et al., mixture-of-softmaxes** (ICLR 2018) — the rank argument used to show the
  integrated multi-β mixture is not reducible to any single β (it is expressively distinct
  and empirically inert; see study 1b).

---

## Quick start

```bash
make setup     # venv + deps
make test      # sanity suite — Proposition 1 gates everything downstream
make demo      # offline milestone demo, <60s
```

Data is not committed (`data/` is gitignored). To reproduce the real-data studies:

```bash
# text — LoCoMo (2.8 MB), for scripts/exp_locomo_anchored.py
curl -L -o data/locomo10.json \
  https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json

# images — Imagenette-160 (99 MB), for scripts/test_imagenet_hopfield.py
curl -L -o data/imagenette2-160.tgz \
  https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz
tar xzf data/imagenette2-160.tgz -C data/
```

Extra deps for these: `torch`, `torchvision`, `sentence-transformers`, `matplotlib`.

## Read these first, in order

| Doc | What it settles |
|---|---|
| [`docs/00_PROJECT_CHARTER.md`](docs/00_PROJECT_CHARTER.md) | Scope, the one-sentence claim, non-negotiables |
<<<<<<< Updated upstream
| [`docs/13_CLAIM_TRACE.md`](docs/13_CLAIM_TRACE.md) | **🔗 Every claim → ledger ID → script → raw output → test → paper location** |
| [`docs/12_ARCHITECTURE.md`](docs/12_ARCHITECTURE.md) | **🏗 The target architecture — full design spec, module contracts, build order** |
| [`docs/10_EXPERIMENT_LEDGER.md`](docs/10_EXPERIMENT_LEDGER.md) | **📒 Every experiment X1–X26c in order, and how the question moved** |
=======
| [`docs/10_EXPERIMENT_LEDGER.md`](docs/10_EXPERIMENT_LEDGER.md) | **📒 Every experiment X1–X19 in order, and how the question moved** |
>>>>>>> Stashed changes
| [`docs/09_DAY0_FINDINGS.md`](docs/09_DAY0_FINDINGS.md) | **⚠️ Empirical results that change what you should build** |
| [`docs/01_RESEARCH_PLAN.md`](docs/01_RESEARCH_PLAN.md) | The science: RQs, hypotheses, novelty argument |
| [`docs/07_CITATION_AUDIT.md`](docs/07_CITATION_AUDIT.md) | Bibliography status + the competitive landscape |
| [`docs/02_TIMELINE.md`](docs/02_TIMELINE.md) | Week-by-week with pass/fail gates |
| [`docs/03_TASK_ALLOCATION.md`](docs/03_TASK_ALLOCATION.md) | Workstream ownership |
| [`docs/04_EXPERIMENT_PROTOCOL.md`](docs/04_EXPERIMENT_PROTOCOL.md) | Pre-registered experiments |
| [`docs/05_VENUE_STRATEGY.md`](docs/05_VENUE_STRATEGY.md) | Targets + deadlines |
| [`docs/06_RISK_REGISTER.md`](docs/06_RISK_REGISTER.md) | What kills this paper and how we avoid it |
| [`docs/08_REVIEW2_CHECKLIST.md`](docs/08_REVIEW2_CHECKLIST.md) | Milestone review deliverables |
| [`experiments/PROTOCOL_CHANGES.md`](experiments/PROTOCOL_CHANGES.md) | Every deviation and withdrawal, dated |

## Layout

```
docs/          plan, protocol, risks, findings, ledger
src/energymem/
  base.py            Retriever protocol, Memory, RetrievalResult   [FROZEN]
  encoders.py        E — frozen encoder + on-disk cache (matched-encoder rule)
  memory/energy.py   M — energy functional, CCCP descent, routing hook, peeling,
                         parallel/repulsive descent
  memory/adaptive.py perplexity-targeted beta + query anchor (anchor=1.0 == cosine)
  memory/multibeta.py per-pattern beta — DEAD END, kept as documented negative result
  memory/lifecycle.py b — strength bias, decay, consolidation
  world/knn_wm.py    P — nonparametric displacement-transfer world model (the one that works)
  world/jepa.py      P — parametric JEPA predictor (documented negative result, X17)
  baselines/         dense top-k, BM25, hybrid RRF
  eval/              metrics, cluster bootstrap, DegradeBench, LoCoMo loader,
                     association probes (build_probes: "what happened next", h hops)
experiments/   run_*.py; paired_test.py (cluster bootstrap); devtest.py (thread-disjoint)
paper/         LaTeX skeleton, section stubs, verified refs.bib
scripts/       confirm_holdout.py          the confirmed image-completion result
               exp_multibeta_scales.py     multi-β scales — negative result
               exp_locomo_anchored.py      the LoCoMo negative result (anchor × pp₀ sweep)
               sweep_beta_transition.py    maps the β leverage band; shows high-β → cosine
tests/         sanity suite — Proposition 1, energy monotonicity, capacity, determinism
```

<<<<<<< Updated upstream
Not yet built, and named here so the gap is visible: **C** (the controller) exists only as
the fixed configurations we sweep by hand, and the **Θ>0 sampler** in `M` is not implemented
— `energy.py` currently exposes deterministic descent plus repulsive parallel descent, which
is a different mechanism (enforced diversity, not thermal noise).

---

## Empirical status

Two modalities, real data, honest protocol. The short version: **energy descent beats cosine
when the query is a degraded copy of a stored pattern and loses when it is not; and the only
mechanism that beats cosine on agent-style memory is running a world model forward.**

### ✅ Proposition 1 — verified

1000/1000 queries agree with exact cosine top-1 to floating-point precision, on synthetic
*and* real embeddings. Gated in CI. Every downstream comparison is exact by construction.

### ✅ One-hop temporal association — externally validated (X16, X17b, X25, X26)

Three independent implementations, two corpora, one of them public. **This is the project's
strongest and only externally validated claim.**

| implementation | corpus | probes | clusters | cosine R@1 | world model | Δ R@1 |
|---|---|---|---|---|---|---|
| (i) 50/50 split | episodic (ours) | 37 | 37 | 0.106 | 0.154 | +0.049 [+0.000, +0.100] p=0.032 |
| (ii) 4-fold | episodic (ours) | 257 | 122 | 0.105 | 0.140 | +0.035 [+0.012, +0.061] p=0.002 |
| **(iii) 4-fold** | **LoCoMo** | **5,233** | **272 sessions** | **0.064** | **0.107** | **+0.043 [+0.036, +0.050] p<1e-4** |

On LoCoMo: **+67% relative** at R@1, and R@5 0.174 → 0.251 (+0.077 [+0.067, +0.087]).
Candidate set is the *full pooled bank* of 5,882 turns across all 10 conversations, with ids
namespaced by conversation. Split unit is the session; every session is in the test set
exactly once; `k_P, τ` chosen per fold on that fold's val sessions. The task is harder than
our own corpus (cosine 0.064 vs 0.105) and the relative effect is **larger** there.

### ❌ Multi-hop composition — WITHDRAWN (X19 → X25b → X26b)

An earlier run reported 2-hop R@5 **0.446 vs 0.262**, Δ +0.185, p<0.001, +70%. It does not
survive. Three measurements now:

| | protocol | probes | result |
|---|---|---|---|
| (a) | episodic, 50/50 — **script lost** | 65 | 0.446 vs 0.262, Δ +0.185 |
| (b) | episodic, 4-fold | 135 | 0.230 vs 0.207, spans zero |
| (c) | **LoCoMo, 4-fold** | **5,013** | 0.041 vs 0.043, **Δ −0.002 [−0.008, +0.004] p=0.77** |

**The 0.446 figure must not be cited.** Association helps one step ahead and does not
compose: compounding prediction error exceeds whatever compositional signal the second step
carries. A residual +0.012 [+0.004, +0.019] survives at R@5 on LoCoMo, an order of magnitude
below the one-hop effect; we do not build on it.

### ✅ Image occlusion recovery — confirmed positive

400 Imagenette memories, ResNet18 features, queries = occluded held-out images. β selected on
a disjoint tune half; one pre-committed paired test. n = 5,847 instances from 1,949 images.

| occlusion | cosine | energy β=40 | paired Δ | 95% CI |
|---|---|---|---|---|
| 30% | 0.730 | 0.733 | +0.003 | [−0.008, +0.014] n.s. |
| 50% | 0.542 | 0.559 | +0.017 | [+0.003, +0.031] **sig** |
| 70% | 0.386 | 0.407 | +0.021 | [+0.007, +0.034] **sig** |
| **pooled** | **0.553** | **0.566** | **+0.014** | **[+0.006, +0.021] sig** |

The advantage **grows with corruption severity** — what pattern completion predicts — and is
null at light corruption where there is nothing to reconstruct. Costs ~8× cosine's latency.
Tune-half estimate (+0.016) matched held-out (+0.014): no selection bias.

Descent wins R@1 but **loses badly on R@5/MRR** (0.605 vs cosine 0.759): it converges to one
attractor, so ranks 2–5 reflect proximity to that point rather than to the query. A better
top-1, a worse shortlist. This is the same structural fact that motivates modules `P`+`D`.

### ❌ Negative results that ship

| claim | verdict | evidence |
|---|---|---|
| Multi-step descent beats one step on corrupted cues | **dead** | identical above the β threshold (X2) — exactly what Prop 1 predicts |
| Readout variants (peeling, repulsion, voting) beat dense top-k on text | **dead** | +0.042 (tuned=tested, n=37) → **+0.004, p=0.43** with thread-disjoint selection (X15). The advantage was hyperparameter overfitting |
| Folding the predictor into the energy beats running it alongside | **dead** | never exceeds external; η=1 merely recovers it (X18). This was the stated novelty against PAM, and it is not supported |
| Multiple β scales beat one tuned β | **dead** | integrated mixture vs ensemble: +0.0003 [−0.0014, +0.0021], despite disagreeing on 37% of queries. Expressively distinct, performance-neutral |
| Retrieval margin predicts correctness ("the memory knows when it is unsure") | **dead** | correlation ≈ 0, accuracy non-monotonic across margin quartiles, both methods |
| Per-memory / learned β helps | **dead** | learned values land above the leverage band; a shuffle control scores identically |
| Descent helps on question→evidence retrieval (LoCoMo) | **dead** | monotone dose–response: every increment of descent costs accuracy, optimum at the cosine boundary (0.336 → 0.252 across the anchor sweep, 1,977 questions) |

### The diagnosis that unifies them

The descent step computes a weighted average of stored patterns and moves the state onto
their span. That pays only when **the query is a degraded copy of something already stored** —
then moving toward the memory manifold restores lost signal. Agent memory retrieval is
question → evidence: the query is not a corrupted turn, so there is nothing to complete and
averaging only smears the query's discriminative signal.

> Associative completion requires the cue to be a degraded instance of a stored pattern.
> Image occlusion satisfies that premise and descent wins, by more as damage increases.
> Question-based retrieval violates it, and descent loses monotonically.

Which is precisely why the architecture's centre of gravity moved from the landscape to the
**predictor in front of it**: every readout variant we tried changed only how you read an
answer *off* the landscape. None of them changed the landscape. A world model does.

### Corrections to our own earlier claims

Logged rather than quietly overwritten (see `experiments/PROTOCOL_CHANGES.md`):

1. **Data bug, invalidating.** LoCoMo `dia_id`s are unique only *within* a conversation —
   5,882 turns carry 1,033 distinct ids. A capacity sweep that pooled banks without
   namespacing let distractors satisfy gold ids and score as hits. Its "adaptive β improves
   as the bank grows" result was an artifact.
2. **Confounded comparison.** `pp_start=32` forces step 1 to average ~32 memories — the
   broad-start annealing already shown to be harmful. Removing it moves pure descent from
   0.201 to 0.252, reversing a stated conclusion.
3. **Missing query anchor.** Descent without an anchor discards the cue after step 1, which
   made the earlier comparison a test of "is a blend of memories a better query than the
   query." Replaced by the full anchor sweep.
4. **Underpowered image verdicts withdrawn.** 80 distinct images, unpaired CIs ~5× the
   effect size, β tuned on the test images. Superseded by the held-out study above.
5. **A statistical error worth naming.** A naive bootstrap over (seed × question) rows
   reported p=0.0005 for a result that a cluster bootstrap over questions puts at p=0.076.
   Five seeds reusing the same questions is pseudo-replication. Cluster bootstrap is now
   mandatory.

---

## What the architecture obliges us to test next

Ordered by how much of the above they can invalidate. Experiments are chosen from this list,
not added to it ad hoc.

1. **Re-log the mode-averaging diagnostics.** Off-manifold-ness of averaged predictions,
   multi-modality rate of the conditional next-state distribution, and mixture-vs-averaged
   readout at 1 and 2 hops. Currently measured but not in the ledger, and module `M`'s `Θ>0`
   regime rests on them. **Gate.**
2. **Re-run the headline text negative inside the β leverage band.** The `β∈[16,48]` band was
   established after X15 ran; a comparison at β above the band is a comparison of cosine
   against cosine, and the project's most-cited negative is uninterpretable until it is
   re-run with held-out selection and a cluster bootstrap. **Gate.**
3. **External validity for the positive result.** Build the association probes on a public
   corpus (LoCoMo, LongMemEval). Everything in the composition regime is currently measured
   on a purpose-built 146-question corpus we wrote ourselves. Internally rigorous;
   externally unestablished. **Gate on any headline claim.**
4. **Implement the `Θ>0` sampler and measure distributional recall.** Not "did the single
   output match the gold?" but "did the `S` chains *cover* the gold?" — on a corpus with
   genuinely branching trajectories, which is where deterministic prediction must fail by
   construction.
5. **The occlusion test for text.** Swap LoCoMo queries from questions to *masked turns* on
   the same corpus and encoder. The diagnosis predicts descent regains its advantage — same
   data, same embeddings, only the query type changes. Cheap, and it either confirms the
   unifying mechanism or breaks it.
6. **Rollout-depth sweep.** `H = 0,1,2,3,4`. X19 shows 2 hops beat 1; where does compounding
   error overtake compositional gain?
7. **Budget-matched comparison.** Every accuracy number against its latency and matmul count.
   Descent already costs ~8× cosine; `S` chains multiply that again.
8. **Controller ablation.** Fixed configuration vs regime-classified budget, across all three
   regimes at once. This is the test that the controller is architecture and not decoration.
9. **`λ` ablation.** Sweep the coupling term to confirm or overturn X18 under the sampler,
   where the prediction enters as a bias on a distribution rather than as an interpolation
   inside a deterministic step.
=======
## Current status (2026-09-04)

- ✅ Energy core, test suite — **5/5 passing**; **Proposition 1 verified** (1000/1000)
- ✅ **Real sentence embeddings wired up** (`all-mpnet-base-v2`) — F6 discharged
- ✅ Real-text corpus, 1043 episodes / 37 multi-target questions (`eval/corpus.py`)
- ✅ BM25 + hybrid RRF baselines; peeling, repulsion, vote-projection readouts
- ✅ 5 seeds, **paired cluster-bootstrap** CIs (`experiments/paired_test.py`)
- ✅ Question bank 37 → **146** (past the power threshold), dev/test split by thread
- 🛑 **RESULT: no readout beats dense top-k.** Held-out Δ = **+0.004** [−0.033, +0.041],
  p=0.43. The +0.042 trend was hyperparameter overfitting. See **F15**.
- ✅ That is a clean, well-powered negative result and it is publishable alongside Prop 1
- ✅ **World model (G3) built and evaluated.** Nonparametric displacement-transfer model
  over the episode stream (`world/knn_wm.py`); parametric JEPA also tried (`world/jepa.py`).
- 🎯 **STRONGEST RESULT: 2-hop association.** Rolling the world model forward reaches
  memories two steps away that similarity cannot: Recall@5 **0.446 vs 0.262**,
  Δ +0.185 [+0.076, +0.297], **p<0.001**, held-out threads. **+70% relative.** See X19.
- 🛑 Integrating the predictor *inside* the energy landscape buys nothing over running it
  alongside (X18). That novelty claim against PAM is not supported.
- ❌ LongMemEval / LoCoMo — external validation of the negative result (network was down)
>>>>>>> Stashed changes

## The rules

1. Matched encoder across every method. No exceptions.
2. Never report an accuracy win without its latency cost.
3. ≥5 seeds, bootstrap 95% CIs on every headline number; cluster over questions, never over
   (seed × question) rows.
4. Hyperparameters selected on a split the score never sees.
5. No claim without an ablation, and every module must have a null setting that reduces to
   the configuration below it.
6. Every citation verified against the actual paper.
7. Negative results ship. If it did not work, that goes in the abstract.
