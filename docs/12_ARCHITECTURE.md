# 12 · Target Architecture

**The Predictive–Associative Stack (PAS)**

Status: **design specification.** This document describes the system we intend to build,
not the system that exists. Where a component already exists it says so and points at the
file; where it does not, it says that too. Nothing here is a result — results live in
[`10_EXPERIMENT_LEDGER.md`](10_EXPERIMENT_LEDGER.md), and every design decision below cites
the experiment that forced it.

Read [`10_EXPERIMENT_LEDGER.md`](10_EXPERIMENT_LEDGER.md) first if you have not. This
document assumes X1–X19.

---

## 1. What this architecture is for

An agent accumulates a stream of episodes. At some later moment it needs a few of them.
Every deployed system answers that with one matrix multiplication — embed the query, take
the top-k nearest stored vectors — and puts all of its sophistication into *what gets
written and evicted*. The retrieval step itself has been a frozen nearest-neighbour lookup
for a decade.

We showed (Proposition 1) that this lookup is one corner of a much larger family: the
`T=1, β→∞, b=0, R=I` corner of an energy-based associative memory. That converts a vague
question ("is energy-based memory better?") into a sharp one:

> **Which departures from the dense-retrieval corner pay for themselves, and in which
> regime?**

Nineteen experiments answer it, and the answer is uncomfortable for the original thesis:

- Every departure that changes only **how you read an answer off the landscape** — more
  descent steps, residual peeling, repulsive parallel descent, vote counting, multi-scale β
  — fails to beat dense retrieval on agent-style text (X2, X15) and is inert on images
  beyond a single well-tuned β.
- The one departure that works is not a readout at all. It is a **model of what follows
  what**, run forward: +46% at one hop, **+70% at two** (X17b, X19).
- And folding that model *into* the energy functional bought nothing over running it
  beside (X18).

So the architecture we want is not "a better landscape." It is a **predictor in front of a
memory**, with the landscape retained for the one regime where it demonstrably wins
(completion under corruption), a **stochastic** sampler so that branching futures are
covered rather than averaged, and a **controller** that decides which regime it is in.

### 1.1 The three regimes, and why one fixed pipeline cannot serve them

This is the load-bearing observation of the whole design.

| regime | the query is… | what wins | required budget |
|---|---|---|---|
| **Completion** | a degraded copy of a stored pattern (occluded image, masked turn) | multi-step energy descent | `H=0, T>1, β∈[16,48]` |
| **Composition** | a cue whose answer lies *h* steps downstream in a storyline | predictor rollout | `H≥1, S>1, Θ>0` |
| **Semantic match** | a question about stored evidence | plain cosine | `H=0, T=1, β→∞` |

Three regimes, three mutually exclusive optimal settings, all three present in a real
agent's workload. Any system with a fixed configuration must lose two of them. That is the
argument for module **C**, and it is why this is an architecture rather than a method.

---

## 2. Design invariants

Non-negotiable. A design that violates one of these is rejected regardless of its numbers.

**I1 · Nesting.** Every module has a null setting under which the system reproduces the
configuration below it **exactly**, and each equivalence is asserted numerically in the test
suite. This is what makes every baseline a point in our own parameter space instead of a
separate codebase we could accidentally advantage.

**I2 · Matched frozen encoder.** One encoder, frozen, identical across every method and
baseline. No result may depend on representation quality.

**I3 · Dimensionless hyperparameters.** Every scale that crosses dimensionalities is
normalised before use (`σ_eff = σ/√d`, noise scaled `√(2Θ/d)`). A missing `√d` once turned a
working mechanism into a reported zero (X10) — that near-miss is the reason this is an
invariant and not a style note.

**I4 · Never average modes.** A distribution over futures is carried to the readout intact.
No stage may collapse hypotheses to their mean.

**I5 · Composition over summation.** The predictor runs *beside* the memory. Coupling it
*into* the energy is available as an explicit term whose null setting restores the separated
pipeline (X18 says the term is worth nothing; it is kept ablatable rather than removed, so
that the negative stays testable).

**I6 · No circular seeding.** Nothing in the pipeline may be initialised or ranked by cosine
similarity to the query in a way that reintroduces the baseline as the mechanism. Stored
patterns are exact fixed points of the descent, so seeding descent at cosine top-k makes it
a no-op and the ranking must then come from cosine itself (X7). Seeds come from the
predictor or from the cue, never from the baseline.

**I7 · Association is temporal, never semantic.** The routing/prediction structure is
learned from co-occurrence in the agent's own stream. Training it on embedding similarity
would collapse it into the baseline and silently invalidate the central claim.

**I8 · Evaluation discipline.** Thread-disjoint train/val/test, hyperparameters selected on
a split the score never sees, cluster bootstrap over questions (never over seed × question
rows), latency reported beside every accuracy number.

---

## 3. System overview

Six components. Two at the boundaries (encoder, decoder), one passive (the store), three
active (controller, predictor, memory).

```
              ┌──────────────────────────────────────────────────────────┐
   write path │   experience x_t ──► E ──► z_t ──► S (store)             │
              │                                    · bank X, strengths b │
              │                                    · transitions (t→t+1) │
              └──────────────────────────────────────────────────────────┘

   read path

     cue x                                                     ranked memories
       │                                                       + predicted state
       ▼                                                              ▲
   ┌───────┐                                                    ┌───────────┐
   │   E   │  q ∈ S^(d−1)                                       │     D     │
   │encoder├──────────┬──────────────────────────────────────►  │  readout  │
   └───────┘          │                                         │  decoder  │
                      ▼                                         └───────────┘
                ┌───────────┐   budget (β, Θ, T, H, S, λ, k)          ▲
                │     C     ├────────────┬──────────────┐             │
                │controller │            │              │             │
                │           │ context c  │              │             │
                └───────────┘            ▼              ▼             │
                      │           ┌───────────┐   ┌───────────┐       │
                      └──────────►│     P     │   │     M     │       │
                                  │ predictor │   │stochastic │       │
                                  │  (JEPA)   │──►│assoc. mem │───────┘
                                  └───────────┘   └───────────┘
                                   {(ẑ_s, w_s)}    {(ξ_s, w_s)}
                                   modes, not         X, b
                                     a mean       energy landscape
```

**Data flow, read path.** Encode the cue. The controller inspects it and the recent stream,
and emits a context vector plus an inference budget. If the budget calls for rollout, the
predictor expands the cue into a weighted set of hypothesis states by iterating learned
dynamics `H` times, keeping the top `S` by weight. Each hypothesis is then relaxed against
the memory landscape — deterministically at `Θ=0`, as a sampler at `Θ>0`. The readout scores
every stored memory against the whole weighted set at once and returns the top-k. The
decoder turns those into text for the agent.

---

## 4. Formal specification

### 4.1 Notation

| symbol | meaning |
|---|---|
| `d` | embedding dimension |
| `N` | number of stored memories |
| `z ∈ S^(d−1)` | a unit-norm embedding; `Π[·]` is projection back to the sphere |
| `X ∈ R^(d×N)` | the memory bank, columns are stored patterns |
| `b ∈ R^N` | per-memory strength bias (usage, recency, surprise) |
| `β` | retrieval sharpness (inverse temperature of the softmax over patterns) |
| `Θ` | **sampling** temperature (thermal noise of the Markov chain) — *not* `1/β` |
| `T` | descent/sampling steps per hypothesis |
| `H` | rollout depth (hops of the predictor) |
| `K` | branching factor: modes expanded per hypothesis per hop |
| `S` | hypotheses carried to the readout |
| `λ` | predictor→memory coupling strength |
| `τ, k_P` | predictor kernel sharpness and neighbourhood size |

`β` and `Θ` are different temperatures and conflating them is a bug. `β` sharpens *which
stored patterns* the state is pulled toward. `Θ` is how much the state is allowed to wander
while being pulled. `β → ∞` with `Θ = 0` is argmax; `β → ∞` with `Θ > 0` is a chain hopping
between argmax basins.

### 4.2 Energy

```
E_M(ξ) = −(1/β)·lse( β·(Xᵀξ + b) ) + ½‖ξ‖²                         (memory)

E(ξ | ẑ) = E_M(ξ) + (λ/2)·‖ξ − ẑ‖²                                 (coupled, λ>0)
```

`E_M` is the modern-Hopfield energy already implemented in `memory/energy.py`. The coupled
form adds a quadratic pull toward the predictor's output; `λ=0` recovers `E_M` exactly.

### 4.3 Inference

```
retrieve(x, k):
    q ← Π[E(x)]                                              # encode
    (c, β, Θ, T, H, S, λ) ← C(q, z_≤t)                       # budget

    Ξ ← Rollout(q, H, K, S)                                  # {(ẑ_s, w_s)}, Σw = 1
    Ξ' ← { (Relax(ẑ_s | β, Θ, T, λ), w_s) : (ẑ_s, w_s) ∈ Ξ } # {(ξ_s, w_s)}

    score(m) ← logsumexp_s [ β·⟨ξ_s, z_m⟩ + log w_s ]        # mixture readout
    return top-k by score
```

```
Rollout(q, H, K, S):
    Ξ ← {(q, 1)}
    repeat H times:
        Ξ ← { (ẑ_i, w·w_i) : (z,w) ∈ Ξ, (ẑ_i,w_i) ∈ topK modes of P(·|z) }
        Ξ ← prune Ξ to the S heaviest, renormalise
    return Ξ
```

```
Relax(ẑ | β, Θ, T, λ):
    ξ ← ẑ
    repeat T times:
        p    ← softmax( β·(Xᵀξ + b) )
        ξ_h  ← X·p                                # Hopfield / CCCP step
        ξ    ← (1−λ)·ξ_h + λ·ẑ                    # coupling (λ=0 default)
        if Θ > 0:
            ξ ← ξ + √(2Θ/d)·ε ,   ε ~ N(0, I_d)   # Langevin noise, dimensionless
            [optional] Metropolis-accept on E(ξ' | ẑ) at temperature Θ
        ξ ← Π[ξ]
        if ‖Δξ‖ < tol and Θ = 0: break
    return ξ
```

Three notes on correctness, stated because they are easy to get wrong.

1. **`H = 0` means `Ξ = {(q, 1)}`** — a single hypothesis which is the cue itself. The
   pipeline then reduces to plain relaxation of the query, and with `T=1, β→∞, Θ=0, λ=0` to
   exact cosine top-1. That is Proposition 1 and it must remain a passing test.
2. **The noise is scaled `√(2Θ/d)`** so that the displacement norm is `≈√(2Θ)` independent of
   dimension (I3). Writing `√(2Θ)·ε` with `ε ~ N(0, I_d)` gives a perturbation of norm
   `√(2Θd)` — at `d=768` that is a 28× error, and it is exactly the class of mistake that
   produced a false zero in X10.
3. **The `Θ=0, λ=0` update is CCCP**, so energy decreases monotonically and convergence is
   guaranteed. Neither property is inherited automatically once `λ>0` or `Θ>0`: the coupled
   update is a proximal step on `E(·|ẑ)` and the noisy update is a Markov chain whose
   stationary distribution is `∝ exp(−E/Θ)` **only if Metropolis-adjusted**. Any claim about
   sampling the Gibbs measure must use the correction; without it we are running an
   approximate sampler and must say so.

---

## 5. Module specifications

Each module states: its contract, what forces its design, its interface, its null setting,
and whether it exists.

### 5.1 E — Encoder

**Contract.** `x ↦ z ∈ S^(d−1)`. Frozen. Identical across every method and baseline (I2).
Text and images both land on the unit sphere so the rest of the stack is modality-agnostic.

**Forced by.** X6 (synthetic vectors have no headroom — cosine scores 1.000 on every
condition including 70% masking, so nothing can be measured against it), X12 (real
embeddings are anisotropic — unrelated pairs sit at cosine 0.258, and that anisotropy is
what creates room to improve).

**Interface.** Existing — `src/energymem/encoders.py`, with an on-disk cache. Every
`Memory.embedding` is populated before any retriever sees it; `EnergyMemory.index()` raises
if it is not.

**Null setting.** N/A — the encoder is never varied within a comparison.

**Status.** ✅ Exists.

---

### 5.2 S — Store

**Contract.** Holds the bank `X`, the strengths `b`, and — new — the **transition log**: the
ordered pairs `(t, t+1)` of episodes the agent actually experienced in sequence.

**Forced by.** The transition log is what makes `P` possible at all, and it is free
supervision: temporal adjacency requires no labelling. I7 forbids ever deriving it from
similarity.

**Interface.**

```python
@dataclass
class Memory:            # exists, base.py
    id, text, timestamp, source, salience, embedding,
    usage_count, last_access, metadata

# new
class EpisodeStore:
    def append(self, m: Memory) -> None: ...
    def transitions(self, split: str) -> list[tuple[int, int]]: ...   # thread-disjoint
    def strengths(self) -> np.ndarray: ...       # -> b, via memory/lifecycle.py
```

**Null setting.** `b = 0` (no strength modulation) recovers an unweighted bank.

**Status.** ⚠️ Partial. `Memory`, `lifecycle.strength_bias/reinforce/decay/consolidate`
exist; the transition log is currently reconstructed ad hoc inside experiment scripts and
needs to become a first-class object with the split logic attached, because "which
transitions is this model allowed to see" is exactly the question X17 shows is fatal to get
wrong.

---

### 5.3 C — Latent controller

**Contract.** `(q, z_≤t) ↦ (c, β, Θ, T, H, S, λ)`. Emits a context vector and an inference
budget. It emits **control, not tokens**: it decides *how much inference this query
deserves*, the way a reasoning model decides how long to think. This is the "LRM-shaped"
part of the stack and it is the reason the system can serve all three regimes at once.

**Forced by.** §1.1. Completion, composition and semantic match want mutually exclusive
budgets, and all three occur in one workload.

**Interface.**

```python
class Controller(Protocol):
    def budget(self, q: np.ndarray, recent: np.ndarray | None) -> Budget: ...

@dataclass(frozen=True)
class Budget:
    beta: float; theta: float; steps: int
    hops: int; branch: int; hypotheses: int
    coupling: float; context: np.ndarray | None
```

Two implementations, in order of ambition:

- **`FixedController(Budget)`** — returns a constant. This *is* the null setting, and it is
  how every result to date was produced. Ships first.
- **`RegimeController`** — a classifier over cheap query features (is the cue an episode or a
  question? how far is it from its nearest stored neighbour? what is the entropy of the
  similarity distribution?) mapping to one of three budget presets, trained thread-disjoint.
- *(later)* a learned continuous budget.

**Null setting.** `FixedController` — must reproduce the current fixed-configuration numbers
bit-for-bit.

**Status.** ❌ Does not exist. Today "the controller" is the three of us picking settings by
hand per experiment, which is precisely the thing that has to become a measured component
rather than an experimenter degree of freedom.

**Open question.** The controller is also the most dangerous module for I8: a budget selected
per query on test data is test-set tuning with extra steps. The regime classifier must be fit
on train threads and frozen before the test split is touched.

---

### 5.4 P — Predictor (joint-embedding, distributional)

**Contract.** `q ↦ {(ẑ_i, w_i)}` — a **weighted set of modes** for the representation of what
comes next. Predicts in embedding space, never in token space. Never returns a mean.

**Forced by.**

- **X16 — the gate.** "What happened next" is not a similarity relation: cosine gets
  Recall@1 = 0.105 with median gold rank **24 of 1319**, and its errors are lexical
  collisions across unrelated storylines. Oracle-restricting to the correct thread still only
  reaches 0.488. If cosine had solved this, `P` would have no reason to exist.
- **X17 — the form.** A free-form parametric map does not generalise here. Rank 2→32 all
  memorised: **0.784** on threads it trained on, **0.008** on unseen ones, against a cosine
  baseline of 0.106. The diagnostic says why — displacements between consecutive episodes are
  near-orthogonal (mean pairwise cosine 0.006), so there is no global "and then" direction to
  fit, and ~107 transitions in 768 dimensions cannot teach one.
- **X17b — the fix.** Carry the transitions instead of compressing them. Locally-weighted
  displacement transfer: **0.154 vs 0.106** on held-out threads, +46%, `k` and `τ` chosen on
  validation threads only.
- **X19 — why it must compose.** Two-hop Recall@5 **0.446 vs cosine 0.262**, Δ +0.185
  [+0.076, +0.297], p<0.001. The model saw only 1-step transitions, so 2 hops requires
  composing the dynamics with themselves.

**Mechanism (default).** Nonparametric kernel over observed transitions:

```
Δ_i = z_next(i) − z_cue(i)          for each observed transition i
w   = softmax( τ · Z_cᵀ q )         restricted to the k_P nearest cues
mode_i = Π[ q + Δ_i ]               weight w_i
```

**Why modes and not a mean — the central design decision.** The conditional next-state
distribution is multi-modal: one situation can legitimately lead to several mutually
exclusive futures. A point estimate returns their average, and in embedding space the
average of two distinct memories is a location where nothing is stored — a ridge between
basins rather than a basin. Preliminary diagnostics support this strongly (averaged
predictions sit at cosine ≈0.44 from their own nearest stored neighbour, against ≈0.73 for
real episodes; a mixture readout recovers 2-hop exact match that an averaged one loses) but
**these are not yet in the ledger** and re-running them is gate E1 in §9.

**Interface.**

```python
class Predictor(Protocol):
    def fit(self, Z: np.ndarray, transitions: list[tuple[int,int]]) -> "Predictor": ...
    def modes(self, q: np.ndarray, K: int) -> list[tuple[np.ndarray, float]]: ...
    def rollout(self, q: np.ndarray, H: int, K: int, S: int
                ) -> list[tuple[np.ndarray, float]]: ...
```

`KNNWorldModel` (`world/knn_wm.py`) already implements `fit`/`predict`/`rollout` but returns
a **single averaged point**. The change required is exactly `predict → modes`: stop collapsing
the kernel, return its components. This is a small edit with the largest expected payoff in
the whole design.

**Null setting.** `H = 0` — the predictor is bypassed entirely and `Ξ = {(q, 1)}`.

**Status.** ⚠️ Exists but averages (I4 violation). `world/jepa.py` retains the parametric
variant as a documented negative result; it stays in the repo because "we tried the obvious
thing and it memorised" is a finding, not dead code.

---

### 5.5 M — Stochastic associative memory

**Contract.** `ẑ ↦ ξ`, relaxing a hypothesis against the stored landscape. One dial makes it
a Hopfield network, another makes it a Boltzmann machine, a third couples it to the
predictor.

**Forced by.**

- **Deterministic descent earns its place in one regime only.** On image occlusion it beats
  cosine, +0.014 pooled [+0.006, +0.021], **and the advantage grows with corruption** (+0.003
  n.s. at 30%, +0.021 sig at 70%) — precisely what pattern completion predicts, and null
  where there is nothing to reconstruct.
- **`β` has leverage only in a band.** Below ≈16 the state collapses to the corpus centroid
  on step one; above ≈50 descent degenerates into cosine (Proposition 1), so the apparent
  plateau above the band is the method dissolving into the baseline rather than a working
  regime. The band is a property of the encoder and must be re-measured per encoder — X3 gave
  ~16 on synthetic vectors, X11 needed ≥64 on `all-mpnet-base-v2`.
- **Multi-scale β is ruled out.** Integrated mixture vs. ensemble: +0.0003 [−0.0014, +0.0021]
  despite disagreeing on 37% of individual queries. Expressively distinct (confirmed by a
  divergence check), performance-neutral. One well-tuned β is sufficient inside the winning
  regime.
- **`Θ > 0` is the fix for mode-averaging.** Noise kicks the state off the ridge into one
  basin or the other, so `S` chains **cover** the branching futures instead of collapsing to
  their invalid mean. This is the Boltzmann-machine regime of the same energy.
- **`λ = 0` is the default because X18 says so.** Interpolating the routing operator inside
  the descent step never exceeded running the predictor externally; at `η=1` it merely
  *recovers* it, by replacing the Hopfield step entirely. This was the stated novelty against
  PAM and it is not supported. The term is retained as an ablatable knob (I5), not deleted.

**Interface.**

```python
class EnergyMemory:                       # exists, memory/energy.py
    def __init__(self, n_steps, beta, beta_schedule, beta_final,
                 eta, tol, shortlist, encoder): ...
    def index(self, memories) -> None: ...
    def set_strength_bias(self, b) -> None: ...
    def set_routing(self, R) -> None:      # matrix or callable
    def retrieve(self, query, k) -> RetrievalResult: ...
    def retrieve_multi(self, query, k) -> RetrievalResult: ...        # residual peeling

    # new
    def relax(self, z_init, *, beta, theta, steps, coupling, anchor
              ) -> np.ndarray: ...
    def sample(self, z_init, *, n_chains, theta, metropolis=False
               ) -> np.ndarray: ...        # (n_chains, d)
```

**Null settings.** `Θ=0` → deterministic CCCP descent. `λ=0` → uncoupled. `T=1, β→∞, b=0` →
exact cosine top-1.

**Status.** ⚠️ Partial. Deterministic descent, residual peeling, the strength bias and the
routing hook all exist. The `Θ>0` sampler does **not**. Note that
`retrieve_multi_parallel` — repulsive parallel descent — is a *different* mechanism
(enforced diversity via a repulsion term, X7/X8) and must not be confused with thermal
sampling; both produce multiple states, for unrelated reasons, and the paper needs to
distinguish them explicitly.

---

### 5.6 D — Readout and decoder

**Contract.** `{(ξ_s, w_s)} ↦ ranked memories ↦ text`.

**Mixture-aware scoring.**

```
score(m) = logsumexp_s [ β·⟨ξ_s, z_m⟩ + log w_s ]
```

and **never** `⟨mean_s ξ_s, z_m⟩`. Averaging hypotheses before scoring reintroduces exactly
the failure the sampler exists to fix (I4). Two implementation traps, both of which we have
already fallen into and which cost a day each: multiply-then-max instead of max-inside-the-
logsumexp scores identically zero; and applying each mode's displacement to *its own
neighbour's* cue rather than to the query makes every candidate a literal training memory,
which also scores zero. Both are silent.

**Multi-answer readout.** For queries with several correct answers, single-state convergence
destroys the multiplicity the cue carried (X4: recall 0.258 vs 1.000). Residual peeling —
commit to what was found, project it out of the cue, re-descend — restores it (X5), and it
produces a ranking for free because each round is ordered by how much of the cue it
explained. Parallel chains have no such signal.

**Decoding modes.**

- **`extractive`** (default) — return the stored source text of the ranked memories. Keeps
  every number attributable to retrieval; nothing is laundered through a generator.
- **`generative`** — an LLM conditioned on the retrieved set, for end-task evaluation only.
- **`imagined`** — decode the predicted latent `ẑ` that has no stored counterpart. This is the
  diagnostic that says whether a rollout went somewhere plausible or off-manifold, and it is
  how we will report the mode-averaging result visually.

**Null setting.** `S=1` with `extractive` decoding reduces to ordinary top-k readout.

**Status.** ⚠️ Partial. Peeling and plain top-k readout exist; the mixture scorer and the
three decode modes do not.

---

## 6. The reduction ladder

Every rung reproduces the rung below it exactly under its null setting, and each equivalence
is a test.

| rung | configuration | is |
|---|---|---|
| **R0** | `T=1, β→∞, b=0, H=0, S=1, Θ=0, λ=0` | dense top-k — **Proposition 1**, verified 1000/1000, gated in CI |
| **R1** | `T>1`, finite β in band | modern Hopfield descent |
| **R2** | `b ≠ 0` | strength-modulated basin depth — usage, recency and surprise as one physical term instead of three bolted-on heuristics |
| **R3** | `H ≥ 1`, predictor external | world-model rollout |
| **R4** | `Θ > 0, S > 1`, mixture readout | Boltzmann sampling over branching futures |
| **R5** | controller emits a per-query budget | latent-reasoning frontend |
| **R6** | `λ > 0` | predictor coupled into the energy — **currently unsupported (X18), kept ablatable** |

> **Proposition 1′ (extended reduction).** Dense top-k retrieval is the corner
> `(β→∞, T=1, H=0, S=1, Θ=0, λ=0)` of PAS.

Proposition 1 is what makes the baseline uncontestable; Proposition 1′ is what makes the
*research question* well-posed. Everything the project does is a walk away from that corner
with a measurement attached.

---

## 7. Hyperparameter register

Anything not listed here is a constant. Anything listed is swept, and swept on a split the
score never sees.

| symbol | module | dimensionless? | range | selected on |
|---|---|---|---|---|
| `β` | M | yes | 16–48 (encoder-specific band; re-measure per encoder) | val threads |
| `Θ` | M | yes (noise scaled `√(2Θ/d)`) | 0 … 0.1 | val threads |
| `T` | M | yes | 1 … 16 | val threads |
| `λ` | M | yes | 0 (default) … 1 | ablation only |
| `k_P`, `τ` | P | yes | k 5–20, τ 1–10 | val threads |
| `H` | P | yes | 0 … 4 | matched to probe hop |
| `K`, `S` | P/D | yes | K 2–5, S 4–16 | val threads |
| `b` | S | yes | from lifecycle stats | not tuned |

Anything with units of "embedding space" is normalised before it is used. This is I3 and it
has already cost us one false negative.

---

## 8. What is learned, from what

Very little is learned, and that is deliberate — X17 is a direct demonstration that at this
data scale a free-form learned map memorises.

| component | learned? | from | split discipline |
|---|---|---|---|
| E | **no** — frozen | — | — |
| S · `b` | statistics only | usage, recency, surprise counters | — |
| P | **nonparametric** — stores `(Z_c, Δ)` | transitions observed in **train threads only** | thread-disjoint; this is the X17 failure mode |
| P · `τ, k_P` | selected | val threads | never test |
| M · `β, Θ, T` | selected | val threads | never test |
| C | classifier | train threads | frozen before test is touched |
| D | **no** | — | — |

**The rule X17 bought us:** any result of this kind reported without a thread-disjoint split
is meaningless. The parametric predictor looked spectacular (0.784) on threads it had seen
and was worse than useless (0.008) on new ones. Same model, same numbers, opposite
conclusions, and only the split tells them apart.

---

## 9. Failure modes this design guards against

Each one is a mistake we actually made, with the guard that now prevents it.

| # | failure | guard |
|---|---|---|
| 1 | "You reinvented attention over a database" | State it first, as Proposition 1. The baseline is a corner of our own parameter space, so there is nothing to argue about |
| 2 | Mode averaging → off-manifold predictions | I4; mixture readout; `Θ>0` sampler |
| 3 | Parametric predictor memorises threads | Nonparametric default; thread-disjoint splits mandatory (X17) |
| 4 | Scale hyperparameters that do not survive a change of `d` | I3 — `σ/√d`, `√(2Θ/d)` (X10) |
| 5 | Running descent outside the β band and concluding it does not work | Band declared per encoder and swept; note that above the band descent **is** cosine, so such a comparison is cosine against cosine |
| 6 | Pseudo-replication (seeds reusing the same questions) | Cluster bootstrap over questions — took a result from p=0.0005 to p=0.076 |
| 7 | Tuning on the test set | Thread-disjoint dev/test — took an effect from +0.042 to +0.004 |
| 8 | Circular seeding from cosine top-k | I6 (X7) |
| 9 | Learning association from similarity | I7 — it would collapse the claim into the baseline |
| 10 | Silent zero-scoring readout bugs | Fixture tests on the mixture scorer with hand-computed expected values; two distinct bugs of this shape have already occurred |

---

## 10. Implementation order

Dependency-ordered. Each step ends in a measurement, and steps 1–3 are gates: if a gate
fails, the steps below it are re-planned rather than executed.

| # | build | unblocks | gate? |
|---|---|---|---|
| **1** | `P.modes()` — stop averaging the kernel; re-log the mode-averaging diagnostics into the ledger | R4 entirely | **GATE** — R4 rests on an unlogged result |
| **2** | β-band re-run of the headline text negative, held-out selection, cluster bootstrap | the project's most-cited negative | **GATE** — see failure mode 5 |
| **3** | Association probes on a public corpus (LoCoMo / LongMemEval) | any headline claim in the composition regime | **GATE** — external validity |
| **4** | `M.sample()` — Langevin update with `√(2Θ/d)` scaling, MH correction behind a flag | R4 | |
| **5** | `D` mixture scorer + fixture tests | R4 | |
| **6** | `EpisodeStore` with a first-class transition log and split logic | clean re-runs of everything above | |
| **7** | `FixedController`, then `RegimeController` | R5 | |
| **8** | `λ` ablation under the sampler | R6 / the X18 re-test | |

Steps 1–5 together are R4; step 7 is R5. R6 is deliberately last because we expect it to fail
again, and it is cheap to run once everything else is instrumented.

---

## 11. Evaluation contract

A configuration is not "better" until all of the following are reported together.

1. **Regime**, named. Completion / composition / semantic match. A win in one is not a win.
2. **Recall@1 and Recall@5.** Descent is known to improve top-1 while *degrading* the
   shortlist — 0.605 vs cosine 0.759 on images — because it converges to one attractor, so
   ranks 2–5 measure proximity to that point rather than to the query. Reporting one without
   the other hides half the behaviour.
3. **Distributional recall** for `S>1`: not "did the single output match?" but "did the `S`
   chains cover the gold?" This is the metric the Boltzmann regime is designed to move, and
   scoring it with a point metric would understate it by construction.
4. **Paired Δ with a cluster-bootstrap 95% CI**, clustered over questions.
5. **Latency and matmul count.** Descent already costs ~8× cosine; `S` chains multiply that
   again. A win that costs 30× is a different claim from a win that costs 2×.
6. **The split**, explicitly: what was tuned, on what, and whether the test split was touched
   before the number was frozen.
7. **The reduction check** for whichever rung is being claimed, passing.

---

## 12. Open design questions

Honest list of what is not settled.

- **Is `Θ>0` actually necessary, or does `K`-mode expansion in `P` already supply the
  diversity?** The predictor's modes are a discrete mixture; the sampler is a continuous one.
  They may be redundant. The ablation is `S` hypotheses from `P` with `Θ=0` versus one
  hypothesis with `S` chains at `Θ>0` versus both.
- **Where should branching live?** Same question as X7/X8 in a new costume: enforced
  diversity can go in the dynamics (repulsion), in the readout (peeling), or in the noise
  (Θ). Peeling won on synthetic data and lost on real; there is no reason to assume the
  ordering transfers here.
- **Does the coupled energy `λ>0` behave differently from X18's interpolation?** X18 tested a
  linear interpolation inside a *deterministic* step. A quadratic bias on a *distribution* is
  a different mechanism. It probably still fails, but "probably" is not a measurement.
- **Can the controller be learned without leaking the test split?** The obvious formulation —
  pick the budget that maximises the score — is test-set tuning unless the classifier is
  fitted and frozen on train threads.
- **Does the completion/composition split hold within a single modality?** The cleanest test
  is text occlusion: same corpus, same encoder, queries swapped from questions to masked
  turns. The diagnosis predicts descent regains its advantage. If it does not, the unifying
  mechanism in §1.1 is wrong.
- **Capacity.** The image advantage is size-dependent — cosine overtakes descent past
  N≈800 at fixed β. Whether that is a β-scaling artefact or a real capacity limit is
  unresolved, and it matters for any claim about agent-scale memory.

---

## 13. Relation to prior work

| work | relation |
|---|---|
| Ramsauer et al., *Hopfield Networks is All You Need* (arXiv 2020; ICLR 2021) | The anchor for Proposition 1 — the equivalence between modern Hopfield networks and attention. `M` at `Θ=0` **is** their update rule |
| LeCun, *A Path Towards Autonomous Machine Intelligence* (2022) | JEPA. Justifies `P` predicting representations rather than tokens |
| Hinton & Sejnowski, Boltzmann machines (1985); Welling & Teh, SGLD (2011) | The `Θ>0` regime of `M` |
| Packer et al., *MemGPT* (2023) | The standard being critiqued: sophisticated write/evict policy over an unmodified nearest-neighbour read. All the intelligence is on the write path; PAS puts it on the read path |
| Gutiérrez et al., *HippoRAG* (NeurIPS 2024) | The nearest rival — also moves past flat cosine, but via a **spatial** association structure (a knowledge graph). Ours is **temporal**: association learned from the agent's own trajectory, no graph construction step |
| PAM | Keeps its predictor external to retrieval. We claimed integrating it was our novelty; **X18 does not support that claim** and it is withdrawn. What remains is the rollout (`H≥1`) and the distributional treatment (`K, S, Θ`) |
| Yang et al., mixture-of-softmaxes (ICLR 2018) | The rank argument showing the integrated multi-β mixture is not reducible to a single β — expressively distinct, empirically inert |

---

## 14. One-paragraph summary

PAS is a memory read path in which the query is first expanded by a **nonparametric
joint-embedding predictor** into a weighted set of hypothesis futures, each of which is then
relaxed against an **energy landscape over stored episodes** — deterministically when the
task is pattern completion, as a **temperature-`Θ` sampler** when the future genuinely
branches — and scored by a **mixture readout** that never averages the hypotheses. A
**controller** in front chooses the inference budget per query, because the three regimes
present in an agent's workload want mutually exclusive settings. Dense top-k retrieval is the
`(β→∞, T=1, H=0, S=1, Θ=0, λ=0)` corner of this system, which makes every comparison exact by
construction and the research question — *which departures pay for themselves, and where* —
answerable rather than rhetorical.
