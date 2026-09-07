# Experiment Ledger

Every experiment, in order, with what it asked, what it answered, and how the research
question moved as a result. **Append only.** An entry is never edited after its result is
in — if a later run overturns it, that is a new entry pointing back.

The purpose is that anyone (including a reviewer, including us in three weeks) can read
top to bottom and see *why* the project is where it is, without reconstructing it from
commit messages.

Columns: **ID** · what was asked · what came back · what it changed.

---

## Phase 0 — Does the formalism hold at all? (synthetic vectors, 2026-08-11)

| ID | Question | Result | Consequence |
|---|---|---|---|
| **X1** | Does `T=1, β→∞, b=0, R=I` reproduce exact cosine top-1? | **Yes, 1000/1000.** | Proposition 1 is real. Baseline becomes a corner of our own parameter space; every comparison is exact by construction. Gated in CI. |
| **X2** | Does multi-step descent beat one step on corrupted cues? | **No.** Identical above the β threshold. | G1 refuted. Do not build a paper section on multi-step alone. |
| **X3** | Is there a β below which descent breaks? | **Yes**, ~16 on synthetic. Below it the state collapses to the global centroid on step 1. | Annealing schedules that *start* low are harmful — inverts the Review-1 intuition. |
| **X4** | Can single-state descent serve multi-answer queries? | **No.** Recall 0.258 vs cosine 1.000 at k=4. | Structural, not tuning. Convergence destroys the multiplicity the cue carries. |
| **X5** | Does residual peeling fix X4? | **Yes**, restores 1.000. | New mechanism, not in any prior work. Promoted to a contribution. |
| **X6** | Is the synthetic corpus hard enough to measure on? | **No.** Cosine scores 1.000 on every condition including 70% masking. | Everything in Phase 0 is provisional. Real embeddings move to the front of the queue. |

## Phase 1 — Is the multi-answer fix the only one? (synthetic, 2026-09-03/04)

| ID | Question | Result | Consequence |
|---|---|---|---|
| **X7** | Does dropping many balls in parallel work instead of peeling? | Noise alone **fails at every σ** (collapse or uniform scatter). Cosine-seeded **is circular** — stored memories are exact fixed points, so descent is a no-op. With **repulsion** it works (0.910). | Diversity must be *enforced*. Two places to put it: sequentially+exactly, or simultaneously+softly. |
| **X8** | *(teammate's proposal)* Rank by **vote count** — how many balls land in each basin? | Real signal: **0.621** vs 0.142 for the readout X7 tested. But banning the winner gives a gold 2nd answer only **2–10%** of the time. With cue **projection** instead: 1.000. | Corrected an overclaim: parallel descent *does* have a ranking signal. The lesson is that diversity belongs in the **dynamics**, not the **readout**. |
| **X9** | Does anything beat cosine under heavy cue corruption? | Peeling **0.961 vs 0.916** at 70% masking. | First sign of headroom anywhere. Pointed at corruption as the regime to test. |

## Phase 2 — Does any of it survive real embeddings? (2026-09-04)

| ID | Question | Result | Consequence |
|---|---|---|---|
| **X10** | Do the synthetic hyperparameters transfer to `all-mpnet-base-v2` (768-d)? | **No.** σ was not dimension-normalized: at d=768, `σ=0.35` is a perturbation ~10× the cue. First run reported `vote_proj=0.000` as if it were a measurement. | Every scale hyperparameter must be dimensionless before it crosses dimensionalities. Would have been published as a false negative. |
| **X11** | Does the β threshold (X3) transfer? | **No.** On real embeddings β=16 blends **377 memories**; β≥64 required. | X3 stands as a mechanism, dies as a number. Report the threshold per encoder, never as a constant. |
| **X12** | Is there headroom on real embeddings? | **Yes.** Unrelated memories sit at cosine **0.258** (not 0.0). Dense scores **0.248**, gold median rank 10/1043. | X6's problem is solved. Real anisotropy is what creates the room to improve. |
| **X13** | Does the X5/X7 ranking hold on real data? | **Inverted.** Peeling 0.268, repulsion 0.290. | Peeling is no longer "the correct readout". Likely because real embeddings are non-orthogonal, so projecting one gold out damages the others. |
| **X14** | Does repulsion beat dense, with correct statistics? | Naive bootstrap: **p=0.0005**. Cluster bootstrap over questions: **p=0.076**. | The naive test was pseudo-replication — 5 seeds reusing the same 37 questions. Cluster bootstrap is now mandatory. Power analysis: 136 questions needed, we had 37. |
| **X15** | Does it survive at 146 questions with **held-out** hyperparameter selection? | **No.** +0.042 (tuned=tested, n=37) → +0.027 (tuned=tested, n=146) → **+0.004, p=0.43** (dev/test split by thread). | The advantage was hyperparameter overfitting. **No readout variant beats dense top-k.** This is the current headline. |

**Where Phase 2 leaves the claim:** dense retrieval is a special case of the energy
formulation (X1), and *every departure we have tested changes only the readout* — G1, peeling,
repulsion, voting. None of them pays for itself (X15). That is a clean negative result, and it
is bounded in an important way: **we have never changed the landscape itself.**

**Standing limitation (stated, not hidden):** all Phase-2 numbers are on a purpose-built
146-question corpus written by us, not a public benchmark. It is internally rigorous
(dev/test by thread, cluster bootstrap, held-out hyperparameters) but external validity is
unestablished. LongMemEval / LoCoMo remain required for any headline number.

---

## Phase 3 — From retrieval to a world model (2026-09-05)

**The turn.** Everything above treats memory as a *static store* to be searched. Phase 3
asks a different question: what if memory is a **learned model of how experience unfolds**,
and retrieval is inference in that model?

An agent's episode stream is a trajectory through the world. A model trained to predict
*the representation of what comes next* is a world model over that trajectory. This is the
JEPA formulation — predict in embedding space, not in token space — and it is the one
mechanism that can retrieve by **association** (what co-occurs) rather than **similarity**
(what looks alike). Cosine provably cannot do this; it is the lightning→thunder case.

Critically, it slots into the existing formalism as the routing operator `R`, which the
energy core already has a hook for — the difference is that `R` becomes a *learned nonlinear
predictor* instead of a matrix, and descent becomes a rollout.

| ID | Question | Result | Consequence |
|---|---|---|---|
| **X16** | Do consecutive episodes in a storyline need association, or does cosine already find them? | **Cosine gets Recall@1 = 0.105**, median gold rank **24 of 1319**. Its errors are lexical collisions across unrelated storylines — a cottage-drive cue returns "forty minutes each way" from the apartment thread. Oracle-restricting candidates to the correct thread still only reaches 0.488. | **Gate passed.** "What happened next" is genuinely not a similarity relation. Enormous headroom, and a task cosine is structurally unable to do. |
| **X17** | Can a *parametric* JEPA predictor trained on some threads generalise to unseen ones? | **No, at any capacity.** rank-32: 0.784 on trained threads, **0.008 on unseen** (cosine 0.106). Swept rank 2→32; val Recall@1 was 0.000–0.037 everywhere. Diagnostic: displacement vectors between consecutive episodes are near-orthogonal (mean pairwise cosine **0.006**), top-10 PCs hold 24% of variance. | There is no global "next step" direction to fit, and 107 transitions in 768 dimensions cannot teach one. **A free-form learned map is the wrong inductive bias here.** |
| **X17b** | Does a *nonparametric* world model — transfer displacements from remembered transitions — do better? | **Yes.** k-NN displacement transfer, k and τ chosen on **val threads only**: Recall@1 **0.154 vs 0.106**, Δ +0.049 [+0.000, +0.100], p=0.032. **+46% relative** on held-out threads. | Carry the transitions, do not compress them into weights. First honestly-evaluated positive signal in the project. |
| **X18** | Does folding the predictor *into* the energy landscape beat keeping it external? | **No.** Routing inside descent: η=0.5 → 0.130, η=1.0 → 0.154 — which merely *recovers* the external predictor, since η=1 replaces the Hopfield step entirely. Never exceeds it. | **The integration claim is not supported.** Association helps; putting it inside the energy functional rather than beside it does not. This weakens the stated novelty against PAM and must be reported. |
| **X19** | Does rolling the world model forward recover multi-hop chains? | **Yes, strongly.** On 2-hop probes, Recall@5: cosine 0.262, one prediction step 0.400 (Δ +0.138 [+0.063, +0.219], p<0.001), **two-step rollout 0.446** (Δ **+0.185** [+0.076, +0.297], **p<0.001**). **+70% relative.** The model saw only 1-step transitions, so 2 hops requires *composing* the dynamics. | **The strongest result in the project.** This is where "world model" stops being a metaphor — it is inference by iterating learned dynamics, reaching memories similarity search cannot. |

**Where Phase 3 leaves the claim.** Two things separate cleanly:

- ❌ **The energy-integration claim is dead** (X18). Putting the predictor inside the landscape buys nothing over running it alongside. Say so.
- ✅ **The world-model claim is alive and significant** (X16, X17b, X19). Retrieval by *composing learned dynamics* beats similarity search on multi-hop association by +70% relative at p<0.001, with hyperparameters selected on held-out threads.

The paper's centre of gravity moves from *"energy-based retrieval"* to *"memory retrieval as inference in a learned world model, with dense retrieval as the degenerate zero-step case."* Proposition 1 still anchors it; the negative results (X15, X18) bound it; X19 is the positive claim.

**Also settled in Phase 3:** parametric vs nonparametric is not a detail. The parametric model looked spectacular on threads it had seen (0.784) and was worse than useless on new ones (0.008). Any result of this kind reported without a thread-disjoint split is meaningless.

