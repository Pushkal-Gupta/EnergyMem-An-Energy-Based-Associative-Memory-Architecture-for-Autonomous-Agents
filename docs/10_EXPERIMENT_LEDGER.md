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


---

## Phase 4 — The mode-averaging gate (2026-09-12)

**Why this phase exists.** `docs/12_ARCHITECTURE.md` makes three design commitments —
invariant I4 (*never average modes*), the mixture readout in module D, and the whole `Θ>0`
sampling regime of module M — that all rest on one claim: *the conditional next-state
distribution is multi-modal, so a point estimate lands between basins where nothing is
stored.* That claim had been measured in an earlier session and **never written into the
ledger**, so it was supporting three design decisions from outside the record. Phase 4
re-measures it from scratch. `experiments/x23_mode_diagnostics.py`, results in
`experiments/results/x23_mode_diagnostics.json`.

**Protocol change, applies to every entry below.** A single 50/25/25 thread split leaves
~60 hop-1 and ~30 hop-2 test probes, which cannot distinguish any effect at stake here.
These runs use **thread-level 4-fold cross-validation**: every storyline is in the test set
exactly once and is never in the train or val of the fold that tests it. `k_P, τ` for the
world model and `K, β, α` for the mixture are selected **per fold on that fold's val
threads**. The cue is excluded from the candidate set for every method identically.
Cluster bootstrap over threads.

| ID | Question | Result | Consequence |
|---|---|---|---|
| **X23** | Do averaged predictions land off-manifold? | **Barely.** Mean nearest-neighbour cosine: a real stored episode **0.482**, an averaged prediction **0.462**, a single un-averaged mode **0.619**. | The off-manifold gap is −0.021, not the large gap the design assumed. A single mode is *closer* to the manifold than a real episode is — which X23b explains, and not in the architecture's favour. |
| **X23b** | Does averaging do something **other** than collapse the distribution? | **Yes, and it is the dominant effect.** A remembered displacement has norm **1.157** against a unit-norm cue, so `Π[q + Δ_i]` is dominated by the displacement: the top-1 of a raw mode is a **training** episode **100.0%** of the time (bank prior 14.0%). The kernel average shrinks the step to **0.394**, ×0.34. | **The first version of this experiment was confounded and its result was an artifact of my own code.** Un-matched, the mixture lost to averaging by −0.115 at hop 1. Matching step size via `α` removed most of that gap. Any mixture-vs-average comparison that does not match the step measures the step. `modes()` now takes `α`; the doc is corrected. |
| **X24** | Is the conditional distribution actually multi-modal? | **Yes.** Mean mutual cosine among the top-3 modes **0.353**; **97%** of cues have modes that disagree (<0.5). | The premise is confirmed. There genuinely is a distribution for averaging to destroy — which makes X25 the more surprising. |
| **X25** | With step size matched, does keeping the modes beat averaging them? | **No — it loses.** Hop 1 (n=257 probes / 122 threads): cosine R@1 0.105, **averaged 0.140** (Δ +0.035 [+0.012, +0.061], **p=0.002**), mixture 0.078 (Δ −0.027). Mixture − averaged: **−0.062 [−0.094, −0.034], p≈1.000**. Hop 2 (n=135 / 112 threads): cosine 0.096, averaged 0.089, mixture 0.089 — **nothing beats cosine**, every interval spans zero. | **GATE FAILED. Invariant I4 is not supported on text.** Averaging the modes is not a failure to be fixed; on this data it is *what makes the predictor work*. The shrinkage it applies keeps the query in control of where the prediction lands, and the distributional treatment removes that. |
| **X25b** | Does X19 — the project's strongest result — replicate under this protocol? | **No.** X19 reported 2-hop Recall@5 **0.446 vs 0.262** (Δ +0.185, p<0.001, +70%). Here 2-hop R@5 is **cosine 0.207, averaged 0.230, mixture 0.237**, all intervals spanning zero. The 1-hop result **does** replicate: 0.140 vs 0.105, +33% relative, p=0.002 (X17b reported 0.154 vs 0.106, +46%). | **The headline claim is in question.** The original X19 script was lost before it could be committed, so this is a fresh reimplementation rather than a re-run, and the difference could lie in probe construction, the exclusion policy, or hyperparameter selection. It cannot be resolved by argument. **Until 2-hop is reproduced under a pre-registered protocol, X19 must not be cited as a result.** |

**Where Phase 4 leaves the architecture.**

- ❌ **I4, the mixture readout (module D), and the case for `Θ>0` (module M) lose their
  evidential basis.** Rung **R4 is not supported** and must not be built on. The multi-modality
  is real (X24); exploiting it distributionally still loses (X25).
- ⚠️ **Rung R3 survives at one hop only.** The nonparametric predictor beats cosine on
  "what happened next" (p=0.002) and that is now the project's only replicated positive.
- ⚠️ **X19 is suspended, not withdrawn.** Reproducing it is now gate 1.
- ✅ **The method worked.** A confound in our own implementation (X23b) produced a large,
  clean, entirely false negative on the first run; matching step size removed it. The
  finding survived the correction and reversed in magnitude but not in sign.

**Consequence for the build order.** `docs/12_ARCHITECTURE.md` §10 step 1 is complete and
**failed**. Steps 4 (`M.sample()`) and 5 (mixture scorer) were conditional on it and are
now **deferred** — the code exists and is tested, but nothing licenses turning it on. The
new step 1 is X19 reproduction under a pre-registered protocol; step 2 (the β-band re-run)
and step 3 (public-corpus probes) are unchanged and now carry more weight, because the
one-hop positive is all that is left to externally validate.

---

## Phase 5 — Is the 2-hop claim real, across more than one draw? (2026-09-15)

**Why this phase exists.** Phase 4 gate build order names a pre-registered X19
reproduction as gate 1. `experiments/x26_hop2_confirmatory.py` is that gate. It also asks a
question X25b could not: `eval/corpus.py`'s storylines are fixed hand-authored content —
`build(seed, ...)` only varies filler-sentence sampling and thread interleaving — so even a
careful single rerun is one specific noise draw. This phase repeats the identical X23/X25
protocol (same grids, same scoring functions, same cluster-bootstrap machinery, imported
verbatim rather than reimplemented) across **5 filler seeds × 3 independent thread-fold
partitions = 15 realizations per hop**.

**Decision rule, fixed before any result was computed:** a hop is CONFIRMED only if a
majority of the 15 realizations individually show a significant effect AND the
meta-analytic mean across realizations excludes zero. This rule turned out to be
**miscalibrated** — see the consequence below — and that miscalibration is reported
rather than quietly patched, per this file's own append-only rule.

| ID | Question | Result | Consequence |
|---|---|---|---|
| **X26** | Under the pre-registered rule, do hop 1 and hop 2 confirm? | **Neither, by the letter of the rule.** Hop 1: 4/15 realizations individually significant (averaged method). Hop 2: 1/15. Majority required 8/15. Both **NOT CONFIRMED** as declared. | The rule as written cannot detect a small, real, *consistent* effect — it is the same underpowered-single-draw problem the image-occlusion experiment hit at n=80, reintroduced by requiring per-realization rather than pooled significance. |
| **X26 (disclosed post-hoc, not a re-vote)** | What does the raw sign pattern across the 15 realizations show? | **Hop 1: 15/15 positive, 0 negative** (sign-test p≈6×10⁻⁵). Mean effect +0.031, meta-CI [+0.027, +0.035]. **Hop 2: 10 positive, 5 negative** (sign-test p≈0.30) — no consistent direction. Mean effect +0.017, meta-CI [+0.004, +0.030], but a lower bound that close to zero from 5 negative draws is fragile, not robust. | **Not pre-registered, so not treated as a confirmatory result** — flagged explicitly to avoid the exact post-hoc-reinterpretation trap this ledger exists to prevent. As a transparent secondary read: the 1-hop association effect now has two independent lines of evidence pointing the same way and should be treated as materially more solid than a single replication. The 2-hop composition claim shows no consistent direction across 15 independent tries, which is a *stronger* basis for treating X19 as dead than X25b's single null was — inconsistency across many draws is harder to explain away as bad luck than one null result is. |

**Where Phase 5 leaves the claim.**

- ✅ **1-hop association (R3) is now the project's best-supported result**, replicated once
  under rigorous protocol (X25) and shown sign-consistent across 15 further independent
  realizations (X26). Effect size is modest (~+0.03 R@5 absolute, ~+15-30% relative
  depending on draw) but the direction is not in serious doubt.
- ❌ **2-hop composition (X19) does not recover under any test applied so far.** Not
  underpowered — inconsistent in sign across realizations, which a larger sample would not
  fix. Downgrade from "suspended" to "no supporting evidence found across two independent
  gates (X25b, X26)."
- ⚠️ **Methodological note for future gates:** a pre-registered decision rule needs its own
  power check before being written down — "majority of realizations individually
  significant" was intuitive but wrong for this effect size. A pooled or meta-analytic test
  should be the primary pre-registered criterion next time, not a secondary one.
## Phase 5b — External validity (2026-10-07)

**Why.** Every composition-regime number in this project (X16–X19, X25) is measured on a
corpus we wrote. The one-hop positive is the project's only replicated claim, and its
external validity was unestablished — the standing limitation, and the one a reviewer was
most entitled to press. `experiments/locomo_assoc.py`, results in
`experiments/results/x26_locomo_assoc.json`.

**Design.** Same probe shape as X16, on LoCoMo. Cue = a conversational turn; gold = the turn
that followed it. Candidate set = the **full pooled bank of 5,882 turns across all 10
conversations**, so a prediction must beat distractors from nine unrelated conversations as
well as its own. Ids namespaced by `sample_id` (the bug that invalidated an earlier capacity
sweep). Split unit = **session** (272 of them); 4-fold rotation, every session in test
exactly once; world model fitted on train-session transitions only; `k_P, τ` and `K, β, α`
selected per fold on that fold's val sessions; cluster bootstrap over sessions.

| ID | Question | Result | Consequence |
|---|---|---|---|
| **X26** | Does the one-hop association result hold on a public benchmark? | **Yes, and more strongly.** n=5,233 probes over 272 sessions. Cosine R@1 **0.064**, world model **0.107**, Δ **+0.043 [+0.036, +0.050], p<0.0001** — **+67% relative**. R@5: 0.174 → 0.251, Δ +0.077 [+0.067, +0.087], p<0.0001. | **The standing limitation is discharged for one hop.** The effect replicates on data built by another group, at ~20× the probe count and 2.2× the cluster count, with a larger relative effect than either of our own implementations (+46%, +33%). This is now the project's strongest claim and the only one with external validity. |
| **X26b** | Does two-hop composition hold on LoCoMo? | **No.** n=5,013. R@1: cosine 0.043, world model 0.041, Δ **−0.002 [−0.008, +0.004], p=0.77**. R@5 shows a small positive (+0.012 [+0.004, +0.019]) an order of magnitude below the one-hop effect. | **X19 is resolved, negatively.** X25b suspended the 2-hop claim pending a pre-registered protocol; this is that protocol, on public data, at n=5,013. The 2-hop claim is **withdrawn**, not suspended. The original 0.446 figure must not be cited. |
| **X26c** | Does the mixture readout lose on public data too? | **It wins, but by a quarter as much.** Mixture R@1 0.075 (Δ +0.011 [+0.006, +0.016] vs cosine) against averaging's 0.107. So mixture beats cosine and still loses decisively to averaging (−0.032). | **X25 replicates in sign on public data.** Keeping the modes is better than nothing and worse than averaging them. I4 stays falsified; R4 stays unsupported. |

**Where Phase 5 leaves the claim.**

- ✅ **One-hop association is now an externally validated positive**, replicated in three
  independent implementations (+46% own corpus 50/50 split; +33% own corpus K-fold; **+67%
  LoCoMo K-fold**), on two corpora, one of them public.
- ❌ **Two-hop composition is dead.** Three measurements now: +0.185 (original, lost script),
  null (own corpus K-fold), null (LoCoMo, n=5,013). The original was an artifact.
- ❌ **I4 / mixture readout: falsified twice**, on both corpora.
- 📉 **Absolute numbers are much lower on LoCoMo** (cosine R@1 0.064 vs 0.105 on our corpus),
  i.e. the public task is harder — a 5,882-turn pooled bank with nine conversations of
  distractors, and speaker-alternating turns. The effect is *larger* in relative terms on
  the harder task, which is the direction that favours the mechanism.

---

## Phase 6 — Closing the unreproducible rows (2026-10-08)

**Why.** Two drivers were lost before they were committed, so two results in the paper had
no reproducible path: X17 (parametric predictor fails) and X17b (the one-hop positive under
a single 50/50 split, Table 3 row i). `world/jepa.py` and `world/knn_wm.py` both survived —
only the drivers were missing — so both were rewritten rather than left as caveats.

| ID | Question | Result | Consequence |
|---|---|---|---|
| **X17-r** | Does the parametric-failure result reproduce? | **Yes, and more starkly.** `experiments/x17_parametric.py`, 134 train transitions, 61 unseen storylines. Fitted Recall@1 reaches **1.000** at rank ≥8; unseen Recall@1 is **0.000** at ranks 8/16/32 and peaks at **0.016** (rank 2) against a cosine baseline of **0.131**. Diagnostic: displacements are near-orthogonal (mean pairwise cosine **+0.004**) and the top-10 principal components carry only **30.8%** of their variance. | **Closed.** The original reported 0.784 fitted / 0.008 unseen; the re-run is 1.000 / 0.000, i.e. the same finding with a cleaner separation. The paper now cites the reproducible numbers. A free-form learned map is the wrong inductive bias here, and that is what forces module P to be nonparametric. |
| **X17b-r** | Does Table 3 row (i) reproduce? | **No.** `experiments/x17b_single_split.py`, same 50/50 storyline split, `(k_P, τ)` chosen on dev only. Cosine R@1 **0.106** — identical to the original — but the world model reaches **0.130**, not 0.154, giving **+0.024 [−0.032, +0.083], p=0.238** over 123 probes and 61 storylines. The original recorded +0.049 [+0.000, +0.100], p=0.032 at **n=37**. | ⚠️ **The original row (i) is superseded.** Its n=37 does not match the 123 probes this split actually yields, so the two are not the same measurement and the discrepancy cannot be resolved — the original code is gone. The re-run value replaces it in the paper, marked n.s. |

**What this does to the claim.** Nothing, and that is the point worth stating. The one-hop
positive never rested on row (i): it is significant at n=257 on our corpus (p=0.002) and at
n=5,233 on LoCoMo (p<10⁻⁴). What the re-run shows is that a 123-probe split **cannot resolve
an effect of this size**, which is exactly what a real small effect looks like — the point
estimate stays positive and the interval shrinks as n grows:

| protocol | n | Δ Recall@1 | |
|---|---|---|---|
| episodic, 50/50 (re-run) | 123 | +0.024 [−0.032, +0.083] | n.s. |
| episodic, 4-fold | 257 | +0.035 [+0.012, +0.061] | p=0.002 |
| LoCoMo, 4-fold | 5,233 | +0.043 [+0.036, +0.050] | p<10⁻⁴ |

**A sixth protocol error, for the record.** The original row (i) was a significant result at
n=37 that does not survive re-measurement at n=123 on the same split. We do not know what
the original driver did differently. The lesson is the one Phase 5 already drew and this
makes concrete: **an uncommitted script is not a result.** Both re-runs exist precisely so
that no row in the paper depends on code nobody can run.

---

## Phase 7 — The LoCoMo QA task, families A–D of the backlog (2026-10-09)

**Why.** LoCoMo ships 1,986 QA pairs annotated with the dialogue turns that answer them.
`src/energymem/eval/locomo.py` has parsed them into `LocomoQuery` since Phase 5b and
**nothing had ever imported it** — all three LoCoMo scripts used only the turn bank. That
is 1,977 resolvable gold-annotated queries on a third-party benchmark, with a pre-labelled
difficulty axis (category 1 multi-hop, mean 3.13 gold turns, 95.4% cross-session;
category 4 single-hop, 1.06), unused. Phase 7 runs the whole of backlog families A–D on it.

**New shared machinery.** `src/energymem/eval/setmetrics.py` (full-set and mean-gold recall;
blocked permutation; two-level bootstrap) plus 10 fixture tests — every pre-existing metric
assumed a single gold item, so none of them could see a multi-target failure.
`experiments/locomo_qa.py` holds the banks, folds, methods and the sweep/selection
machinery. Protocol: leave-one-conversation-out (10 folds), hyperparameters chosen on the
nine training conversations only, conversation as the bootstrap cluster, primary metric
`gold@10`, primary test a paired permutation with signs flipped within conversation.
**Ten clusters is few and the intervals are correspondingly wide.**

### Family A — the task and the method

| ID | Question | Result | Consequence |
|---|---|---|---|
| **X27** | Is the single-query failure an expressivity limit or a query-formation limit? | **Query formation.** The dilution ceiling `sqrt((1+(m−1)ε)/m)` binds for **0.0%** of questions in every category (cat 1: ceiling 0.835 vs 10th-distractor 0.452), while the **actual** encoder query leaves a gold turn below the 10th distractor for **87.9%** of cat 1 and 37.5% of cat 4. | A certificate, not an impossibility result. There is large headroom and it is not blocked by the single-query form. The bound must never be written up as an impossibility theorem — at the measured ε it is vacuous. |
| **X28** | Is that headroom reachable by association? | **Yes, with an ORACLE anchor.** Residual Recall@10 (anchor excluded): query **16.6%** → anchor alone **24.2% [19.8%, 29.0%]**, +46% relative; cat 1 16.2% → 23.0%. Finite-β `logsumexp` (22.7–23.5%) does **not** beat the anchor alone (24.2%). | Upper bound only — the anchor is chosen using gold labels. Adding the query back as a mode *hurts*. The reachable headroom exists; nothing here shows it is reachable without labels. |
| **X29** | What is the real baseline? | **RRF, by a wide margin.** `gold@10`: cosine 0.527, BM25 0.531, MMR 0.527, **RRF 0.651** (Δ vs cosine **+0.124 [+0.098, +0.152], p=0.002**). BM25 vs cosine is null on `gold@10` (+0.004, p=0.87) despite winning `gold@1` (0.242 vs 0.170). **MMR is a no-op: +0.0004, p=0.91.** | **The bar is RRF at 0.651, not cosine at 0.527.** Every claim in this phase is measured against it. MMR being null is the one piece of good news for the mechanism question: diversity reranking is not what is on the table. |
| **X30** | Does the metric choice change the method ordering? | **Yes, on single-hop.** By `gold@1`: rrf > bm25 > cosine > mmr. By `full@10`: rrf > cosine > mmr > bm25. BM25 moves from 2nd to 4th. | Single-gold `R@1` and full-set recall are not interchangeable. Reporting only `R@1`, as every earlier table in this project does, can invert a ranking. |
| **X31 / X33** | Does two-stage expansion beat cosine? | **No. It backs off to the null in 10/10 folds, on every category.** With `K0=0` available to every method, per-fold selection chose the cosine null for mixture, mixture-averaged and Rocchio alike — no expansion setting beat plain cosine on the training side. Forced non-trivially (first run), Rocchio was *worse* than cosine on cat 1 (−0.017, p=0.016) and cat 4 (−0.027, p=0.012). | **Mode expansion does not work on this task.** Against RRF the gap is −0.124 [−0.152, −0.098], p=0.002. |
| **X32** | The pre-registered interaction (`PREREG_X32.md`). | **Prediction 1 NOT met; there is no effect to decompose.** Δ₁ = +0.0000, Δ₄ = +0.0000, interaction +0.0000, because both methods selected the null. | **I4 is not rescued by the multi-target regime.** The pre-registration's own outcome table assigns this reading: "Δ₁ ≈ 0 → I4 stays falsified." It now stays falsified on a third corpus, in the regime most favourable to it, with gold labels and n=281 multi-hop questions. |

> **A false positive this phase produced and then killed.** The first X31 run reported
> `mixture − mixture_avg = +0.051 [+0.020, +0.082], p=0.008` on cat 4 and +0.055 p=0.024 on
> cat 2 — an apparent I4 confirmation. It was an artifact of the grid. `K0` ranged to 10
> while the primary metric was `gold@10`; with `with_query=True` the `K0` feedback modes
> *are* cosine's top-10 and each scores 1.0 against itself, so they are lifted above
> everything else and **the top-10 set is pinned to cosine's by construction**. Selection
> picked `K0=10` for all 10 folds, mixture became a no-op scoring exactly cosine
> (`Δ = +0.0000`, zero-width interval — the signature), while `mixture_avg`, whose grid also
> lacked `K0=0`, was forced into a genuinely worse configuration. The "gap" was a no-op
> beating a handicapped baseline. Fixed by capping `K0 < k` and giving every method a
> reachable null; `x31_expansion.py` now asserts both before producing a number.
> **Seventh protocol error: a metric cutoff at or below the expansion depth is blind to the
> expansion, and a zero-width confidence interval means a bug, not a null.**

### Family B — geometry (`x5x_geometry.py`)

| ID | Result | Consequence |
|---|---|---|
| **X50** | Gold evidence sets are **tighter** than random size-matched sets: ε 0.518 vs 0.394 on cat 1 (76.1% of questions), against a bank background of 0.398. | This is *why* X27's bound is vacuous. High ε keeps the single-query ceiling high. The bound would bite on a benchmark whose evidence sets were near-orthogonal; LoCoMo's are not. |
| **X51** | Gold sets are linearly separable from the bank in **40/40** cases tested, on the pooled 5,882-turn bank against the 2,500 hardest distractors with d=768. | **Expressivity is definitively not the constraint.** Reinforces X27 with a non-vacuous test. ⚠️ A first version used a 400-distractor cap against d=768; with fewer constraints than dimensions feasibility is automatic by Cover's theorem, and the 100% it reported was an artifact of the cap. The cap must exceed d. |
| **X52** | Displacement coherence: global **+0.270**, local (within the cue's 10-NN) **+0.430**, ratio 1.6×. | Local exceeds global, so a local kernel is the right shape. **Not comparable to X17-r's +0.004** — that measured adjacent-turn displacements, this measures question→gold-evidence displacements. +0.270 is *not* near zero, and a global offset is not refuted by this measurement. |
| **X53** | Expansion gain is **not predictable** from geometry: Spearman ρ ≤ 0.064 for top-1 margin, score entropy, local coherence and top-1 score, all n.s. on cat 1. Mean gain at the probed setting is **negative** (−0.051 on cat 1). | The geometry-gated-α idea (backlog X53) has no basis. No pre-answer signal identifies the questions expansion would help, because at this setting it helps none of them. |
| **X54** | Residual gold within the kNN graph: at K=10, hop-1 24.2%, hop-2 18.2%, hop-3 13.8%, **unreached 43.8%**; at K=20, unreached 20.4%. | Roughly 40–60% of missed evidence sits within two association hops. The structure X28 exploits is real; the problem is selecting the anchor without labels. |
| **X55** | Cat 1 questions are more anisotropic than cat 4: top-10 PC variance 41.7% vs 36.6%; effective rank 69 vs 98. Bank turns 32.9% / 134. | Multi-hop questions occupy a lower-dimensional subspace than single-hop ones. |
| **X56** | Hubness present (k-occurrence skew 1.3–2.6, max in-10NN up to 88), and **hubs are enriched for gold** (50% of top-20 hubs are gold vs a 19–32% base rate). | Hubness is ruled out as the explanation for the failure — here the hubs are mostly the *right* answers, not distractors. |
| **X57** | Of top-10 false positives: **85.9%** same speaker as a gold turn, 21.3% same session, 10.6% within 2 turns (cat 1). | A speaker prior is useless (speakers alternate, so it is near-constant). This predicted X62's result before X62 ran. |

### Family C — the energy axis (`x6x_energy.py`)

| ID | Result | Consequence |
|---|---|---|
| **X60** | Top-1 **decision agreement** with cosine rises 0.0% (β=1) → 30.4% (β=16) → 94.9% (β=64) → **99.9% (β=512)**. But `gold@10` peaks at **0.397** (β=32) and settles at 0.356 — never reaching cosine's **0.527**. | Proposition 1 is confirmed **at the decision**: the argmax converges to cosine's. It is *not* confirmed for the ranking, because this driver ranks by similarity to the converged state ξ rather than to the query — a readout choice of the driver, not a refutation. Stated so the gap is not mistaken for evidence against Prop 1. |
| **X61** | `T>1` **never** helps, at any β: Δ(T=10 − T=1) is −0.0004 (β=1) to **−0.203 (β=16)**, p=0.002 at every β≥8. | Corollary 1 confirmed and strengthened — multi-step descent is not merely degenerate at high β, it is actively harmful at the β where the trajectory actually moves. |
| **X62** | **Every** strength-bias prior hurts, monotonically in scale: recency −0.077/−0.252/−0.408 at scale 0.05/0.15/0.50; centrality, session-size and length likewise (all p=0.002, except length@0.05 null). | Module G2 buys nothing on this task. Predicted by X57. |
| **X63** | **Every** learned routing hurts: diagonal R −0.029 (p=0.006) to −0.108; full R −0.211 to −0.471, all p=0.002. | `R=I` — the corner — is the best routing available. Module G3 buys nothing. The least-explored knob is now explored and it is empty. |
| **X64** | Matching pursuit: r=3 null (+0.003, p=0.83), r=5 −0.014, r=10 **−0.169** (p=0.002). | Deflation does not beat top-k. |
| **X65** | Adaptive β beats **fixed β=16** (+0.067, p=0.004 for β∼1/std). | ⚠️ Reads as a win and is not one: fixed β=16 scores 0.241 against cosine's 0.527. This is "less bad than a bad reference". Recorded to stop it being cited as a positive. |
| **X66** | Trajectory: ‖ξ₈−ξ₁‖ = 0.031 (β=1), **0.502 (β=16)**, 0.185 (β=64), 0.055 (β=256); top-1 changes after step 1 in 93.3% of cases at β=16 and **0.0%** at β=256. | Corollary 1 measured on the trajectory rather than argued from algebra. The band where descent does something is precisely the band where accuracy is worst (0.241 at β=16). |

### Family D — predictor variants (`x7x_predictor.py`)

World model fit on the nine training conversations' question→gold displacements, applied to
the held-out tenth. Banks are disjoint, so this is a genuine generalization test.

| ID | Result | Consequence |
|---|---|---|
| **X70 / X71** | Best cell is `mixture, K=5, α=0.10`: **+0.0063 [−0.0016, +0.0137], p=0.049**. The matched averaged collapse is **+0.0070, p=0.014** — i.e. **averaging is marginally better than keeping the modes.** At α=1.0 everything collapses (−0.094 to −0.179, p=0.002). | I4 falsified again, in sign, on a third corpus. ⚠️ **24 cells were tested; 3 reach nominal p<0.05 where 1.2 are expected by chance, and the Bonferroni threshold is 0.002. None survive.** Treated as null. Also `averaged` and `predict` are algebraically identical here and their rows are duplicates, not independent evidence. |
| **X72** | Rollout on cat 1: H=1,K=2 **+0.030 [−0.005, +0.070], p=0.098**; H=2 +0.017/+0.027 n.s.; H=3 negative. | The largest point estimate in family D, and still not significant at n=281. Consistent with X26b: composition does not help. H=1 being best is the one-hop story again. |
| **X73** | Step norms on this field: mean per-mode ‖Δ‖ **1.074**, ‖Σ wΔ‖ **0.722**, implied α **0.672**. | The implicit shrinkage is 0.672 here, not the 0.34 measured in X23b on the adjacent-turn field. α must be recalibrated per displacement field; it is not a constant of the method. |
| **X74** | Transition source: QA-derived +0.002 (n.s.); **adjacent-turn −0.023 (p=0.002)**; same-session −0.063 (p=0.002). | The label-free transition sources that worked in X26's next-turn task actively hurt on QA retrieval. The predictor does not transfer across task shape. |
| **X75** | Repulsive modes: 0.0 null, 0.3 null, **0.6 −0.035/−0.023 (p=0.002/0.006)**. | The principled version of diversity fails, consistent with MMR's null in X29. |

### What Phase 7 establishes

- ✅ **A genuinely new task is now wired up** — 1,977 gold-annotated LoCoMo QA queries, set
  metrics, cluster-correct inference, 24 experiments, 11 result JSONs.
- ✅ **The headroom is real and well-characterized**: not an expressivity limit (X27, X51),
  partly reachable in principle (X28, X54), and the nearest thing to a cause is
  query formation.
- ❌ **Nothing in the architecture closes it.** Expansion backs off to the null (X31/X32);
  every energy knob — `T`, `b`, `R`, β schedule, matching pursuit — is neutral or harmful
  (X61–X66); every predictor variant is null after multiplicity (X70–X75).
- ❌ **I4 stays falsified**, now on a third corpus, in the multi-target regime built to
  favour it, with the matched-mode comparison that isolates the readout.
- 📉 **The strongest method on this task is RRF**, a standard lexical–dense hybrid that
  predates this project, at `gold@10` 0.651 vs the architecture's 0.527.
- 🧹 Two bugs in **this phase's own code** were found and killed before they became claims
  (the `K0 ≥ k` metric blindness; the sub-dimensional separability cap), and one unresolved
  **merge conflict committed into this ledger** since Phase 5b was resolved by keeping both
  phases and renumbering the 2026-10-07 one to Phase 5b.

**Nothing from Phase 7 belongs in the submitted paper.** The paper's claims are unaffected:
it does not claim anything about LoCoMo QA evidence retrieval. If any of this is written up
it is a second paper, and its honest headline is that the reduction's corner is hard to
improve on and a lexical hybrid beats the architecture on a task the architecture was
designed for.

---

## Phase 8 — Families E, F, G: the controls that should have come first (2026-10-09)

Phase 7 published a negative result and then listed its own controls as unrun. Family G's
charter in the backlog is one sentence — *"run these before believing anything in A or
D"* — and Phase 7 did the believing first. Phase 8 runs them, triages the two families
that families A–D had already made unanswerable, and corrects two things Phase 7 believed
about its own instruments.

Drivers: `experiments/x9x_controls.py`, `experiments/x8x_robustness.py`,
`experiments/x90_cost.py`. Results: `x9x_controls.json`, `x8x_robustness.json`,
`x90_cost.json`.

### The triage: four things killed on paper, not by measurement

A result can make an experiment *unanswerable* rather than merely unpromising, and that
distinction is worth being strict about. X84, X85 and X93 were each written as
*"interesting if the gain …"*. X31 found the gain is exactly zero — at the fold-selected
configuration the expansion **is** cosine, in 10/10 folds, in every category — so the
robustness of the gain, the growth of the gain and the survival of the gain under an ANN
index are all questions about a quantity that does not exist. X90's stated headline
("accuracy per millisecond, the quantifiable-impact number") presupposed the same thing.

| killed | reason | the live remainder, relocated |
|---|---|---|
| X84 query perturbation | its selling point was "more robust **than cosine's**"; the method *is* cosine | → X29: is **RRF's** +0.124 paraphrase-robust? |
| X85 bank contamination | "the gain grows"; and padding changes the task, not its size | → X37 (scope), X92 (latency, ran) |
| X93 approximate NN | nothing to survive; an index is a deployment choice | → benchmark on deployment |
| X90's headline | accuracy per ms can only fall at zero accuracy delta | → ran with the sign reversed |

Three further arms are **not runnable as written**, which is a different thing and worth
recording as such: X82's un-normalized arm (the frozen encoder normalizes, and changing
that invalidates every number in the project), X83's window and summary arms (LoCoMo
annotates evidence at *turn* level, so a window containing a gold turn is neither hit nor
miss), and X87's leave-one-session-out (a bank **is** a conversation, and 95.4% of cat-1
gold sets span sessions, so dropping a session changes the task rather than the split —
fold *count* was swept instead).

### Family G — every control passes

| ID | result |
|---|---|
| **X38 shuffled gold** (the P0) | **PASS.** Gold replaced by a random set of matched size: `gold@10` = 0.0179 against a closed-form chance of 0.0172. Gold sets *permuted between questions of the same conversation* — which keeps the real gold-set geometry and breaks only the question↔evidence link: 0.0284, i.e. ~1.6× chance and ~20× below the real 0.527. **Family A's numbers are citable.** |
| X95 random modes | Real modes beat size-matched random bank turns in **9/9** forced configurations, by +0.27 to +0.42 (`K0=5, β=64`: 0.467 vs 0.042). So the modes carry real signal. **And real modes still never reach cosine** — best cell −0.051. The modes are informative; the readout cannot spend them. |
| X96 label-leakage audit | **PASS**, two ways. Static (AST walk): no `s_*` scoring function touches a label-bearing attribute. Functional, which is the one that matters: destroy `gold`, `answers` and `cats`, re-score, and all **7** methods return bit-identical vectors. Selection verified to aggregate train indices only. |
| X97 exclusion symmetry | **PASS.** Every method scores all *N* items — no masking, no `-inf` survivors, no NaN. The QA task has nothing to exclude (the query is not a bank item), so the real asymmetry risk is ties: 1 tie straddling the k=10 boundary in 40 probes, BM25 only. |
| X98 reduction ladder | **10/10 rungs exact** on all 1,977 queries. |
| X99 metric cross-check | **0 mismatches** across 1,326 (query, method) rankings × 4 metrics against independently written loop-based implementations. |

### Correction 1 — `p = 0.002` was a floor all along

The blocked permutation flips **one sign per conversation**. With ten conversations the
null distribution therefore has exactly **2¹⁰ = 1024** points, and the all-flipped pattern
always ties the observed statistic, so the smallest attainable two-sided p is

    2 / 1024 = 0.00195

`exact_blocked_permutation` now enumerates it instead of sampling it, and X86 confirms the
Monte Carlo was already converged (20 seeds, p ∈ [0.0014, 0.0024] around the exact
0.00195). **Every "p = 0.002" in Phase 7 is this floor.** It says "as extreme as the
blocking permits", not "p = 0.002 worth of evidence", and from here it is written
`p = 2/1024 (floor)`. Nothing in Phase 7's conclusions moves — the floor was reached by
the RRF margin and by the harmful energy knobs, both of which were already large — but the
*reported precision* was an illusion, and ten clusters cannot produce more.

### Correction 2 — Proposition 1 is a top-1 statement, and its guard test is misnamed

`tests/test_reduction.py::test_reduction_to_cosine_topk` tests `k = 1`. Its own docstring
says "must reproduce top-1 cosine exactly"; the function name says `topk`. X98 measured
both ends on real LoCoMo banks:

- `descend(T=1, β=512, b=0, R=I)` top-1 == cosine top-1: **100/100**
- the same state's top-10 == cosine top-10: **0/100**

This is not a failure, it is the proposition read correctly. At β→∞ the descent **returns
the nearest stored pattern**, so ranking by similarity to the converged state ranks *that
pattern's* neighbourhood, not the query's. The reduction is exact at k=1 and silent above
it — which is precisely **why `T > 1` hurt in family C** (−0.203 at β=16). Family C
recorded that as an empirical finding; it was a prediction of Proposition 1 that nobody had
read off the proposition. The test name invites the misreading and is left alone only
because the paper's Appendix A.5 cites it; it is flagged here instead.

### Family E — the null survives everything that could be varied

| varied | result |
|---|---|
| **X80 encoder** | mpnet (incumbent), **bge-base-en-v1.5 (pre-declared primary in `PREREG_X32.md`, before any of this ran)**, MiniLM-L6-v2: **10/10 null folds under all three.** The null is not encoder-specific. bge is the better encoder in absolute terms (cosine 0.564 vs mpnet's 0.527). |
| **X81 dimension** | PCA to d ∈ {64, 128, 256, 384, 768}: null at every d. Two cells selected one non-null fold out of ten and that fold made things **worse** (−0.008, −0.013). |
| **X82 normalization** | L2, centered, whitened, all-but-the-top-3. See the lead below. |
| **X83 chunking** | Dropping the speaker prefix costs cosine 0.527 → 0.379. |
| **X86 seeds** | The retrieval path has no seed and is bit-deterministic across re-runs. Resampling seeds move p within [0.0014, 0.0024] and CI endpoints by ±0.003. |
| **X87 folds** | Null at **2, 4, 5 and 10** folds alike. And with **no** selection at all (`K0=3, β=16` fixed) the mixture is **−0.0987 [−0.1251, −0.0730]**, p = floor. Selection is not hiding an effect; the non-null configurations are actively harmful. |

**X81 also closes X51's open caveat.** X51 existed with a dimension warning: at d=768
against n=588 per conversation, separability is automatic by Cover's theorem. Pooled gold
sets are now separable **15/15 at every d including d=64**, where 5,882 pooled points
vastly outnumber the dimensions. Separability is a property of the gold sets, not of the
ambient dimension. The expressivity question is settled.

### The one lead, and why it is only a lead

Expansion is selected over the null in exactly two places, and both are regimes where the
representation was **first damaged**:

| regime | cosine | mixture | Δ (95% CI) | exact p | null folds |
|---|---|---|---|---|---|
| whitened, all categories | 0.5160 | 0.5220 | +0.0060 [−0.0105, +0.0224] | 0.32 | **0/10** |
| whitened, cat 1 | 0.2730 | 0.2753 | +0.0023 [−0.0297, +0.0367] | 0.86 | **0/10** |
| text-only chunks, cat 1 | 0.1534 | 0.1627 | +0.0093 [−0.0290, +0.0454] | 0.50 | **0/10** |

Not one is significant, every interval covers zero, and in both regimes the
damaged-plus-expansion score sits **below the undamaged baseline** (0.522 < 0.527,
0.163 < 0.329). What it suggests, as a conjecture and not a result: **mode expansion is
selected only where cosine has been crippled, and then it recovers a fraction of the
damage rather than adding anything of its own.** Chasing it requires a pre-registration
whose endpoint is an absolute win over the *undamaged* baseline — a win over the cripple
is not a result, and writing it up as one would be the eighth protocol error.

A side observation from the same table: whitening and all-but-the-top both *raise* RRF's
cat-1 margin to nominal significance (+0.050 p=0.039; +0.055 p=0.045) from +0.032 p=0.17.
Flattening the dominant directions makes the lexical channel matter more. Untested as a
mechanism.

### Family F — the cost of buying nothing

Per query, 419-turn bank, mpnet d=768, laptop CPU (hardware in the JSON):

| method | ms | × cosine | accuracy it buys |
|---|---|---|---|
| cosine | 0.012 | 1.0× | — |
| energy descent T=1 | 0.045 | 3.7× | nothing (top-1 identical, X98) |
| rocchio K0=3 | 0.061 | 5.1× | worse when forced non-trivial (X33) |
| mixture_avg K0=3 | 0.091 | 7.7× | marginally **better** than the mixture (X70–X75) |
| **mixture K0=3** | **0.187** | **15.7×** | **nothing — selection picks K0=0** |
| energy descent T=10 | 0.304 | 25.5× | −0.203 at β=16 (X61) |
| **RRF, true cost** | **0.415** | **34.8×** | **+0.124 `gold@10` — the only method that buys anything** |
| MMR pool=50 | 1.897 | 159× | +0.0004, p=0.91 |
| cross-encoder top-50 | 280 | ~23,500× | cost proxy only; the cached model is an NLI head, no accuracy claim |

Memory (X91): the expansion path stores **nothing** extra — its modes are bank vectors it
already holds, so its entire cost is time. Separately, the project stores float64
embeddings at 6,144 B/memory: 2× float32 and 8× int8, for no measured benefit.

**And the scar repeated itself in miniature.** The first X90 run reported BM25 at
**0.000 ms/query, 0.01× cosine**. `locomo_qa.Bank.bm25()` computes the score matrix once
and caches it, and the warm-up call paid for it — so the timer was measuring an array
index. This is the same failure mode as the paper's claimed 8× that measured 21×: a cost
number that was never a clock on the thing being claimed. Timed against the index
directly, BM25 is 0.361 ms/query with a 3.8 ms one-off build, and RRF's honest cost is
34.8× cosine rather than 4.5×. **If a cost number looks too good, check what is cached.**

### A nuance Phase 7's headline hid

RRF's +0.124 is an **all-category** number, and it is driven by single-hop. On cat 1
(multi-hop) RRF's margin is +0.032 p=0.17 under mpnet, **−0.007 p=0.61 under bge**, and
+0.034 p=0.03 under MiniLM. So: *RRF beats the architecture overall and on single-hop
decisively (+0.114, p = floor), and on multi-hop its advantage is not established.* Say it
that way. Nobody has a method that works on multi-hop — which is the honest state of this
task, and a better framing for a second paper than "RRF wins".

### What Phase 8 establishes

- ✅ **Phase 7's negative is no longer provisional.** It survives three encoders, five
  dimensions, four normalizations, two chunkings and four fold counts, with every control
  passing: the metric rewards nothing structural (X38), no method sees the labels (X96),
  the candidate sets are symmetric (X97), the ladder reduces exactly (X98), and the metrics
  agree with independent implementations (X99).
- ✅ **Expressivity is settled.** X81 closes X51's dimension caveat at d=64.
- ✅ **The modes are not noise** (X95: +0.27 to +0.42 over random) — and that makes the
  failure sharper, not softer. The information is there and the readout cannot use it.
- 📉 **Two instruments were over-read**: the p-value floor and the top-1 scope of
  Proposition 1. Both corrections tighten the writing without moving a conclusion, and the
  second one *explains* a family C finding that had been recorded as a surprise.
- ⚠️ **One lead, pre-registration required**: expansion is selected only on damaged
  representations and never recovers past the undamaged baseline.
- 💸 **The architecture's path costs 15.7× cosine for an identical ranking.** The only
  method on the board that buys accuracy charges 34.8× for it.

**Nothing from Phase 8 belongs in the submitted paper either.** The two corrections are
about this project's instruments, not about the paper's claims; Proposition 1 as *stated*
in the paper is a top-1 claim and is exactly right. The one thing worth carrying into a
future revision is the `p = 2/1024` floor: any ten-cluster blocked-permutation p-value in
this project should be written as a floor, not as a measurement.

**Families A–G are closed.** The only P0 left is X102 (HotpotQA / 2WikiMultihop) — after
Phase 8 there is nowhere else on LoCoMo to look.

---

## Phase 9 — Families H and I: other corpora, other modalities (2026-10-09)

**Why.** After Phase 8 there was nowhere left to look on LoCoMo. The one live hypothesis was
X32's: LoCoMo's evidence is too similar (ε=0.52) for a mixture to help, so the effect
could exist on corpora built for multi-hop. Family I carried the paper's own open item:
does the image-occlusion positive (C3) survive on text?

**Protocol.** `experiment-backlog/PREREG_HI.md` was **committed alone (18e7150) before any
of these corpora was opened** — the fix for PREREG_X32's "written before, committed
with" caveat. Each corpus goes through the unchanged `locomo_qa.py` methods and grids via
`experiments/corpus_qa.py`, with leave-one-group-out selection on the training side only
and per-fold picks in every JSON. Drivers: `x10x_corpora.py`, `x101_msc.py`,
`x102_checks.py`, `x11x_modalities.py`. One deviation is logged in the prereg: the
X102 robustness checks were chosen **after** seeing X102, so they are exploratory.

**Environment.** This was a fresh checkout on a different machine (Apple M3). Before
anything new ran, cosine `gold@10` on LoCoMo reproduced Phase 7's 0.5269 exactly.

### Family H — the second (to sixth) corpus

| ID | Question | Result | Consequence |
|---|---|---|---|
| **X102 HotpotQA** | Does expansion help on purpose-built multi-hop? (2,000 dev q, 100 q pooled per bank, ~1k paragraphs, 20 clusters) | **Yes, the first time in this project.** Bridge (n=1,591): the mixture is selected over the null in **0/10** folds (K0∈{1,3}, β=16). Cosine 0.889 → mixture **0.912**, Δ **+0.023 [+0.010, +0.037]**. It beats Rocchio and averaging (I4 Δ +0.006 bridge, **+0.009 [+0.002, +0.016] all**, p=0.0025). Comparison (n=409) backs off to null 10/10. **RRF is still higher: 0.924, and mixture − RRF = −0.017 [−0.030, −0.005] (all).** | **P-H1 met on HotpotQA, P-H3 met.** It holds only against cosine and the averaging baselines. RRF, the one lexical method, still wins in absolute terms. |
| **X102 2Wiki** | Same, plus the within-corpus cardinality axis m=2 vs m=4. | **Large gains, organized by hop *type* rather than by m.** Inference: cosine 0.724 → mixture **0.894** (+0.171), beating RRF by **+0.094 [+0.045, +0.140]**. Compositional: +0.067, with **averaging better than the mixture** (I4 −0.022). Comparison: null 10/10. **Bridge_comparison (m=4): null**. All: I4 **+0.027 [+0.017, +0.036]**. **P-H2's interaction comes out reversed: Δ(m=4) − Δ(m=2) = −0.023 [−0.044, −0.004].** | **P-H1 met, P-H2 not met; it is the opposite sign.** The gain sits on *chain* questions, where the second document is reachable from the first rather than from the question, and not on questions with more targets. |
| **X102-checks** (exploratory) | Does X102 survive a disjoint sample, the bge encoder, banks ½× and 2×, and random modes? | **The expansion-over-cosine gain survives all of them**: HotpotQA bridge +0.020 to +0.026 in every variant; 2Wiki inference +0.10 to +0.19 and compositional +0.07 to +0.17. **Random modes lose by +0.13 to +0.39 (HotpotQA) and +0.16 to +0.36 (2Wiki), 9/9 cells each.** **The I4-specific part does not survive**: HotpotQA's mixture-over-averaging advantage drops to −0.000 under bge and +0.004 n.s. at 200 q/bank. Mixture − RRF flips sign with the encoder (−0.017 mpnet, **+0.021 bge**). | **The robust finding is association-driven expansion on chain-shaped multi-hop questions.** Whether the modes are kept apart (mixture) or averaged is a second-order, encoder-dependent detail. |
| **X103 QASPER** (substitutes for QuALITY/NarrativeQA, which have no evidence labels; declared in the prereg) | Long scientific documents, bank = one paper (~47 paragraphs). | **Expansion hurts where m>1.** Multi-evidence (m=3.31): mixture **−0.027 [−0.049, −0.010]** vs cosine, I4 −0.041. Single-evidence: +0.036 (p=0.04). RRF best overall: **0.659 vs cosine 0.593**. Interaction **−0.036 [−0.070, −0.001]**. | A third corpus where more targets do not help a mixture. With a 47-paragraph bank the top-K0 modes are mostly redundant neighbours. |
| **X104 BEIR** | SciFact (300 q), NFCorpus (323 q; m≈41). | SciFact: **RRF wins (0.832 vs cosine 0.790, p=0.025)**; the mixture is n.s. NFCorpus multi-relevant: **mixture +0.017 [+0.008, +0.027], p=0.0003**, which ties RRF (+0.008, n.s.); I4 +0.011 [+0.002, +0.022]. | PRF-style expansion helps on the one corpus with huge relevant sets. It does not beat RRF on either corpus. |
| **X105 transfer** | Do LoCoMo-selected configs transfer? | **For expansion, trivially, because LoCoMo selected the null.** Transferring K0=0 gives up the entire in-corpus gain (HotpotQA bridge −0.023, 2Wiki inference −0.171, 2Wiki all −0.043). RRF's k=10 transfers within 0.012 everywhere. | **P-H5 not met for expansion.** The LoCoMo null is a property of LoCoMo, not of the method. Any deployment would have to tune on its own corpus. |
| **X101 MSC** | Does X26's one-hop association replicate on a second public corpus? (30,320 pooled turns, 2,505 sessions, 501 dialogues) | **Yes.** Hop 1, n=27,815: cosine R@1 **0.049**, world model **0.065**, Δ **+0.016 [+0.014, +0.018]**, p<10⁻⁴ (**+33% relative**); R@5 +0.029. Mixture +0.006, again below averaging (as in X26c). Hop 2: +0.005 [+0.003, +0.006], small but positive, where LoCoMo was null. | **P-H4 met. The project's one positive claim now replicates on two public corpora**, here with a bank 5× LoCoMo's. The mixture-loses-to-averaging pattern (I4 falsified) replicates too. |
| **X100 LongMemEval-S** | Chat-assistant memory, each question with its own ~490-turn haystack (479 q with gold turns, question = cluster). | **Null for expansion, decisive for RRF.** Mixture − cosine **−0.007 [−0.032, +0.018]** (all); multi-session −0.012; knowledge-update −0.049 [−0.083, −0.014]. Interaction (multi-session − single-session) **+0.002 [−0.040, +0.045]**. **RRF − cosine +0.156 [+0.125, +0.189]** overall, **+0.255** on temporal-reasoning, +0.218 on knowledge-update; mixture − RRF −0.163. | **P-H1/P-H3 not met here.** The second conversational-memory corpus shows LoCoMo's pattern again, more strongly: the lexical signal (dates, names, exact phrases) is what memory QA over chat needs, and association adds nothing to it. Expansion helps on encyclopedic chains (X102), not on chat. |

### Family I — other modalities

| ID | Question | Result | Consequence |
|---|---|---|---|
| **X110 image arm** | Re-run the C3 image result with a k-NN class-vote control. | **Reproduced exactly, then explained away.** Energy − cosine **+0.0135 [+0.0055, +0.0219]**, identical to `image_holdout.json`. The k-NN vote (k=10 chosen on the tune half) gets **+0.045 [+0.035, +0.055]** vs cosine, and **beats energy descent by +0.031 [+0.023, +0.040]**, at every occlusion level (30/50/70%). | **The C3 positive is class smoothing, and a weak form of it.** Cosine top-1 is 1-NN classification, and the task scores class match. Energy descent recovers about a third of what a plain k-NN vote gets. The paper's "memory" framing for C3 does not survive this control. |
| **X110 text arm** | Does span occlusion of text behave like image occlusion? (Yahoo Answers, 10 classes, 400-text bank, 4,000 held-out × 3 levels) | **No.** Energy − cosine **+0.003 [−0.000, +0.006]**, n.s., with deletion and with `[MASK]` alike. k-NN gains **+0.104** (deletion) and **+0.111** (`[MASK]`). | **Prediction not met.** In the backlog's words, "the image result may not be about memory at all", and that is now the reading: on text the energy path does nothing, and the smoothing it partially captured on images is available in full from k-NN. |
| **X111 image mosaics** | X32's interaction with m set by construction: 2×2 mosaics of m∈{1..4} bank images, gold = the sources. | **Null for the mixture at every m.** It backs off to cosine 10/10 in every stratum, so the interaction is exactly 0. **MMR wins: +0.070 / +0.069 / +0.068 at m=2/3/4** (p at the 2/1024 floor). | **Prediction not met. This is PREREG_X32's kill criterion: on constructed multi-target retrieval the useful mechanism is diversity, not association.** |
| **X112 cross-modal** | CLIP ViT-B/32, Flickr8k test. t→i (m=1) vs i→t (m=5). | t→i: everything backs off to null. i→t: **Rocchio +0.019** and averaging +0.012 beat cosine; the mixture is n.s. (+0.004); **I4 −0.008 [−0.016, −0.001]**. Interaction (i→t − t→i) **−0.008 [−0.013, −0.004]**. | **Prediction not met, reversed.** The five gold captions are near-paraphrases (high ε), the regime where averaging should and does win. |

### What Phase 9 establishes

- ✅ **One-hop association has external validity on two public corpora** (LoCoMo X26, MSC
  X101). This is still the project's strongest claim.
- ✅ **Two-stage expansion works on chain-shaped multi-hop text** (HotpotQA bridge, 2Wiki
  compositional/inference). It survives a disjoint sample, a second encoder and two
  bank sizes, and random modes lose to it by large margins. This is new, and it is a
  method result rather than a mechanism result: Rocchio (averaged PRF) gets much of it
  too.
- ❌ **I4 is not resurrected.** The pre-registered cardinality interaction came out
  **reversed** on 2Wiki (m=4 < m=2), QASPER (multi < single) and Flickr8k (m=5 < m=1),
  and was null on mosaics. Keeping modes apart beats averaging on some strata and loses
  on others, and the advantage disappears under bge. **What predicts a gain is hop type
  (is the second piece of evidence reachable from the first?), not m.**
- ❌ **C3's framing must change.** The image positive reproduces, but a k-NN class vote
  beats it by 3 points, and on text the energy path is null. The honest reading is
  "nonparametric class smoothing, partially captured by energy descent", not "memory".
- 📉 **RRF stays the bar in most places, and on conversational memory it is the only thing
  that works**: +0.124 on LoCoMo and +0.156 on LongMemEval, where expansion is null. It loses to
  expansion on 2Wiki and to the mixture under bge. It wins on HotpotQA (mpnet), QASPER,
  SciFact and LongMemEval.

**Families H and I are closed.** Family J needs an API key and was not run.
