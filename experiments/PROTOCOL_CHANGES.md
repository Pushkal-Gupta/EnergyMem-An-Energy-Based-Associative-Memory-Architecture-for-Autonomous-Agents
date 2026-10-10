# Protocol Deviations

The protocol in `docs/04_EXPERIMENT_PROTOCOL.md` is pre-registered. Any change after
seeing results gets logged here with a date and a reason. Reviewers respect a logged
deviation and punish a silent one.

| Date | Change | Reason |
|---|---|---|
| 2026-08-11 | Stage A moves from synthetic to real sentence embeddings in Week 1 | Day-0 finding F6: cosine scores 1.000 on every synthetic condition, so there is no headroom to measure anything |
| 2026-08-11 | DegradeBench D5 is no longer the G1 discriminator | Day-0 finding F4: single-state descent is structurally *worse* than top-k for multi-target queries. D5 now motivates residual peeling instead |
<<<<<<< Updated upstream
| 2026-09-03 | Stage-A real-embedding validation run on **image** data (Imagenette-160, ResNet18 features) instead of sentence embeddings | Fastest path to real (non-synthetic) embeddings ahead of Review 2; also matches the project's actual framing of Hopfield-network memory over image/ImageNet-style data. Text/sentence-embedding validation (per `04_EXPERIMENT_PROTOCOL.md` Stage A) is still owed and not covered by this run. |
| 2026-09-03 | Added a capacity sweep (N=100..4,000) not in the pre-registered protocol | Ad hoc follow-up to the occlusion result: needed to check whether a fixed beta=32 (tuned informally at N=300) generalizes to larger memory banks. It does not -- the occlusion-recovery advantage over cosine shrinks/reverses past ~N=800. Finding: beta requires re-tuning (or an adaptive schedule) as corpus size grows; this was not anticipated in the original plan and is now a candidate future-work item rather than a settled result. |
| 2026-09-03 | Added a confidence-calibration test (retrieval margin vs. correctness) not in the pre-registered protocol | Exploratory addition to test whether the energy margin could serve as an uncertainty signal for an agent. Result is negative for both cosine and energy retrieval on this run (near-zero / negative correlation, non-monotonic accuracy across margin quartiles) -- reported as-is rather than dropped, per the project's rule that negative results ship. |
| 2026-09-05 | Per-memory beta (`memory/multibeta.py`) tried and rejected | Attempted fix for the capacity limit: set each pattern's beta_i from its nearest-neighbour gap. Failed badly (Recall@1 0.150 vs 0.471 for fixed beta at N=4000). Diagnosis: in `softmax_i(beta_i * s_i)` a heterogeneous beta_i rescales pattern i against every other candidate, so it controls basin *depth*, not just *sharpness*; crowded patterns win on multiplier magnitude rather than similarity. Retained in-tree as a documented dead end, not deleted. |
| 2026-09-05 | E6 brought forward: LoCoMo (ACL 2024) evaluated now rather than in Weeks 2-3 | The image study could not settle the agent-memory claim, and `04_EXPERIMENT_PROTOCOL.md` already designates LoCoMo/LongMemEval as the real test (RQ6). 1,977 questions over 5,882 turns, MiniLM encoder, retrieval scored against gold `evidence` turn ids. |
| 2026-09-05 | **Invalidated result**: the first LoCoMo capacity sweep (`exp_locomo_adaptive.py` part B) is withdrawn | Data bug. LoCoMo `dia_id`s are unique only within a conversation: 5,882 turns carry 1,033 distinct ids, and 132/134 of conversation 0's gold ids also occur in other conversations. Pooling distractor banks without namespacing let unrelated turns satisfy gold ids and count as hits, inflating scores in proportion to the distractors added -- the exact axis under test. Its "adaptive beta improves as the bank grows" conclusion was an artifact. Fixed by namespacing ids with `sample_id` in `eval/locomo.py`; corrected sweep shows adaptive beta *degrades* with N (0.250 -> 0.231), more gracefully than fixed beta (0.244 -> 0.216). Invalid plot deleted rather than kept. |
| 2026-09-05 | Added `anchor` to the descent (`memory/adaptive.py`), superseding the unanchored comparison | Without an anchor the state is replaced by a blend of stored patterns on step 1 and the cue never re-enters, so readout ranks by proximity to the blend. The earlier comparison therefore tested "is a blend of memories a better query than the query", not "does descent improve retrieval". `anchor=1.0` reduces to exact cosine (verified 196/196), making the baseline a point in our own parameter space. |
| 2026-09-05 | Corrected claim: adaptive beta is **better** than fixed beta, not worse | The earlier ranking was confounded by `pp_start=32`, which forces step 1 to average ~32 memories -- the broad-start annealing F2 already flagged as harmful. With `pp_start=4`, pure descent goes 0.201 -> 0.252 Coverage@5, above fixed beta=32 (0.236). Both still lose to cosine (0.336). |
| 2026-09-05 | Coverage@k denominator changed to `min(k, |gold|)` | With k=5 and up to 19 gold turns, dividing by `|gold|` capped multi-hop coverage at 5/19 and deflated absolute numbers. Affects all methods identically, so it changes reported magnitudes, not the ranking. |
| 2026-09-06 | **All image-modality "tie"/"negative" verdicts before this date are WITHDRAWN as underpowered** | Three compounding methodology errors, all ours. (1) POWER: only 80 distinct val images, inflated to "720 queries" by reusing them across 3 occlusion levels x 3 seeds; those are correlated, so effective n was ~80 and bootstrap CIs were ~0.072 wide while the differences under test were ~0.014 -- nothing was resolvable in either direction. (2) WRONG TEST: methods were compared via independent marginal CIs although all methods score the *same* queries; the shared difficulty is common variance that cancels only under a paired test. (3) SELECTION BIAS: beta was tuned on the same images it was then scored on, across 7-10 candidate values. Superseded by `scripts/confirm_holdout.py`. |
| 2026-09-06 | Beta sweep re-run densely through the transition zone; earlier "plateau above beta=32" reading corrected | The old sweep jumped 16 -> 32 and never sampled between. Dense sampling shows a smooth rise (12: 0.100, 20: 0.289, 28: 0.487, 40: 0.510, 48: 0.514) and, critically, that beta=512 reproduces cosine to three decimals -- the high-beta "plateau" is the method degenerating into the baseline (Proposition 1), not a working regime. beta has leverage only around 16-48, which is why per-pattern beta values of 51-88 (`multibeta`, `beta_training`) were inert by construction: they were searched outside the zone where beta can matter. |
| 2026-09-06 | **CONFIRMED POSITIVE RESULT** (images): energy descent beats cosine under pattern completion | Protocol: beta selected on a disjoint TUNE half (1,948 images), single pre-committed paired comparison on the HELD-OUT half. n=5,847 query instances from 1,949 distinct images, the full Imagenette val split. beta=40 vs cosine: **+0.014, paired 95% CI [+0.006, +0.021]**, wins 287 / losses 208. Per level: 30% +0.003 (n.s.), 50% +0.017 (sig), 70% +0.021 (sig) -- the advantage grows with corruption severity, as the completion mechanism predicts. Tune-half estimate (+0.016) matched held-out (+0.014), indicating no residual selection bias. |
| 2026-09-06 | The LoCoMo (text) negative result is NOT affected by the above | Its effect size (Coverage@5 0.336 cosine vs 0.252 pure descent) is ~10x the noise floor that obscured the image effect, and the anchor sweep gives a monotone dose-response across six settings rather than a single comparison. Re-powering would not reverse it. The two modalities together support the mechanistic claim: descent helps when the cue is a degraded copy of a stored pattern and hurts when it is not. |
| 2026-09-06 | Multi-scale β (mixing several temperatures during retrieval) investigated and ruled out, third time this mechanism has been tested | Literature check first (Mixture-of-Softmaxes, Yang et al. ICLR 2018; MultiMax ICML 2024; CT-MoS; Hopfield-Fenchel-Young 2025): the general mixing mechanism is established elsewhere, not previously applied to Hopfield-style memory retrieval. Three variants precisely distinguished and implemented: single beta, an INTEGRATED mixture (`p_t = sum_j w_j softmax(beta_j s_t)` inside every descent step, one coupled trajectory), and a NAIVE ENSEMBLE (m independent trajectories fused only at readout). Verified these are genuinely different mechanisms, not accidentally identical: agree 99.5% at T=1, diverge to 63.0% top-1 agreement at T=8 (composition and mixing do not commute for T>1). Reduction check passed (equal betas -> exact single-beta reduction, max diff 4e-13). Held-out result (tune/test split, bootstrap resampling images): integrated mixture vs ensemble = +0.0003 [-0.0014, +0.0021] -- statistically identical despite disagreeing on 37% of individual queries. Neither beats a single well-tuned beta beyond noise. The predicted-to-fail control ([1,5,20], values below the measured beta leverage band) collapsed to exact chance for the integrated version and gave identical accuracy at every occlusion level, worse than its ensemble counterpart (0.282) -- confirms the coupling is real (it can make things worse) even though it does not make things better. |
| 2026-09-04 | Stage-A **sentence**-embedding run carried out (`all-mpnet-base-v2`), separate from the 2026-09-03 image run | The text-side Stage-A validation the image run explicitly left owed. Findings F10-F15, `experiments/run_real.py`. Note the two runs are different modalities and are **not** interchangeable evidence. |

## 2026-09-04 — first sentence-embedding run discarded

The run at `20260904T174209Z` was executed before the F10 fix (perturbation scale not
dimension-normalized). It reported `vote_proj=0.000` and `repulsion=0.016`, which were
units artifacts rather than measurements. Deleted rather than kept, so nobody cites it.
Superseded by `20260904T174411Z`. Finding recorded as F10.

## 2026-09-04 — significance testing changed to a cluster bootstrap

Originally CIs were bootstrapped over all (seed × question) measurements. Seeds reshuffle
the corpus but reuse the same questions, so those rows are correlated and the naive
bootstrap is pseudo-replication — it reported p=0.0005 where the correct cluster bootstrap
over questions reports p=0.076. **All significance testing now resamples questions.**

## 2026-09-05 — hyperparameters must be selected on a held-out thread split

Consequence of F15. β had been chosen by sweeping on the same questions it was then scored
on, which inflated the reported effect from +0.004 to +0.042. Every reported number now
comes from `experiments/devtest.py`, which splits **by thread** so no storyline spans the
boundary, tunes on dev, and reports on test.

## 2026-09-05 — Phase 3 opened: world-model experiments outside the pre-registered E1–E6

`docs/10_EXPERIMENT_LEDGER.md` X16–X19 test retrieval as inference in a learned world model
(`src/energymem/world/`). This is a new axis, not one of the pre-registered experiments, and
was opened because every pre-registered departure (G1, peeling, repulsion) had been measured
and found not to pay (F15). Logged here rather than folded silently into E1–E6.

## Cross-modality caveat

Two "real embedding" runs now exist — images (2026-09-03) and sentences (2026-09-04/05).
The β-versus-corpus-size finding from the image run and the β-versus-dimensionality finding
(F11) from the sentence run are **consistent in direction**: β does not transfer across
conditions and must be re-tuned per setting. Neither run substitutes for the other, and the
paper must say which modality each number came from.
=======

## 2026-09-04 — deviation 3: first real-embedding run discarded
The run at 20260904T174209Z was executed before the F10 fix (perturbation scale not
dimension-normalized). It reported `vote_proj=0.000` and `repulsion=0.016`, which were
units artifacts rather than measurements. Deleted rather than kept, so nobody cites it.
Superseded by 20260904T174411Z. Finding recorded as F10.

## 2026-09-04 — deviation 4: significance testing changed to a cluster bootstrap
Originally CIs were bootstrapped over all (seed × question) measurements. Seeds reshuffle
the corpus but reuse the same 37 questions, so those rows are correlated and the naive
bootstrap is pseudo-replication — it reported p=0.0005 where the correct cluster bootstrap
over questions reports p=0.076. **All significance testing now resamples questions.**
>>>>>>> Stashed changes
