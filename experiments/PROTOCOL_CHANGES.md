# Protocol Deviations

The protocol in `docs/04_EXPERIMENT_PROTOCOL.md` is pre-registered. Any change after
seeing results gets logged here with a date and a reason. Reviewers respect a logged
deviation and punish a silent one.

| Date | Change | Reason |
|---|---|---|
| 2026-08-11 | Stage A moves from synthetic to real sentence embeddings in Week 1 | Day-0 finding F6: cosine scores 1.000 on every synthetic condition, so there is no headroom to measure anything |
| 2026-08-11 | DegradeBench D5 is no longer the G1 discriminator | Day-0 finding F4: single-state descent is structurally *worse* than top-k for multi-target queries. D5 now motivates residual peeling instead |
| 2026-09-03 | Stage-A real-embedding validation run on **image** data (Imagenette-160, ResNet18 features) instead of sentence embeddings | Fastest path to real (non-synthetic) embeddings ahead of Review 2; also matches the project's actual framing of Hopfield-network memory over image/ImageNet-style data. Text/sentence-embedding validation (per `04_EXPERIMENT_PROTOCOL.md` Stage A) is still owed and not covered by this run. |
| 2026-09-03 | Added a capacity sweep (N=100..4,000) not in the pre-registered protocol | Ad hoc follow-up to the occlusion result: needed to check whether a fixed beta=32 (tuned informally at N=300) generalizes to larger memory banks. It does not -- the occlusion-recovery advantage over cosine shrinks/reverses past ~N=800. Finding: beta requires re-tuning (or an adaptive schedule) as corpus size grows; this was not anticipated in the original plan and is now a candidate future-work item rather than a settled result. |
| 2026-09-03 | Added a confidence-calibration test (retrieval margin vs. correctness) not in the pre-registered protocol | Exploratory addition to test whether the energy margin could serve as an uncertainty signal for an agent. Result is negative for both cosine and energy retrieval on this run (near-zero / negative correlation, non-monotonic accuracy across margin quartiles) -- reported as-is rather than dropped, per the project's rule that negative results ship. |
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
