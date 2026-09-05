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
