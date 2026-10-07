# 13 · Claim traceability

Every headline claim in the paper, traced to the experiment, the script, the raw output
file, the statistical test, and where it appears. This is the audit a reviewer cannot
perform and we can, and it is the table that catches a claim drifting away from its
evidence during revision.

**Rule:** a claim with no row here does not go in the paper. A row whose "Raw output"
column is blank means the number is typed rather than generated, and is a submission
blocker.

| Claim | Ledger | Script | Raw output | Test | Paper |
|---|---|---|---|---|---|
| **C1** Dense top-k is the `(T=1, β→∞, b=0, R=I)` corner | X1 | `tests/test_reduction.py` | `results/reduction_check.json` | exact top-1 equality: **1000/1000** synthetic *d*=64, **838/838** real mpnet *d*=768 | Prop. 1, §3.2, App. A |
| **C2** β leverage band ≈[16,48]; β=512 → 0.4995 vs cosine 0.4996 | X3, X11 | `scripts/sweep_beta_transition.py` | ⚠️ stdout only | descriptive sweep; no interval claimed | Cor. 1, §3.4 |
| **C2** Effective blend size 377 patterns at β=16 | X11 | `experiments/run_real.py` | ⚠️ stdout only | descriptive | §3.4 |
| **C3** Image completion +0.014 pooled [+0.006,+0.021] | — (pre-ledger) | `scripts/confirm_holdout.py` | `results/image_holdout.json` | paired bootstrap, **resampling images**; β on disjoint tune half | Tab. 1 *(generated)*, §5.1 |
| **C3** Dose-response: +0.003 n.s. → +0.021 sig across occlusion | — (pre-ledger) | `scripts/confirm_holdout.py` | `results/image_holdout.json` | per-level paired bootstrap | Tab. 1 *(generated)*, §5.1 |
| **C3** LoCoMo monotone 0.336 → 0.252 across anchor sweep | — (pre-ledger) | `scripts/exp_locomo_anchored.py` | `results/locomo_anchored.json` | marginal bootstrap CI per condition; monotonicity descriptive | Tab. 2 *(generated)*, §5.2 |
| **C3** Reduction check 196/196 at anchor=1.0 | — (pre-ledger) | `scripts/exp_locomo_anchored.py` | `results/locomo_anchored.json` | exact top-5 set equality | §5.2 |
| **C4** Multi-step descent ≡ one step above threshold | X2 | `experiments/run_E1.py` | `experiments/results/real/` | descriptive equality | Tab. 4 |
| **C4** Peeling/repulsion inversion on real embeddings (0.268 / 0.290) | X13 | `experiments/run_real.py` | `results/real/20260904T174411Z.json` | paired cluster bootstrap over questions | Tab. 4 |
| **C4** Repulsion +0.004, p=0.43 held-out | X15 | `experiments/devtest.py` | `results/real/devtest.json` | cluster bootstrap over questions, thread-disjoint dev/test | Tab. 4 |
| **C4** Noise-seeded descent fails at every σ | X7 | `experiments/run_E2.py` | `experiments/results/real/` | descriptive sweep | Tab. 4 |
| **C4** Cosine-seeded descent = chance (circularity) | X7 | `experiments/run_E2.py` | `experiments/results/real/` | descriptive | Tab. 4 |
| **C4** Multi-scale β +0.0003 [−0.0014,+0.0021] | — (pre-ledger) | `scripts/exp_multibeta_scales.py` | `results/multibeta.json` | paired bootstrap resampling images | Tab. 5 *(generated)*, §5.4 |
| **C4** Per-memory/learned β inert; shuffle control identical | — (pre-ledger) | `scripts/train_cluster_beta.py`, `verify_trained_beta.py` | ⚠️ stdout only | shuffle control | Tab. 4 |
| **C4** Coupling predictor into energy never exceeds external | X18 | `experiments/run_E5.py` | `experiments/results/real/` | η sweep, same probes | Tab. 4 |
| **C5** One-hop association, own corpus 50/50: +0.049 [+0.000,+0.100] p=0.032 | X17b | ⚠️ script lost | ⚠️ none | cluster bootstrap over threads | Tab. 3 row (i) |
| **C5** One-hop, own corpus 4-fold: +0.035 [+0.012,+0.061] p=0.002 | X25 | `experiments/x23_mode_diagnostics.py` | `results/x23_mode_diagnostics.json` | cluster bootstrap over threads; per-fold val selection | Tab. 3 row (ii) |
| **C5** One-hop, **LoCoMo** 4-fold: +0.043 [+0.036,+0.050] p<1e-4 | **X26** | `experiments/locomo_assoc.py` | `results/x26_locomo_assoc.json` | cluster bootstrap over 272 sessions; per-fold val selection; namespaced ids | Tab. 3 row (iii), Fig. 1 |
| **C5** Two-hop **withdrawn**: LoCoMo −0.002 [−0.008,+0.004] p=0.77 | **X26b** | `experiments/locomo_assoc.py` | `results/x26_locomo_assoc.json` | as above, n=5,013 | Tab. 6, §5.3 |
| **C5** Parametric JEPA memorises: 0.784 fitted / 0.008 unseen | X17 | `src/energymem/world/jepa.py` + driver | ⚠️ driver lost | rank sweep 2→32, thread-disjoint | §5.3 |
| **C5** X16 gate: cosine R@1 0.105, gold median rank 24/1319 | X16 | `experiments/x23_mode_diagnostics.py` (reproduced) | `results/x23_mode_diagnostics.json` | descriptive | §5.3, App. C |
| **C6** σ not dimensionless: 0.016 → 0.277 | X10 | `experiments/run_real.py` | `experiments/results/real/` | before/after on identical probes | Tab. 7 |
| **C6** Pseudo-replication: p=0.0005 → 0.076 | X14 | `experiments/paired_test.py` | `results/real/paired_vs_dense.json` | naive vs cluster bootstrap, same data | Tab. 7 |
| **C6** Test-set selection: +0.042 → +0.027 → +0.004 | X15 | `experiments/devtest.py` | `results/real/devtest.json` | three nested protocols | Tab. 7 |
| **C6** Asymmetric exclusion → exact 0.000 | X18 | `experiments/run_E5.py` | `experiments/results/real/` | before/after fix | Tab. 7 |
| **C6** Unmatched step size: −0.115 → −0.062 | X23b, X25 | `experiments/x23_mode_diagnostics.py` | `results/x23_mode_diagnostics.json` | α matched vs α=1, same probes | Tab. 7, §5.5 |
| **I4 falsified** mixture − averaged −0.062 [−0.094,−0.034] | X25 | `experiments/x23_mode_diagnostics.py` | `results/x23_mode_diagnostics.json` | cluster bootstrap over threads | §5.5 |
| **I4 falsified again** on LoCoMo: mixture 0.075 vs averaged 0.107 | X26c | `experiments/locomo_assoc.py` | `results/x26_locomo_assoc.json` | cluster bootstrap over sessions | §5.5 |
| **Premise confirmed** 97% of cues multi-modal, mutual cosine 0.353 | X24 | `experiments/x23_mode_diagnostics.py` | `results/x23_mode_diagnostics.json` | descriptive | §5.5 |

## Open blockers visible in this table

⚠️ marks every row whose number is **typed, not generated**. Six remain, down from nine.

**Closed on 2026-10-07:**

- ✅ **C1 now leaves an artefact.** `tests/test_reduction.py` writes
  `results/reduction_check.json`. Adding it exposed a **false claim in the paper**: the text
  said the reduction was verified on "synthetic (*d*=256) and real (*d*=768) embeddings,
  1000/1000 in both settings", but the suite was synthetic-only at *d*=64. A real-embedding
  test now exists and passes 838/838, and all four places in the paper are corrected to the
  true figures. The anisotropic case is the one that carries weight — random
  high-dimensional vectors are near-orthogonal, so the softmax gap δ is large and
  concentration is trivial there.
- ✅ **Table 2 is generated.** `exp_locomo_anchored.py` emits
  `results/locomo_anchored.json`; the re-run reproduced all eight typed values and the
  196/196 reduction check exactly.

- ✅ **Tables 1 and 5 are generated.** `confirm_holdout.py` and `exp_multibeta_scales.py`
  now emit JSON. The image re-run reproduced all eight typed values of Table 1 exactly
  (pooled +0.014 [+0.006, +0.021], per-level +0.003/+0.017/+0.021) — C3 confirmed against
  a fresh run on freshly downloaded data.
- ✅ **`make_tables.py` validates its own output.** Twice a generator emitted `\\` inside a
  cell, which LaTeX reads as a row break: the table silently splits and the document still
  compiles with zero errors. A clean build is not evidence a generated table is correct, so
  the generator now rejects malformed rows before they reach the paper.

**Still open:**

1. **Two scripts are lost** — the X17b driver and the X17 JEPA driver. X17b is Table 3
   row (i); it is corroborated by rows (ii) and (iii) and could be dropped with no loss to
   the claim. X17 (parametric failure) has no surviving driver and should be re-run before
   it is cited as a result.
2. **Table 4 (negatives) stays hand-assembled**, deliberately: it is a prose synthesis of
   the ledger, not a numeric table. Its individual numbers trace to rows above.
