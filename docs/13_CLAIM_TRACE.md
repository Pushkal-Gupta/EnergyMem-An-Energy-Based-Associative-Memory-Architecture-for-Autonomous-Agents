# 13 · Claim traceability

Every headline claim in the paper, traced to the experiment, the script, the raw output
file, the statistical test, and where it appears. This is the audit a reviewer cannot
perform and we can, and it is the table that catches a claim drifting away from its
evidence during revision.

**Rule:** a claim with no row here does not go in the paper. A row whose "Raw output"
column is blank means the number is typed rather than generated, and is a submission
blocker.

> ⚠️ **The "Paper" column has drifted, and only the rows added 2026-10-11 are current.**
> Sections and tables have been renumbered twice as the paper grew (27 → 30 → 36 pages), and
> the older rows still carry their numbering from before. **Resolve any reference through the
> label table below rather than trusting a `Tab. N` in a row written earlier.** Regenerate
> this table from `main.aux` after any build:
> `grep -o 'newlabel{\(tab:[a-z]*\|sec:[a-z]*\)}{{[^}]*}' main.aux`
>
> | label | № | | label | № |
> |---|---|---|---|---|
> | `tab:images` | 1 | | `sec:method` | 3 |
> | `tab:completion` | 2 | | `sec:band` | 3.4 |
> | `tab:locomo` | 3 | | `sec:setup` | 4 |
> | `tab:assoc` | 4 | | `sec:results` | 5 |
> | `tab:twohop` | 5 | | `sec:completion` | 5.1 |
> | **`tab:corpora`** | **6** | | `sec:semantic` | 5.2 |
> | **`tab:cardinality`** | **7** | | `sec:composition` | 5.3 |
> | `tab:negatives` | 8 | | **`sec:corpora`** | **5.4** |
> | `tab:multibeta` | 9 | | `sec:negatives` | 5.5 |
> | `tab:regimes` | 10 | | `sec:modes` | 5.6 |
> | `tab:errors` | 11 | | `sec:taxonomy` | 6 |
> | `tab:grids` | 12 | | `sec:occlusion` | 6.2 |
> | | | | `sec:protocol` | 7 |
> | | | | `sec:limitations` | 8 |
> | | | | `sec:conclusion` | 9 |
>
> Fixing the stale rows is a separate pass, not folded in here: rewriting a reference whose
> referent I would have to guess at is how a trace table stops being evidence.

| Claim | Ledger | Script | Raw output | Test | Paper |
|---|---|---|---|---|---|
| **C1** Dense top-k is the `(T=1, β→∞, b=0, R=I)` corner | X1 | `tests/test_reduction.py` | `results/reduction_check.json` | exact top-1 equality: **1000/1000** synthetic *d*=64, **838/838** real mpnet *d*=768 | Prop. 1, §3.2, App. A |
| **C2** β band; **99.4% top-1 decision agreement with cosine at β=512** | X3, X11 | `scripts/sweep_beta_transition.py` | `results/beta_band.json` | decision-agreement sweep (accuracy cannot see the upper edge) | Cor. 1, §3.4, Fig. 1 |
| **C2** Effective blend size 377 patterns at β=16 | X11 | `experiments/run_real.py` | ⚠️ stdout only | descriptive | §3.4 |
| **C3** ~~Image completion +0.014 pooled [+0.006,+0.021]~~ **WITHDRAWN as evidence (X113)** — number reproduces, interpretation does not | — (pre-ledger) | `scripts/confirm_holdout.py` | `results/image_holdout.json` | paired bootstrap, **resampling images**; β on disjoint tune half. **Metric is same-CLASS over 10 classes; cue's source image is not in the bank** | Tab. 1 *(generated)*, §5.1, §4.4 |
| **C3** ~~Dose-response: +0.003 n.s. → +0.021 sig across occlusion~~ **WITHDRAWN as evidence (X113)** | — (pre-ledger) | `scripts/confirm_holdout.py` | `results/image_holdout.json` | per-level paired bootstrap | Tab. 1 *(generated)*, §5.1 |
| **C3** Matched baseline: soft k-NN class vote **beats descent −0.036 [−0.043,−0.028]**, negative at all 3 levels | X113a | `experiments/x113_class_vote.py` | `results/x113_class_vote.json` | paired bootstrap resampling query instances; β, k, τ all selected on the same disjoint tune half; k=1 reproduces cosine exactly (internal null); fast descent path = `EnergyMemory.retrieve` 200/200 | Tab. 1 *(generated)*, §4.4 |
| **C3** Mechanism: first step blends **12.7 patterns (median) of 400**, 48% of mass same-class, at the published β=40 | X113c | `experiments/x113_class_vote.py` | `results/x113_class_vote.json` | descriptive (participation ratio) | §4.4, §3.4 |
| **C3** True completion (cue = occluded **stored** image, same-image recall@1): **−0.003 [−0.008,+0.000]** | X113b | `experiments/x113_class_vote.py` | `results/x113_class_vote.json` | paired bootstrap; identity control cosine R@1 = **1.0000**; β selected on disjoint half → **80, top of grid**; bank-path reproduction max \|Δ\| = 0.00e+00 | Tab. 2 *(generated)*, §4.4 |
| **C6** Seventh protocol error: unmatched baseline, +0.014 → −0.036 | X113 | `experiments/x113_class_vote.py` | `results/x113_class_vote.json` | the two arms differ in ensembling, not in mechanism | Tab. 6, §7 |
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
| **C5** One-hop, own corpus 50/50: **+0.024 [−0.032,+0.083] p=0.24 (n.s.)** | **X17b-r** | `experiments/x17b_single_split.py` | `results/x17b_single_split.json` | cluster bootstrap over storylines; (k_P,τ) on dev only | Tab. 3 row (i) *(generated)* — **supersedes** the unreproducible +0.049 |
| **C5** One-hop, own corpus 4-fold: +0.035 [+0.012,+0.061] p=0.002 | X25 | `experiments/x23_mode_diagnostics.py` | `results/x23_mode_diagnostics.json` | cluster bootstrap over threads; per-fold val selection | Tab. 3 row (ii) |
| **C5** One-hop, **LoCoMo** 4-fold: +0.043 [+0.036,+0.050] p<1e-4 | **X26** | `experiments/locomo_assoc.py` | `results/x26_locomo_assoc.json` | cluster bootstrap over 272 sessions; per-fold val selection; namespaced ids | Tab. 3 row (iii), Fig. 1 |
| **C5** Two-hop **withdrawn**: LoCoMo −0.002 [−0.008,+0.004] p=0.77 | **X26b** | `experiments/locomo_assoc.py` | `results/x26_locomo_assoc.json` | as above, n=5,013 | Tab. 6, §5.3 |
| **C5** One-hop, **MSC** 4-fold: **+0.016 [+0.014,+0.018] p<1e-4** (+33% rel.) | **X201** | `experiments/x101_msc.py` | `results/x101_msc.json` | cluster bootstrap over 2,505 sessions; LoCoMo's grid unchanged; bank 30,320 turns (5× LoCoMo) | Tab. 4 row (iv) *(generated)*, §5.3 |
| **C5** Two-hop on MSC: **+0.005 [+0.003,+0.006]**, excludes zero but 1/18 of the withdrawn +0.185 | **X201** | `experiments/x101_msc.py` | `results/x101_msc.json` | as above, n=25,310 — the only two-hop measurement that resolves; reported *with* its magnitude caveat | Tab. 5 row (d) *(generated)*, §5.3 |
| **C5** Mixture below averaging on MSC too (+0.006 vs +0.016) | X201 | `experiments/x101_msc.py` | `results/x101_msc.json` | same folds; 4th corpus reproducing the I4 ordering | §5.3, §5.6 |
| **C7** Expansion beats cosine, **HotpotQA bridge +0.023 [+0.010,+0.037]** | **X202** | `experiments/x10x_corpora.py` | `results/x10x_hotpot.json` | LOGO over 10 banks, train side only; **non-null K₀ selected 10/10 folds**; cluster bootstrap over banks | Tab. 6 *(generated)*, §5.4 |
| **C7** Expansion beats cosine, **2Wiki inference +0.171 [+0.118,+0.219]**; beats RRF +0.093 | **X202** | `experiments/x10x_corpora.py` | `results/x10x_2wiki.json` | as above, n=246 | Tab. 6 *(generated)*, §5.4 |
| **C7** Comparison strata **self-null 10/10** on both corpora (Δ exactly 0 by selection, not by measurement) | X202 | `experiments/x10x_corpora.py` | `results/x10x_hotpot.json`, `results/x10x_2wiki.json` | `folds_backed_off_to_null` = 10; no interval is placed on these cells | Tab. 6 *(generated)*, §5.4 |
| **C7** Random-mode control: real modes win **+0.13 to +0.39, 18/18 cells** | **X202c** | `experiments/x102_checks.py` | `results/x102_checks.json` | 9 (K₀,β) cells per corpus; **exploratory — chosen after seeing X202**, logged as a deviation in `PREREG_HI.md` | §5.4 |
| **C7** Gain survives disjoint sample, bge, ½×/2× banks (bridge +0.020→+0.026) | X202c | `experiments/x102_checks.py` | `results/x102_checks.json` | exploratory; **the I4 sub-claim does *not* survive bge (+0.001 [−0.004,+0.006])** | §5.4 |
| **C7** Expansion **hurts** on small banks: QASPER multi-evidence **−0.027 [−0.049,−0.010]** | **X203** | `experiments/x10x_corpora.py` | `results/x10x_qasper.json` | LOGO over 268 single-paper banks (~47 paragraphs) | Tab. 6 *(generated)*, §5.4 |
| **C7** No transfer: LoCoMo's K₀=0 gives up the whole gain (−0.023, −0.171); RRF transfers within 0.012 | **X205** | `experiments/x10x_corpora.py` | each `results/x10x_*.json` (`transfer/vs_selected`) | per-corpus transfer contrast | §5.4, §8 |
| **C7** ~~Cardinality: more targets should help more~~ **WITHDRAWN — reversed on 4 of 7 tests** | X202–X204, X211, X212 | `experiments/x10x_corpora.py`, `x11x_modalities.py` | `results/x10x_*.json`, `x111_mosaic.json`, `x112_crossmodal.json` | pre-registered difference-of-differences per corpus; **2 of 7 rows rest on a 23-query stratum and are flagged in the caption** | Tab. 7 *(generated)*, §5.4 |
| **C4** LongMemEval: expansion null **−0.007 [−0.032,+0.018]**; **RRF +0.156 [+0.125,+0.189]** | **X200** | `experiments/x10x_corpora.py` | `results/x10x_longmemeval.json` | LOGO over 479 per-question haystacks; benchmark's own strata | Tab. 6 *(generated)*, §5.4 |
| **C4** Mosaics: expansion self-nulls at every m; **MMR +0.07** | **X211** | `experiments/x11x_modalities.py` | `results/x111_mosaic.json` | m set by construction; p at the 2/1024 floor | Tab. 7 *(generated)*, §5.4 |
| **C4** Cross-modal (CLIP, Flickr8k): averaging +0.012 beats the mixture (n.s. +0.004) | **X212** | `experiments/x11x_modalities.py` | `results/x112_crossmodal.json` | matched encoder *within* the comparison (both arms CLIP) | Tab. 7 *(generated)*, §5.4 |
| **C3** X113's image finding, measured 2 days earlier: descent **−0.0313 [−0.0397,−0.0227]** vs the vote | **X210** | `experiments/x11x_modalities.py` | `results/x110_occlusion.json` | ⚠️ **NOT an independent replication of the image arm** — same 5,847 cues as X113a, agreeing to 16 decimals. Differs only in the vote's grid (k=10 vs k=20) | §6.2 |
| **C3** X210's genuine addition: **text** occlusion, descent +0.003 [−0.000,+0.006] **n.s.**, vote **+0.104** | **X210** | `experiments/x11x_modalities.py` | `results/x110_occlusion.json` | Yahoo Answers, 10 classes, 2,000 held-out texts × 3 levels; deletion and `[MASK]` both | §6.2 |
| **C5** Parametric JEPA memorises: **1.000 fitted / 0.000 unseen** (rank 32); best unseen 0.016 vs cosine 0.131 | **X17-r** | `experiments/x17_parametric.py` | `results/x17_parametric.json` | rank sweep 2→32, storyline-disjoint | §5.3 |
| **C5** X16 gate: cosine R@1 0.105, gold median rank 24/1319 | X16 | `experiments/x23_mode_diagnostics.py` (reproduced) | `results/x23_mode_diagnostics.json` | descriptive | §5.3, App. C |
| **C6** σ not dimensionless: 0.016 → 0.277 | X10 | `experiments/run_real.py` | `experiments/results/real/` | before/after on identical probes | Tab. 7 |
| **C6** Pseudo-replication: p=0.0005 → 0.076 | X14 | `experiments/paired_test.py` | `results/real/paired_vs_dense.json` | naive vs cluster bootstrap, same data | Tab. 7 |
| **C6** Test-set selection: +0.042 → +0.027 → +0.004 | X15 | `experiments/devtest.py` | `results/real/devtest.json` | three nested protocols | Tab. 7 |
| **C6** Asymmetric exclusion → exact 0.000 | X18 | `experiments/run_E5.py` | `experiments/results/real/` | before/after fix | Tab. 7 |
| **C6** Unmatched step size: −0.115 → −0.062 | X23b, X25 | `experiments/x23_mode_diagnostics.py` | `results/x23_mode_diagnostics.json` | α matched vs α=1, same probes | Tab. 7, §5.5 |
| **I4 falsified** mixture − averaged −0.062 [−0.094,−0.034] | X25 | `experiments/x23_mode_diagnostics.py` | `results/x23_mode_diagnostics.json` | cluster bootstrap over threads | §5.5 |
| **C1** Reduction is exact at *k*=1 and **ranks 2..k do not agree**: top-1 100/100, top-10 **0/100** | **X98** | `experiments/x9x_controls.py` | `results/x9x_controls.json` | exact set equality, `descend(T=1, β=512, b=0, R=I)` vs cosine, real LoCoMo banks | Prop. 1, §3.2, App. A Remark (`rem:shortlist`) |
| **C4** G2 (strength bias `b`) hurts under every prior, monotonically: −0.004 [−0.035,+0.030] gentlest → −0.495 [−0.533,−0.459] strongest | **X62** | `experiments/x6x_energy.py` | `results/x6x_energy.json` | two-level cluster bootstrap over 10 conversations; 4 priors × 3 strengths | §8 Limitations |
| **C4** G3 fitted routing hurts: diagonal −0.029 [−0.046,−0.013], dense −0.471 [−0.512,−0.424]; `R=I` is the best routing found | **X63** | `experiments/x6x_energy.py` | `results/x6x_energy.json` | as above; full vs diagonal `R`, η ∈ {1,10,100} | §8 Limitations |
| **C4** Query-conditional β **loses** to a fold-selected fixed β: −0.121 [−0.149,−0.098] and −0.089 [−0.121,−0.061]; sign-reversed control −0.390 [−0.433,−0.348] | **X65-r** | `experiments/x65r_adaptive_beta.py` | `results/x65r_adaptive_beta.json` | leave-one-conversation-out selection of fixed β on train folds only (unanimous β=32) | §6, taxonomy |

| **C3** Text occlusion: **0/45** (β, level) cells beat cosine on R@1; selected-β held-out Δ=−0.0007 [−0.0013,+0.0000] at 70% masking | **X110**, X110b | `experiments/x110_text_occlusion.py` | `results/x110_text_occlusion.json`, `x110b_beta_band_text.json` | β on a disjoint tune half, one pre-committed held-out comparison; paired bootstrap over conversations; pre-reg `PREREG_X110.md` | §6 (`sec:occlusion`), abstract, intro C3, conclusion |
| **C3** Two-target image splice: descent Δgold@5 = **−0.245** [−0.275,−0.215] | **X111** | `experiments/x111_image_multitarget.py` | `results/x111_image_multitarget.json` | β on a disjoint tune half; bootstrap over cues | §6 (`sec:occlusion`), abstract |
| **I4 confirmed-but-irrelevant** logsumexp > averaging in **20/21** cells (+0.178 [+0.152,+0.205] at the selected cell); mixture > cosine in **0/21** | **X111** | `experiments/x111_image_multitarget.py` | `results/x111_image_multitarget.json` | matched hypotheses, cell selected on the tune half | §5.5 |
| **C-limitation** RRF's +0.124 evidence recall@10 → **+0.007 [−0.013,+0.027]** answer coverage@10 | **X35** | `experiments/x3x_family_a.py` | `results/x3x_family_a.json` | substring coverage of the gold answer; blocked permutation over conversations | §8 Limitations |
| **C-caveat** Our Rocchio/PRF beats cosine in **0/16** swept cells on SciFact, its home turf | **X104** | `experiments/x104_beir.py` | `results/x104_beir.json` | selected on a train half of queries, held-out bootstrap | §8 Limitations |

**⚠️ Correction to X65.** The X65 row that stood here before reported `β ~ 1/std` at
**+0.067, p=0.004** as a win. Its driver printed the column as *"delta vs best fixed"* while
hard-coding the reference to β=16, which X60 — in the same driver and the same JSON — shows
is not the best fixed β (0.241 vs β=32's 0.397). Against a fixed β selected on training
folds the effect is **−0.089 [−0.121,−0.061]**. The superseded row is removed rather than
kept, because unlike Table 3 row (i) it was never published; X65-r supersedes it.

**⚠️ `p` has a floor on the LoCoMo-QA task.** The blocked permutation flips one sign per
conversation, so ten conversations give 2¹⁰ = 1024 null points and a smallest attainable
two-sided p of `2/1024 = 0.00195`. Every `p = 0.002` from that task is this floor. Rows
above therefore quote intervals, and the paper quotes intervals, not p.
| **I4 falsified again** on LoCoMo: mixture 0.075 vs averaged 0.107 | X26c | `experiments/locomo_assoc.py` | `results/x26_locomo_assoc.json` | cluster bootstrap over sessions | §5.5 |
| **Premise confirmed** 97% of cues multi-modal, mutual cosine 0.353 | X24 | `experiments/x23_mode_diagnostics.py` | `results/x23_mode_diagnostics.json` | descriptive | §5.5 |

## Open blockers visible in this table

⚠️ marks every row whose number is **typed, not generated**. Two remain, down from nine.

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

- ✅ **Both lost drivers rewritten** (`x17_parametric.py`, `x17b_single_split.py`). The
  modules had survived; only the drivers were missing. **No row in the paper now depends on
  code nobody can run.** X17 reproduced more starkly than recorded (1.000 / 0.000 vs the
  original 0.784 / 0.008). X17b **did not reproduce** — same split, cosine baseline exact,
  but n=123 not 37 and the effect is n.s. The re-run replaces it and the supersession is
  stated in §5.3. Logged as ledger Phase 6 and as the sixth entry in the error taxonomy.
- ✅ **C2 measured properly.** Decision agreement, not accuracy: at β=512 descent returns
  cosine's own memory on 99.4% of queries. Figure 1.

- 📉 **C3's positive instance is gone (X113).** The row above is kept struck through rather
  than deleted, because the claim was in a submitted artifact and the trace has to show
  what happened to it. The number is reproducible and correct; what it measures is ten-way
  classification on cues whose source was never stored, and a classifier beats descent at
  it. Pre-registered in `PREREG_X113.md` before the driver existed; the fourth row of that
  document's outcome table is the one that fired.

- 📈 **C7 is new, and it contradicts the thesis (X202).** One departure from the corner
  does pay. The rows above are deliberately explicit about what bounds it: the comparison
  strata self-null, the gain does not transfer, the robustness checks are exploratory, and
  the cardinality explanation we had pre-registered for it came out reversed. A claim that
  goes *our* way gets the same scrutiny as one that does not, or the record is worthless.

- 🛠 **Sixteen rows above were absent from this file for a day (ledger Phase 11).** The
  experiments ran in `4262715`; a rebase resolved two concurrently-appended ledger sections
  in favour of one, and the surviving documentation described seven of these corpora as
  "not run" while their JSONs sat in `experiments/results/`. The rule this file now carries:
  **a claim is verified against the experiment's output, never against prose about it.**
  Renumbered to X200–X212; `scripts/verify_paper_numbers.py` asserts 66 typed prose numbers
  against the JSONs on every run and caught one mis-rounded figure (+0.094 → +0.093) while
  this section was being written.

### Phase 12 rows — the unstratified (label-free) expansion result

Added 2026-10-11. No new experiment: these cells were already in the committed JSONs and
had not been reported. Selection *and* scoring on the `strata/all` pool, so no question-shape
label is used at any stage.

| Claim | Ledger | Script | Raw output | Test | Paper |
|---|---|---|---|---|---|
| **C7 survives label removal** HotpotQA unstratified **+0.015 [+0.004,+0.025]**, n=2,000, null 0/10 | X202 | `experiments/x10x_corpora.py` | `results/x10x_hotpot.json` :: `strata/all/mixture/vs_cosine` | LOGO selection on train side only, cluster bootstrap over banks | §5.4, Tab. 6 (*unstratified* row), abstract, C7, §9 |
| **C7 survives label removal** 2Wiki unstratified **+0.043 [+0.029,+0.056]**, n=2,000, null 0/10 | X203 | `experiments/x10x_corpora.py` | `results/x10x_2wiki.json` :: `strata/all/mixture/vs_cosine` | as above | §5.4, Tab. 6 (*unstratified* row), abstract, C7, §9 |
| **C7 bounded** QASPER unstratified +0.023 [−0.002,+0.047] **n.s.** (no chain stratum) | X204 | `experiments/x10x_corpora.py` | `results/x10x_qasper.json` :: `strata/all/mixture/vs_cosine` | as above | §5.4 |
| **C7 bounded** NFCorpus unstratified +0.009 [−0.004,+0.021] **n.s.** (no chain stratum) | X205 | `experiments/x10x_corpora.py` | `results/x10x_nfcorpus.json` :: `strata/all/mixture/vs_cosine` | as above | §5.4 |

All fourteen numbers above are asserted by `scripts/verify_paper_numbers.py` (80/80 total).

### Phase 12 — statements narrowed, with no number attached

These are corrections to claims, not results, so they have no raw output to trace. Recorded
here because a reader checking the paper against this file should not conclude they are
undocumented.

| Claim as it stood | What it is now | Where |
|---|---|---|
| Eq. (2) is "the CCCP update" and "decreases (1) monotonically for η=0" | Descent belongs to the unprojected step, now Eq. (2); the projected/blended map is Eq. (3) and claims no energy. Π preserves rankings (so Prop. 1 is untouched) and acts as βₜ = β/‖v‖ ≥ β across steps | §3.1 (Eqs. 1-3), App. A.5 |
| Corollary 1: "stored patterns are exact fixed points" | Hypotheses stated (unit norm, pairwise distinct, b=0, η=0, unique argmax), "in the limit", finite-β rate O((N−1)e^{−βδ_j}) governed by **min_j δ_j** | §3.4 (Cor. 1), App. A.3 |
| Taxonomy implied a usable regime selector | States that every regime label here is a benchmark annotation or our own construction; no detector built or evaluated; Table 10 is post hoc explanatory | §6 |
| "Every number currently reported in this paper is reproducible" | Defines the term: backed by a committed driver that runs ≠ re-derived per build. Nine tables generated; prose numbers typed and checked | App. repro |

### Phase 13 rows — the two mechanisms that had no table row

Added 2026-10-12. Both were named in the abstract's count of mechanisms but had no row in
the negatives table and no number in the paper. Both are real and both are losses. They are
now rows of `tab:negatives`, which is the authoritative inventory of Family A (ten rows).

| Claim | Ledger | Script | Raw output | Test | Paper |
|---|---|---|---|---|---|
| **Family A** Query-conditional β from the retrieval margin loses **−0.121 [−0.149,−0.098]** vs a fold-selected fixed β (β∼1/std: −0.089 [−0.121,−0.061]; sign-reversed control collapses to 0.007) | **X65-r** | `experiments/x65r_adaptive_beta.py` | `results/x65r_adaptive_beta.json` :: `rows/beta ~ 1\/margin/fold_selected` | fixed β selected on each fold's nine training conversations (unanimously 32) | §5.5, Tab. 8 |
| **Family A** Vote-count readout: 0.621 synthetic vs 0.142, does not transfer to real embeddings | **X8** (→ X10, X13) | `experiments/run_real.py` (F8) | `docs/09_DAY0_FINDINGS.md`, X10/X13 ledger rows | synthetic orthogonal patterns; refuted by the non-dimensionless σ of X10 and the inverted ordering of X13 | §5.5, Tab. 8 |

**Count correction.** The paper previously said "eight mechanisms" in three places and no two
lists agreed; the abstract's eight and the table's eight differed by two members each. The
true Family A inventory is **ten**, and the table's row count is now the only source for it.
See ledger Phase 13.

### Phase 13 — editorial, no number attached

| Claim as it stood | What it is now | Where |
|---|---|---|
| Seven co-equal contributions C1–C7 | Three scientific (C1 reduction + fair-comparison, C2 Family A evaluation + withdrawal, C3 the two positives) plus one explicitly-not-co-equal methodological item | §1 |
| Contributions implied a usable regime selector | New "What the taxonomy is, and is not": explanatory taxonomy, **no** selector demonstrated, and none added that we could not evaluate cleanly | §1, §6 |
| "never been evaluated"; "the two literatures have not met"; "quietly circular wherever attempted"; "every mechanism" | Narrowed to what we can support: no literature survey claim, and "all ten mechanisms we tested" | Abstract, §1, §2 |
| "Every number is backed by a driver that is committed and runs" | Three tiers. Tier 0 needs no data; Tier 2 is nine corpora a reader must fetch. Verified by running it: `x10x_corpora.py` fails with `FileNotFoundError` on a fresh checkout. `scripts/check_data.py` reports 2/11 inputs present | App. repro, README |

**Still open:**

1. **Table 5 (negatives) stays hand-assembled**, deliberately: it is a prose synthesis of
   the ledger, not a numeric table. Its individual numbers trace to rows above.
2. **Three pre-ledger studies carry no X identifier** — the image same-class study, the
   anchored LoCoMo sweep, multi-scale β. All three now emit JSON and are cited by script name.
   Deliberately *not* back-numbered: inserting entries into an append-only record
   retrospectively is the manipulation the record exists to prevent.
