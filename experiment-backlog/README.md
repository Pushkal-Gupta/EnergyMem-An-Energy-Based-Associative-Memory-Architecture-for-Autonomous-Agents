# Experiment Backlog

A self-service queue of runnable experiments. Pick an ID, run it, report it. Nothing in
here is in the paper, and nothing in here is promised a place in the paper — the point is
coverage of the space, cheaply and in parallel.

**This folder holds the backlog only.** Drivers go in `experiments/`, outputs in
`experiments/results/`, ledger rows in `docs/10_EXPERIMENT_LEDGER.md`. Don't put code here.

**Numbering.** New IDs start at **X27**. `X1`–`X26` are spent (see
`docs/10_EXPERIMENT_LEDGER.md`); `E1`–`E6` are taken by `experiments/run_E*.py` and are a
different, older scheme. Do not reuse either.

---

## 0. Ground rules

Four rules, all of them learned the hard way. `docs/10_EXPERIMENT_LEDGER.md` has the
post-mortems.

1. **An uncommitted script is not a result.** Two drivers were lost before they were
   committed and one of the two results did not reproduce when rewritten. Commit the
   driver in the same commit as its output JSON, or the number cannot be cited.
2. **One encoder for every method under comparison.** Always
   `energymem.encoders.SentenceEncoder` (`all-mpnet-base-v2`, cached on disk). If two
   methods see different vectors, the experiment measures the encoders.
3. **Never tune on the evaluation split.** Hyperparameters are chosen per fold on that
   fold's training side only. Report the per-fold selections, as `x26_locomo_assoc.json`
   does — a reader must be able to see what was picked where.
4. **Clusters, not samples, drive the interval.** Questions inside a conversation are
   dependent. Bootstrap over conversations (or sessions), never over questions. LoCoMo
   has only 10 conversations, so say so and prefer a paired test blocked within
   conversation as the primary.

### Setup

```bash
make setup                 # venv + editable install
make test                  # 16 tests; Proposition 1 gates everything
.venv/bin/python -c "import torch, sentence_transformers; print('ok')"
```

Known environment traps, both already cost a day:

- **No CA bundle.** `urllib` and `torch.hub` fail with `CERTIFICATE_VERIFY_FAILED`.
  Export `SSL_CERT_FILE` from `certifi`, or pre-fetch weights with `curl`.
- **`torchvision` is not installed by `make setup`.** The two image experiments need it.

### How to report

Write one JSON per run to `experiments/results/<id>_<slug>.json`, including:

```json
{"id": "X31", "git_sha": "...", "utc": "...", "encoder": "all-mpnet-base-v2",
 "n_queries": 1977, "n_clusters": 10, "folds": [{"train": [...], "picked": {...}}],
 "metrics": {...}, "seed": 12345}
```

Then append a ledger row: **ID · date · question · setup · n · result with interval ·
verdict.** Negative and null results get the same row as positive ones — the ledger is
append-only, so a wrong entry is corrected by a *new* row that points back, never by an
edit.

Tables and figures are **generated**, never typed: add your JSON to
`scripts/make_tables.py`, which prints every uncovered table on each run.

### Shared helpers you will want (write once, reuse)

Nothing in the repo does multi-target retrieval yet — every metric in
`src/energymem/eval/metrics.py` and `experiments/assoc_harness.py:recall_at` assumes a
**single** gold. Family A needs set metrics. Write them once, in
`src/energymem/eval/setmetrics.py`, and have everyone import them:

- `full_set_recall_at_k(ranked, gold, k)` — 1.0 only if **all** gold in top k
- `mean_gold_recall_at_k(ranked, gold, k)` — fraction of gold in top k
- `blocked_permutation(diffs, blocks, n=100_000)` — paired test, permuting signs within
  conversation
- `hierarchical_bootstrap(values, clusters, n=20_000)` — resample conversations, then
  questions within

`experiments/assoc_harness.py:cluster_ci` is the existing one-level version.

---

## 1. Where the opportunity is

Two things found on 2026-10-08, by scratchpad scripts that were **never committed** —
so by rule 1 these numbers are *leads, not results*. Re-deriving them is X27 and X28.

**LoCoMo ships 1,986 QA pairs with gold evidence turn ids. The project has never used
them.** `src/energymem/eval/locomo.py` already parses them into `LocomoQuery` with
`gold_ids`, `category` and `answer` — and `LocomoQuery` is imported by nothing. All
three existing LoCoMo scripts use only the turn bank. 1,977 gold-annotated queries with a
pre-labelled difficulty axis are sitting unread:

| cat | meaning | n | mean gold turns | ≥2 gold | spans >1 session |
|-----|---------|-----|------|------|------|
| 1 | multi-hop | 282 | 3.13 | 97.9% | **95.4%** |
| 2 | temporal | 321 | 1.17 | 12.5% | 8.7% |
| 3 | open-domain | 96 | 2.21 | 49.0% | 32.3% |
| 4 | single-hop | 841 | 1.06 | 5.5% | 0.1% |
| 5 | adversarial | 446 | 1.03 | 3.1% | 0.0% |

**Lead 1 — the corner is not at its own ceiling.** For a gold set of size *m* with mean
pairwise similarity ε, any single unit query obeys

&nbsp;&nbsp;&nbsp;&nbsp;`min_i cos(q, z_i) ≤ ‖Σ z_i‖ / m = sqrt((1 + (m−1)ε) / m) =: B(m, ε)`

and a mixture with `K ≥ m` modes attains 1 by taking `ξ_s = z_s`. On cat 1 the bound is
**not binding** — ε=0.53 puts `B`=0.835, far above the 0.452 that the 10th-best
distractor reaches, so `B < d10` for **0.0%** of questions. But the *actual* encoder
query leaves a gold turn below `d10` for **87.9%** of them (cat 4: 37.5%). So the gap is
a **query-formation** failure with a certificate that it is not an expressivity limit.
Do not write the bound up as an impossibility result; its value is as the certificate.

**Lead 2 — the headroom is reachable by association.** Anchor on the gold turn the query
ranks best, exclude it, then try to reach the *rest* of the evidence. Residual Recall@10,
cat 1: query **16.2%** → anchor alone, no query at all, **23.0%** (+42% relative). All
multi-target: 16.6% → 23.9%. Naive `max(query, anchor)` scored 22.2%, *below* the anchor
alone — so the combination rule is the open problem, not a formality. The anchor is an
**oracle**; the non-oracle version is X31.

**Lead 3 — why I4 may have been tested in the wrong regime.** Invariant I4 ("never
average modes") was falsified on both corpora — but every experiment behind that
falsification had a **single** target, where a mixture cannot help by construction. Cat 1
gives *m*=3.13 with gold labels, and cat 4 gives *m*=1.06 as a built-in control. The
sharpest available result in this backlog is the **interaction** (X32), not a main
effect: a theory that predicts the sign of its own earlier refutation is worth more than
one that never failed.

---

## 1b. STATUS — families A–D are done (2026-10-09)

Ran in Phase 7 of `docs/10_EXPERIMENT_LEDGER.md`. **Do not redo these.** E, F and G are
done too — see **1c** — so the live work is **H, I, J**, the sweep grids, or an open
thread below.

| ID | verdict | where |
|---|---|---|
| X27 headroom certificate | ✅ ran — ceiling binds 0.0%, actual query fails 87.9% (cat 1) | `x27_headroom.json` |
| X28 reachability | ✅ ran — oracle anchor 24.2% vs query 16.6% residual R@10 | `x28_reachability.json` |
| X29 baselines | ✅ ran — **RRF wins, 0.651 vs cosine 0.527**; MMR a no-op | `x29_baselines.json` |
| X30 set metrics | ✅ ran — ordering does change on single-hop | `x29_baselines.json` |
| X31 expansion | ❌ **null** — backs off to the cosine null in 10/10 folds | `x31_expansion.json` |
| X32 interaction | ❌ **prediction not met** — no effect to decompose; I4 stays falsified | `x32_interaction.json` |
| X33 Rocchio | ❌ worse than cosine when forced non-trivial | `x31_expansion.json` |
| X50–X57 geometry | ✅ ran — all eight | `x5x_geometry.json` |
| X60–X66 energy | ❌ **every knob neutral or harmful** (T, b, R, β, pursuit). X65's one positive cell was **withdrawn in Phase 8b** — it was scored against β=16 while X60 in the same JSON shows β=32 is better. Against a fold-selected fixed β it is −0.089 [−0.121,−0.061] | `x6x_energy.json`, `x65r_adaptive_beta.json` |
| X70–X75 predictors | ❌ **null after multiplicity** (3/24 cells nominal, Bonferroni 0.002) | `x7x_predictor.json` |

**Headline: the corner is hard to beat.** The headroom is real and is *not* an expressivity
limit, but nothing in the architecture closes it, and a standard lexical–dense hybrid (RRF)
beats it by +0.124 `gold@10`. New machinery available to reuse:
`src/energymem/eval/setmetrics.py` and `experiments/locomo_qa.py`.

**Open threads worth more than a fresh ID**

- **X28 without the oracle.** The anchor is picked with gold labels. Everything turns on
  selecting it label-free; X31 shows top-K0 is not the answer. A calibrated
  "is my top-1 actually relevant?" gate is the live question.
- **Why does RRF win?** It is the only method using lexical signal. Mode expansion on top of
  RRF rather than cosine was never tried, and RRF's margin is bigger than any effect
  this project has measured.
- **X54 says 40–60% of missed evidence is within two hops.** The structure exists; the
  selection policy is what is missing.
- Two of this phase's own bugs are written up as the seventh protocol error in the ledger.
  Read it before adding a sweep: a metric cutoff at or below the expansion depth is blind
  to the expansion, and a zero-width interval means a bug, not a null.

---

## 1c. STATUS — families E, F, G are done, and some of E and F were killed on paper (2026-10-09)

Ran in Phase 8 of `docs/10_EXPERIMENT_LEDGER.md`:
`experiments/x9x_controls.py`, `experiments/x8x_robustness.py`, `experiments/x90_cost.py`.

**First, the triage.** Three arms were marked DEAD without being run, because families
A–D had already made them unanswerable rather than merely unpromising. Each one was
written as *"interesting if the gain …"*, and X31 found the gain is exactly zero: at the
fold-selected configuration the expansion **is** cosine, in 10/10 folds, in every
category. You cannot measure the robustness, growth, or index-survival of zero.

| ID | why it is dead | what survived, and where it went |
|---|---|---|
| X84 query perturbation | "the gain is more robust than cosine's" — it *is* cosine, so its robustness is cosine's, identically | does **RRF's** +0.124 survive paraphrase? → **X29**, still open |
| X85 bank contamination | "the gain grows" — no gain; and padding changes the task, not its size | bank scope → **X37** (open, P1); latency vs size → **X92**, ran |
| X93 approximate NN | "does the gain survive HNSW/IVF" — nothing to survive; an index is a deployment choice, not a claim | benchmark it if someone ships it |
| X90's *headline* | "accuracy per millisecond" as the impact number — at zero accuracy delta that ratio only falls | the measurement ran, sign reversed: the overhead that buys nothing |

Three more arms are **not runnable as written**, which is different from dead, and the
reason is in each case a property of the benchmark rather than of the architecture:

- **X82's "no normalization"** — `SentenceEncoder` sets `normalize_embeddings=True` and is
  frozen; un-normalized vectors would change every number in the project.
- **X83's sliding-window and session-summary arms** — LoCoMo annotates evidence at *turn*
  level. A window containing a gold turn is neither a hit nor a miss, so these need the
  benchmark re-annotated, not just re-chunked.
- **X87's leave-one-session-out** — a bank *is* a conversation and 95.4% of cat-1 gold sets
  span sessions, so dropping a session removes bank items. That is a different task, not a
  different split. Fold **count** was swept instead (2/4/5/10).

**Second, the results.**

| ID | verdict | where |
|---|---|---|
| **X38 shuffled gold** | ✅ **PASS — the P0 is cleared.** Random gold: 0.0179 vs closed-form chance 0.0172. Permuted gold: 0.0284, ~20× below the real 0.527. Family A is citable | `x9x_controls.json` |
| X95 random modes | ✅ ran — real modes beat random by +0.27 to +0.42 at every forced config, 9/9 cells. **But** real modes never reach cosine (best −0.051) | `x9x_controls.json` |
| X96 leakage audit | ✅ **PASS** — static (AST) and functional: all 7 methods bit-identical after the gold sets are destroyed; selection scores train folds only | `x9x_controls.json` |
| X97 exclusion symmetry | ✅ **PASS** — every method scores all N, no masking, no NaN. 1 tie at the k=10 boundary (BM25) in 40 probes | `x9x_controls.json` |
| X98 reduction ladder | ✅ **10/10 rungs exact** on 1,977 real queries — **and a correction, below** | `x9x_controls.json` |
| X99 metric cross-check | ✅ **0 mismatches** in 1,326 rankings × 4 metrics — **and a correction, below** | `x9x_controls.json` |
| X80 encoder swap | ❌ null is **not** encoder-specific: 10/10 null folds under mpnet, bge (the pre-declared primary) and MiniLM | `x8x_robustness.json` |
| X81 dimension | ❌ null holds at d ∈ {64,128,256,384,768}. **X51's caveat is now closed**: separability 15/15 at every d, including d=64 against n=5,882 | `x8x_robustness.json` |
| X82 normalization | ⚠️ the **one** regime that selects a non-null config — see the lead below | `x8x_robustness.json` |
| X83 chunking | ⚠️ same pattern; `text_only` costs cosine 0.527 → 0.379, and cat 1 then selects non-null | `x8x_robustness.json` |
| X86 seed sensitivity | ✅ retrieval path bit-deterministic; p over 20 seeds ∈ [0.0014, 0.0024] around the exact 0.00195 | `x8x_robustness.json` |
| X87 fold granularity | ❌ null at **every** fold count (2, 4, 5, 10). No selection at all is −0.0987, p=0.002 **worse** | `x8x_robustness.json` |
| X90–X92 cost | ✅ ran — mixture 15.7× cosine for an identical ranking; RRF 34.8× for its +0.124 | `x90_cost.json` |

### Two corrections to things Phase 7 believed

**1. `p = 0.002` is a floor, not a measurement.** The blocked permutation flips one sign
per conversation, so with 10 conversations the null has exactly **2¹⁰ = 1024** points and
the smallest attainable two-sided p is `2/1024 = 0.00195`. X99 now computes it by
enumeration (`exact_blocked_permutation`) instead of sampling it. **Every "p = 0.002" in
Phase 7 is this floor** — it means "as extreme as the blocking permits", not "p = 0.002
worth of evidence". Report it as `p = 2/1024 (floor)`.

**2. Proposition 1 is a top-1 statement, and the test that guards it is misnamed.**
`tests/test_reduction_to_cosine_topk` tests `k=1`; its own docstring says so. X98
measured both: `descend(T=1, β=512, b=0, R=I)` matches cosine's top-1 **100/100**, and
matches cosine's **top-10 0/100**. At β→∞ the descent returns the nearest stored pattern,
so ranking by the converged state ranks *that pattern's* neighbourhood, not the query's.
**This is the mechanism behind family C's finding that T > 1 hurts** — it was a prediction
of the proposition all along, not a surprise.

### The one lead, stated with its own caveat

Expansion gets selected over the null in exactly two places, and both are regimes where
the representation was **first damaged**:

| regime | cosine | mixture | Δ (95% CI) | exact p | null folds |
|---|---|---|---|---|---|
| whitened, all | 0.5160 | 0.5220 | +0.0060 [−0.0105, +0.0224] | 0.32 | **0/10** |
| whitened, cat 1 | 0.2730 | 0.2753 | +0.0023 [−0.0297, +0.0367] | 0.86 | **0/10** |
| text-only chunks, cat 1 | 0.1534 | 0.1627 | +0.0093 [−0.0290, +0.0454] | 0.50 | **0/10** |

**Read it honestly.** Not one of these is significant, every interval covers zero, and in
each case the damaged-plus-expansion score is *below* the undamaged baseline (0.522 <
0.527; 0.163 < 0.329). The generalization worth testing is a conjecture, not a result:
**mode expansion is selected only where cosine has been crippled, and it recovers a
fraction of the damage rather than adding anything.** If you want to chase it, the
pre-registration has to come first and the endpoint has to be an *absolute* win over the
undamaged baseline — not a win over the cripple.

### Cost, since the headline had to be rewritten

Per query, one 419-turn bank, mpnet d=768, laptop CPU (hardware recorded in the JSON):

| method | ms | × cosine | what it buys |
|---|---|---|---|
| cosine | 0.012 | 1.0× | the baseline |
| energy descent T=1 | 0.045 | 3.7× | nothing (top-1 identical to cosine) |
| rocchio K0=3 | 0.061 | 5.1× | worse than cosine when forced |
| mixture_avg K0=3 | 0.091 | 7.7× | marginally *better* than the mixture (X70–X75) |
| **mixture K0=3** | **0.187** | **15.7×** | **nothing — selection picks K0=0** |
| energy descent T=10 | 0.304 | 25.5× | −0.203 at β=16 (X61) |
| **RRF (true cost)** | **0.415** | **34.8×** | **+0.124 gold@10 — the only method that buys anything** |
| MMR pool=50 | 1.897 | 159× | +0.0004, p=0.91 |
| cross-encoder top-50 | 280 | ~23,500× | cost proxy only, no accuracy claim |

BM25's per-query cost was **0.000 ms** on the first run — `locomo_qa` caches the score
matrix, so the timer was measuring an array index. That is the 8×-vs-21× scar repeating
itself in miniature; it is now timed against the index directly (0.361 ms/query, 3.8 ms
one-off build). **If a cost number looks too good, check what is cached.**

Memory: the expansion path stores **nothing** extra — its modes are bank vectors already
held. Separately, the project stores float64 embeddings at 6,144 B/memory, which is 2×
what float32 would need and 8× int8.

### What E, F, G change about the conclusion

Nothing, and that is the point: the Phase 7 negative now has its controls behind it. The
null survives three encoders, five dimensions, four normalizations, two chunkings and four
fold counts; the metric is not rewarding structure (X38); no method sees the labels (X96);
every rung of the ladder reduces exactly (X98). **The negative is no longer provisional.**

---

## 1d. STATUS — families H and I are done (2026-10-10)

Ran in Phase 9 of `docs/10_EXPERIMENT_LEDGER.md`, against `PREREG_HI.md`, which was
**committed alone before any corpus was opened**. Drivers: `x10x_corpora.py` (shared
machinery in `corpus_qa.py`), `x101_msc.py`, `x102_checks.py`, `x11x_modalities.py`.
Family G was already closed (1c). Family J needs an API key and was not run.

| ID | verdict | where |
|---|---|---|
| X100 LongMemEval | ❌ expansion null (−0.007); **RRF +0.156**, +0.255 on temporal | `x10x_longmemeval.json` |
| **X101 MSC** | ✅ **one-hop association replicates**: +0.016 R@1 [+0.014, +0.018], +33% rel., 30k-turn bank | `x101_msc.json` |
| **X102 HotpotQA / 2Wiki** | ✅ **first non-null expansion**: bridge +0.023; 2Wiki inference +0.171 and beats RRF by +0.094. ❌ cardinality interaction **reversed** (m=4 < m=2) | `x10x_hotpot.json`, `x10x_2wiki.json` |
| X102-checks (exploratory) | ✅ gain survives a disjoint sample, bge, ½× and 2× banks, and random modes. ❌ the I4 (mixture > averaging) part does **not** survive bge | `x102_checks.json` |
| X103 QASPER (for QuALITY/NarrativeQA, which have no evidence labels) | ❌ expansion hurts on multi-evidence (−0.027); RRF best | `x10x_qasper.json` |
| X104 BEIR | ⚠️ SciFact: RRF wins. NFCorpus (m≈41): mixture +0.017, ties RRF | `x10x_scifact.json`, `x10x_nfcorpus.json` |
| X105 transfer | ❌ LoCoMo selects the null, so transfer gives up every in-corpus gain | in each `x10x_*.json` |
| **X110 occlusion** | ❌ **C3 is class smoothing**: image energy +0.0135 reproduces, but **k-NN vote beats it by +0.031**; on text energy is null and k-NN +0.10 | `x110_occlusion.json` |
| X111 image mosaics | ❌ mixture null at every m; **MMR +0.07** (diversity, not association) | `x111_mosaic.json` |
| X112 cross-modal | ❌ interaction reversed: on i→t (m=5) averaging beats the mixture | `x112_crossmodal.json` |

**Headline.** I4 is not resurrected by multiple targets anywhere: the pre-registered
interaction was reversed on three corpora and null on a fourth. What *does* predict an
expansion gain is **hop type**: when the second piece of evidence is reachable from the
first rather than from the question (encyclopedic bridge/chain questions), two-stage
expansion is robust and sometimes beats RRF. On conversational memory (LoCoMo,
LongMemEval), lexical fusion is what works and expansion is null. Separately, the
paper's image positive (C3) is beaten by a k-NN class vote and does not transfer to
text, so its "memory" framing needs rewriting.

**Open threads worth more than a fresh ID**

- **Expansion on top of RRF**, on 2Wiki/HotpotQA. Both work there and they may stack.
- **A hop-type predictor.** Can a label-free signal tell a chain question from a
  comparison question before retrieval? That would make expansion deployable.
- **C3 rewrite.** Report k-NN vote next to energy descent wherever the image result
  appears.

---

## 2. Priority

> **Superseded 2026-10-10:** X102 and the rest of H and I have run (see 1d). The live items
> are the open threads listed there, plus family J (needs an API key) and the sweep grids.

Revised after families A–D came back null (see 1b). Rough cost on a laptop CPU; the
embedding cache makes everything after the first run much cheaper.

| tier | IDs | why | cost each |
|---|---|---|---|
| **P0** | X102 | HotpotQA / 2WikiMultihop: multi-hop by construction, large *m*, huge *n*. **The one place the X32 effect could still exist**, and now the only P0 — the controls are done | hours |
| **P1** | X29-perturbation | the live half of the dead X84: is **RRF's** +0.124 robust to paraphrase, typos, truncation? RRF is the only thing here that works, so it is the only thing worth stress-testing | 20 min |
| **P1** | X34–X37, X39–X41 | the family-A entries not reached: displacement on QA, answer coverage, adversarial specificity, bank scope, metadata, paraphrase modes | 10–40 min |
| **P1** | "expansion on RRF" | mode expansion was only ever stacked on cosine. RRF's margin is larger than any effect this project has measured and nobody has tried expanding *its* top-k | 30 min |
| **P2** | the X82/X83 lead | pre-register first (see 1c): the endpoint must be an absolute win over the **undamaged** baseline, not over the cripple | 30 min |
| **P2** | sweep grids | volume; low expected value now the knobs are mapped | 5–30 min |
| **P3** | families I, J, rest of H | modalities, reader-in-the-loop, other corpora | hours–days |

**If you only run one thing: X102.** The interaction X32 was built to find could still be
real on a benchmark whose evidence sets are genuinely near-orthogonal, which LoCoMo's
(ε=0.52) are not — and after E/F/G there is nowhere else on LoCoMo left to look.

**Families A–G are closed. Do not reopen one without saying which result you are
overturning.**

---

## Family A — LoCoMo QA evidence retrieval

The untouched benchmark. Bank = one conversation's turns (419–689, mean 588); query =
the question text; gold = `evidence` turn ids. Stratify every result by category and
report cat 1 and cat 4 separately — they are different tasks.

### X27 — Headroom certificate
**Run:** `experiments/x27_headroom.py`. Per category compute *m*, ε, `B(m,ε)`,
`gold_min`, `gold_max`, the 1st/10th/*m*-th best distractor cosine; fractions where the
*bound* binds vs where the *actual* query fails.
**Metric:** table above. **Interesting if:** it reproduces the 0.0% / 87.9% split. This
is a benchmark characterization that stands alone, independent of any method.

### X28 — Reachability from an oracle anchor
**Run:** `experiments/x28_reachability.py`. Anchor = gold turn the query ranks best,
excluded from all rankings. Rank the residual gold under `cos(q,·)`, `cos(z_a,·)`,
`max`, and `logsumexp` at several β.
**Metric:** median residual rank, residual R@10. **Interesting if:** anchor > query
again, and if `logsumexp` beats `max` (it should; `max` is the β→∞ limit).

### X29 — Baseline grid
**Run:** every retriever already in the repo, matched encoder:
`baselines/dense.py:DenseTopK`, `baselines/lexical.py:BM25`, `HybridRRF`, plus MMR
(write it) and a cross-encoder reranker.
**Metric:** R@1/5/10, full-set recall, mean gold recall, MRR, nDCG, by category.
**Interesting if:** cosine is not already the best — and more importantly this is the
table every later claim is measured against. **MMR is the critical baseline:** if
diversity reranking captures the whole gain, the mechanism is diversity, not
association, and X31–X32 have to be reframed.

### X30 — Set metrics vs R@1
**Run:** same rankings as X29, scored with `full_set_recall_at_k` vs `recall_at_k`.
**Interesting if:** method ordering *changes* between the two. Then R@1 was hiding the
multi-target problem all along, and so was every table in the paper.

### X31 — Operational two-stage associative expansion (the method)
**Run:** stage 1 retrieve top-`K₀` by cosine; stage 2 treat the retrieved turns as
**modes** `ξ_s` (optionally displaced, `ξ_s = normalize(q + α·D[s])`), weight
`w_s ∝ exp(β·cos(q, z_s))`; read out with `readout.py:mixture_scores`. `K₀=1, α=0`
must reduce exactly to cosine — assert it, it is a rung of the reduction ladder.
**Anchor policies to compare:** top-1 · top-`K₀` · confidence-gated (expand only when
the top-1 margin is small) · oracle (upper bound, label it).
**Metric:** as X29. **Interesting if:** it beats cosine *and* beats Rocchio (X33).

### X32 — The interaction (pre-register before running)
**Run:** X31's mixture vs X33's averaged expansion, on cat 1 and cat 4, as a
difference-of-differences. Primary test: blocked permutation within conversation.
**Pre-registered prediction:** Δ(mixture − averaged) **> 0 on cat 1**, **≈ 0 on cat 4**.
**Interesting if:** either direction. A confirmed interaction resurrects I4 as a
regime-dependent law and explains the paper's own negative result. A null leaves I4
falsified and sharpens its statement. **Write the prediction into the ledger and commit
it before you run this.** Its whole value is that the prediction preceded the number.

### X33 — Rocchio / pseudo-relevance feedback
**Run:** classical PRF — average the top-`K₀` into the query, renormalize, re-retrieve.
Sweep `K₀`, the Rocchio α/β/γ weights.
**Why it matters:** this is the prior art a reviewer will name, and it is *exactly* the
mode-averaging baseline. X31 vs X33 is the whole architectural argument in one row.

### X34 — Displacement step on the QA task
**Run:** add the X26 world-model offset to X31's modes. Sweep α on the training folds.
`mean_step_norm()` reports the step size; the raw displacement has ‖Δ‖≈1.157 against a
unit cue, so α≈0.34 is where the kernel average implicitly sits. Check α=0 separately.
**Interesting if:** α>0 beats α=0 on cat 1. That is the predictor earning its keep on
third-party labels.

### X35 — Answer-string coverage (downstream proxy)
**Run:** normalized substring match of the gold `answer` inside the concatenated top-k
retrieved turns.
**Metric:** coverage@{1,5,10,20}. **Report as coverage, not QA accuracy** — it is a
deterministic, API-free proxy, and calling it accuracy overclaims.
**Interesting if:** retrieval gains survive into coverage. If they don't, the retrieval
gain may not matter downstream, which is worth knowing before anyone writes it up.

### X36 — Adversarial specificity (cat 5)
**Run:** cat 5 questions are adversarial/unanswerable. Measure whether expansion makes
methods *more* confidently wrong: score distribution, top-1 margin, abstention rate
under a threshold.
**Interesting if:** the method's gain on cat 1 comes with a specificity cost on cat 5.
A reviewer will ask. Better to find it ourselves.

### X37 — Bank scope
**Run:** per-conversation bank (588) vs pooled bank (5,882, namespaced ids — see the
loader comment: dia_ids collide across conversations, 5,882 turns carry 1,033 distinct
ids, and pooling without namespacing lets an unrelated turn satisfy a gold id).
**Interesting if:** the gain grows with bank size. That is the scaling argument.

### X38 — Shuffled-gold control — **RAN: PASS (Phase 8)**
**Run:** replace each gold set with a random size-matched set; rerun everything.
**Interesting if:** anything beats chance. Then a metric is broken. Run this *before*
believing X31.

### X39 — Temporal and speaker metadata
**Run:** the loader keeps `session`, `speaker`, `timestamp`, `local_id`. Add recency
priors, same-speaker priors, session-adjacency priors as a `b` strength bias
(`EnergyMemory.set_strength_bias`).
**Interesting if:** cat 2 (temporal, n=321) improves. Nothing in the project has tried
to exploit time yet.

### X40 — Cross-session structure
**Run:** split cat 1 by whether gold spans 1 vs >1 session (269 vs 13 — the 13 will be
underpowered, say so).
**Interesting if:** the gain concentrates in cross-session questions, which is the
regime the architecture was designed for.

### X41 — Question rewriting as a mode source
**Run:** instead of expanding from retrieved turns, expand from *paraphrases* of the
question (back-translation, or template rewrites — no LLM needed for templates).
**Interesting if:** paraphrase modes match retrieval modes. Then the gain is query
diversity, not memory association, and the story changes. This is X29's MMR concern in a
second form, and it is the strongest alternative explanation available.

### X42–X48 — Per-category deep dives
One ID per category × {baseline grid, expansion, set metrics}. Small, parallelizable,
good first tasks. X42 cat 1 · X43 cat 2 · X44 cat 3 · X45 cat 4 · X46 cat 5 ·
X47 all-pooled · X48 the 4 questions with ≥10 gold turns (anecdotal, but they are where
single-query retrieval must fail worst).

---

## Family B — Geometry diagnostics

Model-free characterizations. Cheap, and they are what turn a number into an explanation.

### X50 — Gold-set geometry
*m* vs ε scatter; ε against the bank's background similarity; are gold sets tighter than
random size-matched sets? **Interesting if:** cat 1 gold sets are *not* tighter — then
"topical coherence" isn't what makes them findable.

### X51 — Linear separability of gold sets
Is gold linearly separable from its complement (LP feasibility, `scipy.optimize.linprog`,
margin fixed to 1 — feasibility is scale-invariant)?
**Watch out:** with `d`=768 > `n`=588 per conversation, *every* subset is separable in
general position, so this is vacuous per-conversation and only meaningful on the pooled
5,882-turn bank. Report the dimension caveat or the result is noise.

### X52 — Local displacement coherence
X17-r measured global displacement mutual cosine at +0.004 — essentially zero. Measure
it *within* k-NN neighbourhoods instead. **Interesting if:** locally coherent while
globally incoherent. That would explain both the 1-hop win and the 2-hop failure with one
mechanism.

### X53 — Geometry-gated α
Predict per-question gain from local coherence alone, then gate α on it (α large where
coherent, 0 where not). **Interesting if:** gain is predictable *before* seeing the
answer. That is a falsifiable mechanism claim, not a tuned knob.

### X54 — Reachability depth
At what hop does gold become reachable — 1 hop from the anchor, 2, never? Build the k-NN
graph over the bank and measure graph distance from anchor to residual gold.
**Interesting if:** most residual gold is at hop 1–2. That bounds how much machinery is
worth building.

### X55 — Intrinsic dimension and anisotropy
Participation ratio, top-10 PC variance share (X17-r measured 30.8%), per-category
effective rank. **Interesting if:** cat 1 queries live in a different subspace than cat 4.

### X56 — Hubness
Which turns are universal near-neighbours? Measure hub scores and whether hubs are
over-retrieved. Hubness is the standard explanation for exactly this failure mode in
high-dimensional retrieval, so rule it in or out.

### X57 — Distractor anatomy
What *are* the false positives — same session, same speaker, topically adjacent?
Cheap, and it tells the next person which prior to try.

---

## Family C — The energy axis (Proposition 1, Corollary 1)

Proposition 1: dense top-k retrieval is exactly the `(T=1, β→∞, b=0, R=I)` corner.
Corollary 1: at high β stored patterns are exact fixed points, so multi-step descent ≡
one-step ≡ cosine. Every experiment here is a search for a regime where that degeneracy
*lifts*.

### X60 — β leverage band on LoCoMo QA
Top-1 **decision agreement** against cosine as β sweeps (not accuracy — agreement; 99.4%
at β=512 on the existing corpora). **Interesting if:** the band sits somewhere different
on a multi-target task.

### X61 — Multi-step descent at low β
`T` sweep at β below the band. Corollary 1 says `T` is dead at high β, so this is the
only place iteration can pay. **Interesting if:** `T>1` ever beats `T=1` at matched β.

### X62 — Strength bias `b`
`EnergyMemory.set_strength_bias` — recency, frequency, session priors. Never swept.

### X63 — Routing `R`
`EnergyMemory.set_routing` — the `R=I` null is the corner. Try learned/low-rank/diagonal
`R`. **Interesting if:** anything beats `R=I`; this is the least-explored knob.

### X64 — Parallel multi-retrieve
`retrieve_multi` vs `retrieve_multi_parallel` vs matching pursuit, on cat 1.

### X65 — Per-query and per-cluster β — **RAN, then CORRECTED: see X65-r (Phase 8b)**
`memory/query_beta.py`, `memory/multibeta.py`, `memory/beta_training.py`,
`scripts/train_query_beta.py`, `scripts/train_cluster_beta.py` all exist and none has
been pointed at the QA task.

### X66 — Energy-landscape descent trace
Instrument `_descend`: energy per step, step norms, basin identity. Diagnostic, but it
is how you find out whether descent is doing anything at all.

### X67–X69 — β × T × K joint sweeps
The full corner-to-interior grid. See the sweep appendix.

---

## Family D — Predictor variants

### X70 — kNN world model vs JEPA predictor
`world/knn_wm.py:KNNWorldModel` vs `world/jepa.py:JEPAPredictor` (InfoNCE-trained), same
pairs, same encoder. **Interesting if:** the parametric predictor ever wins. X17-r found
it catastrophically overfitting — fitted 1.000 / unseen 0.000 at rank ≥8, best unseen
0.016 against cosine's 0.131. Confirm or break that on the QA task.

### X71 — `modes()` vs `predict()`
`predict()` is the averaging baseline; `modes()` preserves them. Head-to-head at matched
`K`, α.

### X72 — Rollout horizon
`rollout(q, H, K, S, α)` — sweep `H`, `K`, `S`. X26 found 2-hop composition unsupported
(Δ=−0.002 [−0.008,+0.004], p=0.77) on a *constructed* hop-2 task. Cat 1 is a real
multi-evidence task with gold labels — a fairer test of the same idea.

### X73 — Step-size calibration
The confound that cost a day: ‖Δ‖=1.157 against a unit cue, so the raw mode lands on a
training episode 100% of the time, and the kernel average implicitly applies α≈0.34.
Sweep α on the QA task and report `mean_step_norm()` alongside.

### X74 — Transition definition
`fit(Z, transitions)` currently uses adjacent turns. Try: same-session pairs,
same-speaker pairs, QA-evidence pairs (careful — that leaks labels; train-fold only),
coreference-linked pairs.

### X75 — Stein-variational / repulsive modes
Add a repulsion term so modes don't collapse. **Interesting if:** it beats top-`K₀`
selection — that is the principled version of MMR.

### X76 — Matching pursuit readout
Greedy residual-based selection instead of logsumexp.

---

## Family E — Robustness

Each of these is the same experiment with one thing changed. They are the cheapest way to
find out whether a result is real.

> **TRIAGED 2026-10-09 against families A-D (Phase 8).** Most of this family was written
> as *"does the gain survive X"*. Families A-D found **no gain**: at the fold-selected
> configuration every expansion method **is** cosine, in 10/10 folds, in every category.
> A robustness check on an effect of exactly zero can only confirm that zero is still
> zero, so the gain-conditional arms are marked **DEAD** below rather than run. The arms
> that do not presuppose a gain -- X80, X81, X82, X83, X86, X87 -- were run: see
> `experiments/x8x_robustness.py` and `results/x8x_robustness.json`. Do not re-run a DEAD
> arm hoping for a different answer; if you think one is live again, say which A-D result
> you are overturning first.

### X80 — Encoder swap — **RAN (Phase 8): the null is not encoder-specific**
`all-MiniLM-L6-v2` (d=384), `bge-base-en-v1.5`, `e5-base-v2`, `gte-base`. One swap is the
minimum; three is convincing. **Pre-register which one counts as primary** or this
becomes a search for the encoder that agrees with you.

### X81 — Dimension — **RAN (Phase 8): null at every d; closes X51's caveat**
Random projection / PCA of mpnet to d ∈ {64, 128, 256, 384, 768}. Note that the X51
separability caveat is a function of `d` vs `n`.

### X82 — Normalization — **RAN (Phase 8): the one non-null selection — see 1c**
L2 vs none vs whitening vs all-but-the-top. Whitening often erases exactly the anisotropy
the mechanism exploits — a good falsification attempt.

### X83 — Chunking — **RAN (Phase 8), two arms; two not runnable as written**
Per-turn (current) vs per-utterance vs sliding windows vs per-session summaries. The
loader builds `"speaker: text"`; try text alone and speaker-prefixed variants.

### X84 — Query perturbation — **DEAD (gain-conditional)**
Paraphrase, typos, truncation, keyword-only. **Interesting if:** the gain is more robust
than cosine's — that is a practical selling point.

**Why it is dead.** The selling point was *"more robust than cosine's"*. X31 showed the
method **is** cosine at its selected configuration, so its robustness is cosine's
robustness identically, for every perturbation, with no measurement needed. The
surviving question — does **RRF's** +0.124 survive paraphrase? — is a baseline question
and belongs to **X29**, not here. If you want it, run it there and say so.

### X85 — Bank contamination — **DEAD as framed (gain-conditional)**
Add 10×–100× irrelevant filler turns from other conversations.
**Interesting if:** the gain grows. Pairs with X37.

**Why it is dead.** *"The gain grows"* presupposes a gain. There is none (X31). Padding
also changes the task rather than its size — a turn from another conversation is a
distractor the annotation never considered — so a drop here would not be interpretable
either. The two live pieces are elsewhere: **bank scope** is **X37** (still open, P1),
and **latency vs bank size** was measured latency-only in **X92**.

### X86 — Seed sensitivity — **RAN (Phase 8): retrieval path is bit-deterministic**
Everything that touches a seed, ×20 seeds. `SPLIT_SEED = 12345` is fixed for the life of
the project — changing it is a protocol deviation, so sweep *other* seeds.

### X87 — Fold-count sensitivity — **RAN (Phase 8): null at 2, 4, 5 and 10 folds**
4-fold (current) vs 10-fold leave-one-conversation-out vs leave-one-session-out.
**Interesting if:** conclusions move. With only 10 conversations they might.

---

## Family F — Cost

The paper already has one scar here: a claimed 8× that was matmul-count reasoning never
checked against a clock, and measured 21× when it finally was.

> **TRIAGED 2026-10-09 (Phase 8).** The family survives; its **headline does not**. X90
> was written to aim at *"accuracy per millisecond"* as the quantifiable-impact number.
> With a zero accuracy delta (X31) that ratio can only fall as milliseconds are added,
> so the measurement was made with the sign reversed: **this is the overhead that buys
> nothing, quantified**, next to the overhead RRF charges for the +0.124 it does buy.
> X90-X92 ran (`experiments/x90_cost.py`); **X93 is DEAD**.

### X90 — Wall-clock per query — **ran; the stated headline is dead**
cosine vs expansion vs MMR vs cross-encoder vs the energy path. Measure it. Report
hardware. ~~**Headline to aim for:** accuracy per millisecond — if expansion costs ~2×
cosine and matches a cross-encoder at ~100×, that is the quantifiable-impact number.~~
**That headline presupposed an accuracy gain and there is none.** The measurement stands
as a cost audit; the cross-encoder figure is a forward-pass **cost proxy only** (the
cached model is an NLI head, not a reranker) and carries no accuracy claim.

### X91 — Index size and memory — **RAN (Phase 8)**
Bytes per memory for each method.

### X92 — Scaling curves — **RAN (Phase 8), latency only**
Latency and accuracy vs bank size, 500 → 50,000 turns (pad with filler).

### X93 — Approximate NN — **DEAD (gain-conditional, and not a claim)**
Does the gain survive HNSW/IVF instead of exact search? Everything here currently
assumes exact.

**Why it is dead.** No gain to survive anything (X31), and an index choice is an
engineering decision about a deployment, not a claim about the architecture. If someone
ships this, benchmark the index then — it is not evidence either way.

---

## Family G — Controls and falsification

Run these *before* believing anything in A or D.

> **ALL RAN 2026-10-09 (Phase 8)** -- `experiments/x9x_controls.py`,
> `results/x9x_controls.json`. Nothing in this family was invalidated by A-D; it is the
> family A-D most needed, because Phase 7 shipped its numbers *before* its controls.
> **X38 is in here too**, even though it is numbered into family A, because it is the
> control that gated family A, and family G's own charter is "run these first".

### X95 — Random modes — **RAN (Phase 8): real modes win, but never reach cosine**
Replace retrieved modes with random bank turns at matched `K`. **Must not** win.

### X96 — Label-leakage audit — **RAN (Phase 8): PASS, static and functional**
Grep every driver for test-side fitting. X17-r exists because this class of bug is real
here.

### X97 — Exclusion symmetry — **RAN (Phase 8): PASS**
`readout.py:rank` excludes before ranking for a reason: an asymmetric exclusion once
produced an exact 0.000 that looked like a finding. Assert every method gets the identical
exclusion set.

### X98 — Reduction-ladder nulls — **RAN (Phase 8): 10/10 exact, plus a correction**
Every rung R0–R6 must reproduce the rung below at its null setting. `tests/test_reduction.py`
covers the bottom; extend to the expansion rungs added in X31.

### X99 — Metric implementation cross-check — **RAN (Phase 8): 0 mismatches, plus the p-floor**
Two independent implementations of full-set recall, asserted equal on fixtures. The
`\\`-in-a-table-cell bug silently split a table with a zero-error build; metrics deserve
the same suspicion.

---

## Family H — Second corpus (external validity) — **RAN (Phase 9), see 1d**

The single biggest open item in `paper/SUBMISSION.md`. One public corpus is one corpus.

### X100 — LongMemEval
### X101 — MSC (Multi-Session Chat)
### X102 — HotpotQA / 2WikiMultihopQA — purpose-built multi-hop, big *m*, huge n
### X103 — QuALITY / NarrativeQA — long-document, different hop structure
### X104 — BEIR subset — standard IR, where PRF baselines are well calibrated
### X105 — Cross-corpus transfer: tune on LoCoMo, evaluate elsewhere, no retuning

For each: the headroom certificate (X27), the baseline grid (X29), the interaction (X32).
**X102 is the sharpest test of the interaction** — it is multi-hop by construction, so if
the X32 effect is real it must appear there, and if it doesn't, the effect was LoCoMo.

---

## Family I — Other modalities — **RAN (Phase 9), see 1d**

`data/imagenette2-160` is present; `scripts/test_imagenet_hopfield.py`,
`exp_multibeta_scales.py`, `confirm_holdout.py` already run on it. Needs `torchvision`.

### X110 — Text-occlusion falsification
The open item from `paper/SUBMISSION.md`. Occlusion works on images; does masking spans
of text behave the same way? **Interesting if:** it doesn't — the image result may not
be about memory at all.

### X111 — Image multi-target retrieval — the cat-1 analogue, with multiple gold images
### X112 — Cross-modal — text query, image bank

---

## Family J — Reader in the loop

Needs an API key; everything above does not. Keep these last so the core results stay
API-free and cheap to reproduce.

### X120 — LLM reader QA accuracy
Feed top-k into a reader, score answers against gold. The honest version of X35. Use the
latest Claude models; fix the model id and prompt in the driver and record both in the
JSON, or the number is not reproducible.
### X121 — Context-length ablation — k ∈ {1,3,5,10,20,50}
### X122 — Reader robustness to distractors — does better retrieval help, or does the
reader paper over retrieval errors? If it papers over them, the retrieval gain may not
matter, and that is the most important negative result in this backlog.
### X123 — LLM-judge agreement with X35's substring coverage

---

## Appendix — Sweep grids

This is where the volume is. The families above are ~90 templates; crossed with these
grids they are a few thousand runs. Start from the defaults that
`experiments/locomo_assoc.py` already uses, and **sweep on training folds only**.

```python
K0_GRID     = [1, 2, 3, 5, 10, 20]          # stage-1 depth (1 = the cosine null)
K_GRID      = [1, 2, 3, 5, 8]               # modes kept
BETA_GRID   = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
ALPHA_GRID  = [0.0, 0.1, 0.25, 0.34, 0.5, 0.75, 1.0]   # 0.0 = no displacement
TAU_GRID    = [1.0, 2.0, 5.0, 10.0, 20.0]   # kNN kernel width
KP_GRID     = [5, 10, 20, 50]               # world-model neighbours
T_GRID      = [1, 2, 3, 5, 10]              # descent steps (1 = the corner)
TOPK_GRID   = [1, 5, 10, 20, 50]            # evaluation cutoffs
CATS        = [1, 2, 3, 4, 5]
ENCODERS    = ["all-mpnet-base-v2", "all-MiniLM-L6-v2", "bge-base-en-v1.5"]
SEEDS       = list(range(20))
```

Dimensionless forms travel across scales; prefer `σ/√d` and `√(2Θ/d)` over raw `σ` and
`Θ` when reporting.

**Cheap first.** The embedding cache (`.cache/emb`, keyed by model+text) makes repeat
runs fast, so do all mpnet work before touching a second encoder. A full β×K×α grid on
one category is minutes; the same grid across encoders and corpora is hours.

**Claiming an ID.** Open a PR that adds the driver and its JSON, titled with the ID. If
two people want the same family, split by category — cat 1 and cat 4 are independent and
both are needed for X32.
