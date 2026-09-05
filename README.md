# EnergyMem

**An Energy-Based Associative Memory Architecture for Autonomous Agents**

BCSE497J Project-I · Fall 2026–27 · SCOPE, VIT Vellore · Project ID 20113UG01
Pushkal Gupta (23BCT0253) · Rajit Agrawal (23BCE2371) · Anshveer Singh (23BCE0703)
Guide: Dr. Thangaramya K

---

## What this is

Agent long-term memory is almost universally retrieved by top-k embedding similarity.
Associative-memory theory offers an alternative — recall as convergence to an attractor in
a learned energy landscape — but it has never been evaluated as the *persistent memory
layer* of an agent under matched conditions.

We show that **dense top-k retrieval is a special case of the energy formulation**
(`T=1, β→∞, b=0, R=I`), then measure which departures from that corner actually pay for
themselves.

## Quick start

```bash
make setup     # venv + deps
make test      # sanity suite — Proposition 1 gates everything downstream
make demo      # Review-2 demo, offline, <60s
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
| [`docs/09_DAY0_FINDINGS.md`](docs/09_DAY0_FINDINGS.md) | **⚠️ Empirical results that change what you should build** |
| [`docs/01_RESEARCH_PLAN.md`](docs/01_RESEARCH_PLAN.md) | The science: RQs, hypotheses, novelty argument |
| [`docs/07_CITATION_AUDIT.md`](docs/07_CITATION_AUDIT.md) | Bibliography status + the 2026 competitive landscape |
| [`docs/02_TIMELINE.md`](docs/02_TIMELINE.md) | Week-by-week with pass/fail gates |
| [`docs/03_TASK_ALLOCATION.md`](docs/03_TASK_ALLOCATION.md) | Who owns what |
| [`docs/04_EXPERIMENT_PROTOCOL.md`](docs/04_EXPERIMENT_PROTOCOL.md) | Pre-registered experiments E1–E6 |
| [`docs/05_VENUE_STRATEGY.md`](docs/05_VENUE_STRATEGY.md) | Targets + deadlines |
| [`docs/06_RISK_REGISTER.md`](docs/06_RISK_REGISTER.md) | What kills this paper and how we avoid it |
| [`docs/08_REVIEW2_CHECKLIST.md`](docs/08_REVIEW2_CHECKLIST.md) | Panel deliverables |

## Layout

```
docs/          plan, protocol, risks, findings
src/energymem/
  base.py            Retriever protocol, Memory, RetrievalResult   [FROZEN]
  memory/energy.py   energy functional, CCCP descent, residual peeling
  memory/adaptive.py perplexity-targeted beta + query anchor (anchor=1.0 == cosine)
  memory/multibeta.py per-pattern beta — DEAD END, kept as documented negative result
  memory/lifecycle.py strength bias, decay, consolidation
  baselines/         dense top-k (+ BM25, hybrid, reranker, PAM, HeLa-Mem — Week 1)
  eval/              metrics, bootstrap CIs, DegradeBench, LoCoMo loader
experiments/   run_E1..E6.py; results/ committed as JSON; cache/ gitignored (regenerable)
paper/         LaTeX skeleton, section stubs, verified refs.bib
scripts/       demo_review2.py
  confirm_holdout.py          the confirmed image-completion result (tune/test split)
  exp_multibeta_scales.py     multi-β scales vs single β vs ensemble — negative result
  exp_locomo_anchored.py      the LoCoMo negative result (anchor × pp₀ sweep)
  sweep_beta_transition.py    maps the β leverage band; shows high-β → cosine
tests/         sanity suite — Proposition 1, energy monotonicity, capacity, determinism
```

## Current status (2026-09-06)

- ✅ Energy core, dense baseline, test suite — **5/5 passing**
- ✅ **Proposition 1 verified numerically**, on synthetic *and* on real embeddings
- ✅ Real embeddings, both modalities: images (Imagenette-160 + ResNet18) and text
      (LoCoMo + MiniLM). The synthetic-ceiling problem in F6 is resolved.
- ✅ **Confirmed, honestly-powered positive result**: energy descent beats cosine on
      image pattern completion, +0.014 [+0.006, +0.021], growing with corruption severity.
- ❌ Multiple β scales (mixture or ensemble) do not beat one well-tuned β — checked
      three ways (per-memory, per-region/trained, multi-scale) and ruled out each time.
- ⚠️ The same descent that wins on images loses monotonically on LoCoMo questions.
      **The claim is conditional, not general** — see the diagnosis below.
- ❌ Learned routing (G3), lifecycle (decay/consolidation) — still untested

---

## Empirical status: what we actually found

Real data, two modalities. The short version: **energy descent beats cosine when the
query is a degraded copy of a stored pattern, and loses when it is not — and we can say
precisely why.** Both halves are established: +0.014 [+0.006, +0.021] on held-out image
completion, −0.084 on LoCoMo question→evidence retrieval. Within the winning regime, one
well-tuned β is already sufficient: mixing multiple β scales, however combined, does not
improve on it (study 1b).

### 1. Images — occlusion recovery ✅ CONFIRMED POSITIVE (`scripts/confirm_holdout.py`)

400 Imagenette images as memories, ResNet18 features, queries = occluded held-out images.
β selected on a **disjoint tune half**; one pre-committed paired test on the held-out half.
n = 5,847 query instances from 1,949 distinct images (the full val split).

| occlusion | cosine | energy β=40 | paired Δ | 95% CI |
|---|---|---|---|---|
| 30% | 0.730 | 0.733 | +0.003 | [−0.008, +0.014] n.s. |
| 50% | 0.542 | 0.559 | +0.017 | [+0.003, +0.031] **sig** |
| 70% | 0.386 | 0.407 | +0.021 | [+0.007, +0.034] **sig** |
| **pooled** | **0.553** | **0.566** | **+0.014** | **[+0.006, +0.021] sig** |

The advantage **grows with corruption severity** — exactly what pattern completion
predicts, and null at light corruption where there is nothing to reconstruct. Costs ~8×
cosine's latency. Tune-half estimate (+0.016) matched held-out (+0.014): no selection bias.

β is only influential in a narrow band. `scripts/sweep_beta_transition.py`: β=12 → 0.100
(centroid collapse), β=40 → 0.510, β=512 → 0.4995 vs cosine 0.4996 — high β *degenerates
into cosine* (Proposition 1), so the apparent "plateau" above β≈50 is the method
dissolving into the baseline, not a working regime.

> ⚠️ Earlier image-modality verdicts in this repo were **underpowered and are withdrawn**
> (80 distinct images, unpaired CIs ~5× the effect size, β tuned on the test images). See
> `experiments/PROTOCOL_CHANGES.md`, 2026-09-06.

### 1b. Images — does mixing multiple β *scales* beat one tuned β? ❌ NO (`scripts/exp_multibeta_scales.py`)

Different question from 2's per-memory β: here every candidate β sees the *whole* bank,
and the question is whether combining several global scales beats one well-tuned scale.
Three mechanisms, precisely distinguished (math in the script header):

- **single β** — one temperature, one trajectory.
- **integrated mixture** — `p_t = Σⱼ wⱼ·softmax(βⱼ·s_t)` inside *every* descent step; one
  coupled trajectory. Provably not reducible to any single β (mixture-of-softmaxes
  rank argument, Yang et al. ICLR 2018) — confirmed empirically: identical to the
  ensemble at T=1 (99.5% top-1 agreement) but **diverges at T=8 (63.0% agreement)**,
  since composition and mixing don't commute for T>1.
- **naive ensemble** — m fully independent trajectories, fused only at readout. Costs
  m× the matmuls of the other two.

Literature check first (Yang et al. 2018 MoS, MultiMax ICML 2024, CT-MoS, Hopfield-
Fenchel-Young 2025): the general mechanism is established elsewhere; applying it to
Hopfield-style memory retrieval specifically does not appear to be published.

Tune/test split as in study 1, resampling **images** in the bootstrap (not query
instances, which are correlated across occlusion levels/seeds of the same image):

| condition | R@1 | R@5 | MRR | matmuls | Δ vs single β |
|---|---|---|---|---|---|
| cosine | 0.543 | **0.759** | **0.643** | 1 | — |
| single β=56 (tuned) | 0.555 | 0.605 | 0.589 | 16 | — |
| integrated mix [32,40,48] | **0.559** | 0.591 | 0.585 | 16 | +0.0036 [−0.0033, +0.0106] n.s. |
| ensemble [32,40,48] | 0.559 | 0.599 | 0.586 | 48 | +0.0032 [−0.0038, +0.0103] n.s. |
| *control* mix [1,5,20] | 0.108 | 0.319 | 0.217 | 16 | predicted collapse — confirmed |
| *control* ensemble [1,5,20] | 0.282 | 0.296 | 0.304 | 48 | predicted collapse — confirmed, less severe |

**Answer: no.** Integrated mixture vs. ensemble: **+0.0003 [−0.0014, +0.0021]** —
statistically identical despite disagreeing on 37% of individual queries. The extra
expressiveness is real (confirmed by the divergence check) but performance-neutral: it
changes *which* queries are answered correctly, not *how many*. Single β=56 already
matches cosine's advantage; mixing buys nothing further and costs 2–3× more compute.

Two findings kept from the controls: (i) the integrated `[1,5,20]` control collapsed to
exact chance (0.108 = 1/10) and gave *identical* accuracy at every occlusion level — one
bad scale inside a shared trajectory poisons the whole state, unlike the ensemble version
(0.282) where it only dilutes one vote among independent runs; this is direct evidence the
coupling is real and can hurt as easily as it can (fail to) help. (ii) Descent wins R@1 but
**loses badly on R@5/MRR** (0.599–0.605 vs cosine 0.759) — it converges to one attractor,
so ranks 2–5 reflect proximity to that point rather than the query (F4/F5, confirmed on
real images): a better top-1, a worse shortlist.

### 2. Images — capacity and calibration (`scripts/test_capacity_confidence.py`)

- **Capacity**: the advantage above is size-dependent. Growing the bank 100 → 4,000 at
  fixed β=32, cosine overtakes descent past N≈800.
- **Calibration** (negative): retrieval margin does *not* predict correctness, for either
  method (correlation ≈ 0; accuracy is non-monotonic across margin quartiles). The
  "memory knows when it is unsure" idea does not survive contact with data.
- **Per-memory β** (`memory/multibeta.py`): two distinct failures, both since diagnosed.
  (i) The original parameterization could only express a *uniform* β of 4.0 — below the
  centroid-collapse cliff — so uniform β=32 was unreachable and the search could not fall
  back to the baseline it was meant to generalize. Fixed: `β_i = β_base·(1 + λ·z_i)`, with
  λ=0 reproducing fixed β exactly (verified 40/40). (ii) `softmax_i(β_i·s_i)` lets a
  heterogeneous β_i rescale pattern i against all others — controlling basin *depth* when
  only *sharpness* was intended. Fixed by max-centering, `exp(β_i·(s_i − max_j s_j))`.
  Even corrected, heterogeneity is **inert here**: the learned β values (51–88) sit above
  the band where β has any leverage at all, and a shuffle control (same values, randomly
  reassigned to regions) scores identically.
- **Gradient-trained β** (`memory/beta_training.py`, `memory/query_beta.py`): β learned by
  backprop, per-region and as a continuous function of the descent state. Both land within
  noise of a well-tuned scalar β. Conditioning β on the query is *motivated* — the optimal
  β genuinely differs by corruption level — but the learned function converged to a near
  constant (β≈31 at every occlusion level).

### 3. Text — LoCoMo, the real agent benchmark (`scripts/exp_locomo_anchored.py`)

1,977 real questions over 5,882 conversational turns. `anchor` interpolates between pure
cosine (`anchor=1.0`) and pure descent (`anchor=0.0`), so the baseline is a point in our
own parameter space and the sweep measures exactly what descent adds.

| condition | Coverage@5 |
|---|---|
| **anchor=1.00 — *is* cosine** | **0.336** [0.316, 0.355] |
| anchor=0.90 | 0.331 |
| anchor=0.75 | 0.321 |
| anchor=0.50 | 0.304 |
| anchor=0.25 | 0.273 |
| anchor=0.00, pp₀=4 | 0.252 |
| energy fixed β=32 | 0.236 |
| anchor=0.00, pp₀=32 | 0.201 |

Monotone dose-response: **every increment of descent costs accuracy**, optimum at the
boundary. Reduction check passes 196/196 (anchor=1.0 reproduces cosine's top-5 exactly),
so this is a property of the method, not a coding error.

### The diagnosis

One mechanism explains both modalities. The descent step computes a *weighted average of
stored patterns* and moves the state onto their span; with no anchor the cue never
re-enters. That pays off only when **the query is a degraded copy of something already
stored** — then moving toward the memory manifold restores lost signal (images at 60–80%
occlusion, the one win). Agent memory retrieval is question → evidence: the query is not
a corrupted turn, so there is nothing to complete and averaging only smears the query's
discriminative signal.

> **Associative completion requires the cue to be a degraded instance of a stored pattern.
> Image occlusion satisfies that premise and descent wins, by more as damage increases.
> Question-based agent retrieval violates it, and descent loses monotonically.**

Both halves are now measured rather than argued: +0.021 at 70% occlusion (held-out,
significant) and −0.084 on LoCoMo (monotone across the anchor sweep). The remaining
confirmation, cheap and not yet run: swap LoCoMo queries from questions to *masked turns*
on the same corpus and encoder. The diagnosis predicts descent regains its advantage —
same data, same embeddings, only the query type changes.

### Corrections to our own earlier claims

Logged rather than quietly overwritten (see `experiments/PROTOCOL_CHANGES.md`):

1. **Data bug, invalidating.** LoCoMo `dia_id`s are unique only *within* a conversation —
   5,882 turns carry 1,033 distinct ids, and 132/134 of conversation 0's gold ids recur in
   other conversations. The first capacity sweep pooled banks without namespacing, so
   distractors satisfied gold ids and scored as hits. Its "adaptive β improves as the bank
   grows" result was an artifact; with namespaced ids adaptive β *degrades* (0.250 → 0.231),
   just less steeply than fixed β (0.244 → 0.216). `exp_locomo_adaptive.py` is superseded.
2. **Confounded comparison.** `pp_start=32` forces step 1 to average ~32 memories — the
   broad-start annealing F2 already flagged as harmful. Removing it moves pure descent
   from 0.201 to 0.252, so "adaptive β is worse than fixed β" was wrong: corrected, it is
   *better* (0.252 vs 0.236). Both still lose to cosine.
3. **Missing query anchor.** Descent without an anchor discards the cue after step 1, which
   made the earlier comparison a test of "is a blend of memories a better query than the
   query." The `anchor` sweep replaces that single mis-specified point with a curve.

## The rules

1. Matched encoder across every method. No exceptions.
2. Never report an accuracy win without its latency cost.
3. ≥5 seeds, bootstrap 95% CIs on every headline number.
4. No claim without an ablation.
5. Every citation verified against the actual paper.
6. Negative results ship. If it did not work, that goes in the abstract.
