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
experiments/   run_E1..E6.py; results/ committed as JSON
paper/         LaTeX skeleton, section stubs, verified refs.bib
scripts/       demo_review2.py
tests/         sanity suite — Proposition 1, energy monotonicity, capacity, determinism
```

## Current status (2026-09-05)

- ✅ Energy core, dense baseline, test suite — **5/5 passing**
- ✅ **Proposition 1 verified numerically**, on synthetic *and* on real embeddings
- ✅ Real embeddings, both modalities: images (Imagenette-160 + ResNet18) and text
      (LoCoMo + MiniLM). The synthetic-ceiling problem in F6 is resolved.
- ⚠️ **The headline hypothesis does not hold on real data.** See below before planning work.
- ❌ Learned routing (G3), lifecycle (decay/consolidation) — still untested

---

## Empirical status: what we actually found

Three studies on real data. The short version: **energy descent does not beat cosine
top-k for agent memory retrieval, and we can say precisely why.**

### 1. Images — occlusion recovery (`scripts/test_imagenet_hopfield.py`)

300 Imagenette images as memories, ResNet18 features, queries = occluded held-out images.

| occlusion | cosine | energy T=8 β=32 | energy T=8 β=8 |
|---|---|---|---|
| 0% | 0.863 | 0.838 | 0.100 |
| 60% | 0.407 | **0.465** | 0.100 |
| 80% | 0.285 | **0.333** | 0.100 |

The **only** setting where descent wins. β=8 collapses to chance (0.100 = 1/10 classes):
too flat a softmax averages the whole corpus on step 1. Reproduces F2 on real features.

### 2. Images — capacity and calibration (`scripts/test_capacity_confidence.py`)

- **Capacity**: the advantage above is size-dependent. Growing the bank 100 → 4,000 at
  fixed β=32, cosine overtakes descent past N≈800.
- **Calibration** (negative): retrieval margin does *not* predict correctness, for either
  method (correlation ≈ 0; accuracy is non-monotonic across margin quartiles). The
  "memory knows when it is unsure" idea does not survive contact with data.
- **Per-memory β** (`memory/multibeta.py`, negative): giving each pattern its own β_i
  made things *worse* at every N (0.150 vs 0.471 at N=4,000). Reason: in
  `softmax_i(β_i·s_i)` a heterogeneous β_i rescales pattern i against all others, so it
  controls basin *depth* when only *sharpness* was intended. Crowded patterns win on
  multiplier size rather than similarity. Kept in the repo as a documented dead end.

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
> Question-based agent retrieval violates that premise, so the method does not transfer.**

Falsifiable and not yet run: swap LoCoMo queries from questions to *masked turns* on the
same corpus and encoder. The diagnosis predicts descent regains its advantage.

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
