# Findings log

**F1–F9 (2026-08-11 to 09-04) — SYNTHETIC vectors. F10–F14 (2026-09-04) — REAL embeddings.**

Read F10–F14 first. Three of the synthetic findings did not survive contact with real
sentence embeddings, and one of them (F10) was a units bug that would have been published
as a false negative. The synthetic section is kept in full because the *mechanisms* it
identified are still the right mechanisms — but **no number above the real-embeddings
divider is citable.**

---

## F1 — Proposition 1 holds. ✅

`T=1, β=1e6, b=0, R=I` reproduces exact cosine top-1 on **500/500** and **1000/1000**
queries. The reduction is real and the test is in CI (`tests/test_reduction.py`).

This is now the paper's most secure contribution. It also means the baseline comparison is
exact by construction.

## F2 — β has a sharp stability threshold, and below it descent destroys the cue. ⚠️

Recall@1 at 70 % cue masking, 1 000 memories, planted clusters:

| β | T=1 | T=2 | T=4 | T=8 | T=16 |
|---|---|---|---|---|---|
| **4** | 0.967 | 0.200 | 0.000 | 0.000 | 0.000 |
| **16** | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 32–256 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

At β=4 the softmax is too flat, so `X·softmax(βXᵀξ)` ≈ the mean of all memories. The state
collapses to the global centroid on step 1 and never recovers.

**Practical consequence:** any β-annealing schedule that *starts* low is actively harmful.
The Review-1 intuition of "explore broadly, then commit" is wrong for this operator — broad
exploration means averaging over the entire corpus, not surveying candidates.
**Use high constant β, or anneal within a shortlist.**

This cost me three misleading experiments before I diagnosed it. It will cost you a week if
you meet it in September instead of August.

## F3 — Multi-step descent buys nothing over cosine on plain retrieval. 🔴

Above β=16, `T=1` and `T=16` give **identical** results. This is exactly what Proposition 1
predicts — once β is high enough to commit on step 1, further steps are fixed-point iterations
that do not move.

**G1 alone is not a contribution.** Do not build a paper section around "multi-step descent
gives pattern completion." On embedding-space retrieval it does not.

## F4 — Single-state descent is structurally incapable of multi-target retrieval. 🔴

Composite cue (query = normalized mean of k memories, gold = all k), Recall over the gold set:

| k | cosine top-k | endpoint, T=8 |
|---|---|---|
| 2 | 1.000 | 1.000 |
| 3 | 1.000 | **0.336** |
| 4 | 1.000 | **0.252** |

**Mechanism, and it is not fixable by tuning:** descent converges to *a single point* `ξ*`.
Reading top-k by similarity to `ξ*` ranks by proximity to one attractor, whereas the original
query still carries evidence for all k targets. Convergence *destroys* the multiplicity that
top-k preserves. This is geometry-independent — it will hold on real embeddings.

This kills the DegradeBench-D5 story as originally conceived (`04_EXPERIMENT_PROTOCOL.md`
claimed D5 was the G1 discriminator — **that claim is now known to be backwards**).

## F5 — Residual peeling recovers it. ✅ (and this is a genuine idea)

Descend, commit to an attractor, **project it out of the residual cue, re-descend** —
matching-pursuit over the energy landscape:

| k | cosine top-k | endpoint T=8 | **residual peeling** |
|---|---|---|---|
| 2 | 1.000 | 1.000 | **1.000** |
| 3 | 1.000 | 0.336 | **1.000** |
| 4 | 1.000 | 0.252 | **1.000** |

Full recovery. This is the right way to do multi-target retrieval under energy dynamics, it
is a natural consequence of the formalism (not a hack), and **no paper in the citation audit
does it.** Worth pursuing as a contribution in its own right.

Caveat: it *matches* cosine here, it does not beat it — because of F6.

## F6 — Random synthetic embeddings have no headroom. Stop using them. 🔴

Cosine top-k scores **1.000 on every condition tested**, including 70 % masking and 4-way
composite cues. Random Gaussian vectors in d=256 are near-orthogonal, so retrieval is trivial
and no method can distinguish itself.

## F7 — Parallel descent is a real alternative to peeling, but only with repulsion. ⚠️

*Added 2026-09-03, prompted by Pushkal: "why not roll multiple balls down at once?"*

Peeling is sequential — k descents, one after another. The obvious alternative is to descend
m particles **simultaneously** and read off the distinct attractors they reach. Tested at
k=4, 1 000 memories, β=32, T=8, 200 composite queries:

| readout | Recall@k | latency (CPU, N=1000) |
|---|---|---|
| cosine top-k | 1.000 | 0.02 ms |
| single descent, top-k from endpoint | 0.258 | 0.10 ms |
| **residual peeling** (`retrieve_multi`) | **1.000** | 0.49 ms |
| parallel, m=k, isotropic noise | 0.142 | 0.5 ms |
| parallel, m=4k, isotropic noise | 0.109 | 0.8 ms |
| **parallel + repulsion** (`retrieve_multi_parallel`) | **0.910** | 1.24 ms |
| parallel seeded at cosine top-4k | 0.260 | 0.3 ms |

**Two naive versions fail, and the reasons are structural, not tuning problems.**

*Perturbation alone has no working σ.* The dynamics are deterministic, so identical particles
converge to an identical point — you must inject diversity. But distinct attractors reached
by 16 particles goes 1.00 (σ=0) → 5.72 (σ=0.15) → 15.36 (σ=0.35) → 15.86 (σ=0.9). Small σ
collapses; large σ scatters the particles *uniformly over the corpus*, where the k gold
memories are no likelier to be hit than any other k. Isotropic noise carries no information
about where the other targets are, so there is no sweet spot in between.

*Seeding at the cosine top-m is circular.* Diagnostic: stored patterns are **exact fixed
points** — seed a descent at a stored memory and self-recall is 1.000 at every β from 8 to 256.
So descent is a no-op on those seeds (particles stayed on their seed 1.000 of the time; all 4
gold were among the 16 landings). The 0.260 is therefore not a retrieval failure at all — it
is chance when choosing 4 of 16 (=0.250). Rank the landings by similarity to the original
query and it jumps to 1.000, but that ranking *is* cosine top-k. The descent contributed
nothing.

**Repulsion is the version that works.** Add an inter-particle repulsion term (Stein-
variational descent over E). 1.000 of the 16 particles then land on a gold memory — versus
0.004 if they were spread at random — so even an arbitrary pick of 4 is usually right.

**But it still loses to peeling, for one reason worth stating in the paper.** Peeling gets an
*ordering* for free: each round's winner is ranked by how much of the residual cue it
explained. Parallel descent has no such signal — after convergence every particle looks
equally converged, and the query information that would let you rank them is exactly what
descent destroyed. This is the same mechanism as F4, one level up.

**Generalisation of F4/F5.** Multi-target readout requires *enforced diversity*, and there are
only two places to put it: sequentially and exactly (orthogonal projection — the committed
attractor provably cannot return), or simultaneously and softly (a repulsive force, tuned by
γ). Peeling is exact and self-ranking; repulsion is approximate, needs two extra
hyperparameters, and degrades as β rises (0.945 at β=16 → 0.468 at β=256).

**Do not close this off, for one reason:** peeling is k sequential descents, repulsion is one
batched matmul per step. On CPU at N=1 000 peeling wins on latency by 2–3×, but that ordering
could invert on GPU at realistic corpus sizes. Latency is a reported column in every table —
measure it at scale before declaring peeling the winner.

**Action:** both are now implemented (`retrieve_multi`, `retrieve_multi_parallel`). They go in
the ablation table as competing multi-target readouts, with the two failed variants reported
as the negative result that motivates the design. Re-run on real embeddings with the rest.

---

## F8 — Basin-volume voting is a real ranking signal, but banning is not peeling. ⚠️

*Added 2026-09-04. Proposed by a teammate of Pushkal's: drop m balls, rank memories by how
many balls land in each basin, remove the winner, re-drop, repeat.*

F7 claimed parallel descent has **no** ranking signal. **That claim was too strong and must be
corrected in `method.tex`.** Basin volume — the fraction of perturbed cues that drain to a
given attractor — is a ranking signal, it is query-dependent (39–40 distinct winners across 40
different queries), and it is far better than the readout F7 tested:

| readout | Recall@4 |
|---|---|
| F7 parallel, "strongest particle first" | 0.142 |
| **vote count + ban the winner** (as proposed) | **0.621** at σ=0.30, m=64 |
| vote count + **project the winner out of the cue** | **1.000** |
| residual peeling | 1.000 |

The first winner is almost always correct — at σ≤0.22 with m=32 it is a gold memory
**1.00** of the time. The method fails on rounds 2–4, and the diagnostic is unambiguous:

> **After banning the winner, the second-most-voted attractor is a gold only 0.02–0.10 of the
> time.** The balls are still dropped around the same cue, so removing the winner from the
> *shortlist* merely promotes whichever attractor is spatially adjacent — almost never another
> target.

**The corrected general claim.** Multi-target readout needs enforced diversity in the
*dynamics*, not in the *readout*. Banning changes what you are allowed to report; projection
changes where the next descent starts from. Voting is a better readout, and it is worth exactly
nothing until the cue itself is peeled — at which point it ties peeling and costs m× the
compute. Report all three; the vote/ban variant is the sharpest negative result we have,
because it is the intuitive proposal and it fails for a precise, measurable reason.

## F9 — FIRST SIGN OF HEADROOM: peeling beats cosine under heavy corruption. 🔍

Same 200 composite queries, cue corrupted by random masking:

| mask | cosine | peeling | repulsion | vote+projection |
|---|---|---|---|---|
| 0 % | 1.000 | 1.000 | 0.931 | 1.000 |
| 30 % | 1.000 | 1.000 | 0.770 | 1.000 |
| 50 % | 0.995 | **1.000** | 0.604 | 0.981 |
| 70 % | 0.916 | **0.961** | 0.370 | 0.695 |

**This is the first condition in the entire project where the energy method beats the
baseline.** It is small, single-seed, and synthetic — do not put it in an abstract. But it says
where to look: F6 said synthetic corpora have no headroom, and that is true *at low corruption*.
Push the corruption hard enough and headroom appears, because pattern completion is exactly the
regime attractor dynamics were designed for.

**Action:** heavy-corruption conditions (D1 at severity 4–5) move to the FRONT of DegradeBench,
and this table gets re-run on real embeddings with ≥5 seeds and CIs before anyone repeats the
0.961 out loud.

---

# ══════════════ REAL EMBEDDINGS — 2026-09-04 ══════════════

**F6's condition is now discharged.** Everything above (F1–F9) was measured on random
Gaussian vectors. The findings below are on **real sentence embeddings**
(`all-mpnet-base-v2`, 768-d) over a 1 043-episode natural-language corpus
(`src/energymem/eval/corpus.py`), 5 seeds, cluster-bootstrapped CIs, against BM25 and
hybrid RRF as well as dense top-k. **Three synthetic findings did not survive.**

## F10 — Perturbation scale must be dimension-normalized. Synthetic hyperparameters do not transfer. 🛑

The first real run returned `vote_proj = 0.000` in **every** condition and
`repulsion = 0.016`. That was not a result, it was a units bug, and it would have been
reported as "the parallel methods fail on real data" by anyone who did not check.

An isotropic draw with per-component sd `s` has expected norm `s·√d`. At d=256 (synthetic)
`σ=0.35` is a perturbation of norm 5.6; at **d=768 it is 9.7 — roughly ten times the length
of the unit cue.** Measured `cos(q, q+noise) = 0.124`: the cue is simply gone.

`sigma` now means **the perturbation norm as a fraction of the cue** and is divided by
`√d` internally (`memory/energy.py`, `run_real.py`). After the fix, repulsion went from
0.016 to 0.277.

> **Rule for the paper:** every scale hyperparameter tuned at one dimensionality must be
> re-expressed dimensionlessly before it crosses to another. State this in the setup
> section — it is a reproducibility trap the field falls into constantly.

## F11 — The β threshold does NOT transfer. F2's "β≈16" was a synthetic artifact. 🛑

Effective memories blended at the first step, real embeddings, 1 043 memories:

| β | 16 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|
| effective blend size | **377.0** | 3.26 | 1.18 | 1.01 | 1.00 |

On synthetic data β=16 was already sharp. On real embeddings β=16 blends **377 memories**
— catastrophic — and you need β≥64 to commit. Recall@|gold| tracks it exactly: 1-step
scores 0.135 at β=16 and 0.248 at β≥64.

**F2 stands as a phenomenon and dies as a number.** The stability cliff is real; the
location of the cliff is a property of the embedding geometry, not of the method. Report
the *mechanism* and report the threshold *per encoder*, never as a constant.

## F12 — Real embeddings are strongly anisotropic, and that is where the headroom is. ✅

Pairwise cosine between unrelated memories: **mean 0.258, p99 0.676.** Synthetic random
vectors sat at ≈0.000.

This is why F6 saw no headroom and why real data does. Gold rank distribution on the real
corpus: gold in top-1 **0.182**, top-5 **0.409**, top-10 **0.485**, median gold rank **10 of
1 043**. Dense top-k scores **0.248** on Recall@|gold| — there is now an enormous amount of
room above the baseline, which is exactly the precondition for measuring anything.

## F13 — The method ranking INVERTS from synthetic to real. 🔄

| | synthetic | real |
|---|---|---|
| residual peeling | **1.000** (dominant) | 0.268 |
| repulsive parallel | 0.910 (lost) | **0.290** (leads) |

On synthetic composite cues peeling was exact and repulsion was a distant second. On real
embeddings the order reverses, and repulsion prefers a *softer* β (16) than everything else.
Plausible mechanism: with anisotropic embeddings the gold memories are not orthogonal, so
projecting one out damages the others' components in the cue — the precise assumption
peeling relies on. **F5's claim that peeling is "the correct readout" is not supported on
real data and must be softened in `method.tex`.**

## F14 — THE HONEST HEADLINE: nothing significantly beats dense top-k yet. ⚠️

Paired **cluster** bootstrap over questions, Δ = method − dense, clean cues:

| method | Δ | 95% CI | verdict |
|---|---|---|---|
| BM25 | −0.153 | [−0.284, −0.027] | **loses** |
| hybrid RRF | −0.063 | [−0.207, +0.081] | n.s. |
| residual peeling | +0.020 | [−0.054, +0.095] | n.s. |
| repulsion (β=64) | +0.029 | [−0.025, +0.083] | n.s. |
| **repulsion (β=16)** | **+0.042** | **[−0.014, +0.100]** | **n.s. (p=0.076)** |

**Methodological trap, and it nearly caught us.** The naive bootstrap over all 185
measurements reported repulsion as a significant win at **p=0.0005**. But the 5 seeds
reshuffle the corpus while reusing the *same 37 questions* — 185 rows are 5 correlated
observations of 37 items. Resampling rows is pseudo-replication. **The cluster bootstrap
over questions is the correct unit, and it moves p from 0.0005 to 0.076.** Any paper
reporting the former would be wrong. This analysis is now the default in
`experiments/paired_test.py`.

**Power analysis — exactly what is missing:**

| effect | per-question sd | questions for 80% power | we have |
|---|---|---|---|
| repulsion β=16 (+0.042) | 0.176 | **136** | 37 |
| repulsion β=64 (+0.029) | 0.168 | 266 | 37 |
| peeling (+0.020) | 0.231 | 1 064 | 37 |

The effect is the right size and the right sign; **the sample is four times too small.**
LongMemEval (500 q), LongMemEval-V2 (451 q) and LoCoMo (~7 000 QA) all clear the bar
comfortably. This is no longer a research question, it is a data-loading task.

**What this means for the paper.** The claim is *not* "we beat dense retrieval." It is:
under matched conditions on real embeddings, with proper paired statistics, **repulsive
parallel descent trends above dense top-k by ~17% relative at 6× the latency, and a
properly-powered benchmark run is required to confirm or kill it.** Reporting the trend
honestly with the power analysis is a legitimate result. Reporting it as a win is not.

**Immediate action:** wire up LongMemEval (E6). It is the difference between a trend and
a finding, and it is the highest-value day of work available.

---

## F15 — THE DEFINITIVE RESULT: the win does not survive honest evaluation. 🛑

*2026-09-04. Question bank expanded 37 → 146, then split dev/test by thread.*

F14 reported repulsion trending +0.042 (p=0.076) and concluded the sample was too small.
The sample is no longer too small. Here is what happened as the evaluation got honest:

| evaluation | Δ (repulsion − dense) | p |
|---|---|---|
| 37 questions, β tuned and reported on the same set | **+0.042** | 0.076 |
| 146 questions, β still tuned on the reported set | +0.027 | 0.026 |
| **146 questions, β tuned on dev, reported on held-out test** | **+0.004** | **0.425** |

**On held-out questions the effect is +0.004 with CI [−0.033, +0.041]. It is gone.**

The dev half still looks good (+0.049 at β=16) — which is exactly what an overfit
hyperparameter looks like. β was being fitted to the questions it was then scored on. Split
the threads so no storyline spans dev and test, and the advantage collapses to noise.

Peeling is the same story in the other direction: best on dev at β=128 (+0.013), **−0.019
on test**.

**Conclusion, and this is the paper's central empirical claim:**

> Under matched conditions on real sentence embeddings, with paired cluster-bootstrap
> statistics and hyperparameters selected on held-out data, **none of the three departures
> from the dense-retrieval corner — multi-step descent (G1), residual peeling, or repulsive
> parallel descent — produces a measurable improvement over dense top-k.** BM25 and hybrid
> RRF are significantly *worse*. The energy formulation's advantage over its own `T=1,
> β→∞` corner is, on this evidence, zero.

This is a clean, well-powered, honestly-evaluated negative result, and it is worth
publishing. Combined with Proposition 1 it makes a coherent paper: *dense retrieval is a
special case of energy-based associative memory, and the general case buys you nothing on
agent-memory retrieval.*

**What is still untested and could still change this:** G3, the learned routing operator.
Every result above varies only the *readout*. `R` changes the landscape itself and is the
one mechanism that can retrieve by association rather than similarity — the thing cosine
provably cannot do. It has never been trained. **That is now the entire remaining upside.**

**Methodology that must go in the paper (all three nearly bit us):**
1. Dimension-normalize every scale hyperparameter (F10).
2. Cluster-bootstrap over questions, never over (seed × question) rows (F14).
3. Split dev/test by *thread*, and never report a number whose hyperparameter saw it (F15).

---

**Protocol correction:** the synthetic-first plan in `04_EXPERIMENT_PROTOCOL.md` Stage A is
wrong. **Go to real sentence embeddings in Week 1**, not Week 3. Real embeddings are
anisotropic with a high cosine floor — that is where the difficulty, and any possible
advantage, actually lives. Keep synthetic corpora only for the capacity/stability tests where
you *want* a clean ceiling.

---

## What this means for the paper

The naive thesis — *"energy minimization beats cosine retrieval"* — is **not supported**, and
F3/F4 give principled reasons why not. Do not spend six weeks rediscovering this.

The defensible paper that these findings point to:

> **When does energy-based associative memory help LLM-agent retrieval, and why usually not?**
>
> 1. **Proposition 1** — dense top-k is the `T=1, β→∞` corner of the energy formulation (F1)
> 2. **A stability characterization** — the β threshold below which descent destroys the cue (F2)
> 3. **A structural impossibility** — single-state descent cannot serve multi-target queries,
>    with the mechanism (F4), and **residual peeling as the principled fix** (F5)
> 4. **What actually pays** — whatever G2 (strength) and G3 (routing) deliver on *real*
>    embeddings, measured against PAM / HeLa-Mem / DMHN with a matched encoder

That is a rigorous, honest, useful paper. It is a **characterization + negative result with a
constructive fix**, which is a much better fit for TMLR or Neural Networks than for ICLR
(see `05_VENUE_STRATEGY.md` — ICLR mostly does not reward this shape; TMLR explicitly does).

**Revised recommendation:** treat ICLR 2027 (25 Sep) as a stretch submission only if G3 shows
a real win on real embeddings by Gate G-B. Otherwise target **TMLR** — rolling deadline, no
September crunch, and the exact venue for this kind of result.

## Immediate actions

- [ ] **Rajit** — Week 1, not Week 3: wire up real sentence embeddings (`all-mpnet-base-v2`) and re-run F2–F5. Everything above is provisional until this is done.
- [ ] **Pushkal** — implement residual peeling properly in `memory/energy.py` (F5); it is currently only a prototype in a throwaway script
- [ ] **Pushkal** — bring Gate G-B (does routing beat PAM?) forward; it is now the *only* remaining source of a positive result
- [ ] **Anshveer** — β-stability sweep becomes a real figure; F2 is a publishable diagnostic
- [ ] **All** — update `04_EXPERIMENT_PROTOCOL.md`: D5 is not a G1 discriminator (F4), and Stage A goes to real embeddings (F6)
