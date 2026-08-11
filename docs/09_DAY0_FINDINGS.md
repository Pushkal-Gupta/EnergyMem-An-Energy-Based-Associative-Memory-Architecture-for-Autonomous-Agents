# Day-0 Findings (2026-08-11)

Produced by running the core against synthetic corpora on the day the scaffold was built.
**All of it is on synthetic embeddings and must be re-verified on real sentence embeddings
before it goes in the paper.** But the structural results below are geometry-independent
and will not go away.

Read this before Week 1. It changes what you should build.

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
