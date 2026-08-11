# Timeline — 11 Aug 2026 → 23 Sep 2026 (submission) → 28 Oct 2026 (Review 3)

## Fixed institutional dates (from the SCOPE Fall 2026–27 project calendar)

| Date | Milestone | What the panel expects |
|---|---|---|
| ~**12-08-2026** | **Review 2** | Title, Abstract, Lit Survey, Objectives, **Methodology, Implementation, Demo** — "implementation to be completed"; panel also nominates best projects here |
| **on or before 23-09-2026** | **Project Outcome** | Proof of submission: Patent / Conference (Scopus) / Journal (Scopus) |
| 07-10-2026 | Project Expo | Only if nominated at Review 2 |
| on or before 14-10-2026 | Initial report draft | Report + demo video + poster |
| on or before 21-10-2026 | Report correction | Guide verification |
| on or before 23-10-2026 | Final draft | Guide final approval |
| **28-10-2026** | **Review 3** | Full documents + outcome with proof |

> ⚠️ **Review 2 is dated 12-08 and the team's plan in chat was "CAT ke baad."** Confirm the
> actual slot with Dr. Thangaramya K **today**. If it holds tomorrow, run the
> minimum-viable demo in `08_REVIEW2_CHECKLIST.md` — it is achievable in one evening
> because it only needs G1, which is ~40 lines of code.

## Working schedule

Assumption: real engineering starts **Mon 17 Aug** (post-CAT), with a thin Review-2 demo
carved out before that. Six working weeks to submission. This is tight but not unrealistic
*if* scope holds.

### Pre-week (11–16 Aug) — Review 2 survival + foundations
- [ ] **P** Confirm Review 2 date/venue with guide; get attendance sheet signed
- [ ] **P** Minimum demo: Hopfield descent over ~1k sentence embeddings, masked-query recall vs cosine top-k, one plot (`scripts/demo_review2.py`)
- [ ] **R** Citation audit pass 1 — every arXiv ID in the Review-1 deck resolved or deleted (`07_CITATION_AUDIT.md`)
- [ ] **A** Environment: `uv`/venv, torch, sentence-transformers, faiss; `make test` green
- [ ] **All** Read `01_RESEARCH_PLAN.md` §2 and agree on the reduction framing

### Week 1 (17–23 Aug) — Core + baselines, matched harness
- [ ] **P** `src/energymem/memory/energy.py` — energy functional, CCCP update, β schedule, convergence + blend-rate instrumentation
- [ ] **R** `src/energymem/baselines/` — dense top-k, BM25, hybrid RRF, cosine+cross-encoder reranker. Same encoder, same corpus, same API
- [ ] **A** `src/energymem/eval/` — metrics (Recall@1/@k, MRR, nDCG), seed control, bootstrap CIs, latency timing harness
- [ ] **Gate G-A (Fri 21 Aug):** reduction test passes — `T=1, β=1e6, b=0, R=I` reproduces top-1 cosine **exactly** on 1k queries. If this fails, nothing downstream is trustworthy.

### Week 2 (24–30 Aug) — DegradeBench + **RQ3 first**
- [ ] **R** `DegradeBench` generator: masking, paraphrase (LLM), embedding noise, distractor injection, composite multi-cue. 5 severity levels each
- [ ] **P** **E3 — learned routing on low-cosine temporal pairs.** Run this *before* the easy experiments
- [ ] **A** E5 systems harness: latency/query, steps-to-converge, footprint vs N
- [ ] **Gate G-B (Fri 28 Aug): DOES ROUTING WORK?**
  - Yes → proceed as planned
  - No → **stop and reframe** to the characterization paper (see `06_RISK_REGISTER.md` R-3). Do not spend September hoping.

### Week 3 (31 Aug–6 Sep) — Main experiment sweep
- [ ] **P** E1 (multi-step descent), E2 (strength bias) full sweeps, 5 seeds
- [ ] **A** E4 consolidation: basin merging, compression-vs-recall curve
- [ ] **R** E6 on LongMemEval + LoCoMo subsets
- [ ] **All** Results freeze Friday: every headline number in `experiments/results/` with CIs
- [ ] **Gate G-C:** results table complete. **Paper writing does not start before this gate.**

### Week 4 (7–13 Sep) — Writing sprint
- [ ] **P** Method + Results + Intro
- [ ] **R** Related Work + Experimental Setup (from verified bib only)
- [ ] **A** All figures, camera-ready quality, colourblind-safe, readable at print size
- [ ] **All** Full draft by Fri 11 Sep

### Week 5 (14–20 Sep) — Hardening
- [ ] Ablations complete, appendix filled
- [ ] Reproducibility check: fresh clone → `make all` → numbers match
- [ ] Guide review + external read (ask a senior/PhD student outside the team)
- [ ] Self-review against `06_RISK_REGISTER.md` reviewer attacks — write rebuttals into the paper
- [ ] Limitations section written honestly (this section wins reviewers over)

### Week 6 (21–23 Sep) — Submit
- [ ] Format to target venue, anonymize if required
- [ ] **arXiv preprint posted** (priority + it is the "proof of submission" the panel accepts)
- [ ] **Submit ≤ 23-09-2026**, screenshot the confirmation for the guide

### Post-submission (24 Sep – 28 Oct)
- Report / demo video / poster on the institutional deadlines above
- Handle reviews; if desk-rejected, immediate resubmit to the tier-2 fallback in `05_VENUE_STRATEGY.md`

## Cadence

- **Mon + Thu 21:00 IST** standups (matches the Google Meet habit already in the chat)
- Guide meeting **≥3×/week** — mandatory, individually logged on the attendance sheet.
  This is an evaluation criterion, not a formality. Sheet lives in `docs/attendance/`.
- Everything through PRs. `main` stays green.
