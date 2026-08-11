# Risk Register

Ordered by expected damage. R-1 through R-3 are the ones that decide whether this paper
exists.

---

## R-1 — "This is just attention with extra steps" 🔴 CRITICAL

**The attack.** Modern Hopfield retrieval is `X·softmax(β Xᵀξ)`. Over L2-normalized
patterns that is softmax-weighted cosine similarity — i.e. attention. One step at high β
returns the top-1 cosine neighbour. A reviewer concludes the paper reimplements its own
baseline under a new name.

**Probability it is raised: ~100 %.** It is the first thing anyone who knows the field
will say.

**Mitigation — do not defend, concede and build on it.**
- State the reduction as **Proposition 1** in Section 3, before the reviewer can.
- Frame the contribution as *characterizing which departures from that special case pay off*.
- Make G3 (learned routing `R`) the headline, because it is provably outside the span of
  any reweighting of `Xᵀq` — no reranking of cosine scores can produce it.
- Implement the baseline **as a configuration of our own code**, not as separate code. Then
  the comparison is exact by construction, and you can say so.

**Residual risk after mitigation:** low. This actually becomes a strength.

---

## R-2 — Hallucinated citations 🔴 CRITICAL

**The problem.** The entire 30-paper Review-1 bibliography came out of Gemini, Genspark and
ChatGPT sessions. Several entries look synthesized. One fabricated reference found by a
reviewer contaminates their trust in every number in the paper.

**Mitigation:** `07_CITATION_AUDIT.md`. Every single entry resolved against the actual
arXiv/DOI page. Owner: Rajit. Deadline: before Week 1 ends. Zero tolerance.

**Residual risk:** eliminated if the audit is done properly. Catastrophic if skipped.

---

## R-3 — Learned routing (G3) does not work 🟠 HIGH

**The problem.** If `R` fails to recover low-cosine temporal associations, the only
provably-novel component collapses and the paper is "we ran Hopfield on embeddings."

**Why it might fail:** sentence embeddings are highly anisotropic; a linear `R` may not have
capacity; contrastive training on a small co-occurrence set may not generalize; routing may
degrade normal retrieval more than it helps associative retrieval.

**Mitigation:**
- **Run E3 in Week 2, not Week 4.** This is the single most important scheduling decision
  in the whole plan. Gate G-B exists for exactly this.
- Have escalation ladder ready: linear `R` → low-rank `R` → small MLP → sparse memory-graph
  operator learned from co-occurrence counts.
- **Pre-planned pivot:** if it still fails, the paper becomes
  *"When does energy-based associative memory help LLM-agent retrieval? A controlled study"* —
  a characterization paper with a well-supported negative result, the reduction theorem, and
  DegradeBench as the durable artifact. Retarget TMLR / Neural Networks (both accept this;
  ICLR mostly does not).

**This pivot is not failure.** A rigorous negative result with a released benchmark is a real
publication. What *would* be failure is discovering it on 20 September.

---

## R-4 — Compute cost sinks the practical argument 🟠 HIGH

**The problem.** Iterative descent over N memories, T steps each, versus one ANN lookup.
If EnergyMem is 10× slower for +1 Recall@1, no practitioner adopts it and reviewers notice.

**Mitigation:**
- Two-stage: ANN shortlist (M ≈ 256) → energy descent within the shortlist. Report as an
  intentional design, and show the accuracy cost of the shortlist.
- Report the **accuracy/latency Pareto frontier**, not a single operating point.
- Put the cost in the abstract. Owning a limitation neutralizes it; hiding it amplifies it.

---

## R-5 — Embeddings too correlated for clean attractors 🟡 MEDIUM

**The problem.** Hopfield capacity results assume well-separated patterns. Real sentence
embeddings occupy a narrow cone. Expect metastable blend states rather than clean convergence.

**Mitigation:** measure it. Report blend rate, non-convergence rate, energy margin
distribution. Try whitening / ZCA on the memory matrix as an ablation. **A characterization
of when embedding geometry defeats attractor dynamics is a genuine contribution** — this is
a risk that converts into a result if you instrument it properly.

---

## R-6 — Team velocity / the pattern in the chat history 🟠 HIGH

**The problem.** Six months of chat show a repeated cycle: meeting proposed → date slips →
work compresses into the night before the deadline. That worked for Review 1 (slides). It
will not work for an experimental paper, because experiments have wall-clock that cannot be
compressed.

**Mitigation:**
- Weekly gates with explicit pass/fail (`02_TIMELINE.md`), not open-ended "let's meet"
- The `BLOCKED` escalation rule (`03_TASK_ALLOCATION.md`) — 24 h maximum silent blockage
- Guide meetings ≥3×/week are institutionally mandated anyway; use them as forcing functions
- Fixed Mon/Thu 21:00 standups matching the habit already visible in the chat
- **CAT exams:** budget for a genuine dead week and plan the schedule around it rather than
  pretending it will not happen

---

## R-7 — Baselines too weak 🟡 MEDIUM

**The problem.** Comparing against naive cosine top-k only. Reviewers assume you tuned yours
and not theirs — the most common cause of rejection for papers of this shape.

**Mitigation:** BM25, hybrid RRF, cosine + cross-encoder reranker, and ≥2 published
agent-memory systems with runnable code. Tune baseline hyperparameters with the *same* budget
as EnergyMem's and report that budget explicitly.

---

## R-8 — Scope creep back toward BEM 🟡 MEDIUM

**The problem.** The BEM deck is genuinely exciting and it will be tempting to add the
planner, or the four-memory-system taxonomy, "since we already described it."

**Mitigation:** the charter says no. Anything outside C1–C4 goes into `docs/FUTURE_WORK.md`,
not into the paper. Re-read `00_PROJECT_CHARTER.md` §2 at every gate.

---

## R-9 — Benchmark access / reproducibility friction 🟢 LOW

LongMemEval and LoCoMo licences, LLM API costs for paraphrase generation and the reader model.
Check access in the pre-week; budget API spend early. Have a fully-offline fallback
(local model for paraphrase) so no experiment is blocked on a credit card.

---

## R-10 — Authorship / guide approval friction 🟢 LOW

Settle author order, affiliations and the guide's role **in August**, not in week 6.
Confirm the guide is comfortable with the target venue and with arXiv preprinting.
