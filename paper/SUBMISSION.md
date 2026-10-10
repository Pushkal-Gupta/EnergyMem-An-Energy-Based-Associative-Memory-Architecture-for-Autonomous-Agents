# Submission status

**TMLR template applied. Compiles clean: 30 pages named / 29 anonymous, 0 overfull boxes,
0 undefined references, 0 undefined citations, 0 TODOs.**

> ⚠️ **Phase 10 changed what the paper claims.** X113 withdrew the image result as evidence
> for an associative mechanism, so Claim 1 now has no confirming instance and the paper's
> empirical verdict is uniformly negative for the energy machinery. The surviving positive
> is the nonparametric predictor, which is not associative retrieval. Do not submit a PDF
> built before 2026-10-10 Phase 10.

> Built with `tectonic` on 2026-10-10 (this machine has no `pdflatex`, so
> `scripts/rebuild_paper.sh` cannot run as written here).

```bash
bash scripts/rebuild_paper.sh     # regenerates every number, then builds BOTH PDFs
```

| file | what it is |
|---|---|
| `paper/main-anonymous.pdf` | **submit this.** `\usepackage{tmlr}` — "Under review as submission to TMLR", anonymous, contributions suppressed |
| `paper/main-preprint.pdf` | named version (= `main.pdf`), `\usepackage[preprint]{tmlr}` — for arXiv, circulation, camera-ready |

⚠️ **TMLR's style file is explicit: "Non-anonymous submissions will be rejected without
review."** The default build is the *named* one, so submit `main-anonymous.pdf`, or flip the
line marked `[SUBMIT]` in `main.tex` before building.

## Venue

| | Venue | Why | Risk |
|---|---|---|---|
| **1st** | **TMLR** | Criteria are *claims supported by evidence* + *interesting to some audience*, no novelty bar. A reduction, a taxonomy, eight diagnosed negatives and one externally validated positive is exactly the shape. Rolling submission. | Lower prestige than JMLR. |
| **2nd** | **JMLR** | Top ML journal, takes long-form methodological work. The external-validity hole that would have drawn a major revision is now closed (X26). | Will likely ask for a second public corpus and a downstream-task result. |
| Alt | NeurIPS D&B / position track | If the taxonomy leads over the theorem. | |

**Recommendation: TMLR.**

## Ready

- **Every number is reproducible.** Both lost drivers rewritten; no row depends on code nobody can run. `scripts/rebuild_paper.sh` rebuilds the paper end to end.
- **Every results-backed table generated from JSON.** `make_tables.py` validates its own output and prints any table it could not generate. Only Table 5 (negatives) is hand-assembled, deliberately — it is prose, not numbers.
- **All 24 new citations verified** against Crossref / arXiv / the official ICML 2011 BibTeX. One genuine error found and corrected (`robertson2009bm25`: 3(4):333–389 → **4(1–2):1–174**). Internal provenance moved to `annote` so it no longer prints.
- **Statements written** — code/data availability, reproducibility, funding, competing interests, ethics.
- **Every page read.** Issues found and fixed: a bibliography leaking internal notes, Table 4 silently split by an escaped backslash, a clipped axis label, a stale "two implementations" in the conclusion, a stale C2 bullet, a stale corpus caveat, a Table 5 caption promising a column that did not exist, and a latency figure (8×) that had never been checked against a clock (21×).
- **Appendices A–G complete**: full proof with the `O((N−1)e^{−βδ})` rate; grids with per-baseline budget; corpus construction; ledger provenance; LoCoMo protocol; statements; reproducibility.

## Before you submit

1. **Submit `main-anonymous.pdf`**, not `main.pdf`. See the warning above.
2. **Attach supplementary material:** `docs/10_EXPERIMENT_LEDGER.md` verbatim (Appendix D promises it), and the repository link once it is public.
3. **Fill in the OpenReview link** in `main.tex` (`\def\openreview`) at camera-ready, and switch `[preprint]` to `[accepted]`.
4. **Table 3 row (i) is kept, deliberately.** It is now n.s. (+0.024, p=0.24). Dropping an unfavourable row after seeing the final results is worse than showing it, and the three rows together make the paper's own methodological point: the point estimate stays positive while the interval narrows from [−0.032,+0.083] at n=123 to [+0.036,+0.050] at n=5,233. That is what a real small effect looks like.

## Known limitations, stated in the paper

One public benchmark and one task shape; association does not compose and we do not know why; retrieval quality is not downstream task success; capacity crossover at N≈800 unresolved; the shortlist regression (R@5 0.605 vs cosine 0.759) means the positive is narrower than "better retrieval"; the lifecycle operations built on G2 (decay, consolidation) are implemented and unevaluated; selection among *regimes* is unbuilt.

**Updated 2026-10-10.** Three statements the paper made about itself were stale or wrong and have been corrected (`.tex` edited, all three PDFs rebuilt):

1. **Proposition 1 over-claimed the top-*k* case** in the main text — "reading the top-$k$ by $\langle z_i,\xi^{(1)}\rangle$ therefore returns exactly the dense top-$k$ cosine neighbourhood of $q$" — contradicting Appendix A's own `Remark [Ranks 2..k do not agree, and this is a result, not a gap]`. No test ever covered the sentence (`test_reduction.py` checks k=1; the 196/196 "top-5 set equality" is at `anchor=1.0`, which *is* cosine by construction). Measured: top-1 **100/100**, top-10 **0/100** (X98). The proposition now claims exactness at k=1 only.
2. **G2 and the controller are no longer "unbuilt"** — both were evaluated and both fail (X62, X63, X65-r). Limitations and the taxonomy section say so.
3. **A win was withdrawn**: X65's query-conditional β was scored against β=16 while its own output called that "best fixed". Against a fold-selected fixed β it loses by −0.089 [−0.121,−0.061] (X65-r).

Named build unchanged at 25 pages; the anonymous build goes **24 → 25** (the corrected Limitations and taxonomy paragraphs are longer than the text they replace, and the anonymous build has no Contributions section to absorb it). Verified by rebuilding `HEAD`'s own sources for comparison. 0 overfull boxes, 0 undefined references, 0 undefined citations in both.


## Updated 2026-10-10 (second pass) — the central claim is narrowed

`taxonomy.tex` had nominated a text-occlusion study as *"the experiment most likely to
falsify our own central claim"* and committed in advance to the reading if it failed. It was
pre-registered (`experiment-backlog/PREREG_X110.md`), run, and **it failed**: 0 of 45
(β, level) cells beat cosine on recall@1, 0 of 45 on recall@5. A second new cell — a cue
that is a degraded copy of *two* stored images — loses −0.245. So the completion premise is
now reported as **necessary, not sufficient**: it predicts every loss we measure and does not
predict the one win, which is single-target image completion.

Changed: `abstract.tex`, `intro.tex` (C3 + closing), `taxonomy.tex` (claim narrowed,
§"not yet run" → §`sec:occlusion` with the result, regimes-table row footnoted),
`conclusion.tex`, `limitations.tex` (X35: the retrieval gain does not reach an answer-coverage
proxy, +0.007 [−0.013,+0.027]; and a caveat that our PRF never helps anywhere, including on
SciFact), `results.tex` (I4 is confirmed 20/21 cells in a clean setting **and** irrelevant,
0/21 beat the cue alone).

**Deliberately still out of the paper:** the LoCoMo-QA evidence-retrieval task and RRF
beating the architecture on four corpora. That remains a second paper, as Phases 7–8
decided. Page count 25 → 27 named, 25 → 26 anonymous.


---

## Phase 10 update — the image result is withdrawn as evidence (X113)

Found by reading our own scoring code, not by running an experiment:
`scripts/confirm_holdout.py:120` scores a hit when the retrieved memory shares the query's
**class** (ten classes), and the bank is drawn from `train` while the cues come from `val`,
so **a cue's source image is never stored**. The flagship "image pattern completion" result
was therefore ten-way classification on cues that did not satisfy the premise it was
offered as confirming.

Pre-registered in `experiment-backlog/PREREG_X113.md` before the driver existed. Both
pre-committed endpoints came back negative — the fourth row of that document's outcome
table, written in advance as "a real possibility and the reason this is pre-registered":

| endpoint | result |
|---|---|
| X113a descent − soft k-NN class vote (published metric) | **−0.036 [−0.043,−0.028]**, negative at all three occlusion levels |
| X113b descent − cosine, same-**image** recall@1, cue = occluded **stored** image | **−0.003 [−0.008,+0.000]**, identity control 1.0000, β selected = 80 (top of grid) |

Descent's published +0.014 reproduces exactly. At β=40 its first step blends 12.7 patterns
of 400 with 48% of the mass same-class, i.e. an implicit class vote — and the pre-declared
grid selected k=20 for the explicit one.

Changed: `abstract.tex` (image claim rewritten; the premise now has no confirming
instance), `intro.tex` (premise paragraph, C3, C6, closing), `taxonomy.tex` (claimbox, the
"satisfied and descent wins" paragraph, regimes-table Completion row → **plain cosine**, new
X113 subsection in §`sec:occlusion`), `results.tex` (§completion retitled "descent beats
cosine, and loses to a plain classifier"; both tables; multibeta caption),
`setup.tex` (the two load-bearing properties of the image design, previously unstated),
`method.tex` (band figure caption; the interior optimum re-read as blend width),
`limitations.tex` (the defect as a process limitation), `protocol_errors.tex` (**seventh
error: unmatched baseline**; six → seven propagated through abstract, intro C6,
conclusion), `appendix.tex` (budget-table rows).

Generated tables: `tab_images.tex` gains the vote column, `tab_completion.tex` is new, both
from `experiments/results/x113_class_vote.json` via `scripts/make_tables.py`. The
generator's own `_validate` guard rejected the first version of the new table; `\cmidrule`
is now in its structural exemptions.

**Still deliberately out:** the LoCoMo-QA evidence-retrieval task and RRF beating the
architecture on four corpora. Page count 27 → 30 named, 26 → 29 anonymous.

**One housekeeping item left, and it is the user's call:** `paper/main-article.tex.bak` is
still tracked. It is a backup, not a source; `git rm --cached paper/main-article.tex.bak`
removes it from the index without deleting the file.
