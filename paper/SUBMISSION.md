# Submission status

**TMLR template applied. Compiles clean: 40 pages named / 39 anonymous, 0 overfull boxes,
0 undefined references, 0 undefined citations, 0 TODOs. Anonymity verified by extracting
the anonymous PDF's text layer (0 hits for author name, email, repo URL or local paths;
the review banner appears on all 39 pages).**

> **Phase 13 (2026-10-12) is built and installed.** An editorial pass that found a real
> miscount: "eight mechanisms" was two overlapping lists of eight, and the true Family A
> inventory is **ten**. Table 8 is now the authoritative count and sits beside its own
> section (it had been floating 19 pages away, to p37). Contributions went 7 → 3 + 1.
> Reproducibility is stated in three tiers after testing it: **only 2 of 11 data inputs are
> present in a fresh checkout**, reported by the new `scripts/check_data.py`. See
> `docs/10_EXPERIMENT_LEDGER.md` Phase 13.

> **Phase 12 (2026-10-11) is built and installed.** `main-anonymous.pdf` and
> `main-preprint.pdf` in this directory are current as of Phase 12. Phase 12 narrowed two
> theory statements (the energy-descent claim for the projected update, and Corollary 1's
> fixed-point hypotheses) and added the **unstratified** expansion rows, which answer the
> objection that C7 presupposes a benchmark hop-type label. See
> `docs/10_EXPERIMENT_LEDGER.md` Phase 12.

> ⚠️ **Phases 10 and 11 both changed what the paper claims. The committed PDFs predate
> Phase 11 and MUST be rebuilt before submission.**
>
> **Phase 10 (2026-10-10).** X113 withdrew the image result as evidence for an associative
> mechanism, so Claim 1 has no confirming instance.
>
> **Phase 11 (2026-10-11).** A phase of eleven committed experiments (X200–X212) had been
> lost from the documentation by a rebase and is now in the paper. Two consequences:
> 1. **C5 is stronger.** One-hop association replicates on **MSC** (+0.016 [+0.014,+0.018],
>    27,815 probes, 30,320-turn bank) — a second public corpus.
> 2. **C7 is new, and it contradicts the thesis.** Two-stage query expansion **beats** the
>    dense corner on chain-shaped multi-hop questions (+0.023 HotpotQA bridge, +0.171 2Wiki
>    inference). The paper's old blanket claim that no departure from the corner pays was
>    false and is withdrawn. It is sharply bounded: query expansion rather than attractor
>    dynamics, no cross-corpus transfer, null on both conversational corpora.
>
> `limitations.tex` also carried a **false statement** until Phase 11 — that LongMemEval
> "is the obvious second corpus and we have not run it" — while `x10x_longmemeval.json`
> sat in the tree. Any PDF built before 2026-10-11 contains it.

> Built with `tectonic` on 2026-10-11 (this machine has no `pdflatex`/`latexmk`, so
> `scripts/rebuild_paper.sh` cannot run as written here; `appendix.tex` now records this
> and gives the `tectonic` invocation instead of the old `pdflatex` recipe).

## Pre-submission checks, and how to re-run them

```bash
python scripts/make_tables.py            # regenerate all 9 tables from JSON
python scripts/verify_paper_numbers.py   # 80 TYPED prose numbers vs the JSONs
python scripts/check_data.py             # which corpora a fresh checkout lacks (2/11 present)
python scripts/verify_refs.py /tmp/r.json # 31 citations vs Crossref/arXiv
python -m pytest -q                      # 31 tests, incl. the Prop. 1 gate
```

Last run, 2026-10-11 (Phase 12): **80/80 numbers agree**, 31/31 citations resolve (the one known gap,
`welling2011sgld`, is a Crossref hole for early PMLR and is flagged in `refs.bib`), 31/31
tests pass. `verify_paper_numbers.py` is new this phase and immediately earned itself: it
caught `+0.094` where the file says `0.09350`, a mis-rounding that had propagated from the
ledger into the draft.

```bash
bash scripts/rebuild_paper.sh     # regenerates every number, then builds BOTH PDFs
```

| file | what it is |
|---|---|
| `paper/main-anonymous.pdf` | **submit this.** `\usepackage{tmlr}` — "Under review as submission to TMLR", anonymous, contributions suppressed |
| `paper/main-preprint.pdf` | named version, `\usepackage[preprint]{tmlr}` — for arXiv, circulation, camera-ready |

> ⚠️ **`paper/main.pdf` is a build artefact, not a deliverable.** It is gitignored and is
> whatever the last `tectonic` run produced (currently the *named* build). Submit
> `main-anonymous.pdf`. Do not treat `main.pdf` as equivalent to `main-preprint.pdf`.

⚠️ **TMLR's style file is explicit: "Non-anonymous submissions will be rejected without
review."** The default build is the *named* one, so submit `main-anonymous.pdf`, or flip the
line marked `[SUBMIT]` in `main.tex` before building.

## Venue

| | Venue | Why | Risk |
|---|---|---|---|
| **1st** | **TMLR** | Criteria are *claims supported by evidence* + *interesting to some audience*, no novelty bar. A reduction, a taxonomy, ten diagnosed negatives, one externally validated positive on two public corpora, and one bounded positive that contradicts our own thesis is exactly the shape. Rolling submission. | Lower prestige than JMLR. |
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
