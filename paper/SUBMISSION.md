# Submission status

**TMLR template applied. Compiles clean: 25 pages named / 24 anonymous, 0 overfull boxes,
0 undefined references, 0 undefined citations, 0 TODOs.**

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

One public benchmark and one task shape; association does not compose and we do not know why; retrieval quality is not downstream task success; capacity crossover at N≈800 unresolved; the shortlist regression (R@5 0.605 vs cosine 0.759) means the positive is narrower than "better retrieval"; G2 and the controller are specified and unbuilt.
