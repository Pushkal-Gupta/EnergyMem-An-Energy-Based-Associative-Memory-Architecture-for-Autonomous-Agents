# Submission status

**Compiles clean: 27 pages, zero overfull boxes, zero undefined references or citations.**
Built with `tectonic` (already installed at `/usr/local/bin/tectonic`).

```bash
python3 scripts/make_tables.py && cd paper && tectonic -X compile main.tex
# or: pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## Venue

| | Venue | Why | Risk |
|---|---|---|---|
| **1st** | **TMLR** | Criteria are *claims supported by evidence* + *interesting to some audience*, no novelty bar. A reduction, a taxonomy, eight diagnosed negatives and one externally validated positive is exactly the shape. Rolling submission. | Lower prestige than JMLR. |
| **2nd** | **JMLR** | Top ML journal, takes long-form methodological work. **Now materially more viable** — the external-validity hole that would have drawn a major revision is closed (X26). | Will likely still ask for a second public corpus (LongMemEval) and a downstream-task result. |
| Alt | NeurIPS D&B / position track | If the taxonomy leads over the theorem. | |

## Checklist status

### 🟢 Done this pass
- **Public-corpus association experiment (was the single biggest blocker).** `experiments/locomo_assoc.py`, LoCoMo, 5,882-turn pooled bank, 272 session clusters, 4-fold rotation, namespaced ids. **One hop: +0.043 Recall@1 [+0.036, +0.050], p<1e-4, +67% relative.** Ledger X26.
- **Two-hop resolved, negatively.** Null on public data at n=5,013 (X26b). The `0.446` figure is **withdrawn**, not suspended, and now carries a forward pointer in the ledger.
- **Appendix A: full proof.** Softmax-concentration lemma with the `O((N−1)e^{−βδ})` rate, state convergence, rank-1 agreement, the `k>1` case stated honestly as a *non*-result (it is the mechanism behind the shortlist regression), the tie case tied to low-β centroid collapse, and the scope of the empirical high-β claim separated from the theorem.
- **Appendix B** hyperparameter grids with per-baseline budget. **Appendix C** corpus construction + the three X16 failure cases in full. **Appendix F** LoCoMo protocol and full results.
- **Table/figure generation from JSON.** `scripts/make_tables.py` → `paper/generated/*.tex` + `paper/figures/assoc_hops.pdf`. Tables 3 and 6 are now generated; the script **prints which tables it could not generate** on every run.
- **Claim traceability.** `docs/13_CLAIM_TRACE.md` — 29 claims → ledger ID → script → raw output → test → paper location.
- **Test suite** 15/15.
- **Citation audit against publisher-deposited metadata.** `scripts/verify_refs.py` queries Crossref REST and the arXiv API for all 24 entries. **22 verified, 1 genuine error found and corrected, 2 not machine-verifiable and flagged inline.**
  - 🔴 **`robertson2009bm25` was wrong**: cited as *Found. Trends IR* 3(4):333–389; correct is **4(1–2):1–174** (DOI 10.1561/1500000019). Exactly the class of error the repo's citation policy exists to stop.
  - **`xu2025amem` upgraded** from arXiv preprint to the NeurIPS proceedings version (pp. 20004–20031).
  - DOIs or arXiv ids added to 20 entries; pages added where Crossref had them.
  - ⚠️ **`lecun2022path`** is in neither Crossref nor arXiv, and an arXiv title search returns a **different** document (Dawid & LeCun 2023, arXiv:2306.02572). Flagged inline; confirm the OpenReview URL and version string by hand, and do not merge the two.
  - ⚠️ **`welling2011sgld`** — ICML 2011 predates PMLR DOI coverage. Page numbers need a manual check.
  - Three year "mismatches" are correct as written (arXiv posting year vs proceedings year: ramsauer 2020→ICLR 2021, yang 2017→ICLR 2018, wu 2024→ICLR 2025); noted in the bib so nobody 'fixes' them.

### 🟢 Also done (second pass)
- **Compiled and read the PDF.** Five overfull boxes fixed, a `remark` environment added, three tables moved to `tabularx`.
- **Two rendering bugs caught by reading the pages, not the log.** A generator emitted `\\` inside table cells, which LaTeX reads as a row break: Table 4 silently split and rows spilled into the margin, **with a zero-error build**. `make_tables.py` now validates its own output and refuses malformed rows.
- **Tables 1, 2, 3, 5 and 6 are generated from JSON.** `confirm_holdout.py`, `exp_locomo_anchored.py`, `exp_multibeta_scales.py` and `sweep_beta_transition.py` all emit results files now. Both re-runs reproduced their typed values exactly — Table 1 pooled +0.014 [+0.006, +0.021] and per-level +0.003/+0.017/+0.021; Table 2 all eight conditions and the 196/196 reduction check.
- **C1 leaves an artefact, and doing so exposed a false claim.** The paper said the reduction was verified on "synthetic (*d*=256) and real (*d*=768), 1000/1000 in both". The suite was **synthetic-only at *d*=64**. A real-embedding test now exists (838/838, *d*=768) and all four places in the paper are corrected. The anisotropic case is the one that matters: random high-dimensional vectors are near-orthogonal, so δ is large and concentration is trivial.
- **Appendices D and E written.** Ledger provenance (including why pre-ledger studies are *not* back-numbered) and reproducibility: exact environment, one-command rebuild, the two environmental frictions (no CA bundle for Python; `torchvision` not in `make setup`), and what cannot be reproduced.
- **Citation audit applied** — see below.

### 🔴 Still blocking
1. **Two scripts are lost.** The X17b driver (Table 3 row i) and the X17 JEPA driver. Row (i) is corroborated by rows (ii)–(iii) and can simply be dropped. X17 (parametric failure) should be re-run before it is cited.
2. **Resolve the two manual citations** — `lecun2022path` and `welling2011sgld`, both flagged inline in `refs.bib`.
3. **Read the remaining PDF pages.** I inspected pages 1–2, 9, 11–13. Nobody has read 3–8, 14–27.
5. ~~**Citation audit.**~~ **Done** — see below. Two entries remain flagged for manual check and must be resolved by hand.
6. **Appendices D and E.** Ledger verbatim; reproducibility (commit SHA, hardware, versions, wall-clock, one-command rebuild).

### 🟡 Decide
- **Figures 2–4 from the original plan** (β stability curve, accuracy/latency Pareto, consolidation crossover) are unbuilt. The β curve is the most valuable — it is the visual evidence for C2 — and needs `sweep_beta_transition.py` to emit JSON.
- **Title.** Current: *Dense Retrieval Is Degenerate Associative Memory: A Reduction and a Regime Taxonomy for Agent Long-Term Memory*. "Degenerate" is mathematically exact (Cor. 1) and reads as rhetoric to some. Milder: *Dense Retrieval as a Limit of Associative Memory*. Appendix A.5 now separates the theorem from the empirical claim, which is the substantive protection either way.
- **Downstream task.** Every number is retrieval quality. A reviewer may ask whether +0.043 Recall@1 changes agent behaviour. Stated as a limitation.
- **Author block.** `main.tex` is anonymous; de-anonymised block commented beneath.
