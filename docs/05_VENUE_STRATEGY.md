# Venue Strategy

**Institutional constraint:** Project Outcome — proof of submission to a Patent /
Scopus conference / Scopus journal — due **on or before 23-09-2026**.

**Your stated goal:** Tier 1.

---

## The lucky alignment

**ICLR 2027 abstract deadline: 18 Sep 2026. Full paper: 25 Sep 2026 (AOE).**

That is within days of your 23-09 institutional deadline. ICLR is a genuine Tier-1 venue
(CCF-A, top-3 in ML), it is the *natural* home for an associative-memory/agent-memory
paper, and its OpenReview model means even a rejection produces public reviews that make
the journal version dramatically stronger.

> Note: ICLR 2027 mandates an **AI-use statement**. Given how much of the Review-1 material
> came from Gemini/Genspark/ChatGPT, read that policy carefully and disclose honestly.
> This is a real compliance item, not a formality.

## Recommended plan — dual track

### Primary: ICLR 2027 (conference, Tier 1)
- Abstract by **18 Sep**, paper by **25 Sep**
- Submit the arXiv preprint on ~22 Sep → that is your "proof of submission" for the 23-09
  panel deadline, and it timestamps priority
- 9 pages + unlimited appendix. Our experiment set fits this shape well.

### Secondary: journal version (Tier 1, no deadline pressure)
After ICLR reviews land (~Nov 2026), extend into a journal submission. Ranked by fit:

| Venue | Type | Fit | Notes |
|---|---|---|---|
| **IEEE TNNLS** | Q1, IF ~10 | ★★★★★ | Neural networks + associative memory is *exactly* its scope. Best journal target. |
| **Neural Networks** (Elsevier) | Q1, IF ~6 | ★★★★★ | Hopfield/attractor work's traditional home. Very strong fit. |
| **TMLR** | Rolling, no deadline | ★★★★☆ | Reviews on *correctness and evidence*, not novelty-hype. Ideal if results are honest-but-mixed. Not Scopus-"journal-ranked" in the classic sense — **confirm with your guide it counts**. |
| **IEEE TKDE** | Q1, IF ~8 | ★★★☆☆ | Fits only if the retrieval/indexing framing dominates |
| **Neurocomputing** | Q1, IF ~6 | ★★★☆☆ | Faster, lower bar. Solid fallback. |
| **IEEE TAI** | Q1, newer | ★★★☆☆ | Reasonable fallback |

You already have IEEE Transactions LaTeX experience from the location-privacy paper — the
formatting cost for TNNLS is near zero. Reuse that template.

## Fallback ladder (in order, if things slip)

1. **ICLR 2027** — 25 Sep 2026 ✅ *primary*
2. **ICLR 2027 workshops** — CFPs ~Dec 2026. Lower bar, real venue, keeps momentum
3. **ICML 2027** — deadline typically late Jan 2027. Fits Review-3 (28 Oct) → journal window
4. **TMLR** — rolling, submit any time, ~6-week turnaround
5. **Neurocomputing / Neural Networks** — no deadline, straight journal path

**If Week-2 Gate G-B fails** (routing does not work, see `06_RISK_REGISTER.md` R-3):
skip ICLR, go to TMLR or Neural Networks with the characterization/negative-result framing.
Both venues explicitly welcome rigorous negative results; ICLR mostly does not.

## What "Tier 1" actually requires — a candid word

Tier-1 acceptance rates are 20–30 %, and reviewers are comparing you to full-time PhD
students with GPU clusters. Three things get a paper like yours over the line, and none of
them is scale:

1. **One crisp, falsifiable claim** — you have it (the reduction + which departures pay)
2. **Airtight, matched-baseline experiments** — fully within your control, and the single
   biggest differentiator for a small team
3. **Unusual honesty about limitations** — reviewers reward papers that pre-empt their own
   objections. Your Review-1 QA slide already showed the team has this instinct. Keep it.

What you *cannot* compete on: model scale, compute, breadth. So do not try. The narrow
scope in the charter is not a compromise — it is the strategy.

Realistically: ICLR 2027 is a genuine shot but not a favourite. **TNNLS or Neural Networks
is the higher-probability Tier-1 outcome**, and going through ICLR review first materially
improves that submission. Plan for both; be disappointed by neither.

## Pre-submission checklist

- [ ] Bibliography 100 % verified (`07_CITATION_AUDIT.md`) — no exceptions
- [ ] Reproducibility statement + public repo (release at submission, anonymized if required)
- [ ] AI-use statement (ICLR 2027 requirement)
- [ ] Limitations section — written by us, not extracted by reviewers
- [ ] Ethics/broader-impact: persistent agent memory has real privacy implications.
      Cite the episodic-memory-risk work and say something substantive.
- [ ] All co-authors have read the final PDF end to end
- [ ] Guide (Dr. Thangaramya K) listed and has approved the submission
- [ ] Confirm authorship order and affiliation format with the guide **early**, not in week 6

Sources:
- [ICLR 2027 Call for Papers](https://iclr.cc/Conferences/2027/CallForPapers)
- [ICLR 2027 Author Guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines)
- [AAAI-27 Main Technical Track Call](https://aaai.org/conference/aaai/aaai-27/main-technical-track-call/) (deadline 28 Jul 2026 — already passed)
