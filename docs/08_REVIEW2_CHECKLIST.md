# Review 2 — Checklist

**Calendar date: 12-08-2026 (tentative).** The team's plan in chat was "CAT ke baad."
**Action today (11 Aug): confirm the actual slot with Dr. Thangaramya K.** Note the
standing instruction — *"No re-review will be conducted under any circumstances."*

## What the panel evaluates at Review 2

Per the SCOPE calendar, Review 2 covers everything from Review 1 **plus**:

| # | Item | Status | Source |
|---|---|---|---|
| i | Title | ✅ locked, VTOP-updated | — |
| ii | Abstract | ✅ have it | EnergyMem deck slide 3 |
| iii | Literature Survey | ⚠️ needs the 2026 update | see `07_CITATION_AUDIT.md` |
| iv | Objectives | ✅ have them | deck slide 6 |
| v | **Methodology** | ⚠️ needs the energy formulation written out | `01_RESEARCH_PLAN.md` §2 |
| vi | **Implementation** | ❌ **the gap** | build it |
| vii | **Demo** | ❌ **the gap** | build it |
| viii | Implementation to be *completed* | ❌ partial is fine, be honest | — |
| ix | Best-project nomination | — panel's call | — |

Also mandatory on the day:
- [ ] **Project Attendance Record** (guide-meeting sheet), individually maintained, per member
- [ ] **Guide's approval signature** confirming the PPT has been verified — get this *before* the review
- [ ] Deck approved by the guide in advance

## Minimum viable demo (one evening of work)

You do not need the full architecture. You need something that **runs live, on your laptop,
in under 60 seconds**, and shows the core idea working. Scope it to exactly this:

1. Embed ~1 000 short episodic memories with `all-mpnet-base-v2`
2. **Baseline:** cosine top-1 retrieval
3. **EnergyMem:** Hopfield/CCCP descent, `T=8`, β annealed
4. **The money shot:** take a query, mask 50 % of its tokens, show cosine top-1 fails and
   energy descent recovers the correct memory
5. Plot: masking severity (x) vs Recall@1 (y), two lines
6. Print the honest cost: latency for both

That is `scripts/demo_review2.py`. It is genuinely ~150 lines and it directly demonstrates
items vi, vii and viii.

**Also show the reduction test passing** (`T=1, β→∞` reproduces cosine top-1 exactly). It
takes ten seconds to run and it is the single most impressive thing you can put in front of
a panel: it proves you understand your own baseline is a special case of your method.

## Slides to update from Review 1

Start from `EnergyMem_Review_1_Expanded.pptx` (the narrow one), not the BEM deck.

- **Slide 4 (Lit Review):** add PAM (arXiv:2602.11322), HeLa-Mem (ACL 2026), DMHN
  (arXiv:2506.01303), LongMemEval-V2. Showing you tracked *2026* literature since Review 1
  is exactly the kind of progress a panel rewards.
- **Slide 5 (Research Gap):** replace with the unification framing from
  `01_RESEARCH_PLAN.md` §6. Be explicit that the gap narrowed and you re-scoped in response —
  panels respect that far more than a gap statement that has not moved since July.
- **NEW — Methodology:** the energy functional, the CCCP update, the three generalization
  axes table. One slide.
- **NEW — Implementation:** repo screenshot, module structure, what runs today.
- **NEW — Demo:** live, or screen-recorded as a fallback if the venue Wi-Fi is bad.
  **Record the fallback video the night before regardless.**
- **NEW — Results so far:** the masking curve, even if preliminary. Label it preliminary.
- **NEW — Plan to Review 3:** the `02_TIMELINE.md` gate chart + ICLR 2027 target (25 Sep).
  Naming a specific Tier-1 venue with a specific date reads as seriousness.

## Questions the panel will ask — have answers ready

1. *"How is this different from a vector database?"*
   → Top-k cosine is the `T=1, β→∞` special case of our energy. We generalize it along three
   axes and measure which ones pay. Show the reduction test.
2. *"Isn't Hopfield retrieval just attention?"*
   → Yes, for the one-step case — that is Proposition 1 in our paper, we state it ourselves.
   The contribution is what happens when you leave that corner. (See `06_RISK_REGISTER.md` R-1.)
3. *"What is your novelty given PAM/HeLa-Mem already exist?"*
   → The unification into one energy objective, plus the first matched-encoder comparison of
   all three. **Do not pretend they don't exist** — the panel may know them, and volunteering
   the competitive landscape is far stronger than being caught by it.
4. *"What is the SDG and TRL?"*
   → SDG 9 (Industry, Innovation and Infrastructure). TRL 2 → 3 (concept → experimental proof
   of concept). Already in the Review-1 objectives.
5. *"What is your outcome and when?"*
   → ICLR 2027, paper deadline 25 Sep 2026; arXiv preprint by 22 Sep as proof for the 23-09
   institutional deadline. Journal extension to IEEE TNNLS / Neural Networks after.
6. *"How much is done?"*
   → Be precise and honest. A panel can tell. "Core retrieval and baselines run; DegradeBench
   and the agent benchmarks are Weeks 2–3" is a good answer. "Almost done" is not.

## Day-before checklist

- [ ] Guide signature on attendance sheet — **each member individually**
- [ ] Deck approved by guide
- [ ] Demo runs offline (no Wi-Fi dependency), fallback video recorded
- [ ] Laptop charged, HDMI/USB-C adapter, deck on a USB stick as backup
- [ ] All three members know the answers to the six questions above
- [ ] Print the attendance sheets
