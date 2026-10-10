# Pre-registration — X110, text occlusion

Written **before** the driver was run. Same caveat as `PREREG_X32.md`: written-before,
committed-with, because commits in this repository are the author's and not the agent's.
The ledger entry for X110 records the order.

## Why this one matters more than the rest of the backlog

`paper/sections/taxonomy.tex` names this experiment itself:

> The cheapest discriminating test is a text occlusion study: hold the LoCoMo corpus and
> encoder fixed and swap the queries from questions to *masked turns*. Only the query type
> changes. Claim 1 predicts that descent regains its advantage; if it does not, the premise
> is wrong and the modality explanation is right. **We flag this as the experiment most
> likely to falsify our own central claim**, and its absence as a limitation rather than as
> future work.

So this is not a backlog item. It is the paper's central claim put at risk by the paper's
own choice of test, and the paper has already committed to the interpretation of both
outcomes. That commitment is what makes the result worth anything, and it is why this
pre-registration exists before the driver does.

## The claim under test

**Claim 1 (the completion premise).** *Associative descent improves retrieval when, and
only when, the cue is a degraded instance of a stored pattern.*

The image half is already evidence for it: occlusion is a degradation, and descent's
advantage grows monotonically with severity ($+0.003$ n.s. at 30% → $+0.021$ at 70%). The
semantic-match half is already evidence for it: a LoCoMo question is not a degraded turn,
and descent loses monotonically. **What is missing is the cell that separates the claim
from "descent works on images and not on text."**

A masked LoCoMo turn is a degraded instance of a stored pattern, in text, on the corpus
and encoder where descent currently loses. Claim 1 and the modality explanation make
opposite predictions about it.

## Design

| | |
|---|---|
| Bank | all turns of a LoCoMo conversation, same loader, same encoder (`all-mpnet-base-v2`) as every other experiment |
| Query | a **masked copy of one stored turn** — a contiguous span of its words removed |
| Gold | that turn, and only that turn (single-target: so `recall@1` is the natural metric and the `setmetrics` multi-target machinery is not needed) |
| Mask levels | **0.3, 0.5, 0.7** — the three levels the image study used, to make the dose-response curves comparable. 0.9 is run as exploratory and labelled so. |
| Mask shape | **contiguous span (primary)** — `occlude()` blacks out a square *patch*, so a span is the faithful analogue. Random-word deletion is reported as a secondary variant. |
| Eligible turns | $\geq 8$ words, so that 70% masking leaves something |
| Methods | cosine vs. energy descent, $T{=}8$, matching `EnergyMemory(n_steps=8)` in the image study |
| $\beta$ | selected on a **disjoint tune half of the conversations**, then one pre-committed comparison on the held-out half — the X15 lesson, same staging as `confirm_holdout.py` |
| Inference | paired bootstrap resampling the **conversation**; intervals, not $p$. Ten conversations floor a blocked-permutation $p$ at $2/1024$ (X99) |

## Predictions, committed in advance

Writing $\Delta_\ell$ for descent minus cosine `recall@1` at mask level $\ell$:

1. **$\Delta_{0.7} > 0$** — the primary endpoint. Claim 1 says a heavily degraded copy is
   exactly the regime descent is for.
2. **$\Delta_{0.3} \leq \Delta_{0.5} \leq \Delta_{0.7}$** — monotone dose-response, the
   same shape as the image curve. This is the part that is hard to get by accident.
3. **$\Delta_{0.3} \approx 0$** — little has been removed, so there is little to restore.

## Outcome table, committed in advance

| outcome | reading |
|---|---|
| $\Delta_{0.7} > 0$ **and** monotone | **Claim 1 survives its own hardest test.** The taxonomy is about query type, not modality, and the limitation in `taxonomy.tex` is discharged. |
| $\Delta_{0.7} > 0$, not monotone | Partial support. The premise predicts the sign but not the dose-response; report both and weaken the dose-response language. |
| $\Delta_{0.7} \approx 0$ | **Claim 1 is not supported on text.** The image result is then about images, and the paper's central claim must be narrowed to the modality the evidence covers. |
| $\Delta_{0.7} < 0$ | **Claim 1 is falsified.** Descent loses even where the premise is satisfied. The modality explanation wins and the taxonomy's central row is wrong. |

## Pre-committed kill criteria and controls

- **Cosine must not be at ceiling.** If cosine `recall@1` $> 0.95$ at 70% masking there is
  no headroom and the cell is uninformative regardless of sign; report it as such and read
  the 0.9 level instead. A text span is far more redundant than an image patch and this is
  the most likely way for the experiment to say nothing.
- **The identity control.** At mask level 0, the query *is* the stored turn. Cosine must
  score `recall@1` $= 1.0$. If it does not, the bank or the gold mapping is broken.
- **The shortlist regression must be checked, not assumed.** The image result improves
  rank 1 while degrading ranks 2–5. `recall@5` is reported at every level. If descent wins
  at rank 1 and loses at rank 5 here too, that is the same trade and must be stated with
  the win, not after it.
- **$\beta$ is selected once, on the tune half, before the held-out comparison is
  computed.** No second look. The held-out number stands whatever it is.
- **A sign-reversed sanity check**: descent at a $\beta$ far below the band should be
  clearly worse than cosine. If every $\beta$ looks the same, the descent is not running.
