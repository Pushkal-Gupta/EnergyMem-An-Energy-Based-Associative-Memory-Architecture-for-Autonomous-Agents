# Pre-registration — X32, the mixture × target-cardinality interaction

Written **2026-10-09, before any X31/X32 number existed.** Committed in the same commit
as the drivers and before the result JSONs. If this file and a result disagree, this file
is the record of what was predicted.

**Honest caveat about this file's own status.** It was written before the run inside a
single working session, but it was *not* committed to git before the run, because commits
on this project are made by the author and not by the agent. The strongest available
version of this claim is "written before, committed with" — not "committed before". A
reader should discount accordingly, and anyone repeating this on a new corpus (X102)
should commit the prediction first.

## Background

Invariant I4 of `docs/12_ARCHITECTURE.md` ("never average modes") was **falsified** on
both corpora in X23–X25 and again on public data in X26c (mixture 0.075 vs averaged
0.107). The architecture document carries the `[FALSIFIED]` marker and the paper reports
it as a falsification.

Every experiment behind that falsification had a **single** retrieval target. A mixture
over `K` modes cannot beat a single well-placed query when there is exactly one correct
answer: the extra modes can only add mass away from it. So I4 was tested in the one
regime where it cannot win, and the falsification may be a regime artifact rather than a
refutation of the invariant.

LoCoMo's own annotations give the missing regime for free. Category 1 (multi-hop) has a
mean of 3.13 gold evidence turns per question with 95.4% spanning more than one session;
category 4 (single-hop) has 1.06. Category 4 is therefore a within-benchmark replication
of the regime where I4 was already falsified, measured with the same code, the same
encoder and the same folds as category 1.

## Prediction

Let `Δ_c = mean(mixture) − mean(averaged)` on category `c`, same modes, same weights,
differing **only** in the combination rule (`logsumexp` vs weighted mean — see
`s_mixture` and `s_mixture_avg` in `experiments/locomo_qa.py`, which share their
hypothesis construction line for line).

1. **`Δ_1 > 0`** — mixture beats averaging on multi-hop.
2. **`Δ_4 ≈ 0`**, and specifically `Δ_4` not reliably positive.
3. **`Δ_1 − Δ_4 > 0`** — the interaction. This is the primary endpoint.

Primary metric: **`gold@10`** (mean fraction of the gold set in the top 10). Chosen
because `full@10` floors out on category 1 and loses resolution. Secondary: `full@10`,
`gold@5`, `mrr`, `ndcg@10`.

Primary test: paired permutation on per-question differences, signs flipped **within
conversation** (`blocked_permutation`). Interval: two-level bootstrap over conversations
then questions (`hierarchical_bootstrap`). Significance at `p < 0.05` two-sided.

Hyperparameters (`K0`, `β`, `γ`) selected per fold on the nine training conversations
only, never on the held-out one, and the per-fold selections are reported.

## What each outcome means

| outcome | reading |
|---|---|
| `Δ_1 > 0`, `Δ_4 ≈ 0`, interaction significant | **I4 holds in the multi-target regime and was falsified in the wrong one.** The invariant becomes regime-dependent and the earlier falsification becomes a prediction of the theory. |
| `Δ_1 > 0` and `Δ_4 > 0` | mixtures just help; nothing to do with cardinality, and the mechanism story is wrong even though the method works. |
| `Δ_1 ≈ 0` | **I4 stays falsified.** Report as a second, better-powered falsification on a multi-target task. No change to the paper beyond sharpening the statement. |
| `Δ_1 < 0` | I4 is falsified in the regime most favourable to it, which is the strongest negative result available here. |

## Pre-committed kill criteria

- If `s_mmr` (diversity reranking) matches the mixture gain, the mechanism is
  **diversity, not association**, and X31/X32 must be reframed regardless of `Δ_1`.
- If `s_rocchio` tuned at its best matches the mixture at its best, the contribution is
  **mode preservation, not expansion** — a much narrower claim.
- If the effect does not survive the encoder swap (X80, primary comparator declared in
  advance as `bge-base-en-v1.5`), it is reported as encoder-specific.
- X38 (shuffled gold) must come out at chance. If anything beats chance there, the
  metric is broken and every number in family A is void.

## Deviations

Any departure from the above gets appended here with its date and reason, and the
original text stays. This section is empty as of writing.
