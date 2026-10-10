# PREREG X113 — Is the image win pattern completion, or class smoothing?

**Written before the driver exists.** Git SHA at writing: see `git log -1` (eb3d8ea).
Committed ahead of any X113 run, for the same reason X110 was: this experiment can
withdraw the paper's only surviving positive regime, so the endpoint must be fixed first.

## 1. The discovery that motivates this

The paper's flagship positive result — "+0.014 pooled Recall@1 on image pattern
completion, rising to +0.021 at 70% occlusion" — is produced by
`scripts/confirm_holdout.py`. Its hit criterion is line 120:

```python
out.append(float(id_to_class[retr.retrieve(q, 1).ids[0]] == test_cls[j]))
```

Two facts follow, and neither is stated in the paper:

1. **The metric is same-CLASS, not same-image.** A retrieval counts as correct when the
   returned memory shares the query's Imagenette class. There are 10 classes, so the
   task is 10-way classification performed through a retrieval interface, and chance is
   0.100 — which is exactly the number the paper already reports as the collapse floor.
2. **The cue's source image is not in the bank.** The bank is 400 images from the
   `train` split (`N_PER_CLASS = 40`); the cues are occluded images from the `val`
   split. So the cue is a degraded instance of an image that was never stored.

Fact 2 means the flagship experiment **does not satisfy the premise of Claim 1**. The
taxonomy says descent helps "only if the cue is a degraded instance of a stored
pattern". Here the cue is a degraded instance of an *unstored* pattern that happens to
share a class label with 40 stored ones.

Fact 1 supplies an alternative mechanism with no associative content. At the selected
β≈40 the first descent step returns a softmax-weighted blend of many patterns. On a
class metric, blending with same-class neighbours is what a k-nearest-neighbour class
vote does, and a k-NN vote is a standard classifier, not a memory architecture. Cosine
top-1 is a 1-NN classifier and gets no ensemble at all, so the published comparison is
between a 1-NN classifier and a soft-k-NN classifier — unmatched in a way that favours
descent for reasons unrelated to energy.

## 2. Endpoints, fixed now

### X113a — the matched classifier baseline (primary)

Same cache, same bank, same cues, same tune/test image split, same metric as
`confirm_holdout.py`, adding one arm: a **soft k-NN class vote** that scores each class
by `sum over top-k cosine neighbours of that class of softmax(tau * sim)` and predicts
the argmax. `k` and `tau` are selected on the TUNE half only, exactly as β is.

- **Primary contrast:** descent(β\*) − kNN-vote(k\*, tau\*) on the TEST half, paired
  bootstrap over query instances.
- **Prediction if class smoothing explains the win:** the contrast is ≤ 0.
- **Prediction if descent has a memory-specific advantage:** the contrast is > 0 with a
  CI excluding 0.
- **Pre-committed reading:** if the CI contains 0 or lies below it, the +0.014 is not
  evidence for an associative mechanism, because a classifier with no energy, no
  iteration and no attractor matches or beats it on the paper's own metric.

### X113b — actual pattern completion (primary)

The experiment the paper's prose describes. Occlude the **bank** images themselves, so
the cue is a degraded instance of a pattern that *is* stored, and score **same-IMAGE**
recall@1 (identity of the source), not class.

- **Primary contrast:** descent(β\*) − cosine on same-image recall@1, TEST half,
  paired bootstrap.
- **Prediction from Claim 1:** > 0, and increasing with occlusion severity.
- Report same-class accuracy on the identical cues alongside it, so the two metrics are
  visible on one data set.

### X113c — mechanism (secondary, descriptive)

At the selected β: effective number of patterns in the blend, and the fraction of that
blend's mass sitting on the query's own class. Descriptive only; no endpoint.

## 3. Controls and kill criteria

- **Identity control.** At 0% occlusion, cosine same-image recall@1 on bank cues must
  be 1.0000. Anything else means the cue/bank wiring is broken and X113b is void.
- **Chance control.** Same-class accuracy must sit near 0.100 when the ranking is
  permuted.
- **Selection hygiene.** β, k and tau are chosen on the tune half and used once on the
  test half. The grid for k is pre-declared: {1, 3, 5, 10, 20, 40}; for tau:
  {4, 8, 16, 32, 64}; β reuses the published grid [24..80].
  k=1 with any tau must reproduce cosine exactly — an internal null check.
- **No metric substitution after the fact.** Same-class for X113a (to match the
  published claim), same-image for X113b (to test the premise). Neither is swapped for
  the other if the sign disappoints.

## 4. Outcome table, written in advance

| X113a (descent − kNN) | X113b (descent − cosine, same-image) | Reading |
|---|---|---|
| > 0 | > 0 | The win is real and is pattern completion. Paper stands, metric must still be relabelled. |
| ≤ 0 | > 0 | The published number was an unmatched-classifier artifact, but descent does complete patterns. Replace the flagship experiment with X113b. |
| > 0 | ≤ 0 | Descent beats a classifier at classification but cannot complete a stored pattern. The regime label "completion" is wrong. |
| ≤ 0 | ≤ 0 | The one surviving positive regime does not survive. Claim 1 keeps no confirming instance, and the paper's thesis becomes uniformly negative. |

The fourth row is a real possibility and is the reason this is pre-registered. If it
occurs, the abstract's "+0.014" sentence and contribution C3 are withdrawn, not
softened.
