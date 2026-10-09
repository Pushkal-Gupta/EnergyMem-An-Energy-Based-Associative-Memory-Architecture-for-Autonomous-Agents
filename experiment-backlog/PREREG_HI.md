# Pre-registration — families H and I (X100–X105, X110–X112)

Written **2026-10-09, before any number on any of these corpora existed**. Some files were
still downloading while this was written, and none had been opened, parsed or encoded.
PREREG_X32.md was "written before, committed with", and said that anyone repeating
the test on a new corpus should commit first. **This file is committed by itself, before
any driver for these families runs.** Check `git log -- experiment-backlog/PREREG_HI.md`
against the commits that add the result JSONs.

## What families A–G leave standing

Families A–G are closed (README 1b, 1c). On LoCoMo QA, mode expansion **is** cosine at
its fold-selected configuration (X31: K0=0 in 10/10 folds, every category), the X32
interaction had no effect to decompose, and RRF (+0.124 `gold@10`) is the only method
that buys accuracy, mostly on single-hop. The one open hypothesis is X32's: LoCoMo's
gold sets are not orthogonal enough (ε=0.52) for a mixture to help, so the effect could
exist where evidence is genuinely near-orthogonal. Family H tests that on public corpora
built for multi-hop; X111 tests it on a task where the number of targets is set by
construction.

**Prior.** Given A–G, the most likely outcome everywhere is the same null again. The
predictions below are the hypothesis under test, not what we expect to find.

## Shared protocol (all of H, and X111/X112)

- **Methods**, unchanged from `experiments/locomo_qa.py`: `cosine`, `bm25`, `rrf`,
  `mmr`, `rocchio`, `mixture`, `mixture_avg`. BM25 and RRF are skipped where the query is
  not text (X111, X112 image→text). The grids are the X31 grids line for line: K0 ∈
  {0,1,2,3,5} (K0=0 always reachable, K0 < cutoff 10), β ∈ {4,16,64}, with_query ∈
  {T,F}; rocchio γ ∈ {0,.25,.5,1}; rrf k ∈ {10,60,120}; mmr λ ∈ {.3,.5,.7,.9}, pool ∈
  {20,50}.
- **Encoder:** `all-mpnet-base-v2` for every text corpus. X111 uses the project's
  ResNet-18 image features; X112 uses `clip-ViT-B-32` for both modalities (any method
  comparison inside one experiment shares one encoder).
- **Selection:** grouped leave-one-group-out, with hyperparameters chosen on the training
  groups only, and the per-fold picks reported. **10 groups** per corpus unless the corpus
  has natural clusters, as stated per corpus below.
- **Primary metric `gold@10`**, with `full@10`, `mrr` and `ndcg@10` secondary. BEIR also
  reports `ndcg@10`, since that is its own convention.
- **Tests:** paired, signs flipped within cluster. Exact enumeration up to 22 clusters,
  otherwise 20,000 Monte Carlo sign patterns. Two-level cluster bootstrap for intervals.
  With 10 clusters, p is floored at 2/1024 and is reported as a floor.

## Corpora (H)

| ID | corpus | bank | gold | strata |
|---|---|---|---|---|
| X100 | LongMemEval-S (cleaned), 500 q | the question's own haystack turns | turns flagged `has_answer` | question type; multi-session vs single-session-* is the m axis |
| X101 | MSC test | pooled turns | **next turn** (the X26 probe, since MSC has no QA evidence) | hop 1, hop 2 |
| X102 | HotpotQA distractor dev, 2WikiMultihopQA dev | paragraphs pooled over 100-question groups | supporting-fact paragraphs | HotpotQA: bridge / comparison. 2Wiki: type, so m=2 vs m=4 |
| X103 | **QASPER dev** (substitute) | the paper's paragraphs | annotator-0 evidence paragraphs | 1 vs >1 evidence paragraph |
| X104 | BEIR SciFact, NFCorpus (test) | whole corpus | qrels > 0 | — |
| X105 | transfer | — | — | LoCoMo-selected config applied unchanged |

**Deviation declared in advance:** X103 names QuALITY / NarrativeQA, but neither ships
evidence annotations. Gold-evidence retrieval on them is not runnable as written.
QASPER is the nearest long-document corpus that has evidence paragraphs, and it
replaces them. The substitution is made here, before any data has been seen.

X101 is the **X26 one-hop probe** on MSC (cue turn → next turn, world model fitted on
train sessions, `experiments/locomo_assoc.py` machinery reused). It is the external
replication of the project's one positive claim, not a QA task.

## Predictions

**P-H1 (X102, the primary endpoint of this file).** On HotpotQA and on 2Wiki,
`Δ = gold@10(mixture) − gold@10(mixture_avg)` > 0, at p < 0.05 on each corpus.
**P-H2 (X102 interaction).** On 2Wiki, Δ on m=4 (bridge_comparison) exceeds Δ on m=2,
testing `Δ_{m=4} − Δ_{m=2}` > 0 with a cluster bootstrap.
**P-H3.** In at least one multi-hop corpus (X100 multi-session, X102), `mixture`
**selects a non-null configuration** in a majority of folds. If it backs off to K0=0
everywhere, P-H1 and P-H2 are moot, just as X32 was.
**P-H4 (X101).** On MSC, the one-hop world model beats cosine at R@1 (the X26 sign),
with a CI excluding 0.
**P-H5 (X105).** LoCoMo-selected configurations transfer, meaning the transferred
config's `gold@10` is within 0.01 of the in-corpus fold-selected one. This is a
prediction about the selection procedure, not the method.

## Family I

**X110 — text occlusion.** This mirrors `scripts/confirm_holdout.py` with text
substituted for images. Corpus: Yahoo Answers Topics (10 classes, mirroring
Imagenette's 10). Bank: 40 texts per class. Test: a disjoint held-out set. The query is
the held-out text with a **contiguous span of tokens masked**, covering 30%, 50% or 70%
of tokens (the analogue of the square patch), with the masked tokens deleted. The
`[MASK]` arm is a secondary. Metric: top-1 class match. Energy: `EnergyMemory(n_steps=8,
β)` with β selected on a disjoint tune half from the same grid as the image script; one
comparison is then made on the test half.

- **Prediction (the image result generalizes):** energy − cosine > 0 at 50% and 70%,
  with a CI excluding 0.
- **Control added to both modalities:** a k-NN class vote (k selected on the tune half).
  If k-NN voting matches or beats energy descent, the image effect is **class smoothing,
  not memory retrieval**. That is the backlog's "may not be about memory at all", made
  testable. The image arm is re-run here with this control and reported next to the
  existing `image_holdout.json`, which stays untouched.

**X111 — image multi-target (the cat-1 analogue, with m set by construction).** The bank
holds Imagenette train images. Each query is a 2×2 mosaic built from *m* ∈ {1,2,3,4}
distinct bank images, each from a different class. For m < 4, the free quadrants repeat
the query's own images, cycling, so every query is a full mosaic. The gold set is the *m*
source images. This is instance retrieval of stored memories. The groups are 10 disjoint
image partitions.
- **Prediction (the X32 interaction, where m is controlled rather than observed):**
  Δ(mixture − mixture_avg) increases with m: Δ_{m=4} − Δ_{m=1} > 0, and Δ_{m=1} ≈ 0.

**X112 — cross-modal (Flickr8k test, CLIP ViT-B/32).** Text→image has one gold image
per caption (m=1). Image→text has five gold captions per image (m=5). Groups are 10
image partitions.
- **Prediction:** Δ(mixture − mixture_avg) is larger on image→text (m=5) than on
  text→image (m=1).

## What each outcome means

- P-H1 met and P-H3 met → **I4 is regime-dependent.** The LoCoMo null was the ε=0.52
  regime, and the falsification in the single-target regime becomes a prediction of
  the theory.
- P-H3 not met (back-off to null) → **the X31 null generalizes to purpose-built
  multi-hop corpora.** I4 stays falsified, and nowhere is left to look on text.
- X111 met but X102 not → the mechanism exists when evidence is orthogonal by
  construction, and natural multi-hop text is not orthogonal enough for it.
- X110 energy > cosine but k-NN ≥ energy → the image positive is class smoothing. The
  paper's C3 framing must change from "memory" to "nonparametric denoising".

## Deviations

Append here with date and reason; the original text stays.
