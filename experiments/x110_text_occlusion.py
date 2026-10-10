"""X110 -- text occlusion. The experiment the paper names as its own likeliest falsifier.

PRE-REGISTERED in experiment-backlog/PREREG_X110.md before this driver existed. Read that
first: the predictions, the outcome table and the kill criteria are committed there, so
this file computes and reports, it does not decide what counts.

THE SETUP, in one line: hold the LoCoMo corpus and the encoder fixed, and swap the queries
from questions (where descent loses, monotonically) to MASKED TURNS (where Claim 1 says it
must win, monotonically). Only the query type changes.

Mask shape is a contiguous span because `scripts/test_imagenet_hopfield.py:occlude` blacks
out a square PATCH, not scattered pixels; random-word deletion is the secondary variant.
beta is chosen on a disjoint tune half of the conversations and then used once.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from experiments.locomo_qa import RESULTS, banks                        # noqa: E402
from experiments.x6x_energy import descend                              # noqa: E402
from energymem.encoders import SentenceEncoder                          # noqa: E402

OUT = RESULTS / "x110_text_occlusion.json"
LEVELS = (0.0, 0.3, 0.5, 0.7, 0.9)       # 0.0 is the identity control; 0.9 exploratory
PRIMARY_LEVELS = (0.3, 0.5, 0.7)         # the image study's levels
BETAS = (8.0, 16.0, 32.0, 48.0, 64.0, 128.0, 256.0)
T_DESCENT = 8                            # matches EnergyMemory(n_steps=8) in the image study
MIN_WORDS = 8
SPLIT_SEED = 12345
MASK_SEED = 909


def hr(t: str) -> None:
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


def mask_span(words: list[str], frac: float, rng: np.random.Generator) -> str:
    """Remove a contiguous span covering `frac` of the words -- the span analogue of a patch."""
    n = len(words)
    k = int(round(frac * n))
    if k <= 0:
        return " ".join(words)
    if k >= n:
        k = n - 1
    start = int(rng.integers(0, n - k + 1))
    return " ".join(words[:start] + words[start + k:])


def mask_random(words: list[str], frac: float, rng: np.random.Generator) -> str:
    """Delete a random `frac` of word positions, keeping order -- the secondary variant."""
    n = len(words)
    k = int(round(frac * n))
    if k <= 0:
        return " ".join(words)
    if k >= n:
        k = n - 1
    drop = set(rng.choice(n, size=k, replace=False).tolist())
    return " ".join(w for i, w in enumerate(words) if i not in drop)


def paired_bootstrap(a, b, clusters, n=20_000, seed=0):
    """(mean delta, lo, hi) resampling the CONVERSATION, as the image study resamples images."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    c = np.asarray(clusters)
    uniq = np.unique(c)
    idx = {u: np.flatnonzero(c == u) for u in uniq}
    rng = np.random.default_rng(seed)
    means = np.empty(n)
    for i in range(n):
        take = np.concatenate([idx[u] for u in rng.choice(uniq, len(uniq), replace=True)])
        means[i] = d[take].mean()
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def build_probes(bks, enc, mask_fn, levels=LEVELS):
    """{level: (Qmask, bank_index, target_row)} -- one probe per eligible turn per level.

    The query is a masked copy of a stored turn; the gold is that turn. Both methods see
    the identical bank with nothing excluded (X97: the exclusion must be symmetric, and
    for a completion cue the target MUST stay in the bank -- it is what we are recovering).
    """
    rng = np.random.default_rng(MASK_SEED)
    eligible = []
    for bi, b in enumerate(bks):
        for r, t in enumerate(b.texts):
            body = t.split(": ", 1)[1] if ": " in t else t
            if len(body.split()) >= MIN_WORDS:
                eligible.append((bi, r, t.split(": ", 1)[0] if ": " in t else "?",
                                 body.split()))
    print(f"  {len(eligible)} eligible turns (>= {MIN_WORDS} words) "
          f"of {sum(b.n for b in bks)}")
    probes = {}
    for lvl in levels:
        texts, bidx, rows = [], [], []
        for bi, r, spk, words in eligible:
            texts.append(f"{spk}: {mask_fn(words, lvl, rng)}")
            bidx.append(bi)
            rows.append(r)
        probes[lvl] = (enc.encode(texts), np.asarray(bidx), np.asarray(rows))
        print(f"  encoded mask level {lvl:.1f}")
    return probes


def score(bks, probes, lvl, beta, which):
    """recall@1 and recall@5 per probe. which='cosine' or 'descent'."""
    Q, bidx, rows = probes[lvl]
    r1, r5 = [], []
    for bi in np.unique(bidx):
        b = bks[bi]
        sel = np.flatnonzero(bidx == bi)
        for j in sel:
            q = Q[j]
            if which == "cosine":
                s = b.Z @ q
            else:
                s = b.Z @ descend(b.Z, q, beta, T_DESCENT)
            order = np.argsort(-s)
            tgt = rows[j]
            r1.append(float(order[0] == tgt))
            r5.append(float(tgt in order[:5]))
    return np.asarray(r1), np.asarray(r5)


def clusters_of(probes, lvl, bks):
    _, bidx, _ = probes[lvl]
    return np.asarray([bks[i].sample_id for i in bidx])


def run(bks, enc, mask_fn, tag, res):
    hr(f"X110 [{tag}] -- masked turns as cues, LoCoMo corpus and encoder held fixed")
    probes = build_probes(bks, enc, mask_fn)
    rng = np.random.default_rng(SPLIT_SEED)
    perm = rng.permutation(len(bks))
    tune, test = sorted(perm[: len(bks) // 2].tolist()), sorted(perm[len(bks) // 2:].tolist())
    print(f"\n  tune conversations {[bks[i].sample_id for i in tune]}")
    print(f"  held-out           {[bks[i].sample_id for i in test]}")

    # ---- the identity control, before anything else
    c1, _ = score(bks, probes, 0.0, None, "cosine")
    print(f"\n  IDENTITY CONTROL  mask=0.0 cosine recall@1 = {c1.mean():.4f} "
          f"(must be 1.0000)")
    res[tag] = {"identity_control_cosine_r1": float(c1.mean())}
    if c1.mean() < 0.999:
        print("  *** BROKEN: the cue IS the stored turn and cosine cannot find it. ***")

    # ---- cosine at every level, and the ceiling check
    cos = {}
    print(f"\n  {'level':>7}{'cosine R@1':>13}{'cosine R@5':>13}")
    for lvl in LEVELS:
        a, b5 = score(bks, probes, lvl, None, "cosine")
        cos[lvl] = (a, b5)
        print(f"  {lvl:>7.1f}{a.mean():>13.4f}{b5.mean():>13.4f}")
    ceiling = cos[0.7][0].mean() > 0.95
    print(f"\n  KILL CRITERION -- cosine R@1 at 70% masking > 0.95 (no headroom): {ceiling}")
    res[tag]["cosine"] = {str(k): dict(r1=float(v[0].mean()), r5=float(v[1].mean()))
                          for k, v in cos.items()}
    res[tag]["ceiling_triggered"] = bool(ceiling)

    # ---- STAGE 1: select beta on the TUNE half only, pooled over the primary levels
    print(f"\n  STAGE 1 -- select beta on the TUNE half only (T={T_DESCENT})")
    tune_mask = {lvl: np.isin(probes[lvl][1], tune) for lvl in LEVELS}
    best_b, best_d, tune_rows = None, -np.inf, {}
    for beta in BETAS:
        ds = []
        for lvl in PRIMARY_LEVELS:
            d1, _ = score(bks, probes, lvl, beta, "descent")
            m = tune_mask[lvl]
            ds.append(d1[m].mean() - cos[lvl][0][m].mean())
        d = float(np.mean(ds))
        tune_rows[str(beta)] = dict(delta_pooled=d,
                                    per_level=[float(x) for x in ds])
        print(f"    beta={beta:>6g}  pooled delta on tune = {d:+.4f}   "
              + "  ".join(f"{l:.1f}:{x:+.4f}" for l, x in zip(PRIMARY_LEVELS, ds)))
        if d > best_d:
            best_b, best_d = beta, d
    print(f"\n  selected beta = {best_b:g} (tune pooled delta {best_d:+.4f}) -- used once")
    res[tag]["beta_selection"] = dict(betas=tune_rows, selected=best_b,
                                      tune_delta=best_d,
                                      tune=[bks[i].sample_id for i in tune],
                                      test=[bks[i].sample_id for i in test])

    # ---- STAGE 2: one pre-committed comparison on the held-out half
    print(f"\n  STAGE 2 -- ONE pre-committed comparison on the HELD-OUT half, "
          f"beta={best_b:g} vs cosine")
    print(f"  {'level':>7}{'cos R@1':>10}{'desc R@1':>10}{'delta R@1':>24}"
          f"{'cos R@5':>10}{'desc R@5':>10}{'delta R@5':>24}")
    rows = {}
    for lvl in LEVELS:
        d1, d5 = score(bks, probes, lvl, best_b, "descent")
        m = np.isin(probes[lvl][1], test)
        cl = clusters_of(probes, lvl, bks)[m]
        a1, a5 = cos[lvl][0][m], cos[lvl][1][m]
        g1, g5 = d1[m], d5[m]
        e1 = paired_bootstrap(g1, a1, cl)
        e5 = paired_bootstrap(g5, a5, cl)
        rows[str(lvl)] = dict(n=int(m.sum()), cos_r1=float(a1.mean()),
                              desc_r1=float(g1.mean()), d_r1=e1[0], d_r1_lo=e1[1],
                              d_r1_hi=e1[2], cos_r5=float(a5.mean()),
                              desc_r5=float(g5.mean()), d_r5=e5[0], d_r5_lo=e5[1],
                              d_r5_hi=e5[2],
                              exploratory=bool(lvl not in PRIMARY_LEVELS))
        print(f"  {lvl:>7.1f}{a1.mean():>10.4f}{g1.mean():>10.4f}"
              f"{e1[0]:>+12.4f} [{e1[1]:>+.4f},{e1[2]:>+.4f}]"
              f"{a5.mean():>10.4f}{g5.mean():>10.4f}"
              f"{e5[0]:>+12.4f} [{e5[1]:>+.4f},{e5[2]:>+.4f}]"
              + ("   (exploratory)" if lvl not in PRIMARY_LEVELS else ""))
    res[tag]["heldout"] = rows

    # ---- the pre-registered readings
    d = {l: rows[str(l)]["d_r1"] for l in PRIMARY_LEVELS}
    lo = {l: rows[str(l)]["d_r1_lo"] for l in PRIMARY_LEVELS}
    p1 = lo[0.7] > 0
    p2 = d[0.3] <= d[0.5] <= d[0.7]
    p3 = lo[0.3] <= 0 <= rows["0.3"]["d_r1_hi"]
    print(f"\n  PRE-REGISTERED PREDICTIONS")
    print(f"    1. delta(0.7) > 0 (primary endpoint, interval excludes 0): {p1}"
          f"   [{d[0.7]:+.4f}, CI {lo[0.7]:+.4f}..{rows['0.7']['d_r1_hi']:+.4f}]")
    print(f"    2. monotone delta(0.3) <= delta(0.5) <= delta(0.7):        {p2}"
          f"   [{d[0.3]:+.4f} -> {d[0.5]:+.4f} -> {d[0.7]:+.4f}]")
    print(f"    3. delta(0.3) ~ 0 (interval covers 0):                     {p3}")
    if p1 and p2:
        verdict = "CLAIM 1 SURVIVES its own hardest test"
    elif p1:
        verdict = "PARTIAL: sign predicted, dose-response not"
    elif lo[0.7] <= 0 <= rows["0.7"]["d_r1_hi"]:
        verdict = "CLAIM 1 NOT SUPPORTED on text -- narrow it to the modality measured"
    else:
        verdict = "CLAIM 1 FALSIFIED -- descent loses where the premise is satisfied"
    print(f"\n  VERDICT ({tag}): {verdict}")
    res[tag]["predictions"] = dict(p1_primary=bool(p1), p2_monotone=bool(p2),
                                   p3_null_at_30=bool(p3), verdict=verdict)
    # shortlist trade, stated with the win rather than after it
    sl = {l: rows[str(l)]["d_r5"] for l in PRIMARY_LEVELS}
    print(f"  SHORTLIST TRADE   delta R@5: " +
          "  ".join(f"{l:.1f}:{sl[l]:+.4f}" for l in PRIMARY_LEVELS))
    res[tag]["shortlist_trade_r5"] = {str(k): v for k, v in sl.items()}
    return res


if __name__ == "__main__":
    t0 = time.time()
    enc = SentenceEncoder()
    bks = banks(encoder=enc)
    print(f"{len(bks)} conversations, {sum(b.n for b in bks)} turns, "
          f"encoder all-mpnet-base-v2 d={bks[0].Z.shape[1]}")
    res: dict = {}
    run(bks, enc, mask_span, "span (PRIMARY)", res)
    run(bks, enc, mask_random, "random-word (secondary)", res)
    res["_meta"] = dict(seconds=round(time.time() - t0, 1), levels=list(LEVELS),
                        primary_levels=list(PRIMARY_LEVELS), betas=list(BETAS),
                        T=T_DESCENT, min_words=MIN_WORDS,
                        prereg="experiment-backlog/PREREG_X110.md")
    OUT.write_text(json.dumps(res, indent=2, default=str))
    print(f"\nwrote {OUT}  ({res['_meta']['seconds']}s)")
