"""Family E -- robustness. Only the arms that survive triage against families A-D.

WHAT THIS FAMILY CAN AND CANNOT ASK NOW. Most of family E was written as "does the gain
survive X" (X84, X85, X93 explicitly say "interesting if the gain ..."). Families A-D
found no gain: at the fold-selected configuration every expansion method IS cosine, in
10/10 folds, in every category. A robustness check on an effect of exactly zero is
vacuous -- it can only confirm that zero is still zero -- so those arms are marked dead
in the backlog rather than run.

What survives is the set of questions that do not presuppose a gain:

X80  encoder swap     -- is the NULL encoder-specific? bge-base-en-v1.5 was pre-declared
                         the primary comparator in PREREG_X32 before any of this ran.
X81  dimension        -- does the null hold at lower d, and does X51's separability
                         result depend on d vs n (its stated reason for existing)?
X82  normalization    -- whitening and all-but-the-top erase the anisotropy the
                         mechanism is supposed to exploit. A real falsification attempt,
                         and it needs no gain to be meaningful.
X83  chunking         -- the bank text itself. Changes cosine and RRF too, so it is not
                         gain-conditional.
X86  seed sensitivity -- the retrieval path is deterministic; the seeds that exist are
                         in the resampling. Checked against X99's exact p-value.
X87  fold granularity -- do the conclusions move with the number of folds?
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from experiments.locomo_qa import (                              # noqa: E402
    DATA, RESULTS, Bank, banks, evaluate, folds, fold_select, grid_points,
    s_cosine, s_mixture, s_mixture_avg, s_rocchio, s_rrf, sweep,
)
from energymem.base import l2_normalize                          # noqa: E402
from energymem.encoders import SentenceEncoder                   # noqa: E402
from energymem.eval.setmetrics import (                          # noqa: E402
    blocked_permutation, exact_blocked_permutation, hierarchical_bootstrap, paired_report,
)

OUT = RESULTS / "x8x_robustness.json"
R: dict = {}

# The expansion grid from X31 -- K0 strictly below the k=10 cutoff, and K0=0 reachable,
# which is what makes "backs off to the null" a statement the grid can express.
GRID = {"K0": [0, 1, 2, 3, 5], "beta": [4.0, 16.0, 64.0], "with_query": [True, False]}
K_CUTOFF = 10
assert max(GRID["K0"]) < K_CUTOFF and 0 in GRID["K0"]


def hr(t: str) -> None:
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


def _sel(bks, tab, mask_fn, score_key="gold@10"):
    """Leave-one-conversation-out selection restricted to the questions mask_fn keeps.

    The sweep table is computed ONCE per set of vectors and then filtered, so the
    all-categories panel and the per-category panels never re-run the grid. Selection
    still sees only the nine training conversations.
    """
    vals, clus, picks = [], [], []
    for train, test in folds(bks):
        best, best_s = None, -np.inf
        for key, per_bank in tab.items():
            parts = [per_bank[j][score_key][mask_fn(per_bank[j]["cat"])] for j in train]
            s = float(np.mean(np.concatenate(parts)))
            if s > best_s:
                best, best_s = key, s
        r = tab[best][test]
        m = mask_fn(r["cat"])
        vals.append(r[score_key][m])
        clus.append(np.full(int(m.sum()), bks[test].sample_id))
        picks.append(best)
    return np.concatenate(vals), np.concatenate(clus), picks


def panels(bks: list[Bank], tag: str, cat_sets=((None, "all"), ({1}, "cat 1 multi-hop"),
                                                ({4}, "cat 4 single-hop"))) -> dict:
    """The Phase 7 headline, recomputed under whatever vectors `bks` carries.

    Both RRF and the mixture are fold-selected over their own grids, so the numbers here
    are comparable to X29/X31 rather than to a hand-picked k_rrf.
    """
    mix_tab = sweep(bks, s_mixture, GRID)
    rrf_tab = sweep(bks, s_rrf, {"k_rrf": [10.0, 60.0, 120.0]})
    out = {}
    for cats, name in cat_sets:
        mf = (lambda c: np.ones(len(c), bool)) if cats is None else \
             (lambda c, cs=cats: np.isin(c, list(cs)))
        cos = np.concatenate([evaluate(b, s_cosine)["gold@10"][mf(np.asarray(b.cats))]
                              for b in bks])
        clus = np.concatenate([np.full(int(mf(np.asarray(b.cats)).sum()), b.sample_id)
                               for b in bks])
        mv, mc, mpicks = _sel(bks, mix_tab, mf)
        rv, rc, rpicks = _sel(bks, rrf_tab, mf)
        nulls = sum(1 for p in mpicks if "K0=0" in p)
        rr = paired_report(rv, cos, clus, n_boot=4000, n_perm=20000)
        rm = paired_report(mv, cos, mc, n_boot=4000, n_perm=20000)
        out[name] = dict(n=int(cos.size), cosine=float(cos.mean()),
                         rrf=float(rv.mean()), mixture=float(mv.mean()),
                         rrf_minus_cosine=rr["delta"], rrf_lo=rr["lo"], rrf_hi=rr["hi"],
                         rrf_p=rr["p"], rrf_p_exact=exact_blocked_permutation(rv - cos, clus),
                         mix_minus_cosine=rm["delta"], mix_lo=rm["lo"], mix_hi=rm["hi"],
                         mix_p=rm["p"], mix_p_exact=exact_blocked_permutation(mv - cos, mc),
                         folds_selecting_null=nulls, mix_picks=mpicks, rrf_picks=rpicks)
        o = out[name]
        print(f"  {tag + ' / ' + name:38s} cos {o['cosine']:.4f}  "
              f"rrf {o['rrf']:.4f} ({o['rrf_minus_cosine']:+.4f}, p={o['rrf_p_exact']:.4f})  "
              f"mix {o['mixture']:.4f} ({o['mix_minus_cosine']:+.4f}, p={o['mix_p_exact']:.4f})"
              f"  null-folds {nulls}/10")
    return out


# ====================================================================== X80
# PREREG_X32 declared bge-base-en-v1.5 the primary comparator in advance. bge ships a
# documented query instruction; the pre-declaration did not mention it, so the literal
# pre-declared configuration (no prefix) is primary and the prefixed variant is reported
# beside it rather than instead of it.

BGE_PREFIX = "Represent this sentence for searching relevant passages: "


def reencode_queries(bks: list[Bank], enc: SentenceEncoder, prefix: str) -> list[Bank]:
    out = []
    for b in bks:
        nb = Bank(**{**b.__dict__, "_bm25": None})
        nb.Q = enc.encode([prefix + q for q in b.questions])
        out.append(nb)
    return out


def x80() -> None:
    hr("X80  encoder swap  (primary comparator pre-declared: bge-base-en-v1.5)")
    rows = {}
    models = [("mpnet-base-v2 (incumbent, d=768)", "sentence-transformers/all-mpnet-base-v2"),
              ("bge-base-en-v1.5 (PRIMARY, d=768)", "BAAI/bge-base-en-v1.5"),
              ("MiniLM-L6-v2 (secondary, d=384)", "sentence-transformers/all-MiniLM-L6-v2")]
    for label, name in models:
        enc = SentenceEncoder(name)
        bks = banks(encoder=enc)
        print(f"\n  -- {label}  d={bks[0].Z.shape[1]}")
        rows[label] = dict(d=int(bks[0].Z.shape[1]), panels=panels(bks, label))
        if "bge" in name:
            pb = reencode_queries(bks, enc, BGE_PREFIX)
            print("  -- bge with its documented query instruction (secondary)")
            rows[label]["with_query_instruction"] = panels(pb, "bge+instruction")
    nulls = {k: v["panels"]["all"]["folds_selecting_null"] for k, v in rows.items()}
    print(f"\n  folds selecting the null, per encoder: {nulls}")
    print("  VERDICT: the null is encoder-specific only if some encoder selects a")
    print("           non-null configuration AND beats cosine there.")
    R["X80"] = dict(rows=rows, null_folds=nulls,
                    primary="BAAI/bge-base-en-v1.5",
                    prereg="experiment-backlog/PREREG_X32.md declared this in advance")


# ====================================================================== X81

def separable(Z: np.ndarray, g, cap: int) -> bool:
    """Same LP as X51. `cap` must exceed Z.shape[1] or Cover's theorem answers for you."""
    g = list(g)
    d = Z.shape[1]
    assert cap > d, f"cap {cap} <= d {d}: the test would be vacuous"
    other = [j for j in range(len(Z)) if j not in set(g)]
    if len(other) > cap:
        s = Z[other] @ Z[g].mean(0)
        other = [other[j] for j in np.argsort(-s)[:cap]]
    A = [np.append(-Z[i], 1.0) for i in g] + [np.append(Z[j], -1.0) for j in other]
    bb = [-1.0] * len(g) + [0.0] * len(other)
    r = linprog(np.zeros(d + 1), A_ub=np.asarray(A), b_ub=np.asarray(bb),
                bounds=[(-50, 50)] * (d + 1), method="highs")
    return bool(r.status == 0)


def project(bks: list[Bank], d: int, seed: int = 0) -> list[Bank]:
    """PCA of the POOLED bank vectors to d dims, then renormalize. Label-free."""
    Zp = np.concatenate([b.Z for b in bks])
    mu = Zp.mean(0)
    _, _, Vt = np.linalg.svd(Zp - mu, full_matrices=False)
    P = Vt[:d].T
    out = []
    for b in bks:
        nb = Bank(**{**b.__dict__, "_bm25": None})
        nb.Z = l2_normalize((b.Z - mu) @ P, axis=1)
        nb.Q = l2_normalize((b.Q - mu) @ P, axis=1)
        out.append(nb)
    return out


def x81(bks: list[Bank]) -> None:
    hr("X81  dimension  (and the d-vs-n basis of X51's separability caveat)")
    rows = {}
    Zp_full = np.concatenate([b.Z for b in bks])
    off = np.cumsum([0] + [b.n for b in bks])
    for d in (64, 128, 256, 384, 768):
        pb = bks if d == 768 else project(bks, d)
        rows[str(d)] = dict(panels=panels(pb, f"d={d}"))
        # separability on the POOLED bank at this d, cap chosen to exceed d
        Zp = np.concatenate([b.Z for b in pb])
        cap = max(2500, 2 * d + 100)
        hits = tot = 0
        for bi, b in enumerate(pb):
            for i in range(b.m):
                if b.cats[i] != 1 or tot >= 15:
                    continue
                hits += separable(Zp, [x + off[bi] for x in b.gold[i]], cap=cap)
                tot += 1
        rows[str(d)]["separable_cat1"] = dict(frac=hits / max(tot, 1), n=tot, cap=cap,
                                             n_pooled=int(Zp.shape[0]))
        print(f"      pooled separability cat 1 at d={d}: {hits}/{tot} "
              f"= {hits / max(tot, 1):.0%}  (cap {cap} > d, n={Zp.shape[0]})")
    R["X81"] = rows


# ====================================================================== X82

def renorm_variant(bks: list[Bank], kind: str, k_abtt: int = 3) -> list[Bank]:
    """Label-free post-hoc transforms of the frozen vectors.

    'none' is NOT available here: SentenceEncoder sets normalize_embeddings=True, so
    un-normalized vectors would require re-encoding outside the frozen encoder path,
    which would change every number already recorded in the project. Stated, not faked.
    """
    Zp = np.concatenate([b.Z for b in bks])
    mu = Zp.mean(0)
    if kind == "l2":
        return bks
    if kind == "centered":
        f = lambda X: l2_normalize(X - mu, axis=1)                        # noqa: E731
    elif kind == "whitened":
        C = np.cov((Zp - mu).T) + 1e-6 * np.eye(Zp.shape[1])
        w, V = np.linalg.eigh(C)
        W = V @ np.diag(1.0 / np.sqrt(np.maximum(w, 1e-12))) @ V.T
        f = lambda X: l2_normalize((X - mu) @ W, axis=1)                  # noqa: E731
    elif kind == "abtt":
        _, _, Vt = np.linalg.svd(Zp - mu, full_matrices=False)
        U = Vt[:k_abtt]
        f = lambda X: l2_normalize((X - mu) - ((X - mu) @ U.T) @ U, axis=1)  # noqa: E731
    else:
        raise ValueError(kind)
    out = []
    for b in bks:
        nb = Bank(**{**b.__dict__, "_bm25": None})
        nb.Z, nb.Q = f(b.Z), f(b.Q)
        out.append(nb)
    return out


def x82(bks: list[Bank]) -> None:
    hr("X82  normalization  (whitening is the falsification attempt)")
    rows = {}
    for kind in ("l2", "centered", "whitened", "abtt"):
        pb = renorm_variant(bks, kind)
        rows[kind] = panels(pb, kind)
    print("\n  If whitening/abtt leaves cosine roughly where it was, the anisotropy was")
    print("  not what cosine was living on. If a NON-null config gets selected there,")
    print("  that is the one place the mechanism could still be hiding.")
    R["X82"] = dict(rows=rows, note="'none' unavailable: the frozen encoder normalizes")


# ====================================================================== X83

def rechunk(bks: list[Bank], kind: str, enc: SentenceEncoder) -> list[Bank]:
    out = []
    for b in bks:
        if kind == "text_only":
            texts = [t.split(": ", 1)[1] if ": " in t else t for t in b.texts]
        elif kind == "speaker_prefixed":
            texts = list(b.texts)
        else:
            raise ValueError(kind)
        nb = Bank(**{**b.__dict__, "_bm25": None})
        nb.texts = texts
        nb.Z = enc.encode(texts)
        out.append(nb)
    return out


def x83(bks: list[Bank], enc: SentenceEncoder) -> None:
    hr("X83  chunking  (turn-level only -- windows break the gold mapping)")
    print("  LoCoMo annotates evidence at TURN level. A sliding window or a session")
    print("  summary containing a gold turn is neither a hit nor a miss under that")
    print("  annotation, so those two arms are not runnable without re-annotating the")
    print("  benchmark. The two arms that preserve the mapping are run.\n")
    rows = {}
    for kind in ("speaker_prefixed", "text_only"):
        pb = rechunk(bks, kind, enc)
        rows[kind] = panels(pb, kind)
    R["X83"] = dict(rows=rows,
                    deferred=["sliding_window", "session_summary"],
                    deferred_reason="turn-level gold annotation does not transfer to "
                                    "merged units without re-annotation")


# ====================================================================== X86

def x86(bks: list[Bank]) -> None:
    hr("X86  seed sensitivity  (the retrieval path has no seed; the resampling does)")
    clus = np.concatenate([np.full(b.m, b.sample_id) for b in bks])
    cos = np.concatenate([evaluate(b, s_cosine)["gold@10"] for b in bks])
    rrf = np.concatenate([evaluate(b, s_rrf)["gold@10"] for b in bks])
    d = rrf - cos

    # determinism of the retrieval path itself
    rep = [float(np.concatenate([evaluate(b, s_mixture, K0=3)["gold@10"]
                                 for b in bks]).mean()) for _ in range(3)]
    print(f"  mixture re-run 3x: {rep}  identical={len(set(rep)) == 1}")

    ps = [blocked_permutation(d, clus, n=20_000, seed=s) for s in range(20)]
    cis = [hierarchical_bootstrap(d, clus, n=4000, seed=s) for s in range(20)]
    ex = exact_blocked_permutation(d, clus)
    print(f"  rrf-cosine p over 20 seeds: min {min(ps):.5f} max {max(ps):.5f}  "
          f"exact {ex:.5f}")
    print(f"  CI lower over 20 seeds: [{min(c[1] for c in cis):+.4f}, "
          f"{max(c[1] for c in cis):+.4f}]   upper: "
          f"[{min(c[2] for c in cis):+.4f}, {max(c[2] for c in cis):+.4f}]")
    R["X86"] = dict(retrieval_deterministic=bool(len(set(rep)) == 1), repeats=rep,
                    p_seeds=ps, p_exact=ex,
                    ci_lo=[c[1] for c in cis], ci_hi=[c[2] for c in cis],
                    note="SPLIT_SEED is fixed for the life of the project and was not "
                         "varied; only resampling seeds were")


# ====================================================================== X87

def grouped_folds(bks: list[Bank], n_folds: int):
    idx = np.arange(len(bks))
    parts = np.array_split(idx, n_folds)
    return [([j for j in idx if j not in set(p)], list(p)) for p in parts]


def select_over(bks: list[Bank], tab: dict, fold_list, score_key="gold@10"):
    vals, clus, picks = [], [], []
    for train, test in fold_list:
        best, best_s = None, -np.inf
        for k, per_bank in tab.items():
            s = float(np.mean(np.concatenate([per_bank[j][score_key] for j in train])))
            if s > best_s:
                best, best_s = k, s
        for t in ([test] if isinstance(test, (int, np.integer)) else test):
            r = tab[best][t]
            vals.append(r[score_key])
            clus.append(np.full(len(r[score_key]), bks[t].sample_id))
        picks.append(best)
    return np.concatenate(vals), np.concatenate(clus), picks


def x87(bks: list[Bank]) -> None:
    hr("X87  fold granularity  (the backlog's '4-fold (current)' was stale)")
    print("  Families A-D used leave-one-conversation-out, 10 folds. Leave-one-SESSION-")
    print("  out is not a folding of this task: a bank is a conversation and 95.4% of")
    print("  cat-1 gold sets span sessions, so removing a session removes bank items and")
    print("  changes the task rather than the split. Fold COUNT is swept instead.\n")
    tab = sweep(bks, s_mixture, GRID)
    cos = np.concatenate([evaluate(b, s_cosine)["gold@10"] for b in bks])
    rows = {}
    for n_folds in (2, 4, 5, 10):
        v, c, picks = select_over(bks, tab, grouped_folds(bks, n_folds))
        nulls = sum(1 for p in picks if "K0=0" in p)
        rr = paired_report(v, cos, c, n_boot=4000, n_perm=20000)
        rows[str(n_folds)] = dict(mixture=float(v.mean()), cosine=float(cos.mean()),
                                  delta=rr["delta"], lo=rr["lo"], hi=rr["hi"], p=rr["p"],
                                  null_folds=nulls, n_folds=len(picks), picks=picks)
        print(f"  {n_folds:>2}-fold: mixture {v.mean():.4f}  delta {rr['delta']:+.4f} "
              f"[{rr['lo']:+.4f},{rr['hi']:+.4f}] p={rr['p']:.4f}  "
              f"null-folds {nulls}/{len(picks)}")
    # no selection at all: the grid's own default
    nosel = np.concatenate([evaluate(b, s_mixture, K0=3, beta=16.0)["gold@10"]
                            for b in bks])
    clus = np.concatenate([np.full(b.m, b.sample_id) for b in bks])
    rr = paired_report(nosel, cos, clus, n_boot=4000, n_perm=20000)
    rows["none(K0=3,beta=16)"] = dict(mixture=float(nosel.mean()), delta=rr["delta"],
                                      lo=rr["lo"], hi=rr["hi"], p=rr["p"])
    print(f"  no selection (K0=3,beta=16): {nosel.mean():.4f}  delta {rr['delta']:+.4f} "
          f"[{rr['lo']:+.4f},{rr['hi']:+.4f}] p={rr['p']:.4f}")
    R["X87"] = dict(rows=rows,
                    loso_not_run="a bank is a conversation; gold spans sessions")


if __name__ == "__main__":
    t0 = time.time()
    enc = SentenceEncoder()
    bks = banks(encoder=enc)
    print(f"{len(bks)} conversations, {sum(b.n for b in bks)} turns, "
          f"{sum(b.m for b in bks)} questions, d={bks[0].Z.shape[1]}")
    x80()
    x81(bks)
    x82(bks)
    x83(bks, enc)
    x86(bks)
    x87(bks)
    R["_meta"] = dict(seconds=round(time.time() - t0, 1),
                      dead_arms={"X84": "gain-conditional: 'interesting if the gain is "
                                        "more robust than cosine's'. The gain is 0.",
                                 "X85": "gain-conditional: 'interesting if the gain "
                                        "grows'. Subsumed by X37 for the scope question.",
                                 "X93": "gain-conditional: 'does the gain survive "
                                        "HNSW/IVF'. Engineering, not a claim."})
    OUT.write_text(json.dumps(R, indent=2, default=str))
    print(f"\nwrote {OUT}  ({R['_meta']['seconds']}s)")
