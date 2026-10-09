"""Shared machinery for evidence retrieval on corpora other than LoCoMo (backlog family H).

NOT a driver -- it computes no result. It turns each corpus into the same ``Bank`` objects
``experiments/locomo_qa.py`` uses, so every method, grid, metric and null-setting there
runs unchanged on a new corpus. Nothing in locomo_qa is modified.

What differs from LoCoMo is only the clustering:

  bank     the candidate set a query ranks (one conversation, one haystack, one paper,
           a pool of 100 multi-hop questions' paragraphs, or a whole BEIR corpus)
  group    the unit of leave-one-group-out SELECTION -- hyperparameters are chosen on the
           other groups only (10 groups unless the corpus has fewer natural units)
  cluster  the unit of INFERENCE -- signs are flipped and the bootstrap resamples at this
           level. A bank of pooled questions is one cluster; independent questions (one
           haystack each, or BEIR queries over a shared corpus) are their own clusters.

Pre-registered in experiment-backlog/PREREG_HI.md, committed before any of this ran.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from experiments.locomo_qa import (KS, METHODS, RESULTS, Bank, grid_points,  # noqa: E402
                                   _key)
from energymem.encoders import SentenceEncoder                    # noqa: E402
from energymem.eval.setmetrics import (exact_blocked_permutation,  # noqa: E402
                                       full_set_recall_at_k, hierarchical_bootstrap,
                                       mean_gold_recall_at_k, ndcg_at_k, reciprocal_rank)

DATA = ROOT / "data"
SPLIT_SEED = 12345
N_GROUPS = 10
SHA = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                     capture_output=True, text=True).stdout.strip()

# The X31 grids line for line (K0 < cutoff 10, K0=0 reachable), plus X29's bm25/mmr.
GRIDS = {
    "cosine": {},
    "bm25": {},
    "rrf": {"k_rrf": [10.0, 60.0, 120.0]},
    "mmr": {"lam": [0.3, 0.5, 0.7, 0.9], "pool": [20, 50]},
    "rocchio": {"K0": [0, 1, 2, 3, 5], "gamma": [0.0, 0.25, 0.5, 1.0]},
    "mixture": {"K0": [0, 1, 2, 3, 5], "beta": [4.0, 16.0, 64.0], "with_query": [True, False]},
    "mixture_avg": {"K0": [0, 1, 2, 3, 5], "beta": [4.0, 16.0, 64.0],
                    "with_query": [True, False]},
}
TEXTLESS = ("bm25", "rrf")          # need a text query
PRIMARY = "gold@10"


def stamp(id_: str, **kw) -> dict:
    return dict(id=id_, git_sha=SHA, utc=datetime.now(timezone.utc).isoformat(),
                prereg="experiment-backlog/PREREG_HI.md", **kw)


# ------------------------------------------------------------------ encoding

def encode_unique(texts: list[str], enc: SentenceEncoder | None = None,
                  chunk: int = 2048) -> np.ndarray:
    """Encode with de-duplication. Unique texts are sorted so chunk cache keys are stable."""
    enc = enc or SentenceEncoder()
    uniq = sorted(set(texts))
    pos = {t: i for i, t in enumerate(uniq)}
    parts = []
    # ENERGYMEM_NOCACHE=1: same model, same call, but vectors stay in memory. Used when the
    # disk cannot hold the float64 cache (LongMemEval alone would be 1.1 GB).
    nocache = os.environ.get("ENERGYMEM_NOCACHE") == "1"
    for s in range(0, len(uniq), chunk):
        if nocache:
            parts.append(enc._load().encode(
                uniq[s:s + chunk], batch_size=64, show_progress_bar=False,
                convert_to_numpy=True, normalize_embeddings=True).astype(np.float64))
        else:
            parts.append(enc.encode(uniq[s:s + chunk]))
        print(f"    encoded {min(s + chunk, len(uniq))}/{len(uniq)}", flush=True)
    U = np.concatenate(parts) if parts else np.zeros((0, 768))
    return U[[pos[t] for t in texts]]


def make_banks(specs: list[dict], enc=None) -> list[Bank]:
    """specs: [{sample_id, texts, questions, gold, cats, answers}] -> encoded Banks."""
    all_t = [t for s in specs for t in s["texts"]] + [q for s in specs for q in s["questions"]]
    V = encode_unique(all_t, enc)
    nt = sum(len(s["texts"]) for s in specs)
    Zall, Qall = V[:nt], V[nt:]
    out, a, b = [], 0, 0
    for s in specs:
        n, m = len(s["texts"]), len(s["questions"])
        out.append(Bank(sample_id=s["sample_id"], ids=[f"{s['sample_id']}/{j}" for j in range(n)],
                        texts=s["texts"], Z=Zall[a:a + n], Q=Qall[b:b + m], gold=s["gold"],
                        cats=s["cats"], answers=s.get("answers", [""] * m),
                        questions=s["questions"], sessions=[""] * n, speakers=[""] * n))
        a, b = a + n, b + m
    return out


# ------------------------------------------------------------------ loaders

def _pq(path):
    import pyarrow.parquet as pq
    return pq.read_table(path).to_pylist()


def _multihop(rows, n_q, per_bank, catmap, seed=SPLIT_SEED, offset=0):
    """HotpotQA / 2Wiki: pool the paragraphs of `per_bank` questions into one bank.

    Each question's own 8+ distractor paragraphs stay in the bank, so this is the
    standard distractor setting with ~100x more distractors. Paragraphs are keyed by
    (title, text): a title shared between two questions' contexts with identical text is
    one bank item, and gold is resolved against the question's OWN context only.
    """
    rng = np.random.default_rng(seed)
    rows = [rows[i] for i in rng.permutation(len(rows))[offset:offset + n_q]]
    specs = []
    for bi in range(0, len(rows), per_bank):
        key2i, texts, qs, gold, cats, ans = {}, [], [], [], [], []
        for r in rows[bi:bi + per_bank]:
            local = {}
            for t, sents in zip(r["context"]["title"], r["context"]["sentences"]):
                body = f"{t}: {''.join(sents).strip()}"
                k = (t, body)
                if k not in key2i:
                    key2i[k] = len(texts)
                    texts.append(body)
                local[t] = key2i[k]
            g = sorted({local[t] for t in r["supporting_facts"]["title"] if t in local})
            if not g:
                continue
            qs.append(r["question"]); gold.append(g); ans.append(r["answer"])
            cats.append(catmap[r["type"]])
        specs.append(dict(sample_id=f"b{bi // per_bank:02d}", texts=texts, questions=qs,
                          gold=gold, cats=cats, answers=ans))
    return specs


HOTPOT_CATS = {1: "bridge", 2: "comparison"}
WIKI_CATS = {1: "compositional", 2: "inference", 3: "comparison", 4: "bridge_comparison"}


def load_hotpot(n_q=2000, per_bank=100, offset=0):
    rows = _pq(DATA / "hotpotqa/validation.parquet")
    return _multihop(rows, n_q, per_bank, {"bridge": 1, "comparison": 2},
                     offset=offset), HOTPOT_CATS


def load_2wiki(n_q=2000, per_bank=100, offset=0):
    rows = _pq(DATA / "2wiki/validation.parquet")
    inv = {v: k for k, v in WIKI_CATS.items()}
    return _multihop(rows, n_q, per_bank, inv, offset=offset), WIKI_CATS


LME_CATS = {1: "multi-session", 2: "temporal-reasoning", 3: "knowledge-update",
            4: "single-session-user", 5: "single-session-assistant",
            6: "single-session-preference"}


def load_longmemeval():
    """LongMemEval-S (cleaned): each question has its OWN ~490-turn haystack, so each
    question is one bank and one inference cluster. Gold = turns flagged has_answer.
    Questions with no flagged turn (21 of 500) have nothing to retrieve and are dropped."""
    d = json.loads((DATA / "longmemeval/longmemeval_s_cleaned.json").read_text())
    inv = {v: k for k, v in LME_CATS.items()}
    specs = []
    for q in d:
        texts, gold = [], []
        for s in q["haystack_sessions"]:
            for t in s:
                if t.get("has_answer"):
                    gold.append(len(texts))
                texts.append(f"{t['role']}: {t['content']}")
        if not gold:
            continue
        specs.append(dict(sample_id=q["question_id"], texts=texts, questions=[q["question"]],
                          gold=[gold], cats=[inv[q["question_type"]]], answers=[str(q["answer"])]))
    return specs, LME_CATS


QASPER_CATS = {1: "single-evidence", 2: "multi-evidence"}


def load_qasper():
    """QASPER dev: bank = the paper's paragraphs; gold = annotator-0's evidence paragraphs
    that are paragraphs of the paper (FLOAT SELECTED table/figure captions are not, and
    are dropped). Pre-declared: annotator 0 only, so m is not inflated by a union."""
    d = json.loads((DATA / "qasper/qasper-dev-v0.3.json").read_text())
    specs = []
    for pid, p in sorted(d.items()):
        texts = [para.strip() for sec in p["full_text"] for para in sec["paragraphs"]
                 if para and para.strip()]
        pos = {}
        for i, t in enumerate(texts):
            pos.setdefault(t, i)
        qs, gold, cats, ans = [], [], [], []
        for qa in p["qas"]:
            if not qa["answers"]:
                continue
            a = qa["answers"][0]["answer"]
            g = sorted({pos[e.strip()] for e in a.get("evidence", []) if e.strip() in pos})
            if not g:
                continue
            qs.append(qa["question"]); gold.append(g); cats.append(1 if len(g) == 1 else 2)
            ans.append(a.get("free_form_answer") or " ".join(a.get("extractive_spans", [])))
        if qs and len(texts) > 12:
            specs.append(dict(sample_id=pid, texts=texts, questions=qs, gold=gold, cats=cats,
                              answers=ans))
    return specs, QASPER_CATS


def load_beir(name, n_groups=N_GROUPS):
    """BEIR test split: one shared corpus; queries dealt into n_groups banks (seeded),
    each bank carrying the whole corpus. Queries are independent -> cluster = query."""
    corp = _pq(DATA / f"beir/{name}/corpus.parquet")
    qs = {r["_id"]: r["text"] for r in _pq(DATA / f"beir/{name}/queries.parquet")}
    texts = [f"{r['title']}. {r['text']}".strip(". ") for r in corp]
    pos = {r["_id"]: i for i, r in enumerate(corp)}
    rel = {}
    for line in (DATA / f"beir/{name}/test.tsv").read_text().splitlines()[1:]:
        qid, cid, s = line.split("\t")
        if int(s) > 0 and cid in pos:
            rel.setdefault(qid, set()).add(pos[cid])
    qids = sorted(q for q in rel if q in qs)
    rng = np.random.default_rng(SPLIT_SEED)
    qids = [qids[i] for i in rng.permutation(len(qids))]
    specs = []
    for g in range(n_groups):
        mine = qids[g::n_groups]
        specs.append(dict(sample_id=f"{name}{g}", texts=texts, questions=[qs[q] for q in mine],
                          gold=[sorted(rel[q]) for q in mine],
                          cats=[1 if len(rel[q]) == 1 else 2 for q in mine]))
    return specs, {1: "single-relevant", 2: "multi-relevant"}


# ------------------------------------------------------- evaluation & selection

def evaluate(b: Bank, fn, cats=None, **params) -> dict:
    """locomo_qa.evaluate plus the query index, so per-question clusters are possible."""
    rows = {f"full@{k}": [] for k in KS} | {f"gold@{k}": [] for k in KS}
    rows |= {"mrr": [], "ndcg@10": [], "cat": [], "m": [], "qidx": []}
    for i in range(b.m):
        if cats is not None and b.cats[i] not in cats:
            continue
        g = b.gold[i]
        order = np.argsort(-fn(b, i, **params))
        for k in KS:
            rows[f"full@{k}"].append(full_set_recall_at_k(order, g, k))
            rows[f"gold@{k}"].append(mean_gold_recall_at_k(order, g, k))
        rows["mrr"].append(reciprocal_rank(order, g))
        rows["ndcg@10"].append(ndcg_at_k(order, g, 10))
        rows["cat"].append(b.cats[i]); rows["m"].append(len(g)); rows["qidx"].append(i)
    return {k: np.asarray(v) for k, v in rows.items()}


def sweep(bks, fn, grid, cats=None) -> dict:
    return {_key(p): [evaluate(b, fn, cats=cats, **p) for b in bks] for p in grid_points(grid)}


METRIC_KEYS = [f"full@{k}" for k in KS] + [f"gold@{k}" for k in KS] + ["mrr", "ndcg@10"]


def select(bks, groups, table, cluster_level, score_key=PRIMARY, fixed: str | None = None):
    """Leave-one-group-out selection. Returns every metric on the held-out side.

    ``fixed`` bypasses selection and applies one named config everywhere (X105 transfer).
    ``cluster_level`` is "bank" or "question".
    """
    groups = np.asarray(groups)
    out = {k: [] for k in METRIC_KEYS + ["cat", "m"]}
    clus, picks = [], []
    for g in np.unique(groups):
        test = np.flatnonzero(groups == g)
        train = np.flatnonzero(groups != g)
        if fixed is not None:
            best, best_s = fixed, float("nan")
        else:
            best, best_s = None, -np.inf
            for k, per_bank in table.items():
                v = np.concatenate([per_bank[j][score_key] for j in train])
                s = float(v.mean()) if v.size else -np.inf
                if s > best_s:
                    best, best_s = k, s
        for j in test:
            r = table[best][j]
            for k in out:
                out[k].append(r[k])
            cid = [bks[j].sample_id] * len(r["cat"]) if cluster_level == "bank" else \
                [f"{bks[j].sample_id}:{q}" for q in r["qidx"]]
            clus.append(np.asarray(cid, dtype=object))
        picks.append(dict(group=int(g), picked=best, train_score=best_s))
    return ({k: np.concatenate(v) for k, v in out.items()}, np.concatenate(clus), picks)


# ------------------------------------------------------------------ inference

def _cluster_index(clusters):
    u, inv = np.unique(np.asarray(clusters).astype(str), return_inverse=True)
    return u, inv


def cluster_bootstrap(d, clusters, n=4000, seed=0):
    """Two-level for <=30 clusters (the project's hierarchical_bootstrap); vectorised
    one-level above that, where clusters are mostly singletons and the inner level is
    immaterial."""
    d = np.asarray(d, np.float64)
    u, inv = _cluster_index(clusters)
    if len(u) <= 30:
        return hierarchical_bootstrap(d, inv, n=n, seed=seed)
    sums = np.bincount(inv, weights=d, minlength=len(u))
    cnts = np.bincount(inv, minlength=len(u)).astype(np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(u), size=(n, len(u)))
    m = sums[idx].sum(1) / cnts[idx].sum(1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def sign_flip_p(d, clusters, n=20000, seed=0):
    """Two-sided p, one sign per cluster. Exact enumeration up to 22 clusters."""
    d = np.asarray(d, np.float64)
    u, inv = _cluster_index(clusters)
    if len(u) <= 22:
        return exact_blocked_permutation(d, inv), "exact"
    sums = np.bincount(inv, weights=d, minlength=len(u))
    rng = np.random.default_rng(seed)
    s = rng.choice((-1.0, 1.0), size=(n, len(u)))
    stats = np.abs(s @ sums) / d.size
    return float(((stats >= abs(d.mean()) - 1e-15).sum() + 1) / (n + 1)), "monte-carlo"


def paired(a, b, clusters) -> dict:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    d = a - b
    mean, lo, hi = cluster_bootstrap(d, clusters)
    p, how = sign_flip_p(d, clusters)
    nclu = len(np.unique(np.asarray(clusters).astype(str)))
    return dict(a=float(a.mean()), b=float(b.mean()), delta=mean, lo=lo, hi=hi, p=p,
                p_method=how, p_floor=(2.0 / 2 ** nclu if how == "exact" else None),
                n=int(a.size), n_clusters=nclu)


def dod(d1, c1, d2, c2, n=8000, seed=0) -> dict:
    """Difference of differences mean(d1) - mean(d2), clusters resampled jointly."""
    c1, c2 = np.asarray(c1).astype(str), np.asarray(c2).astype(str)
    convs = np.unique(np.concatenate([c1, c2]))
    i1 = {u: np.flatnonzero(c1 == u) for u in convs}
    i2 = {u: np.flatnonzero(c2 == u) for u in convs}
    rng = np.random.default_rng(seed)
    boot = np.empty(n)
    for t in range(n):
        take = rng.choice(convs, size=len(convs), replace=True)
        e1 = np.concatenate([i1[u] for u in take]); e2 = np.concatenate([i2[u] for u in take])
        boot[t] = (d1[e1].mean() if e1.size else 0.0) - (d2[e2].mean() if e2.size else 0.0)
    obs = float(np.mean(d1) - np.mean(d2))
    return dict(dod=obs, lo=float(np.percentile(boot, 2.5)), hi=float(np.percentile(boot, 97.5)),
                p=float(min(1.0, 2 * min((boot <= 0).mean(), (boot >= 0).mean()))),
                n1=int(len(d1)), n2=int(len(d2)))


# ------------------------------------------------------------------ X27 certificate

def certificate(bks, catnames) -> dict:
    rows = []
    for b in bks:
        S = b.Q @ b.Z.T
        for i in range(b.m):
            g = b.gold[i]; m = len(g)
            if b.n <= m + 10:
                continue
            G = b.Z[g]
            eps = float(((G @ G.T).sum() - m) / (m * (m - 1))) if m > 1 else 1.0
            mask = np.ones(b.n, bool); mask[g] = False
            dist = np.sort(S[i][mask])[::-1]
            rows.append(dict(cat=b.cats[i], m=m, eps=eps,
                             bound=float(np.sqrt((1.0 + (m - 1) * eps) / m)),
                             gold_min=float(S[i][g].min()), d10=float(dist[9]), n=b.n))
    res = {}
    for c in sorted(set(r["cat"] for r in rows)) + ["all"]:
        r = rows if c == "all" else [x for x in rows if x["cat"] == c]
        f = lambda k: float(np.mean([x[k] for x in r]))
        res[str(c) if c == "all" else catnames.get(c, str(c))] = dict(
            n=len(r), mean_m=f("m"), mean_eps=float(np.mean([x["eps"] for x in r if x["m"] > 1]))
            if any(x["m"] > 1 for x in r) else None,
            mean_bound=f("bound"), mean_gold_min=f("gold_min"), mean_d10=f("d10"),
            mean_bank=f("n"),
            frac_bound_binds=float(np.mean([x["bound"] < x["d10"] for x in r])),
            frac_query_fails=float(np.mean([x["gold_min"] < x["d10"] for x in r])))
    return res


# ------------------------------------------------------------------ one corpus, end to end

def run_corpus(name, bks, groups, cluster_level, catnames, strata, methods=None,
               interactions=(), transfer: dict | None = None) -> dict:
    """Certificate, baseline grid, expansion, I4 contrast per stratum, interactions.

    strata: {label: tuple of cat codes, or None for all}
    interactions: [(label_hi, label_lo)] -> DoD of (mixture - mixture_avg) between strata
    transfer: {method: fixed config key} for X105
    """
    methods = methods or list(GRIDS)
    print(f"\n{'#' * 100}\n# {name}: {len(bks)} banks, {sum(b.m for b in bks)} queries, "
          f"mean bank {np.mean([b.n for b in bks]):.0f}, {len(set(groups))} selection groups, "
          f"cluster = {cluster_level}\n{'#' * 100}")
    out = dict(corpus=name, n_banks=len(bks), n_queries=int(sum(b.m for b in bks)),
               mean_bank=float(np.mean([b.n for b in bks])), cluster_level=cluster_level,
               n_groups=len(set(groups)), grids={k: {kk: list(map(str, vv)) for kk, vv in
                                                     GRIDS[k].items()} for k in methods},
               certificate=certificate(bks, catnames), strata={}, interactions={})
    print("X27 certificate:")
    for k, v in out["certificate"].items():
        print(f"  {k:<20} n={v['n']:>5} m={v['mean_m']:.2f} eps={v['mean_eps'] or float('nan'):.3f}"
              f" B={v['mean_bound']:.3f} gmin={v['mean_gold_min']:.3f} d10={v['mean_d10']:.3f}"
              f"  B<d10 {v['frac_bound_binds']:.1%}  query fails {v['frac_query_fails']:.1%}")

    keep = {}
    for label, cats in strata.items():
        print(f"\n== {name} / {label}")
        hdr = f"{'method':<13}{'n':>6}" + "".join(f"{'gold@' + str(k):>9}" for k in KS) \
            + f"{'full@10':>9}{'mrr':>8}{'ndcg10':>8}"
        print(hdr)
        store, vals = {}, {}
        for meth in methods:
            table = sweep(bks, METHODS[meth], GRIDS[meth], cats=cats)
            mets, clus, picks = select(bks, groups, table, cluster_level)
            if mets[PRIMARY].size == 0:
                break
            row = {k: float(mets[k].mean()) for k in METRIC_KEYS}
            nulls = sum(1 for p in picks if p["picked"] and
                        ("K0=0" in p["picked"] or "gamma=0.0" in p["picked"]))
            store[meth] = dict(metrics=row, n=int(len(clus)), picks=picks,
                               folds_backed_off_to_null=nulls if "K0" in GRIDS[meth] else None)
            vals[meth] = (mets[PRIMARY], clus, mets["cat"])
            if transfer and meth in transfer:
                tm, tc, _ = select(bks, groups, table, cluster_level, fixed=transfer[meth])
                store[meth]["transfer"] = dict(config=transfer[meth],
                                               metrics={k: float(tm[k].mean()) for k in METRIC_KEYS},
                                               vs_selected=paired(tm[PRIMARY], mets[PRIMARY], tc))
            print(f"{meth:<13}{len(clus):>6}" + "".join(f"{row['gold@' + str(k)]:>9.3f}" for k in KS)
                  + f"{row['full@10']:>9.3f}{row['mrr']:>8.3f}{row['ndcg@10']:>8.3f}"
                  + (f"   null {nulls}/{len(picks)}" if "K0" in GRIDS[meth] else ""))
        if not vals:
            continue
        cv, cc, _ = vals["cosine"]
        for meth in vals:
            if meth == "cosine":
                continue
            r = paired(vals[meth][0], cv, cc)
            store[meth]["vs_cosine"] = r
            print(f"  {meth + ' - cosine':<24}{r['delta']:>+8.4f} [{r['lo']:>+.4f}, {r['hi']:>+.4f}]"
                  f"  p={r['p']:.4g}{' (floor)' if r['p_floor'] and r['p'] <= r['p_floor'] + 1e-12 else ''}")
        if "mixture" in vals and "mixture_avg" in vals:
            r = paired(vals["mixture"][0], vals["mixture_avg"][0], cc)
            store["I4_contrast"] = r
            print(f"  {'mixture - mixture_avg':<24}{r['delta']:>+8.4f} [{r['lo']:>+.4f}, "
                  f"{r['hi']:>+.4f}]  p={r['p']:.4g}   <-- I4 contrast")
            keep[label] = (vals["mixture"][0] - vals["mixture_avg"][0], cc)
            # the comparisons a positive I4 contrast must also survive
            for other in ("rocchio", "rrf", "mmr"):
                if other in vals:
                    r = paired(vals["mixture"][0], vals[other][0], cc)
                    store[f"mixture_vs_{other}"] = r
                    print(f"  {'mixture - ' + other:<24}{r['delta']:>+8.4f} [{r['lo']:>+.4f}, "
                          f"{r['hi']:>+.4f}]  p={r['p']:.4g}")
        out["strata"][label] = store

    for hi_, lo_ in interactions:
        if hi_ in keep and lo_ in keep:
            r = dod(keep[hi_][0], keep[hi_][1], keep[lo_][0], keep[lo_][1])
            out["interactions"][f"{hi_} - {lo_}"] = r
            print(f"\nINTERACTION Δ[{hi_}] - Δ[{lo_}] = {r['dod']:+.4f} [{r['lo']:+.4f}, "
                  f"{r['hi']:+.4f}] p={r['p']:.4f}")
    return out


def groups_by_bank(bks, n=N_GROUPS, seed=SPLIT_SEED):
    """Assign banks to n selection groups, seeded, as evenly as possible."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(bks))
    g = np.empty(len(bks), dtype=int)
    g[perm] = np.arange(len(bks)) % n
    return g


def write(path_name: str, obj: dict):
    p = RESULTS / path_name
    p.write_text(json.dumps(obj, indent=2, default=float))
    print(f"\nwrote {p}")
