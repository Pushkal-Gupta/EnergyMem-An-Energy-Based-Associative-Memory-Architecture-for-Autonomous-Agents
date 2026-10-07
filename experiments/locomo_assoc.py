"""X26 -- temporal-association probes on LoCoMo. THE EXTERNAL-VALIDITY GATE.

Every composition-regime number in this project is measured on a 1,319-episode corpus we
wrote ourselves (X16-X19, X25). It is internally rigorous and externally unestablished,
and the one-hop positive is the paper's only replicated claim. This script asks the same
question on a public benchmark built by another group.

TASK. Given a conversational turn, retrieve the turn that FOLLOWED it. The candidate set
is the full pooled bank of all 5,882 turns across all 10 conversations, so a prediction
must beat distractors from nine unrelated conversations as well as from its own.

NAMESPACING IS NOT OPTIONAL. LoCoMo dia_ids are unique only within a conversation: 5,882
turns carry 1,033 distinct ids. eval/locomo.py namespaces every id with its sample_id;
a bank pooled without that lets an unrelated turn satisfy a gold id and score as a hit,
which already invalidated one result in this project.

SPLIT UNIT IS THE SESSION. A session is a coherent conversational episode. Transitions
from a session the model was fitted on are never evaluated, and 4-fold rotation puts every
session in the test set exactly once. Cluster bootstrap resamples sessions.

    python experiments/locomo_assoc.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from energymem.base import l2_normalize                 # noqa: E402
from energymem.encoders import SentenceEncoder          # noqa: E402
from energymem.eval.locomo import load_locomo           # noqa: E402
from experiments.assoc_harness import cluster_ci        # noqa: E402

DATA = ROOT / "data" / "locomo10.json"
N_FOLDS = 4
SPLIT_SEED = 12345
CHUNK = 256
MAX_PROBES_PER_SESSION = 25        # keeps folds balanced; clusters drive the interval
KP_GRID = [5, 10, 20]
TAU_GRID = [2.0, 5.0, 10.0]
K_GRID = [1, 2, 3]
BETA_GRID = [16.0, 32.0]
ALPHA_GRID = [0.25, 0.35, 1.0]
VAL_CAP = 600                      # selection subsample, fixed seed


# --------------------------------------------------------------------- loading

def build():
    convs = load_locomo(DATA)
    texts, sess_of, idx = [], [], {}
    for c in convs:
        for m in c.memories:
            idx[m.id] = len(texts)
            texts.append(m.text)
            sess_of.append(f"{m.metadata['sample_id']}/{m.metadata['session']}")
    Z = SentenceEncoder().encode(texts)
    sess_of = np.array(sess_of)

    # probes: consecutive turns inside one session, in stored order
    probes = []          # (cue_idx, gold_idx, session, hop)
    for s in np.unique(sess_of):
        members = np.flatnonzero(sess_of == s)          # already in session order
        for hop in (1, 2):
            for a in range(len(members) - hop):
                probes.append((int(members[a]), int(members[a + hop]), s, hop))
    return Z, sess_of, probes


def folds(sessions):
    s = sorted(set(sessions))
    rng = np.random.default_rng(SPLIT_SEED)
    rng.shuffle(s)
    parts = [set(s[i::N_FOLDS]) for i in range(N_FOLDS)]
    out = []
    for f in range(N_FOLDS):
        test = parts[f]
        val = parts[(f + 1) % N_FOLDS]
        train = set().union(*[parts[j] for j in range(N_FOLDS) if j not in (f, (f + 1) % N_FOLDS)])
        out.append((train, val, test))
    return out


# ------------------------------------------------------- vectorised world model

class WM:
    """Same model as world/knn_wm.py, batched over queries."""

    def __init__(self, Z, transitions, k, tau):
        a = np.array([t[0] for t in transitions])
        b = np.array([t[1] for t in transitions])
        self.Zc = Z[a]
        self.D = Z[b] - Z[a]
        self.k, self.tau = k, tau

    def _kernel(self, Q):
        S = Q @ self.Zc.T
        k = min(self.k, S.shape[1])
        nn = np.argpartition(-S, k - 1, axis=1)[:, :k]
        rows = np.arange(len(Q))[:, None]
        s = S[rows, nn]
        o = np.argsort(-s, axis=1)
        nn, s = nn[rows, o], s[rows, o]
        w = np.exp(self.tau * (s - s[:, :1]))
        return nn, w / w.sum(1, keepdims=True)

    def mean(self, Q):
        nn, w = self._kernel(Q)
        return l2_normalize(Q + (w[..., None] * self.D[nn]).sum(1), axis=1)

    def modes(self, Q, K, alpha):
        nn, w = self._kernel(Q)
        K = min(K, nn.shape[1])
        nn, w = nn[:, :K], w[:, :K] / w[:, :K].sum(1, keepdims=True)
        return [(l2_normalize(Q + alpha * self.D[nn[:, j]], axis=1), w[:, j]) for j in range(K)]


# ------------------------------------------------------------------- evaluation

def _hyps(wm, Q, method, hop, K, beta, alpha):
    if method == "cosine":
        return [(Q, np.ones(len(Q)))]
    if method == "averaged":
        cur = Q
        for _ in range(hop):
            cur = wm.mean(cur)
        return [(cur, np.ones(len(Q)))]
    if method == "mixture":
        beam = [(Q, np.ones(len(Q)))]
        for _ in range(hop):
            nxt = []
            for Zh, wh in beam:
                for Zm, wm_ in wm.modes(Zh, K, alpha):
                    nxt.append((Zm, wh * wm_))
            tot = sum(w for _, w in nxt)
            beam = [(z, w / tot) for z, w in nxt]
        return beam
    raise ValueError(method)


def evaluate(Z, wm, probes, method, hop, K, beta, alpha):
    """Returns (r1, r5) arrays. Cue excluded from candidates for every method alike."""
    cue = np.array([p[0] for p in probes])
    gold = np.array([p[1] for p in probes])
    r1 = np.empty(len(probes)); r5 = np.empty(len(probes))
    for st in range(0, len(probes), CHUNK):
        sl = slice(st, min(st + CHUNK, len(probes)))
        Q = Z[cue[sl]]
        hs = _hyps(wm, Q, method, hop, K, beta, alpha)
        if len(hs) == 1:
            sc = hs[0][0] @ Z.T
        else:
            A = np.stack([beta * (Zh @ Z.T) + np.log(np.maximum(wh, 1e-300))[:, None]
                          for Zh, wh in hs])            # (S,P,N)
            m = A.max(0)
            sc = m + np.log(np.exp(A - m).sum(0))
        sc[np.arange(sc.shape[0]), cue[sl]] = -np.inf
        top = np.argpartition(-sc, 5, axis=1)[:, :5]
        rows = np.arange(sc.shape[0])[:, None]
        top = top[rows, np.argsort(-sc[rows, top], axis=1)]
        g = gold[sl][:, None]
        r1[sl] = (top[:, :1] == g).any(1)
        r5[sl] = (top == g).any(1)
    return r1, r5


def subsample(probes, cap, seed=0):
    rng = np.random.default_rng(seed)
    by = {}
    for p in probes:
        by.setdefault(p[2], []).append(p)
    out = []
    for s, ps in by.items():
        out.extend([ps[i] for i in rng.permutation(len(ps))[:cap]] if len(ps) > cap else ps)
    return out


def main():
    if not DATA.exists():
        sys.exit(f"missing {DATA} -- see the curl command in README.md")
    Z, sess_of, all_probes = build()
    print("=" * 86)
    print("X26  LoCoMo temporal association -- the external-validity gate")
    print("=" * 86)
    print(f"  turns (candidate bank)   {len(Z)}")
    print(f"  sessions                 {len(set(sess_of))}")
    print(f"  conversations            {len({s.split('/')[0] for s in sess_of})}")

    out = {"bank": int(len(Z)), "sessions": len(set(sess_of)), "folds": N_FOLDS}
    splits = folds(sess_of)

    for hop in (1, 2):
        acc = {m: {"r1": [], "r5": []} for m in ("cosine", "averaged", "mixture")}
        clusters, chosen = [], []
        for train, val, test in splits:
            tr = [(a, b) for a, b, s, h in all_probes if h == 1 and s in train]
            vp = subsample([p for p in all_probes if p[3] == hop and p[2] in val],
                           MAX_PROBES_PER_SESSION)
            vp = vp[:VAL_CAP]
            tp = subsample([p for p in all_probes if p[3] == hop and p[2] in test],
                           MAX_PROBES_PER_SESSION)

            # (k, tau) on val, by the averaged predictor's R@5 -- the X17b protocol
            best, kt = -1.0, (KP_GRID[0], TAU_GRID[0])
            for kp in KP_GRID:
                for tau in TAU_GRID:
                    _, r5 = evaluate(Z, WM(Z, tr, kp, tau), vp, "averaged", hop, 1, 1.0, 1.0)
                    if r5.mean() > best:
                        best, kt = float(r5.mean()), (kp, tau)
            wm = WM(Z, tr, *kt)
            # (K, beta, alpha) for the mixture on val
            bb, kba = -1.0, (K_GRID[0], BETA_GRID[0], ALPHA_GRID[0])
            for K in K_GRID:
                for beta in BETA_GRID:
                    for alpha in ALPHA_GRID:
                        _, r5 = evaluate(Z, wm, vp, "mixture", hop, K, beta, alpha)
                        if r5.mean() > bb:
                            bb, kba = float(r5.mean()), (K, beta, alpha)
            chosen.append({"k_P": kt[0], "tau": kt[1], "K": kba[0], "beta": kba[1],
                           "alpha": kba[2], "n_train_transitions": len(tr), "n_val": len(vp)})
            clusters.extend(p[2] for p in tp)
            for m in acc:
                r1, r5 = evaluate(Z, wm, tp, m, hop, *kba)
                acc[m]["r1"].append(r1); acc[m]["r5"].append(r5)

        for m in acc:
            acc[m] = {k: np.concatenate(v) for k, v in acc[m].items()}
        clusters = np.array(clusters)
        print(f"\n  hop {hop}   n={len(clusters)} probes over {len(set(clusters))} sessions")
        print("   per-fold: " + "; ".join(
            f"f{i}(k={c['k_P']},tau={c['tau']:g},K={c['K']},b={c['beta']:g},a={c['alpha']:g},"
            f"tr={c['n_train_transitions']})" for i, c in enumerate(chosen)))
        print(f"  {'method':>12} {'R@1':>8} {'R@5':>8}   {'dR@1 vs cosine (95% CI)':>34}")
        base = acc["cosine"]["r1"]
        res = {}
        for m in ("cosine", "averaged", "mixture"):
            r1, r5 = acc[m]["r1"], acc[m]["r5"]
            if m == "cosine":
                print(f"  {m:>12} {r1.mean():>8.3f} {r5.mean():>8.3f}   {'--':>34}")
                res[m] = {"R@1": float(r1.mean()), "R@5": float(r5.mean())}
            else:
                d, lo, hi, p = cluster_ci(r1 - base, clusters)
                d5, lo5, hi5, p5 = cluster_ci(r5 - acc["cosine"]["r5"], clusters)
                print(f"  {m:>12} {r1.mean():>8.3f} {r5.mean():>8.3f}   "
                      f"{d:>+8.3f} [{lo:+.3f}, {hi:+.3f}] p={p:.4f}")
                res[m] = {"R@1": float(r1.mean()), "R@5": float(r5.mean()),
                          "dR1": d, "lo": lo, "hi": hi, "p": p,
                          "dR5": d5, "lo5": lo5, "hi5": hi5, "p5": p5}
        out[f"hop{hop}"] = {"n": len(clusters), "n_sessions": len(set(clusters)),
                            "selected": chosen, "methods": res}

    dst = ROOT / "experiments" / "results" / "x26_locomo_assoc.json"
    dst.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {dst}")


if __name__ == "__main__":
    main()
