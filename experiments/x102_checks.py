"""X102 follow-up -- is the multi-hop expansion gain real? EXPLORATORY, NOT PRE-REGISTERED.

X102 (x10x_hotpot.json, x10x_2wiki.json) is the first place in the project where mode
expansion is selected over the cosine null: 0/10 null folds on HotpotQA bridge and on
2Wiki compositional/inference, with gains of +0.02 to +0.17 gold@10. This project's
history is that surprising numbers get stress-tested before anyone believes them, and
these checks were chosen AFTER seeing that result -- so they are logged as a deviation in
PREREG_HI.md and read as robustness, not confirmation.

  A  REPLICATION  a disjoint 2,000-question sample (positions 2000-3999 of the same seeded
                  permutation; no question shared with the primary run)
  B  ENCODER      bge-base-en-v1.5, the comparator PREREG_X32 declared primary in advance
  C  BANK SIZE    100 questions per bank -> 50 and 200 (bank ~500 and ~2000 paragraphs)
  D  RANDOM MODES X95 on this corpus: at forced (K0, beta), real top-K0 modes vs size-
                  matched random bank items. Real must beat random, or the gain is mass
                  on the bank, not association.

    python experiments/x102_checks.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments import corpus_qa as C                            # noqa: E402
from experiments.x9x_controls import s_random_modes               # noqa: E402
from energymem.encoders import SentenceEncoder                    # noqa: E402

METHODS = ["cosine", "rrf", "rocchio", "mixture", "mixture_avg"]
STRATA = {
    "hotpot": ({"bridge": (1,), "comparison": (2,), "all": None}, [("bridge", "comparison")]),
    "2wiki": ({"compositional": (1,), "inference": (2,), "comparison": (3,),
               "bridge_comparison (m=4)": (4,), "all": None}, []),
}
LOAD = {"hotpot": C.load_hotpot, "2wiki": C.load_2wiki}


def run(name, label, enc=None, **load_kw):
    specs, cn = LOAD[name](**load_kw)
    bks = C.make_banks(specs, enc)
    strata, inter = STRATA[name]
    return C.run_corpus(f"{name} [{label}]", bks, C.groups_by_bank(bks), "bank", cn, strata,
                        methods=METHODS, interactions=inter)


def random_modes(bks, cats):
    rows = []
    for K0 in (1, 3, 5):
        for beta in (4.0, 16.0, 64.0):
            real, rnd, clus = [], [], []
            for b in bks:
                r = C.evaluate(b, C.METHODS["mixture"], cats=cats, K0=K0, beta=beta)
                x = np.mean([C.evaluate(b, s_random_modes, cats=cats, K0=K0, beta=beta,
                                        seed=s)["gold@10"] for s in range(3)], axis=0)
                real.append(r["gold@10"]); rnd.append(x); clus += [b.sample_id] * len(x)
            pr = C.paired(np.concatenate(real), np.concatenate(rnd), clus)
            rows.append(dict(K0=K0, beta=beta, real=pr["a"], random=pr["b"], delta=pr["delta"],
                             lo=pr["lo"], hi=pr["hi"], p=pr["p"]))
            print(f"    K0={K0} beta={beta:>4}: real {pr['a']:.4f} random {pr['b']:.4f} "
                  f"delta {pr['delta']:+.4f} [{pr['lo']:+.4f}, {pr['hi']:+.4f}]")
    return rows


def main():
    out = C.stamp("X102-checks", status="EXPLORATORY: chosen after seeing X102; see PREREG_HI.md "
                                        "Deviations", methods=METHODS, runs={})
    bge = SentenceEncoder("BAAI/bge-base-en-v1.5")
    for name in ("hotpot", "2wiki"):
        out["runs"][f"{name}/A_replication"] = run(name, "A disjoint sample", offset=2000)
        out["runs"][f"{name}/B_bge"] = run(name, "B bge-base-en-v1.5", enc=bge)
        for pb in (50, 200):
            out["runs"][f"{name}/C_bank{pb}"] = run(name, f"C {pb} q/bank", per_bank=pb)
        specs, _ = LOAD[name]()
        bks = C.make_banks(specs)
        cats = (1,) if name == "hotpot" else (1, 2)
        print(f"\n  D random modes, {name}, cats {cats} (forced configs, with_query=True)")
        out["runs"][f"{name}/D_random_modes"] = random_modes(bks, cats)
    C.write("x102_checks.json", out)


if __name__ == "__main__":
    main()
