"""Paired bootstrap on the DIFFERENCE vs the dense baseline.

Comparing two overlapping confidence intervals is the wrong test when both methods are
evaluated on the SAME questions -- it is far too conservative and hides real effects.
The correct analysis resamples the per-question DIFFERENCE. If that interval excludes
zero, the win is real.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from energymem.base import Memory                       # noqa: E402
from energymem.encoders import SentenceEncoder          # noqa: E402
from energymem.memory.energy import EnergyMemory        # noqa: E402
from energymem.baselines.lexical import BM25, HybridRRF # noqa: E402
from energymem.eval.corpus import build                 # noqa: E402
from experiments.run_real import mask_words, embed_noise, vote_project  # noqa: E402

SEEDS=[0,1,2,3,4]; NF=900; T=8

def paired_ci(diff, qid, n=20000, seed=0):
    """CLUSTER bootstrap over QUESTIONS, not over measurements.

    The 5 seeds reshuffle the corpus but reuse the SAME 37 questions, so the 185 rows
    are 5 correlated observations of 37 items. Resampling rows treats them as
    independent and produces intervals that are far too narrow -- the classic
    pseudo-replication error. Resampling whole questions is the correct unit."""
    a=np.asarray(diff); q=np.asarray(qid)
    uq=np.unique(q); by=[a[q==u] for u in uq]
    rng=np.random.default_rng(seed)
    idx=rng.integers(0,len(by),size=(n,len(by)))
    means=np.array([np.concatenate([by[j] for j in row]).mean() for row in idx])
    return float(a.mean()), float(np.quantile(means,.025)), float(np.quantile(means,.975)), float((means<=0).mean())

def main():
    enc=SentenceEncoder()
    CONDS=[("clean",{}),("mask-50",{"mask":.5}),("mask-70",{"mask":.7})]
    per={c:{} for c,_ in CONDS}
    for seed in SEEDS:
        eps,qs=build(seed,n_filler=NF)
        E=enc.encode([e.text for e in eps])
        mems=[Memory(id=f"e{i}",text=eps[i].text,embedding=E[i]) for i in range(len(eps))]
        dn=EnergyMemory(n_steps=1,beta=1e6); dn.index(mems)
        e64=EnergyMemory(n_steps=T,beta=64.0); e64.index(mems)
        e16=EnergyMemory(n_steps=T,beta=16.0); e16.index(mems)
        bm=BM25(); bm.index(mems); hy=HybridRRF(); hy.index(mems)
        for cn,cfg in CONDS:
            rng=np.random.default_rng(1000+seed)
            for q in qs:
                k=len(q.gold_idx); gold={f"e{i}" for i in q.gold_idx}
                qt=mask_words(q.text,cfg.get("mask",0.),rng)
                qv=embed_noise(enc.encode(qt),cfg.get("noise",0.),rng)
                R=lambda ids: len(gold&set(ids[:k]))/k
                got={
                 "bm25":R(bm.retrieve(qt,k=k).ids),
                 "hybrid_rrf":R(hy.retrieve(qt,qv,k=k).ids),
                 "peeling":R(e64.retrieve_multi(qv,k=k).ids),
                 "repulsion_b64":R(e64.retrieve_multi_parallel(qv,k=k,seed=seed).ids),
                 "repulsion_b16":R(e16.retrieve_multi_parallel(qv,k=k,seed=seed).ids),
                 "vote_proj":len(gold&{f"e{i}" for i in vote_project(e64,qv,k,seed=seed)[:k]})/k,
                }
                base=R(dn.retrieve(qv,k=k).ids)
                for m,v in got.items():
                    per[cn].setdefault(m,[]).append(v-base)
                per[cn].setdefault("_base",[]).append(base)
                per[cn].setdefault("_qid",[]).append(q.text)

    print("="*92)
    print("PAIRED CLUSTER BOOTSTRAP vs DENSE TOP-K  (resampling QUESTIONS)")
    print("Delta = method - dense.  Significant if the 95% interval excludes zero.")
    print("="*92)
    out={}
    for cn,_ in CONDS:
        b=np.mean(per[cn]["_base"])
        print(f"\n[{cn}]   dense baseline = {b:.3f}")
        print(f"  {'method':>15} {'delta':>8}  {'95% CI on delta':>22}  {'p(delta<=0)':>12}  verdict")
        out[cn]={}
        for m in ["bm25","hybrid_rrf","peeling","repulsion_b64","repulsion_b16","vote_proj"]:
            mu,lo,hi,p=paired_ci(per[cn][m], per[cn]["_qid"])
            sig = "** WINS" if lo>0 else ("** LOSES" if hi<0 else "   n.s.")
            out[cn][m]=dict(delta=mu,ci=[lo,hi],p_le0=p)
            print(f"  {m:>15} {mu:>+8.4f}  [{lo:>+7.4f}, {hi:>+7.4f}]  {p:>12.4f}  {sig}")
    Path("experiments/results/real").mkdir(parents=True,exist_ok=True)
    Path("experiments/results/real/paired_vs_dense.json").write_text(json.dumps(out,indent=2))
    print("\nwrote experiments/results/real/paired_vs_dense.json")

if __name__=="__main__":
    main()
