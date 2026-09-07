"""Dev/test split -- the analysis that makes the beta choice legitimate.

beta=16 was picked by sweeping on the same questions we then reported on. That is
test-set tuning and a reviewer will call it out. Here the questions are split in half by
THREAD (so no storyline leaks across the split): beta is chosen on dev, and the number we
report comes from test, which the tuning never saw.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from energymem.base import Memory                        # noqa: E402
from energymem.encoders import SentenceEncoder           # noqa: E402
from energymem.memory.energy import EnergyMemory         # noqa: E402
from energymem.eval.corpus import build                  # noqa: E402
from experiments.run_real import mask_words              # noqa: E402

SEEDS=[0,1,2,3,4]; NF=900; T=8
BETAS=[16.,32.,64.,128.]

def cluster_ci(diff,qid,n=20000,seed=0):
    a=np.asarray(diff); q=np.asarray(qid); uq=np.unique(q)
    by=[a[q==u] for u in uq]
    rng=np.random.default_rng(seed)
    idx=rng.integers(0,len(by),size=(n,len(by)))
    m=np.array([np.concatenate([by[j] for j in r]).mean() for r in idx])
    return float(a.mean()), float(np.quantile(m,.025)), float(np.quantile(m,.975)), float((m<=0).mean())

def main():
    enc=SentenceEncoder()
    # split by THREAD so a storyline never spans dev and test
    eps0,qs0=build(0,n_filler=NF)
    threads=sorted({q.thread for q in qs0})
    rng=np.random.default_rng(12345); rng.shuffle(threads)
    dev_th=set(threads[:len(threads)//2]); test_th=set(threads[len(threads)//2:])
    print(f"threads: {len(dev_th)} dev / {len(test_th)} test")
    print(f"questions: {sum(q.thread in dev_th for q in qs0)} dev / "
          f"{sum(q.thread in test_th for q in qs0)} test\n")

    store={}   # (split, method, beta) -> list of (qid, delta)
    for seed in SEEDS:
        eps,qs=build(seed,n_filler=NF)
        E=enc.encode([e.text for e in eps])
        mems=[Memory(id=f"e{i}",text=eps[i].text,embedding=E[i]) for i in range(len(eps))]
        dn=EnergyMemory(n_steps=1,beta=1e6); dn.index(mems)
        models={b:EnergyMemory(n_steps=T,beta=b) for b in BETAS}
        for b,m in models.items(): m.index(mems)
        for q in qs:
            split="dev" if q.thread in dev_th else "test"
            k=len(q.gold_idx); gold={f"e{i}" for i in q.gold_idx}
            qv=enc.encode(q.text)
            base=len(gold&set(dn.retrieve(qv,k=k).ids[:k]))/k
            for b in BETAS:
                mm=models[b]
                for meth,ids in (("repulsion", mm.retrieve_multi_parallel(qv,k=k,seed=seed).ids),
                                 ("peeling",   mm.retrieve_multi(qv,k=k).ids)):
                    v=len(gold&set(ids[:k]))/k
                    store.setdefault((split,meth,b),[]).append((q.text, v-base))

    print("DEV -- choosing beta (this half is allowed to be looked at)")
    print(f"  {'method':>10} " + " ".join(f"{'b='+str(int(b)):>10}" for b in BETAS))
    best={}
    for meth in ["repulsion","peeling"]:
        row=[]
        for b in BETAS:
            d=[x[1] for x in store[("dev",meth,b)]]
            row.append(float(np.mean(d)))
        print(f"  {meth:>10} " + " ".join(f"{v:>+10.4f}" for v in row))
        best[meth]=BETAS[int(np.argmax(row))]
    print(f"\n  chosen on dev:  repulsion beta={best['repulsion']:.0f}   peeling beta={best['peeling']:.0f}")

    print("\n" + "="*84)
    print("TEST -- the reportable number. Tuning never saw these questions.")
    print("="*84)
    out={}
    for meth in ["repulsion","peeling"]:
        b=best[meth]
        rows=store[("test",meth,b)]
        qid=[r[0] for r in rows]; d=[r[1] for r in rows]
        mu,lo,hi,p=cluster_ci(d,qid)
        sig="** SIGNIFICANT" if lo>0 else "   not significant"
        out[meth]=dict(beta=b,delta=mu,ci=[lo,hi],p_le0=p,n_q=len(set(qid)))
        print(f"  {meth:>10} (beta={b:.0f})  delta={mu:+.4f}  "
              f"[{lo:+.4f}, {hi:+.4f}]  p={p:.4f}  {sig}")
        print(f"  {'':>10}  {len(set(qid))} held-out questions")
    Path("experiments/results/real/devtest.json").write_text(json.dumps(out,indent=2))
    print("\nwrote experiments/results/real/devtest.json")

if __name__=="__main__":
    main()
