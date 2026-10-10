"""X110b -- was the text beta band simply below X110's grid? Fifteen beta on the TUNE half.

X110 selected beta over 8..256 and every value was harmful, with the harm shrinking
monotonically toward beta->infinity (where the descent IS cosine, by Corollary 1). The
obvious objection is that the text band might lie outside that window. This closes it:
beta from 1 to 1024 crossed with the three image-study severities, tune half only, so no
held-out number is spent on answering an objection.

An uncommitted script is not a result (the sixth protocol error), which is why this file
exists rather than living in a scratch directory.
"""
import json, sys, numpy as np
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'src'))
import experiments.x110_text_occlusion as X
from energymem.encoders import SentenceEncoder

enc = SentenceEncoder(); bks = X.banks(encoder=enc)
probes = X.build_probes(bks, enc, X.mask_span, levels=(0.3,0.5,0.7))
rng = np.random.default_rng(X.SPLIT_SEED); perm = rng.permutation(len(bks))
tune = sorted(perm[:len(bks)//2].tolist())
cos = {l: X.score(bks, probes, l, None, "cosine") for l in (0.3,0.5,0.7)}
BET = (1.,2.,4.,8.,16.,32.,48.,64.,96.,128.,192.,256.,384.,512.,1024.)
print(f"\n{'beta':>7}" + "".join(f"{f'd R@1 @{l}':>14}" for l in (0.3,0.5,0.7))
      + "".join(f"{f'd R@5 @{l}':>14}" for l in (0.3,0.5,0.7)))
out={}
for b in BET:
    r1s, r5s = [], []
    for l in (0.3,0.5,0.7):
        d1,d5 = X.score(bks, probes, l, b, "descent")
        m = np.isin(probes[l][1], tune)
        r1s.append(float(d1[m].mean()-cos[l][0][m].mean()))
        r5s.append(float(d5[m].mean()-cos[l][1][m].mean()))
    out[str(b)]=dict(d_r1=r1s, d_r5=r5s)
    print(f"{b:>7g}" + "".join(f"{x:>+14.4f}" for x in r1s) + "".join(f"{x:>+14.4f}" for x in r5s))
best = max(out, key=lambda k: max(out[k]["d_r1"]))
print(f"\n  best single (beta, level) cell for R@1: beta={best}, "
      f"delta={max(out[best]['d_r1']):+.4f}")
print(f"  number of (beta, level) cells with delta R@1 > 0: "
      f"{sum(1 for k in out for x in out[k]['d_r1'] if x > 0)} of {len(out)*3}")
print(f"  number with delta R@5 > 0: "
      f"{sum(1 for k in out for x in out[k]['d_r5'] if x > 0)} of {len(out)*3}")
json.dump(out, open("experiments/results/x110b_beta_band_text.json","w"), indent=2)
print("wrote experiments/results/x110b_beta_band_text.json")
