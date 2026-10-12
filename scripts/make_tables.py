r"""Generate paper tables and figures from experiments/results/*.json. NEVER TYPE A NUMBER.

Repo policy is generate-never-type, and the policy exists because a typo in a table is
indistinguishable from fabrication to a reader. Output goes to paper/generated/*.tex, which
main.tex \input{}s.

    python scripts/make_tables.py

PARTIAL COVERAGE, STATED HONESTLY. Only experiments that emitted JSON are covered here:
X23-X25 (mode diagnostics) and X26 (LoCoMo association). The image-completion, LoCoMo-anchored
and multi-beta studies predate the results-JSON convention and emitted PNGs plus stdout only;
their tables in the paper are still typed from the ledger. Those three scripts must be
re-run with a --json flag before submission. This file prints which tables it could NOT
generate so the gap is visible rather than assumed closed.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "experiments" / "results"
OUT = ROOT / "paper" / "generated"
FIG = ROOT / "paper" / "figures"

# Tables that cannot be generated, with the reason. Printed on every run so a gap in the
# generate-never-type policy is visible rather than assumed closed.
UNCOVERED = {
    "tab:negatives": "prose table, assembled from the ledger by hand (intentional)",
}
# Generated only when their results file exists; the run that produces it is named.
OPTIONAL = {
    "tab_images.tex": ("image_holdout.json", "scripts/confirm_holdout.py"),
    "tab_completion.tex": ("x113_class_vote.json", "experiments/x113_class_vote.py"),
    "tab_multibeta.tex": ("multibeta.json", "scripts/exp_multibeta_scales.py"),
}


def _validate(name: str, body: str) -> None:
    r"""Reject malformed LaTeX before it reaches the paper.

    WHY THIS EXISTS. Twice now a generator has emitted `\\` inside a cell instead of `\`,
    which LaTeX reads as a ROW BREAK: the table silently splits, rows spill into the margin,
    and the document still compiles with zero errors. A clean build is not evidence that a
    generated table is correct, so the generator checks its own output.
    """
    for i, line in enumerate(body.splitlines(), 1):
        if line.lstrip().startswith(("%", "\\begin", "\\end", "\\toprule", "\\midrule",
                                     "\\bottomrule", "\\cmidrule", "\\footnotesize")):
            continue
        if not line.strip():
            continue
        if line.count("\\\\") > 1:
            raise AssertionError(
                f"{name} line {i}: more than one row break -- an escaped backslash leaked "
                f"into a cell and LaTeX will split this row:\n  {line.strip()}")
        if line.strip() and not line.rstrip().endswith("\\\\"):
            raise AssertionError(f"{name} line {i}: table row does not end in a row break:\n  {line.strip()}")


def f3(x):
    return f"{x:.3f}"


def sig(x):
    return f"{x:+.3f}"


def tab_assoc(d26, d25, d17b=None, d101=None):
    """Table: one-hop association across implementations. Every row generated.

    d101 (X201, MSC) is optional: the row appears only when the results file is present,
    so the table degrades to the pre-MSC version rather than inventing a corpus.
    """
    h1 = d26["hop1"]
    a = h1["methods"]["averaged"]
    c = h1["methods"]["cosine"]
    e25 = d25["X25_hop1"]["methods"]
    if d17b is None:
        raise RuntimeError("x17b_single_split.json missing -- run experiments/x17b_single_split.py")
    rows = [
        ("(i)", "episodic (ours), 50/50", str(d17b["n_probes"]), str(d17b["n_clusters"]),
         f3(d17b["cosine_R@1"]), f3(d17b["wm_R@1"]),
         rf"${sig(d17b['delta'])}$ \ci{{{d17b['lo']:+.3f}}}{{{d17b['hi']:+.3f}}}, $p{{=}}{d17b['p']:.2f}$"),
        ("(ii)", "episodic (ours), 4-fold", str(d25["X25_hop1"]["n"]),
         str(d25["X25_hop1"]["n_threads"]), f3(e25["cosine"]["R@1"]), f3(e25["averaged"]["R@1"]),
         rf"${sig(e25['averaged']['dR1'])}$ \ci{{{e25['averaged']['lo']:+.3f}}}{{{e25['averaged']['hi']:+.3f}}}, $p{{=}}{e25['averaged']['p']:.3f}$"),
        ("(iii)", r"\textbf{\textsc{LoCoMo}}, 4-fold", rf"\textbf{{{h1['n']:,}}}",
         rf"\textbf{{{h1['n_sessions']}}}", rf"\textbf{{{f3(c['R@1'])}}}", rf"\textbf{{{f3(a['R@1'])}}}",
         rf"$\mathbf{{{sig(a['dR1'])}}}$ \ci{{{a['lo']:+.3f}}}{{{a['hi']:+.3f}}}, $\mathbf{{p<10^{{-4}}}}$"
         if a["p"] < 1e-4 else rf"${sig(a['dR1'])}$, $p{{=}}{a['p']:.4f}$"),
    ]
    if d101 is not None:
        m1 = d101["hop1"]
        am, cm = m1["methods"]["averaged"], m1["methods"]["cosine"]
        rows.append((
            "(iv)", r"\textbf{\textsc{MSC}}, 4-fold", rf"\textbf{{{m1['n']:,}}}",
            rf"\textbf{{{m1['n_sessions']:,}}}", rf"\textbf{{{f3(cm['R@1'])}}}",
            rf"\textbf{{{f3(am['R@1'])}}}",
            rf"$\mathbf{{{sig(am['dR1'])}}}$ \ci{{{am['lo']:+.3f}}}{{{am['hi']:+.3f}}}, "
            rf"$\mathbf{{p<10^{{-4}}}}$" if am["p"] < 1e-4
            else rf"${sig(am['dR1'])}$, $p{{=}}{am['p']:.4f}$"))
    body = " \\\\\n".join(" & ".join(r) for r in rows)
    return (
        "% GENERATED by scripts/make_tables.py -- do not edit\n"
        "\\footnotesize\\setlength{\\tabcolsep}{4pt}\n"
        "\\begin{tabular}{@{}llrrccl@{}}\n\\toprule\n"
        "& \\textbf{Corpus} & \\textbf{Probes} & \\textbf{Clust.} & \\textbf{Cosine} & "
        "\\textbf{WM} & \\textbf{Paired $\\Delta$ R@1} \\\\\n\\midrule\n"
        f"{body} \\\\\n\\bottomrule\n\\end{{tabular}}\n")


def tab_twohop(d26, d25, d101=None):
    h2 = d26["hop2"]; a = h2["methods"]["averaged"]; c = h2["methods"]["cosine"]
    e = d25["X25_hop2"]["methods"]
    rows = [
        ("(a)", r"episodic, 50/50 \textit{(lost)}", "65", "R@5",
         r"$0.446$ vs.\ $0.262$; $\Delta{+}0.185$"),
        ("(b)", "episodic, 4-fold", str(d25["X25_hop2"]["n"]), "R@5",
         rf"${f3(e['averaged']['R@5'])}$ vs. ${f3(e['cosine']['R@5'])}$; spans zero"),
        ("(c)", r"\textsc{LoCoMo}, 4-fold", f"{h2['n']:,}", "R@1",
         rf"${f3(a['R@1'])}$ vs. ${f3(c['R@1'])}$; $\Delta{sig(a['dR1'])}$ "
         rf"\ci{{{a['lo']:+.3f}}}{{{a['hi']:+.3f}}}"),
    ]
    if d101 is not None:
        m2 = d101["hop2"]; am = m2["methods"]["averaged"]; cm = m2["methods"]["cosine"]
        rows.append((
            "(d)", r"\textsc{MSC}, 4-fold", f"{m2['n']:,}", "R@1",
            rf"${f3(am['R@1'])}$ vs. ${f3(cm['R@1'])}$; $\Delta{sig(am['dR1'])}$ "
            rf"\ci{{{am['lo']:+.3f}}}{{{am['hi']:+.3f}}}"))
    body = " \\\\\n".join(" & ".join(r) for r in rows)
    return ("% GENERATED by scripts/make_tables.py -- do not edit\n"
            "\\footnotesize\\setlength{\\tabcolsep}{4pt}\n"
            "\\begin{tabularx}{\\textwidth}{@{}l>{\\raggedright\\arraybackslash}X rc>{\\raggedright\\arraybackslash}X@{}}\n\\toprule\n"
            "& \\textbf{Corpus / protocol} & \\textbf{Probes} & \\textbf{Metric} & "
            "\\textbf{Result} \\\\\n\\midrule\n"
            f"{body} \\\\\n\\bottomrule\n\\end{{tabularx}}\n")


def tab_images(d, d113):
    """Table 1: the image study with its MATCHED baseline (X113a).

    This table used to have three columns -- occlusion, cosine, descent -- and it was
    read as evidence for an associative mechanism. It was not. The metric is same-CLASS
    top-1 accuracy over ten classes, and a soft k-NN class vote, which has no energy, no
    iteration and no attractor, beats descent at every level. The fourth column is the
    one that decides what the first three mean, so it is not optional.
    """
    a = d113["X113a"]
    rows = []
    for lvl in sorted(d["levels"], key=int):
        v, w = d["levels"][lvl], a["per_level"][lvl]
        star = "$^{*}$" if (w["lo"] > 0 or w["hi"] < 0) else ""
        rows.append((f"{lvl}\\%", f3(v["cosine"]), f3(v["energy"]), f3(w["knn"]),
                     rf"${sig(w['descent_minus_knn'])}$ \ci{{{w['lo']:+.3f}}}{{{w['hi']:+.3f}}}{star}"))
    p, c = d["pooled"], a["contrasts"]["descent_minus_knn"]
    star = "$^{*}$" if (c["lo"] > 0 or c["hi"] < 0) else ""
    pooled = (r"\textbf{pooled}", rf"\textbf{{{f3(p['cosine'])}}}",
              rf"\textbf{{{f3(p['energy'])}}}", rf"\textbf{{{f3(a['knn'])}}}",
              rf"$\mathbf{{{sig(c['delta'])}}}$ \ci{{{c['lo']:+.3f}}}{{{c['hi']:+.3f}}}{star}")
    body = " \\\\\n".join(" & ".join(r) for r in rows)
    return ("% GENERATED by scripts/make_tables.py -- do not edit\n"
            "\\begin{tabular}{@{}lrrrl@{}}\n\\toprule\n"
            "\\textbf{Occlusion} & \\textbf{Cosine} & "
            f"\\textbf{{Descent}} ($\\beta{{=}}{d['beta']:g}$) & "
            f"\\textbf{{$k$-NN vote}} ($k{{=}}{a['selected_k']}$) & "
            "\\textbf{Descent $-$ vote (95\\% CI)} \\\\\n\\midrule\n"
            f"{body} \\\\\n\\midrule\n" + " & ".join(pooled) +
            " \\\\\n\\bottomrule\n\\end{tabular}\n")


def tab_completion(d113):
    """Table: the completion experiment the prose always described (X113b).

    Cues are occluded copies of images that ARE in the bank, so the premise of Claim 1
    holds by construction. Same-image recall@1 is identification; same-class accuracy on
    the identical cues is the published metric, shown beside it for comparison.
    """
    b = d113["X113b"]
    rows = []
    for lvl in sorted(b["per_level"], key=int):
        v = b["per_level"][lvl]
        star = "$^{*}$" if (v["img_lo"] > 0 or v["img_hi"] < 0) else ""
        rows.append((f"{lvl}\\%", f3(v["img_cosine"]), f3(v["img_descent"]),
                     rf"${sig(v['img_delta'])}$ \ci{{{v['img_lo']:+.3f}}}{{{v['img_hi']:+.3f}}}{star}",
                     f3(v["cls_cosine"]), f3(v["cls_descent"]), rf"${sig(v['cls_delta'])}$"))
    si, sc = b["same_image"], b["same_class"]
    star = "$^{*}$" if (si["lo"] > 0 or si["hi"] < 0) else ""
    pooled = (r"\textbf{pooled}", rf"\textbf{{{f3(si['cosine'])}}}",
              rf"\textbf{{{f3(si['descent'])}}}",
              rf"$\mathbf{{{sig(si['delta'])}}}$ \ci{{{si['lo']:+.3f}}}{{{si['hi']:+.3f}}}{star}",
              f3(sc["cosine"]), f3(sc["descent"]), rf"${sig(sc['delta'])}$")
    body = " \\\\\n".join(" & ".join(r) for r in rows)
    return ("% GENERATED by scripts/make_tables.py -- do not edit\n"
            "\\begin{tabular}{@{}lrrlrrr@{}}\n\\toprule\n"
            "& \\multicolumn{3}{c}{\\textbf{Same-image recall@1}} & "
            "\\multicolumn{3}{c}{\\textbf{Same-class accuracy}} \\\\\n"
            "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\n"
            "\\textbf{Occlusion} & Cosine & Descent & $\\Delta$ (95\\% CI) & "
            "Cosine & Descent & $\\Delta$ \\\\\n\\midrule\n"
            f"{body} \\\\\n\\midrule\n" + " & ".join(pooled) +
            " \\\\\n\\bottomrule\n\\end{tabular}\n")


def tab_multibeta(d):
    """Table 5: multi-beta scales, with compute."""
    order = sorted(d["rows"], key=lambda n: (d["rows"][n]["is_control"], -d["rows"][n]["R@1"]))
    con = d.get("contrasts", {})
    best_single = [n for n in d["rows"] if n.startswith("single beta")]
    sgl = best_single[0] if best_single else None
    rows = []
    for name in order:
        v = d["rows"][name]
        label = name.replace("INTEGRATED mix", "integrated mixture").replace("CONTROL ", "")
        if v["is_control"]:
            label = rf"\textit{{control}} {label}"
        delta = "---"
        for k, c in con.items():
            if k.startswith(name + " - ") and sgl and k.endswith(sgl):
                delta = rf"${c['delta']:+.4f}$ \ci{{{c['lo']:+.4f}}}{{{c['hi']:+.4f}}}"
        if v["is_control"]:
            delta = "collapse (predicted)"
        rows.append((label, f3(v["R@1"]), f3(v["R@5"]), f3(v["MRR"]),
                     str(v["matmuls"] or "---"), delta))
    body = " \\\\\n".join(" & ".join(r) for r in rows)
    return ("% GENERATED by scripts/make_tables.py -- do not edit\n"
            "\\footnotesize\\setlength{\\tabcolsep}{4pt}\n"
            "\\begin{tabular}{@{}lrrrrl@{}}\n\\toprule\n"
            "\\textbf{Condition} & \\textbf{R@1} & \\textbf{R@5} & \\textbf{MRR} & "
            "\\textbf{matmuls} & \\textbf{$\\Delta$ vs.\\ single $\\beta$} \\\\\n\\midrule\n"
            f"{body} \\\\\n\\bottomrule\n\\end{{tabular}}\n")


def tab_locomo(d):
    """Table: the anchored LoCoMo sweep. anchor=1.0 IS cosine (reduction check in caption)."""
    sw = d["sweep"]
    # one row per distinct condition, best pp0 per anchor, descending by coverage
    rows, seen = [], set()
    for name, v in sorted(sw.items(), key=lambda kv: -kv[1]["coverage@5"]):
        if name.startswith("energy fixed"):
            label = r"energy, fixed $\beta{=}32$"
        else:
            pp = name.split("pp0=")[1].split()[0]
            anc = name.split("anchor=")[1]
            if anc in seen:
                continue
            seen.add(anc)
            label = (rf"$\texttt{{anchor}}{{=}}{anc}$ \textit{{(is cosine)}}" if anc == "1.00"
                     else rf"$\texttt{{anchor}}{{=}}{anc}$, $\mathrm{{pp}}_0{{=}}{pp}$")
        bold = anc == "1.00" if not name.startswith("energy fixed") else False
        cov = f"{v['coverage@5']:.3f}"
        ci = rf"\ci{{{v['lo']:.3f}}}{{{v['hi']:.3f}}}"
        if bold:
            label, cov = rf"\textbf{{{label}}}", rf"\textbf{{{cov}}}"
        rows.append((label, cov, ci))
    body = " \\\\\n".join(" & ".join(r) for r in rows)
    return ("% GENERATED by scripts/make_tables.py -- do not edit\n"
            "\\begin{tabular}{@{}lrl@{}}\n\\toprule\n"
            "\\textbf{Condition} & \\textbf{Coverage@5} & \\textbf{95\\% CI} \\\\\n\\midrule\n"
            f"{body} \\\\\n\\bottomrule\n\\end{{tabular}}\n")


def _ci(v, lo="lo", hi="hi"):
    return rf"\ci{{{v[lo]:+.3f}}}{{{v[hi]:+.3f}}}"


def _star(v, lo="lo", hi="hi"):
    """A star marks an interval that excludes zero. Read off the CI, never asserted."""
    return "$^{*}$" if (v[lo] > 0) == (v[hi] > 0) and v[lo] != 0 else ""


# X202/X203/X204/X200: (file, row label, stratum key, indent?) in reporting order.
CORPORA_ROWS = [
    ("x10x_hotpot", r"\textsc{HotpotQA}", "bridge", "bridge (chain)", True),
    ("x10x_hotpot", "", "comparison", "comparison", True),
    ("x10x_hotpot", "", "all", r"\textit{all, unstratified}", True),
    ("x10x_2wiki", r"\textsc{2Wiki}", "inference", "inference (chain)", True),
    ("x10x_2wiki", "", "compositional", "compositional (chain)", True),
    ("x10x_2wiki", "", "comparison", "comparison", True),
    ("x10x_2wiki", "", "bridge_comparison (m=4)", r"bridge-comparison ($m{=}4$)", True),
    ("x10x_2wiki", "", "all", r"\textit{all, unstratified}", True),
    ("x10x_qasper", r"\textsc{QASPER}", "single-evidence", "single-evidence", True),
    ("x10x_qasper", "", "multi-evidence", "multi-evidence", True),
    ("x10x_scifact", r"\textsc{SciFact}", "all", "all", True),
    ("x10x_nfcorpus", r"\textsc{NFCorpus}", "multi-relevant", r"multi-relevant ($m{\approx}41$)", True),
    ("x10x_longmemeval", r"\textsc{LongMemEval}", "all", "all", True),
    ("x10x_longmemeval", "", "temporal-reasoning", "temporal-reasoning", True),
]


def tab_corpora(loaded):
    """Table: two-stage expansion across six corpora, stratified by question shape.

    Every cell is read from the corpus JSON. `folds_backed_off_to_null` is reported as a
    column because it is the load-bearing fact: where it is 10/10 the method has selected
    its own null and the delta is exactly zero by construction, not by measurement.
    """
    rows = []
    for fname, corpus, key, label, _ in CORPORA_ROWS:
        d = loaded[fname]
        S = d["strata"][key]
        prim = d["primary"]
        cos = S["cosine"]["metrics"][prim]
        mix = S["mixture"]["metrics"][prim]
        vc = S["mixture"]["vs_cosine"]
        rrf = S.get("rrf", {}).get("vs_cosine")
        bo = S["mixture"]["folds_backed_off_to_null"]
        n = S["cosine"]["n"]
        rows.append((
            corpus, label, f"{n:,}",
            f"{bo}/10" if bo is not None else "--",
            f3(cos), f3(mix),
            rf"${sig(vc['delta'])}$ {_ci(vc)}{_star(vc)}",
            rf"${sig(rrf['delta'])}$" if rrf else "--",
        ))
    body = " \\\\\n".join(" & ".join(r) for r in rows)
    return ("% GENERATED by scripts/make_tables.py -- do not edit\n"
            "\\footnotesize\\setlength{\\tabcolsep}{4pt}\n"
            "\\begin{tabular}{@{}llrrrrll@{}}\n\\toprule\n"
            "\\textbf{Corpus} & \\textbf{Question shape} & \\textbf{$n$} & "
            "\\textbf{Null} & \\textbf{Cosine} & \\textbf{Exp.} & "
            "\\textbf{Expansion $-$ cosine} & \\textbf{RRF} \\\\\n\\midrule\n"
            f"{body} \\\\\n\\bottomrule\n\\end{{tabular}}\n")


# The pre-registered cardinality interaction, one row per corpus that tested it.
CARD_ROWS = [
    ("x10x_2wiki", r"\textsc{2Wiki}", "bridge_comparison (m=4) - m=2 types",
     r"$m{=}4$ $-$ $m{=}2$"),
    ("x10x_qasper", r"\textsc{QASPER}", "multi-evidence - single-evidence",
     r"multi $-$ single"),
    ("x10x_scifact", r"\textsc{SciFact}", "multi-relevant - single-relevant",
     r"multi $-$ single"),
    ("x10x_nfcorpus", r"\textsc{NFCorpus}", "multi-relevant - single-relevant",
     r"multi $-$ single"),
    ("x10x_longmemeval", r"\textsc{LongMemEval}",
     "multi-session - single-session (all 3)", r"multi-sess.\ $-$ single"),
]


def tab_cardinality(loaded, dmos, dx12):
    """Table: every pre-registered test of 'more targets should help', with its sign."""
    rows = []
    for fname, corpus, key, contrast in CARD_ROWS:
        v = loaded[fname]["interactions"][key]
        sign = ("reversed" if v["hi"] < 0 else
                "as predicted" if v["lo"] > 0 else "null")
        rows.append((corpus, contrast, f"{v['n1']:,} / {v['n2']:,}" if "n1" in v else "--",
                     rf"${sig(v['dod'])}$ {_ci(v)}{_star(v)}", sign))
    # X212 cross-modal: the interaction lives at the top level, i2t (m=5) minus t2i (m=1).
    v = dx12["interaction_i2t_minus_t2i"]
    rows.append((r"\textsc{Flickr8k}", r"$m{=}5$ $-$ $m{=}1$",
                 f"{dx12['image_to_text']['n_queries']:,} / {dx12['text_to_image']['n_queries']:,}",
                 rf"${sig(v['dod'])}$ {_ci(v)}{_star(v)}",
                 "reversed" if v["hi"] < 0 else "null"))
    # X211 mosaics: constructed m, so the interaction is exactly zero -- the method
    # selected its own null in 10/10 folds at every m, which is why.
    vm = dmos["interactions"]["m=4 - m=1"]
    rows.append((r"\textsc{Imagenette} mosaics", r"$m{=}4$ $-$ $m{=}1$",
                 f"{dmos['n_queries']:,}", rf"${sig(vm['dod'])}$ {_ci(vm)}",
                 "null (self-nulled)"))
    body = " \\\\\n".join(" & ".join(r) for r in rows)
    return ("% GENERATED by scripts/make_tables.py -- do not edit\n"
            "\\footnotesize\\setlength{\\tabcolsep}{4pt}\n"
            "\\begin{tabular}{@{}llrll@{}}\n\\toprule\n"
            "\\textbf{Corpus} & \\textbf{Contrast} & \\textbf{$n$} & "
            "\\textbf{Interaction (95\\% CI)} & \\textbf{Verdict} \\\\\n\\midrule\n"
            f"{body} \\\\\n\\bottomrule\n\\end{{tabular}}\n")


def tab_modes(d25):
    out = ["% GENERATED by scripts/make_tables.py -- do not edit",
           "\\begin{tabular}{@{}lcccc@{}}", "\\toprule",
           "\\textbf{Diagnostic} & \\textbf{Value} & & \\textbf{Diagnostic} & \\textbf{Value} \\\\",
           "\\midrule"]
    x23, x23b, x24 = d25["X23"], d25["X23b"], d25["X24"]
    pairs = [
        ("NN cosine, real episode", f3(x23["real"]), "$|\\Delta_i|$ (one displacement)", f3(x23b["delta_norm"])),
        ("NN cosine, averaged prediction", f3(x23["averaged"]), "$|\\sum_i w_i \\Delta_i|$", f3(x23b["avg_step_norm"])),
        ("NN cosine, single mode", f3(x23["single_mode"]), "implied shrinkage", f"$\\times${x23b['shrinkage']:.2f}"),
        ("mutual cosine, top-3 modes", f3(x24["mutual_cosine"]), "raw mode lands on train ep.", f"{x23b['raw_mode_on_train']:.0%}"),
        ("cues with disagreeing modes", f"{x24['frac_multimodal']:.0%}", "average lands on train ep.", f"{x23b['avg_on_train']:.0%}"),
    ]
    out += [" & ".join(p) + " \\\\" for p in pairs]
    out += ["\\bottomrule", "\\end{tabular}", ""]
    return "\n".join(out)


def figures(d26, d25):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG.mkdir(parents=True, exist_ok=True)

    # Fig: one-hop vs two-hop, both corpora -- the claim and its bound
    fig, ax = plt.subplots(figsize=(6.2, 3.1))
    labels, deltas, los, his = [], [], [], []
    for tag, d, key in (("episodic\n1 hop", d25["X25_hop1"]["methods"]["averaged"], "ep1"),
                        ("episodic\n2 hops", d25["X25_hop2"]["methods"]["averaged"], "ep2"),
                        ("LoCoMo\n1 hop", d26["hop1"]["methods"]["averaged"], "lc1"),
                        ("LoCoMo\n2 hops", d26["hop2"]["methods"]["averaged"], "lc2")):
        labels.append(tag); deltas.append(d["dR1"]); los.append(d["lo"]); his.append(d["hi"])
    y = range(len(labels))
    ax.axvline(0, color="0.5", lw=1)
    for i, (dd, lo, hi) in enumerate(zip(deltas, los, his)):
        col = "#0d6b7d" if lo > 0 else "#9e2f28" if hi < 0 else "0.45"
        ax.plot([lo, hi], [i, i], color=col, lw=2.2)
        ax.plot([dd], [i], "o", color=col, ms=6)
    ax.set_yticks(list(y)); ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(r"paired $\Delta$ Recall@1 vs. cosine (95% cluster-bootstrap CI)", fontsize=9)
    ax.set_title("Association helps one step ahead and does not compose", fontsize=10)
    ax.tick_params(labelsize=8); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(FIG / "assoc_hops.pdf"); plt.close(fig)
    out = ["assoc_hops.pdf"]

    # Fig: the beta leverage band -- accuracy AND decision agreement on one axis pair.
    # Accuracy alone cannot distinguish "working" from "dissolved into the baseline";
    # the agreement curve is what makes Corollary 1 visible.
    band = RES / "beta_band.json"
    if band.exists():
        d = json.loads(band.read_text())
        betas = [r["beta"] for r in d["betas"]]
        acc = [r["pooled"] for r in d["betas"]]
        agr = [r["top1_agreement_with_cosine"] for r in d["betas"]]
        fig, ax = plt.subplots(figsize=(6.8, 3.3))
        ax.semilogx(betas, acc, color="#0d6b7d", lw=1.8, marker="o", ms=3,
                    label="Recall@1 (descent)")
        ax.axhline(d["cosine"]["pooled"], color="#9e2f28", ls="--", lw=1.2,
                   label=f"cosine ({d['cosine']['pooled']:.3f})")
        ax.set_xlabel(r"$\beta$ (log scale)"); ax.set_ylabel("Recall@1")
        ax.set_ylim(0, 0.62)
        ax2 = ax.twinx()
        ax2.semilogx(betas, agr, color="0.45", lw=1.4, ls=":", marker="s", ms=2.5,
                     label="top-1 agreement with cosine")
        ax2.set_ylabel("top-1 agreement with cosine", fontsize=9)
        ax2.set_ylim(0, 1.02)
        b = d["best"]["beta"]
        ax.axvline(b, color="0.7", lw=0.9)
        ax.annotate(f"optimum $\\beta$={b:g}", xy=(b, 0.56), fontsize=8,
                    ha="center", color="0.3")
        h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=7.5, loc="center left")
        ax.spines[["top"]].set_visible(False); ax2.spines[["top"]].set_visible(False)
        fig.tight_layout(); fig.savefig(FIG / "beta_band.pdf"); plt.close(fig)
        out.append("beta_band.pdf")
    return out


def main():
    d17b = json.loads((RES / "x17b_single_split.json").read_text())
    d25 = json.loads((RES / "x23_mode_diagnostics.json").read_text())
    d26 = json.loads((RES / "x26_locomo_assoc.json").read_text())
    dlc = json.loads((RES / "locomo_anchored.json").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    written = {}
    skipped = []

    # X201 (MSC) adds a row to the two association tables when its file is present.
    p101 = RES / "x101_msc.json"
    d101 = json.loads(p101.read_text()) if p101.exists() else None
    if d101 is None:
        skipped.append(("tab_assoc.tex (MSC row)", "x101_msc.json",
                        "experiments/x101_msc.py"))

    for name, body in (("tab_assoc.tex", tab_assoc(d26, d25, d17b, d101)),
                       ("tab_twohop.tex", tab_twohop(d26, d25, d101)),
                       ("tab_modes.tex", tab_modes(d25)),
                       ("tab_locomo.tex", tab_locomo(dlc))):
        _validate(name, body)
        (OUT / name).write_text(body); written[name] = len(body)

    # X200/X202/X203/X204 (the six-corpus expansion study) and the cardinality
    # interaction. Both are all-or-nothing: a partial corpora table would misrepresent
    # the sweep, so a single missing file skips the whole table and says which.
    corpora_files = sorted({f for f, *_ in CORPORA_ROWS} | {f for f, *_ in CARD_ROWS})
    missing = [f for f in corpora_files if not (RES / f"{f}.json").exists()]
    extra = {n: RES / f"{n}.json" for n in ("x111_mosaic", "x112_crossmodal")}
    missing += [n for n, p in extra.items() if not p.exists()]
    if missing:
        skipped.append(("tab_corpora.tex, tab_cardinality.tex",
                        ", ".join(f"{m}.json" for m in missing),
                        "experiments/x10x_corpora.py, x11x_modalities.py"))
    else:
        loaded = {f: json.loads((RES / f"{f}.json").read_text()) for f in corpora_files}
        dmos = json.loads(extra["x111_mosaic"].read_text())
        dx12 = json.loads(extra["x112_crossmodal"].read_text())
        for name, body in (("tab_corpora.tex", tab_corpora(loaded)),
                           ("tab_cardinality.tex", tab_cardinality(loaded, dmos, dx12))):
            _validate(name, body)
            (OUT / name).write_text(body); written[name] = len(body)

    for fname, (jname, script) in OPTIONAL.items():
        src = RES / jname
        if not src.exists():
            skipped.append((fname, jname, script)); continue
        dj = json.loads(src.read_text())
        if fname == "tab_images.tex":
            x113 = RES / "x113_class_vote.json"
            if not x113.exists():
                skipped.append((fname, "x113_class_vote.json",
                                "experiments/x113_class_vote.py")); continue
            body = tab_images(dj, json.loads(x113.read_text()))
        elif fname == "tab_completion.tex":
            body = tab_completion(dj)
        else:
            body = tab_multibeta(dj)
        _validate(fname, body)
        (OUT / fname).write_text(body); written[fname] = len(body)

    figs = figures(d26, d25)

    print("generated:")
    for k in written:
        print(f"  paper/generated/{k}")
    for f in figs:
        print(f"  paper/figures/{f}")
    print("\nNOT generated -- still typed, must be fixed before submission:")
    for k, why in UNCOVERED.items():
        print(f"  {k:16} {why}")
    for fname, jname, script in skipped:
        print(f"  {fname:16} no {jname} yet -- run {script}")
    if not skipped:
        print("  (every results-backed table is generated)")


if __name__ == "__main__":
    main()
