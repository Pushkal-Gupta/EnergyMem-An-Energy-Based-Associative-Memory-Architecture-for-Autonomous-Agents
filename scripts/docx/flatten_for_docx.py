"""Flatten paper/main.tex into one standalone .tex per variant, for pandoc -> docx."""
import io, re, sys, pathlib

ROOT = pathlib.Path("/Users/pushkalgupta/Desktop/EnergyMem-An-Energy-Based-Associative-Memory-Architecture-for-Autonomous-Agents/paper")
variant = sys.argv[1]          # "anonymous" | "named"
out = pathlib.Path(sys.argv[2])

def expand(text, depth=0):
    assert depth < 6
    def sub(m):
        rel = m.group(1)
        p = ROOT / (rel if rel.endswith(".tex") else rel + ".tex")
        if not p.exists():
            return m.group(0)
        return expand(io.open(p, encoding="utf-8").read(), depth + 1)
    return re.sub(r"\\input\{([^}]*)\}", sub, text)

s = expand(io.open(ROOT / "main.tex", encoding="utf-8").read())

# The \if@accepted guard around the contributions section is TeX-internal; pandoc
# cannot evaluate it. Neither target variant includes contributions (preprint shows
# the byline but not contributions; anonymous shows neither), so drop the block.
s = re.sub(r"\\makeatletter\s*\\if@accepted.*?\\fi\s*\\makeatother", "", s, flags=re.S)

# tmlr.sty is not available to pandoc; keep only packages that affect semantics.
s = s.replace(r"\usepackage[preprint]{tmlr}", "")
s = s.replace(r"\usepackage{tmlr}", "")

# Figures: PDF images do not embed in .docx. Point at the PNG renders.
s = s.replace("figures/assoc_hops.pdf", "figures/assoc_hops.png")
s = s.replace("figures/beta_band.pdf", "figures/beta_band.png")

# Author block per variant.
author_re = re.compile(r"\\author\{.*?\n        \\addr School of Computer Science and Engineering\}", re.S)
TITLE_TEX = (r"Dense Retrieval Is Degenerate Associative Memory: "
             r"A Reduction and a Regime Taxonomy for Agent Long-Term Memory")
# pandoc records \title/\author only in docProps/core.xml, so the Word file opened
# with no visible title or byline. Emit them as real body content instead.
if variant == "anonymous":
    s = author_re.sub(r"\\author{Anonymous authors}", s)
    byline = (r"\textbf{Anonymous authors}\\" "\n"
              r"\emph{Paper under double-blind review.}")
    banner = ("\n\\noindent\\textbf{Under review as a submission to TMLR.}\n\n"
              "\\section*{" + TITLE_TEX + "}\n\n"
              "\\noindent " + byline + "\n\n")
else:
    byline = (r"Pushkal Gupta \quad Rajit Agrawal \quad Anshveer Singh\\" "\n"
              r"\emph{School of Computer Science and Engineering}")
    banner = ("\n\\section*{" + TITLE_TEX + "}\n\n"
              "\\noindent " + byline + "\n\n")

# --- math fidelity -------------------------------------------------------
# pandoc parses these but cannot emit OMML for them, and falls back to printing
# raw TeX into the Word file. Rewrite to forms its docx writer does handle.
s = s.replace(r"\tfrac12", r"\frac{1}{2}").replace(r"\tfrac1\beta", r"\frac{1}{\beta}")
s = re.sub(r"\\tfrac\{([^{}]*)\}\{([^{}]*)\}", r"\\frac{\1}{\2}", s)
s = s.replace(r"\!", "").replace(r"\,", " ").replace(r"\;", " ")
s = s.replace(r"\big(", "(").replace(r"\big)", ")")
s = s.replace(r"\Big[", "[").replace(r"\Big]", "]")
s = s.replace(r"\big[", "[").replace(r"\big]", "]")
# \| -> explicit Vert pairs (pandoc maps \| to \parallel, which it then drops)
def vert(m):
    return r"\lVert " + m.group(1) + r"\rVert "
s = re.sub(r"\\\|([^|]*?)\\\|", vert, s)
s = s.replace(r"\quad\quad", r"\quad")

# natbib -> pandoc citations; \bibliography is handled by --citeproc.
s = s.replace(r"\bibliographystyle{tmlr}", "")
s = s.replace(r"\bibliography{refs}", r"\bibliography{refs.bib}")

# pandoc's LaTeX reader treats `abstract` as METADATA and the docx writer then
# prints it above everything, so the Word file opened with "Abstract" before the
# title. Demote it to an ordinary unnumbered section to preserve document order.
s = s.replace(r"\begin{abstract}", r"\section*{Abstract}")
s = s.replace(r"\end{abstract}", "")

s = s.replace(r"\begin{document}", r"\begin{document}" + banner)

TITLE = ("Dense Retrieval Is Degenerate Associative Memory: "
         "A Reduction and a Regime Taxonomy for Agent Long-Term Memory")
AUTHORS = ("Pushkal Gupta, Rajit Agrawal, Anshveer Singh --- "
           "School of Computer Science and Engineering")
io.open(str(out) + ".meta", "w", encoding="utf-8").write(
    "---\ntitle: |\n  " + TITLE + "\nauthor:\n- " +
    ("Anonymous authors (paper under double-blind review)" if variant == "anonymous"
     else AUTHORS) + "\n---\n")
io.open(out, "w", encoding="utf-8").write(s)
print(f"{variant}: {len(s.splitlines())} lines -> {out}")
