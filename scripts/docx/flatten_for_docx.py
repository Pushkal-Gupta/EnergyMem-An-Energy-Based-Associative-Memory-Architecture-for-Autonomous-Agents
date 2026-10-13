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

# Title and byline. pandoc's docx writer DOES emit the title and the author list
# as body paragraphs (Title and Author styles) when --standalone is given, so the
# earlier banner injected here produced every one of them twice. The banner is
# gone; the metadata file below is the single source of both.
TITLE_TEX = (r"Dense Retrieval Is Degenerate Associative Memory: "
             r"A Reduction and a Regime Taxonomy for Agent Long-Term Memory")

# Drop \title{...} and the tmlr \author{\name ...\addr ...} block: pandoc parses
# the latter without separating the names from the addresses, which is where
# "Pushkal GuptaSchool of Computer Science and Engineering" came from.
s = re.sub(r"\\title\{.*?\}\s*\n", "", s, count=1, flags=re.S)
s = re.sub(r"\\author\{.*?\n        \\addr School of Computer Science and Engineering\}",
           "", s, count=1, flags=re.S)

if variant == "anonymous":
    banner = "\n\\noindent\\textbf{Under review as a submission to TMLR.}\n\n"
else:
    banner = "\n"

# Run-in paragraph headings. \paragraph is a level-4 heading to pandoc, and with
# --number-sections the Word file grew 90 headings numbered "1.0.0.1". The PDF
# sets them run-in and bold, so emit that instead.
def runin(text):
    out, i = [], 0
    while True:
        j = text.find(r"\paragraph{", i)
        if j < 0:
            out.append(text[i:])
            return "".join(out)
        out.append(text[i:j])
        k, depth = j + len(r"\paragraph{"), 1
        while depth:
            if text[k] == "{":
                depth += 1
            elif text[k] == "}":
                depth -= 1
            k += 1
        out.append(r"\noindent\textbf{" + text[j + len(r"\paragraph{"):k - 1] + "}")
        i = k
s = runin(s)

# Caption numbers. pandoc does not number floats in docx, so the captions read as
# bare sentences while the body says "Table 8". Number them in source order,
# which is the order pandoc's own \ref resolution uses.
def number_captions(text):
    counts, env, out, i = {"table": 0, "figure": 0}, [], [], 0
    for m in re.finditer(r"\\begin\{(table|figure)\}|\\end\{(table|figure)\}|\\caption\{", text):
        out.append(text[i:m.start()]); i = m.start()
        tok = m.group(0)
        if tok.startswith(r"\begin"):
            env.append(m.group(1)); counts[m.group(1)] += 1
        elif tok.startswith(r"\end"):
            if env: env.pop()
        elif env:
            kind = env[-1]
            out.append(r"\caption{\textbf{%s %d.} " % (kind.capitalize(), counts[kind]))
            i = m.end()
    out.append(text[i:])
    return "".join(out)
s = number_captions(s)

# Equation numbers. pandoc does not number display equations either, so every
# \eqref printed its raw key: "in Equation [eq:update]". Number the equation
# environments in order, tag each one, and resolve the references by hand.
def number_equations(text):
    nums, n = {}, 0
    for m in re.finditer(r"\\begin\{equation\}(.*?)\\end\{equation\}", text, re.S):
        n += 1
        lab = re.search(r"\\label\{(eq:[^}]+)\}", m.group(1))
        if lab:
            nums[lab.group(1)] = n
    k = [0]
    def tag(m):
        k[0] += 1
        return (r"\begin{equation}" + m.group(1).rstrip() +
                r" \qquad (%d)" % k[0] + "\n\\end{equation}")
    text = re.sub(r"\\begin\{equation\}(.*?)\\end\{equation\}", tag, text, flags=re.S)
    for lab, i in nums.items():
        text = text.replace(r"\eqref{%s}" % lab, "(%d)" % i)
        text = text.replace(r"\ref{%s}" % lab, "%d" % i)
    return text
s = number_equations(s)

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
# \operatorname sets the name upright in LaTeX; pandoc's OMML writer italicises
# it, so "lse" and "softmax" read as products of variables. \mathrm survives.
s = s.replace(r"\operatorname{", r"\mathrm{")

# Thousands separators. The paper writes $5{,}847$, which pandoc turns into three
# math runs -- "5", ",", "847" -- and Word then sets a math comma with a space
# after it, so the Word file read "5, 847". Numbers that are only numbers become
# ordinary text, and a separator left inside real math becomes one upright run.
def _plain_number(m):
    body = m.group(1).replace("{", "").replace("}", "")
    return body if re.fullmatch(r"[\d.,]+", body) else m.group(0)
s = re.sub(r"\$([^$]{1,24})\$", _plain_number, s)
s = re.sub(r"(\d{1,3})\{,\}(\d{3})", r"\\mathrm{\1,\2}", s)


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
# One Author paragraph per list entry. The faculty member is listed as a
# contributor, not as an author, and appears on the named variant only: TMLR
# review is double-blind, so a supervising or contributing faculty name
# identifies the group as surely as an author name does.
BYLINE_NAMED = [
    "Pushkal Gupta  \u00b7  pushkal.gupta2023@vitstudent.ac.in",
    "Rajit Agrawal  \u00b7  rajit.agrawal2023@vitstudent.ac.in",
    "Anshveer Singh  \u00b7  anshveer.singh2023@vitstudent.ac.in",
    "School of Computer Science and Engineering",
    "Contributor: Dr. Thangaramya K, Associate Professor Grade 1",
    "Department of Information Security  \u00b7  thangaramya.k@vit.ac.in",
]
BYLINE_ANON = [
    "Anonymous authors",
    "Paper under double-blind review",
]
authors = BYLINE_ANON if variant == "anonymous" else BYLINE_NAMED
io.open(str(out) + ".meta", "w", encoding="utf-8").write(
    "---\ntitle: |\n  " + TITLE + "\nauthor:\n" +
    "".join('- "' + a + '"\n' for a in authors) +
    # citeproc drops the 35 references at the end with no heading above them.
    "reference-section-title: References\n" + "---\n")
io.open(out, "w", encoding="utf-8").write(s)
print(f"{variant}: {len(s.splitlines())} lines -> {out}")
