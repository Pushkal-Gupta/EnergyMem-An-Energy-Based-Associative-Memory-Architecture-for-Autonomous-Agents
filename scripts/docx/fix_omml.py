"""Drop the m:nor that pandoc emits alongside m:scr inside m:rPr.

CT_MathRPr makes <m:nor/> and <m:scr/> a choice: at most one may appear. pandoc
writes both for \texttt-style math, which Word may report as unreadable content.
m:scr m:val="monospace" already renders upright monospace, so m:nor is redundant.
"""
import io, re, shutil, subprocess, sys, tempfile, pathlib, os

def fix(docx):
    d = pathlib.Path(tempfile.mkdtemp())
    subprocess.run(["unzip", "-q", docx, "-d", str(d)], check=True)
    for link in d.rglob("*"):
        if link.is_symlink():
            link.unlink()
    p = d / "word" / "document.xml"
    s = io.open(p, encoding="utf-8").read()
    before = s.count("<m:nor />")
    s = re.sub(r"<m:nor />(?=<m:scr )", "", s)
    after = s.count("<m:nor />")
    io.open(p, "w", encoding="utf-8").write(s)
    out = pathlib.Path(docx).resolve()
    out.unlink()
    subprocess.run(["zip", "-Xrq", str(out), "."], cwd=d, check=True)
    shutil.rmtree(d)
    print(f"{docx}: removed {before - after} redundant <m:nor/>")

for f in sys.argv[1:]:
    fix(f)
