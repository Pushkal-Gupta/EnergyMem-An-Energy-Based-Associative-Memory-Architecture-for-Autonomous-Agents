#!/usr/bin/env bash
# Rebuild paper/main-anonymous.docx and paper/main-preprint.docx from paper/main.tex.
#
# Word cannot consume the LaTeX sources directly, so this flattens \input, renders
# the PDF figures to PNG (Word will not embed PDF images), rewrites the handful of
# math constructs pandoc cannot turn into OMML, converts with pandoc, and then
# repairs one schema violation pandoc leaves in its math markup.
#
# Needs: pandoc (pip install pypandoc-binary gives one), pdftoppm (poppler).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PY:-$ROOT/.venv/bin/python}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

mkdir -p "$WORK/figures"
for f in assoc_hops beta_band; do
  pdftoppm -png -r 200 -singlefile "$ROOT/paper/figures/$f.pdf" "$WORK/figures/$f"
done
cp "$ROOT/paper/refs.bib" "$WORK/"

for v in anonymous named; do
  "$PY" "$ROOT/scripts/docx/flatten_for_docx.py" "$v" "$WORK/paper-$v.tex"
  mv "$WORK/paper-$v.tex.meta" "$WORK/meta-$v.yaml"
  ( cd "$WORK" && "$PY" - <<PYEOF
import pypandoc
pypandoc.convert_file('paper-$v.tex', 'docx', outputfile='main-$v.docx',
    extra_args=['--citeproc', '--bibliography=refs.bib', '--resource-path=.',
                '--number-sections', '--wrap=preserve', '--standalone',
                '--metadata-file=meta-$v.yaml'])
PYEOF
  )
  "$PY" "$ROOT/scripts/docx/fix_omml.py" "$WORK/main-$v.docx"
  "$PY" "$ROOT/scripts/docx/polish_docx.py" "$WORK/main-$v.docx"
done

cp "$WORK/main-anonymous.docx" "$ROOT/paper/main-anonymous.docx"
cp "$WORK/main-named.docx"     "$ROOT/paper/main-preprint.docx"
# main.docx is the Word counterpart of paper/main.pdf, which is the named build.
cp "$WORK/main-named.docx"     "$ROOT/paper/main.docx"
echo "wrote paper/main-anonymous.docx, paper/main-preprint.docx and paper/main.docx"

# No legacy .doc target. The only converter present here is macOS textutil,
# whose reader ignores OMML, so every quantity typeset as math -- which is every
# effect size and interval in this paper -- is dropped silently, along with both
# figures. A faithful .doc needs LibreOffice: soffice --convert-to doc main.docx.
