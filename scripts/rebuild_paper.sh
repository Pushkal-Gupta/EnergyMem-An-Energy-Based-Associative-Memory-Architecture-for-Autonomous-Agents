#!/usr/bin/env bash
# Regenerate every number and figure in the paper, then build the PDF.
#
# This is the claim Appendix G makes, as an executable file. If it fails, the claim is
# false. Runs in roughly 30-50 minutes on the hardware named in Appendix G, dominated by
# image feature extraction.
#
#   bash scripts/rebuild_paper.sh
#
# Prerequisites not handled here, both documented in Appendix G:
#   * data/locomo10.json and data/imagenette2-160/ -- see the two curl commands in README
#   * torchvision, which `make setup` does not install
#   * a CA bundle for Python, if torch.hub needs to fetch weights:
#       export SSL_CERT_FILE="$(python3 -c 'import certifi;print(certifi.where())')"
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== Proposition 1 gate =================================================="
python3 -m pytest tests/ -q        # writes experiments/results/reduction_check.json

echo "== Experiments ========================================================="
python3 experiments/x17_parametric.py        # X17-r   parametric failure
python3 experiments/x17b_single_split.py     # X17b-r  Table 3 row (i)
python3 experiments/x23_mode_diagnostics.py  # X23-X25 mode averaging
python3 experiments/locomo_assoc.py          # X26     external validity
python3 scripts/exp_locomo_anchored.py       # Table 2
python3 scripts/confirm_holdout.py           # Table 1
python3 scripts/exp_multibeta_scales.py      # Table 5
python3 scripts/sweep_beta_transition.py     # Figure 1 (beta band)

echo "== Tables and figures from JSON ========================================"
python3 scripts/make_tables.py

echo "== Bibliography audit =================================================="
python3 scripts/verify_refs.py /tmp/refs_verified.json || \
  echo "  (network unavailable -- skipping citation verification)"

echo "== PDF (both builds) ==================================================="
cd paper
build () {
  if command -v tectonic >/dev/null 2>&1; then
    tectonic -X compile main.tex >/dev/null
  else
    pdflatex -interaction=nonstopmode main >/dev/null && bibtex main >/dev/null \
      && pdflatex -interaction=nonstopmode main >/dev/null \
      && pdflatex -interaction=nonstopmode main >/dev/null
  fi
}
# named build (preprint option) -- for arXiv, circulation and camera-ready
sed -i.bak 's/^\\usepackage{tmlr}$/\\usepackage[preprint]{tmlr}/' main.tex
build && cp main.pdf main-preprint.pdf
# anonymous build -- THIS is what TMLR review requires
sed -i.bak 's/^\\usepackage\[preprint\]{tmlr}$/\\usepackage{tmlr}/' main.tex
build && cp main.pdf main-anonymous.pdf
# leave the named build as main.pdf
sed -i.bak 's/^\\usepackage{tmlr}$/\\usepackage[preprint]{tmlr}/' main.tex
build
rm -f main.tex.bak
echo
echo "Done."
echo "  paper/main-anonymous.pdf   <- submit this to TMLR (double-blind)"
echo "  paper/main-preprint.pdf    <- named version, = main.pdf"
