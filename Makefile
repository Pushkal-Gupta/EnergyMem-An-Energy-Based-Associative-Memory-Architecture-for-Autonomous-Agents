.PHONY: help setup test demo lint paper clean all

PY := .venv/bin/python
PIP := .venv/bin/pip

help:
	@echo "setup   install deps into .venv"
	@echo "test    run the sanity suite (Proposition 1 gates everything)"
	@echo "demo    Review-2 demo, offline, <60s"
	@echo "lint    ruff"
	@echo "paper   build paper/main.pdf"
	@echo "all     test + every experiment + paper  (the reproducibility claim)"

setup:
	python3 -m venv .venv
	$(PIP) install -q -e ".[all]"

test:
	$(PY) -m pytest tests/ -q

demo:
	$(PY) scripts/demo_review2.py

lint:
	.venv/bin/ruff check src/ scripts/ tests/

# Experiments. Each writes experiments/results/<exp>/<timestamp>_<sha>.json.
# Tables and figures are GENERATED from those files -- never hand-typed.
E1 E2 E3 E4 E5 E6:
	$(PY) experiments/run_$@.py

experiments: E3 E1 E2 E4 E5 E6   # E3 first: it is the gating question (Gate G-B)

paper:
	cd paper && latexmk -pdf main.tex

all: test experiments paper

clean:
	rm -rf paper/*.aux paper/*.log paper/*.out paper/*.fls paper/*.fdb_latexmk
	find . -name __pycache__ -type d -exec rm -rf {} +
