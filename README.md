# EnergyMem

**An Energy-Based Associative Memory Architecture for Autonomous Agents**

BCSE497J Project-I · Fall 2026–27 · SCOPE, VIT Vellore · Project ID 20113UG01
Pushkal Gupta (23BCT0253) · Rajit Agrawal (23BCE2371) · Anshveer Singh (23BCE0703)
Guide: Dr. Thangaramya K

---

## What this is

Agent long-term memory is almost universally retrieved by top-k embedding similarity.
Associative-memory theory offers an alternative — recall as convergence to an attractor in
a learned energy landscape — but it has never been evaluated as the *persistent memory
layer* of an agent under matched conditions.

We show that **dense top-k retrieval is a special case of the energy formulation**
(`T=1, β→∞, b=0, R=I`), then measure which departures from that corner actually pay for
themselves.

## Quick start

```bash
make setup     # venv + deps
make test      # sanity suite — Proposition 1 gates everything downstream
make demo      # Review-2 demo, offline, <60s
```

## Read these first, in order

| Doc | What it settles |
|---|---|
| [`docs/00_PROJECT_CHARTER.md`](docs/00_PROJECT_CHARTER.md) | Scope, the one-sentence claim, non-negotiables |
| [`docs/09_DAY0_FINDINGS.md`](docs/09_DAY0_FINDINGS.md) | **⚠️ Empirical results that change what you should build** |
| [`docs/01_RESEARCH_PLAN.md`](docs/01_RESEARCH_PLAN.md) | The science: RQs, hypotheses, novelty argument |
| [`docs/07_CITATION_AUDIT.md`](docs/07_CITATION_AUDIT.md) | Bibliography status + the 2026 competitive landscape |
| [`docs/02_TIMELINE.md`](docs/02_TIMELINE.md) | Week-by-week with pass/fail gates |
| [`docs/03_TASK_ALLOCATION.md`](docs/03_TASK_ALLOCATION.md) | Who owns what |
| [`docs/04_EXPERIMENT_PROTOCOL.md`](docs/04_EXPERIMENT_PROTOCOL.md) | Pre-registered experiments E1–E6 |
| [`docs/05_VENUE_STRATEGY.md`](docs/05_VENUE_STRATEGY.md) | Targets + deadlines |
| [`docs/06_RISK_REGISTER.md`](docs/06_RISK_REGISTER.md) | What kills this paper and how we avoid it |
| [`docs/08_REVIEW2_CHECKLIST.md`](docs/08_REVIEW2_CHECKLIST.md) | Panel deliverables |

## Layout

```
docs/          plan, protocol, risks, findings
src/energymem/
  base.py            Retriever protocol, Memory, RetrievalResult   [FROZEN]
  memory/energy.py   energy functional, CCCP descent, residual peeling
  memory/lifecycle.py strength bias, decay, consolidation
  baselines/         dense top-k (+ BM25, hybrid, reranker, PAM, HeLa-Mem — Week 1)
  eval/              metrics, bootstrap CIs, DegradeBench
experiments/   run_E1..E6.py; results/ committed as JSON
paper/         LaTeX skeleton, section stubs, verified refs.bib
scripts/       demo_review2.py
tests/         sanity suite — Proposition 1, energy monotonicity, capacity, determinism
```

## Current status (2026-08-11)

- ✅ Energy core, dense baseline, test suite — **5/5 passing**
- ✅ **Proposition 1 verified numerically** (1000/1000 queries)
- ✅ Residual peeling implemented (fixes the multi-target failure, see F4/F5)
- ⚠️ **Day-0 findings materially change the plan — read `docs/09_DAY0_FINDINGS.md`**
- ❌ Real sentence embeddings — Week 1, and now urgent (synthetic has no headroom)
- ❌ DegradeBench, learned routing, agent benchmarks — Weeks 2–3

## The rules

1. Matched encoder across every method. No exceptions.
2. Never report an accuracy win without its latency cost.
3. ≥5 seeds, bootstrap 95% CIs on every headline number.
4. No claim without an ablation.
5. Every citation verified against the actual paper.
6. Negative results ship. If it did not work, that goes in the abstract.
