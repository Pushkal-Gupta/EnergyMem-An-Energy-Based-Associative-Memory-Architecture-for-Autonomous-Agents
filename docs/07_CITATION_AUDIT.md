# Citation Audit

**Owner:** Rajit · **Deadline:** end of Week 1 · **Status:** partial (7 of ~35 verified)

The Review-1 bibliography was assembled from Gemini, Genspark and ChatGPT sessions. Every
entry must be resolved against the real arXiv/DOI page before it enters `paper/refs.bib`.

**Rule:** an entry is `VERIFIED` only when someone has opened the actual paper page and
confirmed title + authors + venue + year. Not the search result. The page.

---

## Verified so far (2026-08-11)

| Cited as | Reality | Status |
|---|---|---|
| arXiv:2602.11322 — Predictive Associative Memory, Dury 2026 | ✅ Exact match. Jason Dury. JEPA-style predictor, temporal co-occurrence, 97 % assoc. P@1, AUC 0.916 vs cosine | **VERIFIED — and it is major prior art, see below** |
| arXiv:2607.18439 — Learnable Sequential Memory in Coupled Oscillators, Guo & Pasqualetti 2026 | ✅ Exact match. Kuramoto phase-locked memory + routing layer | **VERIFIED** |
| arXiv:2605.12493 — LongMemEval-V2, Wu et al. 2026 | ✅ Exact match. 451 curated questions, five memory abilities, agent trajectories | **VERIFIED** |
| arXiv:2604.16839 — HeLa-Mem 2026 | ✅ Real. Zhu, Li, Zhang, Liu, Yang. **Accepted ACL 2026.** Hebbian dynamic graph, episodic + semantic distillation, LoCoMo | **VERIFIED — major prior art, see below** |
| arXiv:2506.01303 | ❌ **MIS-CITED.** Deck says *"Latent Structured Hopfield Network for Semantic Association and Retrieval"*. Actual paper is **"Dynamic Manifold Hopfield Networks for Context-Dependent Associative Memory"** — Li, Zeng, Xue, Feng | **CORRECTED — and also prior art** |
| github.com/EridosAI/PAM-Benchmark | ✅ Real, runnable, **MIT licensed**. 4-layer MLP predictor (2.36 M params), synthetic 20-room/50-object/50 k-state world, shuffle + similarity-matched-negative controls | **VERIFIED — usable as a baseline and eval harness** |
| AAAI-27 deadline (28 Jul 2026) | ✅ Confirmed passed | **VERIFIED** |

**Good news:** the hallucination rate is lower than feared — these are real papers.
**Bad news:** doing the audit properly revealed a mis-citation *and* three pieces of prior
art that sit much closer to EnergyMem than the Review-1 decks acknowledged. Read the next
section before writing a single line of the paper.

---

## ⚠️ Competitive landscape — this changes the positioning

Three papers occupy ground the decks assumed was empty:

### 1. PAM — Dury 2026 (arXiv:2602.11322)
Already does **retrieval beyond similarity via temporal co-occurrence**, which was our C3.
97 % association P@1 where cosine is at chance. Runnable MIT-licensed benchmark.

> **We cannot claim "association beyond similarity" as novel.** PAM owns it.

*What PAM does not do:* it is a **separate feed-forward predictor**, not an energy/attractor
formulation; no memory lifecycle (no decay, reinforcement or consolidation); evaluated only on
a synthetic 20-room world, not on real agent-memory benchmarks.

### 2. HeLa-Mem — Zhu et al., **ACL 2026** (arXiv:2604.16839)
Hebbian dynamic graph for LLM agents: episodic memory evolving through **co-activation**, plus
semantic memory via **Hebbian Distillation**. Evaluated on LoCoMo. That is close to our C2 + C4.

> **We cannot claim "co-activation-driven episodic memory + episodic→semantic distillation
> for LLM agents" as novel.** HeLa-Mem owns it, at a top-tier venue.

*What HeLa-Mem does not do:* it is a **graph** method — retrieval is graph traversal plus
similarity, not energy descent over a continuous landscape. No attractor dynamics, no
pattern completion from degraded cues, no energy-based unification.

### 3. DMHN — Li et al. (arXiv:2506.01303)
**Context-dependent modulation of attractor geometry** — contextual cues deform the energy
landscape. That is adjacent to our C2 (strength-modulated basins). 64 % retrieval at 2N
patterns vs 13 % for modern Hopfield.

*What DMHN does not do:* pattern-recall benchmarks only, not agent memory; context modulation
is not memory *strength* (usage/recency/surprise) and there is no lifecycle.

### The gap that actually survives

Each of the three isolates one mechanism in its own formalism. What none of them does — and
what is now the paper's real claim:

> **A single energy functional in which association topology, memory strength, and
> episodic→semantic consolidation are all terms in the same objective**, rather than three
> separate subsystems bolted together — evaluated on real agent-memory benchmarks with a
> matched encoder against PAM, HeLa-Mem and DMHN directly.

That is narrower and harder than the deck's version, but it is defensible, and it is the only
framing that survives contact with these three papers.

**Consequence for the plan:** `01_RESEARCH_PLAN.md` §2 and `04_EXPERIMENT_PROTOCOL.md` E3
must be updated — PAM is no longer a motivation, it is a **baseline to beat**. See
`docs/FUTURE_WORK.md` for what got displaced.

---

## Still to verify (Rajit — Week 1)

From `BEM_Review_1_FINAL_QA_CHECKED.pptx` slides 18–20, ~60 entries. Priority order:

**Tier A — load-bearing, verify first**
- [ ] arXiv:2008.02217 Ramsauer, Hopfield Networks is All You Need
- [ ] Krotov & Hopfield 2016, Dense Associative Memory, NeurIPS
- [ ] Hopfield 1982, PNAS 79(8):2554
- [ ] arXiv:2501.00663 Titans (check: NeurIPS 2025 proceedings version exists)
- [ ] Millidge et al. 2022, Universal Hopfield Networks, ICML
- [ ] arXiv:2412.05562 Expressive Power of Modern Hopfield Networks — **note the deck cites
      this under two different titles on slides 5 and 19; resolve which is correct**
- [ ] LongMemEval (ICLR 2025), LoCoMo (ACL 2024)
- [ ] HippoRAG (NeurIPS 2024), A-MEM, MemoryBank (AAAI 2024), MemGPT, MemOS

**Tier B — supporting**
- [ ] arXiv:2603.20955 Beyond Expression Similarity (Dury) — same author as PAM, check it is not double-cited
- [ ] arXiv:2606.04492, 2606.27783, 2601.15313, 2510.15828 (GENESIS), 2503.02303
- [ ] LeCun 2006 energy-based learning tutorial; LeCun 2022 JEPA path paper
- [ ] Friston 2016; Parr et al. 2022 — **likely cut entirely** now that planning is out of scope

**Tier C — cut candidates.** Blog posts, Medium articles, ResearchGate PDFs, Scholarpedia and
themoonlight.io entries from slides 18–20 are not citable in a Tier-1 paper. Delete unless
replaced by the primary source.

## Procedure

1. Open the real page (arxiv.org/abs/ID, or the DOI)
2. Confirm title, all authors, venue, year
3. If it has a peer-reviewed version, **cite that**, not the arXiv preprint
4. Paste the official BibTeX into `paper/refs.bib`
5. Tick the box here with the date

Any entry not verified by end of Week 1 is **deleted from the paper**, not left "to check later."
