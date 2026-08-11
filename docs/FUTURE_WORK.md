# Future Work — the parking lot

Everything here was in the BEM Review-1 deck and is **deliberately out of scope** for this
paper (`00_PROJECT_CHARTER.md` §2). It is parked, not discarded.

When someone says "since we already described it, let's just add…", the answer is: it goes
here, and it goes in the paper's one-paragraph Future Work section. Not in the experiments.

## Parked

- **Energy-landscape planner / active inference.** Expected-free-energy planning over the
  learned topology. This is the most exciting item and the most dangerous — it doubles the
  paper's surface area and halves the evidence per claim. Natural Project-II or a second paper.
- **Full memory taxonomy.** Explicit working / episodic / semantic / procedural systems with
  routing between them. HeLa-Mem (ACL 2026) already covers episodic→semantic; a contribution
  here needs to be sharper than "we have four boxes."
- **Multimodal perception + world embeddings.** The Environment→Perception→World-Embeddings
  front end from BEM slide 8.
- **Procedural / policy attractors** for reusable action sequences.
- **Sequential topology via coupled oscillators** (Guo & Pasqualetti 2026) — a principled way
  to store ordered transitions separately from patterns. Genuinely interesting follow-up to G3.
- **Safety of persistent episodic memory.** Real issue (arXiv:2501.11739). Deserves more than
  the broader-impact paragraph we will give it.

## Promoted out of the parking lot

- **Residual peeling** — was not in any deck; emerged from the Day-0 experiments (F4/F5).
  Now a first-class contribution in `method.tex` §3.3.
- **β stability characterization** — emerged as a diagnostic (F2). Now a results figure.

## Rule

An item leaves this file only at a gate review, only if the core claim is already fully
supported, and only by unanimous team agreement. Not at 2 a.m. three days before a deadline.
