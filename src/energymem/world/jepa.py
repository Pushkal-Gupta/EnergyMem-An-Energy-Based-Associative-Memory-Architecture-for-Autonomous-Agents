"""A JEPA-style world model over the episode stream.

WHAT THIS IS. Every mechanism tested in Phases 0-2 varied only the *readout* -- how an
answer is picked once the landscape is fixed. This changes the landscape. It is the G3
axis of docs/01_RESEARCH_PLAN.md, finally instantiated.

THE JEPA FRAMING (LeCun). Do not predict the next episode's *text*; predict its
*representation*. Token-level prediction wastes capacity on surface form that carries no
retrieval signal. Embedding-space prediction learns only what distinguishes one episode
from another, which is exactly what retrieval needs.

    context  z_t          (frozen sentence encoder -- the target encoder is FROZEN, so
    predictor P_phi(z_t)   there is no representation collapse to defend against and no
    target   z_{t+1}       EMA schedule to tune)

WHY IT MATTERS HERE. An agent's episode stream is a trajectory through the world. A model
that predicts the next representation IS a world model over that trajectory, and rolling
it forward is inference in that model. It gives retrieval a mechanism cosine provably does
not have: association by CO-OCCURRENCE rather than by SIMILARITY. Measured on our corpus,
cosine answers "what happened next" at Recall@1 = 0.105 with median gold rank 24 (X16),
and its errors are lexical collisions across unrelated storylines. That gap is what this
is for.

POSITION VS PRIOR WORK. PAM (Dury 2026) also learns association beyond similarity, with a
JEPA-style predictor kept EXTERNAL to retrieval. The claim here is the integration: the
predictor becomes the routing operator inside the energy functional, so association
reshapes the same landscape that retrieval descends. See docs/07_CITATION_AUDIT.md.

CAPACITY. Deliberately small and low-rank. The corpus yields ~128 training pairs on the
dev half against 768 dimensions; a full-width MLP would memorise the dev threads and tell
us nothing about generalisation. The test that matters is transfer to threads the
predictor has never seen.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class JEPAConfig:
    rank: int = 32              # low-rank bottleneck; capacity control, not efficiency
    hidden: int = 256
    lr: float = 3e-4
    weight_decay: float = 1e-2
    epochs: int = 400
    temperature: float = 0.07   # InfoNCE
    dropout: float = 0.1
    patience: int = 40
    seed: int = 0


class JEPAPredictor:
    """Predicts the representation of the NEXT episode from the current one.

    P(z) = normalize(z + delta(z)) -- the model learns a *displacement* in embedding
    space rather than the target outright. With a frozen encoder the target already lives
    on the unit sphere, so predicting the offset is both easier and better conditioned.
    """

    def __init__(self, dim: int, cfg: JEPAConfig | None = None):
        self.dim = dim
        self.cfg = cfg or JEPAConfig()
        self.model = None
        self.history: dict[str, list[float]] = {"train": [], "val": []}

    # ------------------------------------------------------------------ build
    def _build(self):
        import torch
        import torch.nn as nn
        torch.manual_seed(self.cfg.seed)
        c = self.cfg
        return nn.Sequential(
            nn.Linear(self.dim, c.rank, bias=False),   # bottleneck
            nn.LayerNorm(c.rank),
            nn.Linear(c.rank, c.hidden),
            nn.GELU(),
            nn.Dropout(c.dropout),
            nn.Linear(c.hidden, self.dim),
        )

    # ------------------------------------------------------------------ train
    def fit(self, Z: np.ndarray, pairs: list[tuple[int, int]],
            val_pairs: list[tuple[int, int]] | None = None, verbose: bool = True):
        """Z: (N, d) frozen embeddings. pairs: (t, t+1) index pairs to learn from."""
        import torch
        import torch.nn.functional as F

        c = self.cfg
        self.model = self._build()
        opt = torch.optim.AdamW(self.model.parameters(), lr=c.lr, weight_decay=c.weight_decay)

        Zt = torch.tensor(Z, dtype=torch.float32)
        Zt = F.normalize(Zt, dim=1)
        src = torch.tensor([p[0] for p in pairs], dtype=torch.long)
        tgt = torch.tensor([p[1] for p in pairs], dtype=torch.long)
        if val_pairs:
            vsrc = torch.tensor([p[0] for p in val_pairs], dtype=torch.long)
            vtgt = torch.tensor([p[1] for p in val_pairs], dtype=torch.long)

        def infonce(s_idx, t_idx):
            # negatives = the whole corpus. Retrieval is against everything, so the
            # training objective should be too -- in-batch negatives would flatter it.
            pred = F.normalize(Zt[s_idx] + self.model(Zt[s_idx]), dim=1)
            logits = pred @ Zt.T / c.temperature
            logits.scatter_(1, s_idx.unsqueeze(1), -1e4)     # never predict the cue itself
            return F.cross_entropy(logits, t_idx)

        best, best_state, bad = float("inf"), None, 0
        for ep in range(c.epochs):
            self.model.train()
            opt.zero_grad()
            loss = infonce(src, tgt)
            loss.backward()
            opt.step()
            self.history["train"].append(float(loss.detach()))

            if val_pairs:
                self.model.eval()
                with torch.no_grad():
                    vl = float(infonce(vsrc, vtgt))
                self.history["val"].append(vl)
                if vl < best - 1e-4:
                    best, bad = vl, 0
                    best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                else:
                    bad += 1
                    if bad >= c.patience:
                        if verbose:
                            print(f"    early stop at epoch {ep}, best val {best:.4f}")
                        break
            if verbose and ep % 100 == 0:
                v = self.history["val"][-1] if val_pairs else float("nan")
                print(f"    epoch {ep:4d}  train {float(loss):.4f}  val {v:.4f}")

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    # ------------------------------------------------------------------ apply
    def predict(self, z: np.ndarray) -> np.ndarray:
        """One step forward in latent space. Accepts (d,) or (N, d)."""
        import torch
        import torch.nn.functional as F
        single = z.ndim == 1
        Z = torch.tensor(np.atleast_2d(z), dtype=torch.float32)
        Z = F.normalize(Z, dim=1)
        self.model.eval()
        with torch.no_grad():
            out = F.normalize(Z + self.model(Z), dim=1).numpy().astype(np.float64)
        return out[0] if single else out

    def rollout(self, z: np.ndarray, steps: int) -> list[np.ndarray]:
        """Roll the world model forward. This is where 'world model' stops being a
        metaphor: multi-hop chains are recovered by iterating the dynamics, not by
        widening a similarity search."""
        out, cur = [], z
        for _ in range(steps):
            cur = self.predict(cur)
            out.append(cur)
        return out
