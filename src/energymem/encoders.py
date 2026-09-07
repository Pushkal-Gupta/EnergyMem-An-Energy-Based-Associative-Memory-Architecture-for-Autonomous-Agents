"""The single encoder every retriever sees. THE MATCHED-ENCODER RULE LIVES HERE.

Rule 1 of docs/00_PROJECT_CHARTER.md: every method under comparison is handed the same
vectors from the same model. If two methods ever encode differently, the comparison
measures the encoders, not the algorithms, and the result is worthless.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np

_CACHE = Path(os.environ.get("ENERGYMEM_CACHE", ".cache/emb"))


class SentenceEncoder:
    """Frozen sentence encoder with an on-disk cache keyed by (model, text)."""

    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def dim(self) -> int:
        return int(self._load().get_sentence_embedding_dimension())

    def encode(self, texts, batch_size: int = 64) -> np.ndarray:
        single = isinstance(texts, str)
        items = [texts] if single else list(texts)
        key = hashlib.sha256(
            (self.model_name + "\x00" + "\x00".join(items)).encode()
        ).hexdigest()[:32]
        _CACHE.mkdir(parents=True, exist_ok=True)
        f = _CACHE / f"{key}.npy"
        if f.exists():
            arr = np.load(f)
        else:
            arr = self._load().encode(
                items, batch_size=batch_size, show_progress_bar=False,
                convert_to_numpy=True, normalize_embeddings=True,
            ).astype(np.float64)
            np.save(f, arr)
        return arr[0] if single else arr
