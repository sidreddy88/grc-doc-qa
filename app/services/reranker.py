import threading
from typing import Protocol

import numpy as np


class Reranker(Protocol):
    def score(self, query: str, passages: list[str]) -> list[float]:
        """Return a relevance score in [0, 1] for each passage."""
        ...


class CrossEncoderReranker:
    """Scores (query, passage) pairs jointly, which is far more precise than comparing separate embeddings."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None
        self._load_lock = threading.Lock()

    def _get_model(self):
        if self._model is None:
            with self._load_lock:
                if self._model is None:
                    from sentence_transformers import CrossEncoder

                    from app.services.model_utils import realign_parameters

                    model = CrossEncoder(self.model_name, device="cpu")
                    realign_parameters(model.model)
                    self._model = model
        return self._model

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        logits = self._get_model().predict([(query, passage) for passage in passages], convert_to_numpy=True)
        # ms-marco cross-encoders are trained with a binary objective, so the sigmoid is a usable relevance probability.
        return (1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=np.float64)))).tolist()
