"""Deterministic stand-ins for the ML models so tests never download weights or call APIs."""

import hashlib

import numpy as np

from app.services.embeddings import LocalEmbeddings
from app.services.text import tokenize

_DIM = 256


class FakeEmbeddings(LocalEmbeddings):
    """Hashed bag-of-words vectors: texts sharing tokens get high cosine similarity."""

    def __init__(self) -> None:
        super().__init__(model_name="fake", query_prefix="")
        self.calls = 0

    def encode(self, texts: list[str]) -> np.ndarray:
        self.calls += 1
        vectors = np.zeros((len(texts), _DIM), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in tokenize(text):
                vectors[row, int(hashlib.md5(token.encode()).hexdigest(), 16) % _DIM] += 1.0
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.where(norms == 0, 1.0, norms)


class FakeReranker:
    """Relevance = fraction of query tokens present in the passage."""

    def score(self, query: str, passages: list[str]) -> list[float]:
        query_tokens = set(tokenize(query))
        if not query_tokens:
            return [0.0] * len(passages)
        return [len(query_tokens & set(tokenize(passage))) / len(query_tokens) for passage in passages]
