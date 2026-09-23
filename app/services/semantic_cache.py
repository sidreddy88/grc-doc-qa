import time
from collections import OrderedDict
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from app.models import AnswerResult
from app.services.text import fold_for_match, normalize_whitespace


class CacheHit(StrEnum):
    MISS = "miss"
    EXACT = "exact"
    SEMANTIC = "semantic"


@dataclass(slots=True)
class _Entry:
    question_key: str
    vector: np.ndarray
    result: AnswerResult
    expires_at: float


def normalize_question(question: str) -> str:
    return fold_for_match(normalize_whitespace(question))


class SemanticAnswerCache:
    """Caches verified answers per document, looked up by exact or near-identical question.

    Keys always include the document hash: the same question has different answers
    for different documents. The similarity threshold is deliberately high: a
    wrong cache hit would return a confident answer to a different question, which
    is worse than paying for a fresh one.
    """

    def __init__(
        self,
        similarity_threshold: float,
        max_entries_per_document: int,
        max_documents: int,
        ttl_seconds: float,
        clock=time.monotonic,
    ) -> None:
        self._threshold = similarity_threshold
        self._max_entries = max_entries_per_document
        self._max_documents = max_documents
        self._ttl = ttl_seconds
        self._clock = clock
        self._documents: OrderedDict[str, OrderedDict[str, _Entry]] = OrderedDict()

    def lookup(self, doc_hash: str, question: str, vector: np.ndarray) -> tuple[AnswerResult | None, CacheHit]:
        entries = self._documents.get(doc_hash)
        if not entries:
            return None, CacheHit.MISS
        self._evict_expired(entries)
        self._documents.move_to_end(doc_hash)

        key = normalize_question(question)
        exact = entries.get(key)
        if exact is not None:
            entries.move_to_end(key)
            return exact.result, CacheHit.EXACT

        best: _Entry | None = None
        best_score = -1.0
        for entry in entries.values():
            score = float(np.dot(vector, entry.vector))  # vectors are L2-normalized: dot product == cosine
            if score > best_score:
                best, best_score = entry, score
        if best is not None and best_score >= self._threshold:
            entries.move_to_end(best.question_key)
            return best.result, CacheHit.SEMANTIC
        return None, CacheHit.MISS

    def store(self, doc_hash: str, question: str, vector: np.ndarray, result: AnswerResult) -> None:
        if result.error is not None:
            return  # never cache transient failures
        entries = self._documents.setdefault(doc_hash, OrderedDict())
        self._documents.move_to_end(doc_hash)
        key = normalize_question(question)
        entries[key] = _Entry(key, np.asarray(vector, dtype=np.float32), result, self._clock() + self._ttl)
        entries.move_to_end(key)
        while len(entries) > self._max_entries:
            entries.popitem(last=False)
        while len(self._documents) > self._max_documents:
            self._documents.popitem(last=False)

    def _evict_expired(self, entries: OrderedDict[str, _Entry]) -> None:
        now = self._clock()
        for key in [key for key, entry in entries.items() if entry.expires_at <= now]:
            del entries[key]
