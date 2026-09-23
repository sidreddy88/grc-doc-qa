import numpy as np

from app.models import AnswerResult, QuestionError
from app.services.semantic_cache import CacheHit, SemanticAnswerCache


def _vector(*values: float) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


def _cache(**overrides) -> SemanticAnswerCache:
    options = {"similarity_threshold": 0.97, "max_entries_per_document": 10, "max_documents": 2, "ttl_seconds": 60}
    return SemanticAnswerCache(**{**options, **overrides})


ANSWER = AnswerResult(question="Is data encrypted at rest?", answer="Yes, AES-256.")


def test_exact_hit_ignores_case_and_whitespace():
    cache = _cache()
    cache.store("doc", "Is data encrypted at rest?", _vector(1, 0), ANSWER)
    result, hit = cache.lookup("doc", "  is DATA encrypted   at rest?", _vector(0, 1))
    assert (result, hit) == (ANSWER, CacheHit.EXACT)


def test_semantic_hit_above_threshold():
    cache = _cache()
    cache.store("doc", "Is data encrypted at rest?", _vector(1, 0.1), ANSWER)
    result, hit = cache.lookup("doc", "Do you encrypt stored data?", _vector(1, 0.12))
    assert (result, hit) == (ANSWER, CacheHit.SEMANTIC)


def test_similar_but_different_question_misses():
    cache = _cache()
    cache.store("doc", "Which cloud providers do you rely on?", _vector(1, 0.3), ANSWER)
    result, hit = cache.lookup("doc", "Which cloud provider is primary?", _vector(1, 0.6))  # cosine ~0.96
    assert (result, hit) == (None, CacheHit.MISS)


def test_never_shared_across_documents():
    cache = _cache()
    cache.store("doc-a", "Is data encrypted at rest?", _vector(1, 0), ANSWER)
    assert cache.lookup("doc-b", "Is data encrypted at rest?", _vector(1, 0)) == (None, CacheHit.MISS)


def test_errors_are_never_cached():
    cache = _cache()
    failed = AnswerResult(question="Q?", answer="Unable", error=QuestionError(code="upstream_unavailable", message="x"))
    cache.store("doc", "Q?", _vector(1, 0), failed)
    assert cache.lookup("doc", "Q?", _vector(1, 0)) == (None, CacheHit.MISS)


def test_entries_expire():
    now = [0.0]
    cache = _cache(clock=lambda: now[0])
    cache.store("doc", "Q?", _vector(1, 0), ANSWER)
    now[0] = 61
    assert cache.lookup("doc", "Q?", _vector(1, 0)) == (None, CacheHit.MISS)


def test_bounded_per_document_and_across_documents():
    cache = _cache(max_entries_per_document=1, max_documents=1)
    cache.store("doc-a", "Q1?", _vector(1, 0), ANSWER)
    cache.store("doc-a", "Q2?", _vector(0, 1), ANSWER)
    assert cache.lookup("doc-a", "Q1?", _vector(1, 0))[1] is CacheHit.MISS
    cache.store("doc-b", "Q3?", _vector(1, 1), ANSWER)
    assert cache.lookup("doc-a", "Q2?", _vector(0, 1))[1] is CacheHit.MISS
