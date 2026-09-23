from dataclasses import dataclass

import numpy as np

from app.core.config import Settings
from app.models import Chunk
from app.services.indexing import DocumentIndex
from app.services.reranker import Reranker
from app.services.text import tokenize


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    chunk: Chunk
    relevance: float  # cross-encoder probability in [0, 1]
    fused_score: float


def _min_max(scores: dict[int, float]) -> dict[int, float]:
    if not scores:
        return {}
    low, high = min(scores.values()), max(scores.values())
    if high == low:
        return {key: 1.0 for key in scores}
    return {key: (value - low) / (high - low) for key, value in scores.items()}


def fuse_scores(vector: dict[int, float], lexical: dict[int, float], vector_weight: float) -> dict[int, float]:
    """Weighted sum of per-list min-max normalized scores; a chunk missing from one list scores 0 there."""
    vector_norm, lexical_norm = _min_max(vector), _min_max(lexical)
    return {
        chunk_id: vector_weight * vector_norm.get(chunk_id, 0.0) + (1 - vector_weight) * lexical_norm.get(chunk_id, 0.0)
        for chunk_id in vector_norm.keys() | lexical_norm.keys()
    }


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = 60) -> list[int]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=scores.__getitem__, reverse=True)


class HybridRetriever:
    """Dense + BM25 recall, fused, then cross-encoder reranking. CPU-bound; call from a thread.

    Final order is RRF of the hybrid ranking and the cross-encoder ranking. The
    ms-marco cross-encoder is trained on short web queries and, on long multi-clause
    questionnaire items, sometimes buries chunks both retrievers ranked highly.
    Its absolute score is still the best "is anything relevant at all" signal.
    """

    def __init__(self, reranker: Reranker, settings: Settings) -> None:
        self._reranker = reranker
        self._settings = settings

    def warm_up(self) -> None:
        self._reranker.score("warm up", ["warm up"])

    def retrieve(self, index: DocumentIndex, query: str, query_vector: np.ndarray, k: int) -> list[RetrievedChunk]:
        if not index.chunks:
            return []
        candidates = min(self._settings.retrieval_candidates, len(index.chunks))

        vector_hits = index.vector_store.similarity_search_with_score_by_vector(query_vector.tolist(), k=candidates)
        vector_scores = {int(doc.metadata["chunk_id"]): float(score) for doc, score in vector_hits}

        lexical_all = index.bm25.get_scores(tokenize(query))
        top_lexical = np.argsort(lexical_all)[::-1][:candidates]
        lexical_scores = {int(i): float(lexical_all[i]) for i in top_lexical if lexical_all[i] > 0}

        fused = fuse_scores(vector_scores, lexical_scores, self._settings.fusion_vector_weight)
        shortlist = sorted(fused, key=fused.__getitem__, reverse=True)[: self._settings.rerank_candidates]

        scores = self._reranker.score(query, [index.chunks[chunk_id].index_text for chunk_id in shortlist])
        relevance = dict(zip(shortlist, scores, strict=True))
        reranked = sorted(shortlist, key=relevance.__getitem__, reverse=True)
        final = reciprocal_rank_fusion([shortlist, reranked])[:k]
        return [
            RetrievedChunk(chunk=index.chunks[chunk_id], relevance=relevance[chunk_id], fused_score=fused[chunk_id])
            for chunk_id in final
        ]
