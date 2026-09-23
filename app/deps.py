from fastapi import Request

from app.core.config import Settings
from app.services.embeddings import LocalEmbeddings
from app.services.faithfulness import FaithfulnessJudge
from app.services.indexing import IndexService
from app.services.llm import OpenAIStructuredLLM
from app.services.pipeline import QAPipeline
from app.services.qa_chain import AnswerSynthesizer
from app.services.query_classifier import QueryClassifier
from app.services.reranker import CrossEncoderReranker
from app.services.retrieval import HybridRetriever
from app.services.semantic_cache import SemanticAnswerCache


def build_pipeline(settings: Settings) -> QAPipeline:
    llm = OpenAIStructuredLLM(settings)
    embeddings = LocalEmbeddings(settings.embedding_model, settings.embedding_query_prefix)
    return QAPipeline(
        settings=settings,
        index_service=IndexService(embeddings, settings),
        embeddings=embeddings,
        retriever=HybridRetriever(CrossEncoderReranker(settings.reranker_model), settings),
        classifier=QueryClassifier(llm),
        synthesizer=AnswerSynthesizer(llm),
        judge=FaithfulnessJudge(llm),
        answer_cache=build_answer_cache(settings),
    )


def build_answer_cache(settings: Settings) -> SemanticAnswerCache:
    return SemanticAnswerCache(
        similarity_threshold=settings.answer_cache_similarity,
        max_entries_per_document=settings.answer_cache_entries_per_document,
        max_documents=settings.index_cache_entries,
        ttl_seconds=settings.answer_cache_ttl_s,
    )


async def get_pipeline(request: Request) -> QAPipeline:
    """One pipeline per process (built at startup): models load once and caches are shared across requests."""
    return request.app.state.pipeline
