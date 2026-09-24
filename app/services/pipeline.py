import asyncio
import logging
import time
from dataclasses import dataclass, field, replace

import numpy as np

from app.core.config import Settings
from app.core.exceptions import UpstreamServiceError
from app.models import (
    NOT_FOUND_ANSWER,
    AnswerResult,
    CitationData,
    DocumentType,
    ItemResult,
    QuestionError,
    QuestionType,
)
from app.services.embeddings import LocalEmbeddings
from app.services.faithfulness import Claim, FaithfulnessJudge, VerifiedCitation, verify_citations
from app.services.indexing import DocumentIndex, IndexService
from app.services.llm import TokenUsage
from app.services.qa_chain import AnswerSynthesizer, Draft
from app.services.query_classifier import QueryClassifier, split_question_parts
from app.services.retrieval import HybridRetriever, RetrievedChunk
from app.services.semantic_cache import CacheHit, SemanticAnswerCache
logger = logging.getLogger(__name__)

_UNAVAILABLE_ANSWER = "Unable to answer this question right now: the language model is unavailable. Please retry."


@dataclass(slots=True)
class QuestionTrace:
    question_type: str = ""
    classifier_path: str = ""
    top_relevance: float | None = None
    contexts: int = 0
    faithfulness: str = "not_run"
    citations_dropped: int = 0
    outcome: str = ""
    answer_cache: CacheHit = CacheHit.MISS
    usage: TokenUsage = field(default_factory=TokenUsage)


@dataclass(slots=True)
class RequestStats:
    latency_ms: int
    questions: int
    unique_questions: int
    index_cache_hit: bool
    answer_cache_hits: int
    usage: TokenUsage


@dataclass(slots=True)
class PipelineResult:
    results: list[AnswerResult]
    stats: RequestStats


def _to_citations(verified: list[VerifiedCitation]) -> list[CitationData]:
    return [
        CitationData(excerpt=v.excerpt, page=v.chunk.page, source_id=v.chunk.source_id, section=v.chunk.section)
        for v in verified
    ]


class QAPipeline:
    def __init__(
        self,
        settings: Settings,
        index_service: IndexService,
        embeddings: LocalEmbeddings,
        retriever: HybridRetriever,
        classifier: QueryClassifier,
        synthesizer: AnswerSynthesizer,
        judge: FaithfulnessJudge,
        answer_cache: SemanticAnswerCache,
    ) -> None:
        self._settings = settings
        self._answer_cache = answer_cache
        self._index_service = index_service
        self._embeddings = embeddings
        self._retriever = retriever
        self._classifier = classifier
        self._synthesizer = synthesizer
        self._judge = judge

    def warm_up(self) -> None:
        """Load the local models eagerly so the first request doesn't pay for it."""
        self._embeddings.encode_queries(["warm up"])
        self._retriever.warm_up()

    async def answer(self, questions: list[str], document: bytes, doc_type: DocumentType) -> PipelineResult:
        started = time.perf_counter()
        self._synthesizer.check_ready()  # fail fast on missing config rather than after indexing
        index, index_cache_hit = await self._index_service.get_index(document, doc_type)

        unique = list(dict.fromkeys(questions))  # identical questions are answered once
        vectors = await asyncio.to_thread(self._embeddings.encode_queries, unique)
        semaphore = asyncio.Semaphore(self._settings.max_concurrency)

        async def bounded(question: str, vector: np.ndarray) -> tuple[AnswerResult, QuestionTrace]:
            async with semaphore:
                return await self._answer_one(index, question, vector)

        try:
            async with asyncio.TaskGroup() as group:
                tasks = [group.create_task(bounded(q, v)) for q, v in zip(unique, vectors, strict=True)]
        except ExceptionGroup as failures:
            # Systemic failures (bad API key, missing config) abort the request; surface the first one.
            raise failures.exceptions[0] from None

        answered = {question: task.result() for question, task in zip(unique, tasks, strict=True)}
        if all(result.error is not None for result, _ in answered.values()):
            raise UpstreamServiceError("The language model is unavailable; please retry shortly.")

        usage = TokenUsage()
        for _, trace in answered.values():
            usage.add(trace.usage)
        stats = RequestStats(
            latency_ms=round((time.perf_counter() - started) * 1000),
            questions=len(questions),
            unique_questions=len(unique),
            index_cache_hit=index_cache_hit,
            answer_cache_hits=sum(1 for _, trace in answered.values() if trace.answer_cache is not CacheHit.MISS),
            usage=usage,
        )
        return PipelineResult(results=[answered[q][0] for q in questions], stats=stats)

    async def _answer_one(
        self, index: DocumentIndex, question: str, vector: np.ndarray
    ) -> tuple[AnswerResult, QuestionTrace]:
        started = time.perf_counter()
        trace = QuestionTrace()
        cached, trace.answer_cache = self._answer_cache.lookup(index.doc_hash, question, vector)
        if cached is not None:
            trace.outcome = "cache_hit"
            # A semantic hit was answered for a differently worded question; report the one actually asked.
            result = replace(cached, question=question)
        else:
            try:
                result = await self._run(index, question, vector, trace)
                self._answer_cache.store(index.doc_hash, question, vector, result)
            except UpstreamServiceError as error:
                trace.outcome = "error"
                result = AnswerResult(
                    question=question,
                    answer=_UNAVAILABLE_ANSWER,
                    error=QuestionError(code=error.code, message=error.message),
                )
        logger.info(
            "question_answered",
            extra={
                "answer_cache": trace.answer_cache.value,
                "question_type": trace.question_type,
                "classifier_path": trace.classifier_path,
                "top_relevance": None if trace.top_relevance is None else round(trace.top_relevance, 5),
                "contexts": trace.contexts,
                "faithfulness": trace.faithfulness,
                "citations_dropped": trace.citations_dropped,
                "outcome": trace.outcome,
                "llm_calls": trace.usage.calls,
                "input_tokens": trace.usage.input_tokens,
                "output_tokens": trace.usage.output_tokens,
                "latency_ms": round((time.perf_counter() - started) * 1000),
            },
        )
        return result, trace

    async def _run(self, index: DocumentIndex, question: str, vector: np.ndarray, trace: QuestionTrace) -> AnswerResult:
        classification = await self._classifier.classify(question)
        trace.question_type = classification.question_type.value
        trace.classifier_path = classification.path.value
        trace.usage.add(classification.usage)

        hits = await self._retrieve(index, question, vector, classification.question_type, classification.items)
        trace.contexts = len(hits)
        trace.top_relevance = max((hit.relevance for hit in hits), default=0.0)
        if trace.top_relevance < self._settings.min_relevance:
            trace.outcome = "not_found_retrieval"
            return AnswerResult(question=question, answer=NOT_FOUND_ANSWER)

        contexts = [hit.chunk for hit in hits]
        draft = await self._synthesizer.synthesize(question, classification.question_type, classification.items, contexts)
        trace.usage.add(draft.usage)
        if classification.question_type is not QuestionType.CHECKLIST:
            return await self._ground_answer(question, draft, trace)

        result = await self._ground_checklist(question, draft, trace)
        if result.answer != NOT_FOUND_ANSWER:
            return result
        # No option applies. The document may still describe related practice ("none of these, but it monitors
        # uptime and error counts"), which a per-option answer has no place for. Ask again as an open question;
        # the answer passes the same two layers as any other.
        fallback = await self._synthesizer.synthesize(question, QuestionType.EXPLANATORY, [], contexts)
        trace.usage.add(fallback.usage)
        answer = await self._ground_answer(question, fallback, trace)
        if answer.answer == NOT_FOUND_ANSWER:
            return result
        trace.outcome = "answered_open_fallback"
        return replace(answer, items=result.items)

    async def _retrieve(
        self, index: DocumentIndex, question: str, vector: np.ndarray, question_type: QuestionType, items: list[str]
    ) -> list[RetrievedChunk]:
        if question_type is QuestionType.CHECKLIST:
            # Retrieve for each option too, so evidence for every option is in context.
            stem = question.splitlines()[0]
            return await self._retrieve_with_sub_queries(
                index, question, vector, self._settings.top_k_default,
                [f"{stem} {item}" for item in items], self._settings.top_k_checklist_item,
            )

        k = self._settings.top_k_explanatory if question_type is QuestionType.EXPLANATORY else self._settings.top_k_default
        parts = split_question_parts(question)
        if not parts:
            return await asyncio.to_thread(self._retriever.retrieve, index, question, vector, k)
        return await self._retrieve_with_sub_queries(
            index, question, vector, k, parts, self._settings.top_k_question_part
        )

    async def _retrieve_with_sub_queries(
        self,
        index: DocumentIndex,
        question: str,
        vector: np.ndarray,
        k: int,
        sub_queries: list[str],
        sub_k: int,
    ) -> list[RetrievedChunk]:
        """Retrieve for the whole question and for each sub-query, then interleave the hits."""
        sub_vectors = await asyncio.to_thread(self._embeddings.encode_queries, sub_queries)
        batches = [await asyncio.to_thread(self._retriever.retrieve, index, question, vector, k)]
        for query, sub_vector in zip(sub_queries, sub_vectors, strict=True):
            batches.append(await asyncio.to_thread(self._retriever.retrieve, index, query, sub_vector, sub_k))
        merged: dict[int, RetrievedChunk] = {}
        for rank in range(max(len(batch) for batch in batches)):  # interleave so every sub-query keeps its best hits
            for batch in batches:
                if rank < len(batch) and batch[rank].chunk.chunk_id not in merged:
                    merged[batch[rank].chunk.chunk_id] = batch[rank]
        return list(merged.values())[: self._settings.max_context_chunks]

    async def _ground_answer(self, question: str, draft: Draft, trace: QuestionTrace) -> AnswerResult:
        not_found = AnswerResult(question=question, answer=NOT_FOUND_ANSWER)
        if not draft.supported:
            trace.outcome, trace.faithfulness = "not_found_model", "not_applicable"
            return not_found

        verified = verify_citations(draft.citations)
        trace.citations_dropped = len(draft.citations) - len(verified)
        if not verified:
            trace.outcome, trace.faithfulness = "not_found_ungrounded", "failed_citation_check"
            return not_found

        verdicts, usage = await self._judge.judge(
            question, [Claim(claim_id="answer", text=draft.answer, evidence=[v.excerpt for v in verified])]
        )
        trace.usage.add(usage)
        verdict = verdicts.get("answer")
        if verdict is None or not verdict.faithful:
            trace.outcome, trace.faithfulness = "not_found_unfaithful", "failed_judge"
            logger.warning("faithfulness_rejected", extra={"reason": verdict.reason if verdict else "no verdict"})
            return not_found

        trace.outcome, trace.faithfulness = "answered", "passed"
        return AnswerResult(question=question, answer=draft.answer, citations=_to_citations(verified))

    async def _ground_checklist(self, question: str, draft: Draft, trace: QuestionTrace) -> AnswerResult:
        verified_by_item: dict[int, list[VerifiedCitation]] = {}
        for position, item in enumerate(draft.items):
            if not item.supported:
                continue
            verified = verify_citations(item.citations)
            trace.citations_dropped += len(item.citations) - len(verified)
            if verified:
                verified_by_item[position] = verified

        claims = [
            Claim(claim_id=f"item-{position}", text=f"{draft.items[position].item}: {draft.items[position].answer}",
                  evidence=[v.excerpt for v in verified])
            for position, verified in verified_by_item.items()
        ]
        verdicts, usage = await self._judge.judge(question, claims)
        trace.usage.add(usage)

        items: list[ItemResult] = []
        citations: list[CitationData] = []
        for position, item in enumerate(draft.items):
            verdict = verdicts.get(f"item-{position}")
            if position in verified_by_item and verdict is not None and verdict.faithful:
                item_citations = _to_citations(verified_by_item[position])
                items.append(ItemResult(item=item.item, answer=item.answer, supported=True, citations=item_citations))
                citations.extend(c for c in item_citations if c not in citations)
            else:
                items.append(ItemResult(item=item.item, answer=NOT_FOUND_ANSWER, supported=False))

        supported = [item for item in items if item.supported]
        trace.faithfulness = "passed" if len(supported) == len(verified_by_item) else "partially_rejected"
        if not supported:
            trace.outcome = "not_found_model" if not verified_by_item else "not_found_unfaithful"
            return AnswerResult(question=question, answer=NOT_FOUND_ANSWER, items=items)
        trace.outcome = "answered"
        answer = "\n".join(f"{item.item}: {item.answer}" for item in items)
        return AnswerResult(question=question, answer=answer, citations=citations, items=items)
