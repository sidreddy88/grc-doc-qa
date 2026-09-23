import asyncio
from typing import Annotated

from fastapi import APIRouter, Depends, File, Request, UploadFile

from app.api.schemas import (
    ChecklistItem,
    Citation,
    ErrorResponse,
    QAMeta,
    QAResponse,
    QAResult,
    QuestionError,
    Usage,
)
from app.api.uploads import read_upload_limited
from app.core.config import Settings, get_settings
from app.core.exceptions import ProcessingTimeoutError
from app.deps import get_pipeline
from app.models import AnswerResult, CitationData
from app.services.pipeline import QAPipeline
from app.services.validation import detect_document_type, parse_questions, validate_questions_filename

router = APIRouter()

_ERROR_RESPONSES = {
    status: {"model": ErrorResponse}
    for status in (413, 415, 422, 502, 503, 504)
}


def _to_citation(citation: CitationData) -> Citation:
    return Citation(
        page=citation.page, source_id=citation.source_id, section=citation.section, excerpt=citation.excerpt
    )


def _to_result(result: AnswerResult) -> QAResult:
    items = None
    if result.items is not None:
        items = [
            ChecklistItem(item=item.item, answer=item.answer, citations=[_to_citation(c) for c in item.citations])
            for item in result.items
        ]
    error = QuestionError(code=result.error.code, message=result.error.message) if result.error else None
    return QAResult(
        question=result.question,
        answer=result.answer,
        citations=[_to_citation(c) for c in result.citations],
        items=items,
        error=error,
    )


@router.post(
    "/qa",
    response_model=QAResponse,
    response_model_exclude_none=True,
    responses=_ERROR_RESPONSES,
    summary="Answer a list of questions from a PDF or JSON document",
)
async def answer_questions(
    request: Request,
    questions: Annotated[UploadFile, File(description="JSON array of question strings.")],
    document: Annotated[UploadFile, File(description="PDF report or JSON knowledge base to answer from.")],
    settings: Annotated[Settings, Depends(get_settings)],
    pipeline: Annotated[QAPipeline, Depends(get_pipeline)],
) -> QAResponse:
    validate_questions_filename(questions.filename)
    questions_bytes = await read_upload_limited(questions, settings.max_questions_bytes, field="questions")
    document_bytes = await read_upload_limited(document, settings.max_document_bytes, field="document")

    parsed_questions = parse_questions(questions_bytes, settings)
    document_type = detect_document_type(document.filename, document_bytes)

    try:
        async with asyncio.timeout(settings.request_timeout_s):
            output = await pipeline.answer(parsed_questions, document_bytes, document_type)
    except TimeoutError as error:
        raise ProcessingTimeoutError(
            f"Processing exceeded {settings.request_timeout_s:.0f}s. Try fewer questions or a smaller document."
        ) from error

    stats = output.stats
    meta = QAMeta(
        request_id=getattr(request.state, "request_id", None),
        latency_ms=stats.latency_ms,
        questions=stats.questions,
        unique_questions=stats.unique_questions,
        index_cache_hit=stats.index_cache_hit,
        answer_cache_hits=stats.answer_cache_hits,
        usage=Usage(
            llm_calls=stats.usage.calls,
            input_tokens=stats.usage.input_tokens,
            output_tokens=stats.usage.output_tokens,
        ),
    )
    return QAResponse(results=[_to_result(result) for result in output.results], meta=meta)
