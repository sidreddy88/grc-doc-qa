from pydantic import BaseModel, Field


class Citation(BaseModel):
    page: int | None = Field(default=None, description="1-based PDF page number (PDF documents).")
    source_id: str | None = Field(default=None, description="Row id (JSON knowledge-base documents).")
    section: str | None = Field(default=None, description="Section title from the document's table of contents.")
    excerpt: str


class ChecklistItem(BaseModel):
    item: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)


class QuestionError(BaseModel):
    code: str
    message: str


class QAResult(BaseModel):
    question: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    items: list[ChecklistItem] | None = Field(
        default=None, description="Per-item breakdown, present only for 'which of the following' questions."
    )
    error: QuestionError | None = Field(
        default=None, description="Set when this question could not be processed (other questions still succeed)."
    )


class Usage(BaseModel):
    llm_calls: int
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float


class QAMeta(BaseModel):
    request_id: str | None = None
    latency_ms: int
    questions: int
    unique_questions: int
    index_cache_hit: bool
    answer_cache_hits: int
    usage: Usage


class QAResponse(BaseModel):
    results: list[QAResult]
    meta: QAMeta | None = None


class ErrorDetail(BaseModel):
    code: str
    message: str
    request_id: str | None = None
    details: list[dict] | None = None


class ErrorResponse(BaseModel):
    error: ErrorDetail
