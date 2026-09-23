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


class QAResult(BaseModel):
    question: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    items: list[ChecklistItem] | None = Field(
        default=None, description="Per-item breakdown, present only for 'which of the following' questions."
    )


class QAResponse(BaseModel):
    results: list[QAResult]


class ErrorDetail(BaseModel):
    code: str
    message: str
    request_id: str | None = None
    details: list[dict] | None = None


class ErrorResponse(BaseModel):
    error: ErrorDetail
