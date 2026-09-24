from dataclasses import dataclass, field
from enum import StrEnum

NOT_FOUND_ANSWER = "Not found in document"


class DocumentType(StrEnum):
    PDF = "pdf"
    JSON = "json"


class QuestionType(StrEnum):
    BOOLEAN = "boolean"
    FACTUAL = "factual"
    EXPLANATORY = "explanatory"
    CHECKLIST = "checklist"


@dataclass(frozen=True, slots=True)
class Segment:
    """A run of document text with its location, before chunking."""

    text: str
    page: int | None = None
    source_id: str | None = None
    section: str | None = None


@dataclass(frozen=True, slots=True)
class Chunk:
    """A retrieval unit. PDF chunks carry a page; knowledge-base rows carry a source_id."""

    chunk_id: int
    text: str
    page: int | None = None
    source_id: str | None = None
    section: str | None = None
    index_context: str | None = None  # searchable context not shown to users, e.g. a control's criterion wording

    @property
    def index_text(self) -> str:
        # Section title (and any index context) is indexed with the chunk so retrieval sees context the raw
        # text lacks; citations and faithfulness checks still use the raw text.
        return "\n".join(part for part in (self.section, self.index_context, self.text) if part)


@dataclass(slots=True)
class CitationData:
    excerpt: str
    page: int | None = None
    source_id: str | None = None
    section: str | None = None


@dataclass(slots=True)
class ItemResult:
    item: str
    answer: str
    supported: bool
    citations: list[CitationData] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class QuestionError:
    code: str
    message: str


@dataclass(slots=True)
class AnswerResult:
    question: str
    answer: str
    citations: list[CitationData] = field(default_factory=list)
    items: list[ItemResult] | None = None
    error: QuestionError | None = None
