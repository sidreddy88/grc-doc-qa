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
class Chunk:
    """A retrieval unit. PDF chunks carry a page; knowledge-base rows carry a source_id."""

    chunk_id: int
    text: str
    page: int | None = None
    source_id: str | None = None
    section: str | None = None

    @property
    def index_text(self) -> str:
        # Section title is indexed with the chunk so retrieval sees context the raw text lacks;
        # citations and faithfulness checks still use the raw text.
        return f"{self.section}\n{self.text}" if self.section else self.text


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
