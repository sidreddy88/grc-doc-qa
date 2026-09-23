import json
from pathlib import PurePath

from app.core.config import Settings
from app.core.exceptions import (
    EmptyDocumentError,
    InvalidFileTypeError,
    InvalidQuestionsError,
    TooManyQuestionsError,
)
from app.models import DocumentType

_PDF_MAGIC = b"%PDF-"
_UTF8_BOM = b"\xef\xbb\xbf"
_DOCUMENT_EXTENSIONS = {".pdf": DocumentType.PDF, ".json": DocumentType.JSON}


def _extension(filename: str | None) -> str:
    return PurePath(filename or "").suffix.lower()


def sniff_document_type(data: bytes) -> DocumentType | None:
    # The PDF spec allows junk before the header, so look within the first KB rather than at offset 0.
    if _PDF_MAGIC in data[:1024]:
        return DocumentType.PDF
    head = data[:1024].removeprefix(_UTF8_BOM).lstrip()
    if head[:1] in (b"{", b"["):
        return DocumentType.JSON
    return None


def detect_document_type(filename: str | None, data: bytes) -> DocumentType:
    if not data.strip():
        raise EmptyDocumentError("The document file is empty.")

    extension = _extension(filename)
    if extension and extension not in _DOCUMENT_EXTENSIONS:
        raise InvalidFileTypeError(f"Unsupported document type '{extension}'. Upload a PDF or JSON file.")

    sniffed = sniff_document_type(data)
    if sniffed is None:
        raise InvalidFileTypeError("The document content is not a PDF or JSON file.")
    if extension and _DOCUMENT_EXTENSIONS[extension] is not sniffed:
        raise InvalidFileTypeError(
            f"The document has a '{extension}' extension but its content is {sniffed.value.upper()}."
        )
    return sniffed


def validate_questions_filename(filename: str | None) -> None:
    extension = _extension(filename)
    if extension and extension != ".json":
        raise InvalidFileTypeError(f"The questions file must be JSON, got '{extension}'.")


def parse_questions(data: bytes, settings: Settings) -> list[str]:
    if not data.strip():
        raise InvalidQuestionsError("The questions file is empty.")
    try:
        payload = json.loads(data.decode("utf-8-sig"))
    except UnicodeDecodeError as error:
        raise InvalidQuestionsError("The questions file must be UTF-8 encoded JSON.") from error
    except json.JSONDecodeError as error:
        raise InvalidQuestionsError(f"The questions file is not valid JSON: {error.msg} (line {error.lineno}).") from error

    if not isinstance(payload, list):
        raise InvalidQuestionsError(
            'The questions file must be a JSON array of strings, e.g. ["Which cloud providers do you rely on?"].'
        )
    if not payload:
        raise InvalidQuestionsError("The questions list is empty.")
    if len(payload) > settings.max_questions:
        raise TooManyQuestionsError(
            f"Received {len(payload)} questions; the limit is {settings.max_questions} per request."
        )

    questions: list[str] = []
    for index, item in enumerate(payload):
        if not isinstance(item, str) or not item.strip():
            raise InvalidQuestionsError(f"Question at index {index} must be a non-empty string.")
        question = item.strip()
        if len(question) > settings.max_question_chars:
            raise InvalidQuestionsError(
                f"Question at index {index} is {len(question)} characters; the limit is {settings.max_question_chars}."
            )
        questions.append(question)
    return questions
