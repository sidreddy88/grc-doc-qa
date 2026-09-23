import io
import json
import logging
from dataclasses import dataclass
from typing import Any

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.core.config import Settings
from app.core.exceptions import DocumentParseError, DocumentTooLongError, EmptyDocumentError
from app.models import DocumentType
from app.services import pdf_structure
from app.services.text import normalize_whitespace

logger = logging.getLogger(__name__)

# Pages with less text than this after removing headers/footers are cover pages,
# section dividers, or signature pages: nothing a question could be answered from.
_MIN_CONTENT_PAGE_CHARS = 100
_TOC_SEARCH_PAGES = 15
_ID_KEYS = ("id", "_id", "uuid", "question_id", "row_id")


@dataclass(frozen=True, slots=True)
class Segment:
    text: str
    page: int | None = None
    source_id: str | None = None
    section: str | None = None


@dataclass(frozen=True, slots=True)
class LoadedDocument:
    doc_type: DocumentType
    segments: list[Segment]
    unit_count: int


def load_document(data: bytes, doc_type: DocumentType, settings: Settings) -> LoadedDocument:
    """Parse a document into section-aware text segments. CPU-bound; call from a worker thread."""
    if doc_type is DocumentType.PDF:
        return _load_pdf(data, settings)
    return _load_json(data, settings)


def _open_pdf(data: bytes) -> PdfReader:
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted and not reader.decrypt(""):
            raise DocumentParseError("The PDF is password-protected.")
        return reader
    except (PdfReadError, ValueError, KeyError) as error:
        raise DocumentParseError("The PDF could not be read; it may be corrupt or truncated.") from error


def _load_pdf(data: bytes, settings: Settings) -> LoadedDocument:
    reader = _open_pdf(data)
    page_count = len(reader.pages)
    if page_count > settings.max_pdf_pages:
        raise DocumentTooLongError(f"The PDF has {page_count} pages; the limit is {settings.max_pdf_pages}.")

    try:
        raw_texts = [page.extract_text() or "" for page in reader.pages]
    except Exception as error:  # pypdf raises a wide range of errors on malformed content streams
        raise DocumentParseError("Text could not be extracted from the PDF.") from error

    toc_pages, toc_titles = _read_toc(reader, raw_texts)
    header = pdf_structure.find_running_header([normalize_whitespace(t) for t in raw_texts])

    content_pages: list[tuple[int, str]] = []
    for index, raw in enumerate(raw_texts):
        page_number = index + 1
        if page_number in toc_pages:
            continue
        text = pdf_structure.strip_page_furniture(raw, header, page_number)
        if len(text) >= _MIN_CONTENT_PAGE_CHARS:
            content_pages.append((page_number, text))

    if not content_pages:
        raise EmptyDocumentError(
            "No extractable text was found in the PDF. Scanned/image-only PDFs need OCR, which is not supported."
        )

    headings = pdf_structure.locate_headings(content_pages, toc_titles) if toc_titles else []
    segments = [
        Segment(text=text, page=page, section=section)
        for page, section, text in pdf_structure.split_at_headings(content_pages, headings)
    ]
    logger.info(
        "pdf_loaded",
        extra={
            "page_count": page_count,
            "content_pages": len(content_pages),
            "toc_titles": len(toc_titles),
            "headings_located": len(headings),
        },
    )
    return LoadedDocument(doc_type=DocumentType.PDF, segments=segments, unit_count=page_count)


def _read_toc(reader: PdfReader, raw_texts: list[str]) -> tuple[set[int], list[str]]:
    """Find ToC pages near the front of the document and parse their entry titles."""
    for index in range(min(_TOC_SEARCH_PAGES, len(raw_texts))):
        if not pdf_structure.is_toc_start(raw_texts[index]):
            continue
        toc_pages: set[int] = set()
        titles: list[str] = []
        for toc_index in range(index, min(index + 5, len(raw_texts))):
            layout = reader.pages[toc_index].extract_text(extraction_mode="layout") or ""
            if toc_index > index and not pdf_structure.looks_like_toc_page(layout):
                break
            toc_pages.add(toc_index + 1)
            titles.extend(pdf_structure.parse_toc_lines(layout))
        return toc_pages, titles
    return set(), []


def _load_json(data: bytes, settings: Settings) -> LoadedDocument:
    try:
        payload = json.loads(data.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DocumentParseError("The JSON document is not valid UTF-8 JSON.") from error

    rows = _rows_from_payload(payload)
    if len(rows) > settings.max_kb_rows:
        raise DocumentTooLongError(f"The JSON document has {len(rows)} records; the limit is {settings.max_kb_rows}.")

    segments = [segment for index, row in enumerate(rows) if (segment := _row_to_segment(index, row))]
    if not segments:
        raise EmptyDocumentError("The JSON document contains no text to answer from.")
    return LoadedDocument(doc_type=DocumentType.JSON, segments=segments, unit_count=len(rows))


def _rows_from_payload(payload: Any) -> list[dict[str, Any]]:
    """Normalize the common tabular JSON shapes (records, pandas column/index/split orients) into rows."""
    if isinstance(payload, list):
        return [item if isinstance(item, dict) else {"text": item} for item in payload]
    if not isinstance(payload, dict):
        raise DocumentParseError("The JSON document must be an object or an array.")

    if isinstance(payload.get("columns"), list) and isinstance(payload.get("data"), list):
        columns = payload["columns"]
        return [dict(zip(columns, values, strict=False)) for values in payload["data"] if isinstance(values, list)]

    values = list(payload.values())
    if values and all(isinstance(value, dict) for value in values):
        inner_keys = {key for value in values for key in value}
        if inner_keys and all(str(key).isdigit() for key in inner_keys):
            # pandas orient="columns": {"question": {"0": ..., "1": ...}, "answer": {...}}
            ordered = sorted(inner_keys, key=lambda key: int(key))
            return [{column: payload[column].get(row_key) for column in payload} for row_key in ordered]
        # orient="index": {"row-id": {"question": ..., "answer": ...}}
        return [{"id": key, **value} for key, value in payload.items()]

    list_values = [value for value in values if isinstance(value, list)]
    if len(list_values) == 1:
        return _rows_from_payload(list_values[0])
    return [{"id": key, "text": value} for key, value in payload.items()]


def _row_to_segment(index: int, row: dict[str, Any]) -> Segment | None:
    source_id = next((str(row[key]) for key in _ID_KEYS if row.get(key) not in (None, "")), str(index))
    page = row.get("page") if isinstance(row.get("page"), int) else None

    lines = []
    for key, value in row.items():
        if key in _ID_KEYS or key == "page" or value in (None, "", [], {}):
            continue
        rendered = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        lines.append(f"{key}: {normalize_whitespace(rendered)}")
    if not lines:
        return None
    return Segment(text="\n".join(lines), page=page, source_id=source_id)
