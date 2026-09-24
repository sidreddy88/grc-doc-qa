from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import Settings
from app.models import Chunk, DocumentType
from app.services.document_loader import LoadedDocument

_SEPARATORS = ["\n\n", "\n", ". ", "; ", ", ", " ", ""]


def chunk_document(document: LoadedDocument, settings: Settings) -> list[Chunk]:
    chunk_size = settings.chunk_size_chars if document.doc_type is DocumentType.PDF else settings.kb_row_max_chars
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=min(settings.chunk_overlap_chars, chunk_size // 2),
        separators=_SEPARATORS,
        keep_separator="end",
    )

    chunks: list[Chunk] = []

    def add(text: str, page: int | None, source_id: str | None, section: str | None, context: str | None = None):
        text = text.strip()
        if text:
            chunks.append(
                Chunk(
                    chunk_id=len(chunks),
                    text=text,
                    page=page,
                    source_id=source_id,
                    section=section,
                    index_context=context,
                )
            )

    for segment in document.segments:
        pieces = [segment.text] if len(segment.text) <= chunk_size else splitter.split_text(segment.text)
        for piece in pieces:
            add(piece, segment.page, segment.source_id, segment.section)
    # Control test matrix rows (extracted by the PDF loader) are one chunk per control, kept whole unless unusually
    # long: splitting mid-row is what extracting them avoids.
    for row in document.matrix_rows:
        pieces = [row.body] if len(row.body) <= 2 * chunk_size else splitter.split_text(row.body)
        for piece in pieces:
            add(piece, row.page, None, row.section, row.index_context)
    return chunks
