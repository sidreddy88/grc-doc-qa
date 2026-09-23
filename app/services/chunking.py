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
    for segment in document.segments:
        pieces = [segment.text] if len(segment.text) <= chunk_size else splitter.split_text(segment.text)
        for piece in pieces:
            text = piece.strip()
            if text:
                chunks.append(
                    Chunk(
                        chunk_id=len(chunks),
                        text=text,
                        page=segment.page,
                        source_id=segment.source_id,
                        section=segment.section,
                    )
                )
    return chunks
