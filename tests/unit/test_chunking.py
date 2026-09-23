from app.core.config import Settings
from app.models import DocumentType
from app.services.chunking import chunk_document
from app.services.document_loader import LoadedDocument, Segment


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, chunk_size_chars=200, chunk_overlap_chars=40, kb_row_max_chars=300, **overrides)


def test_short_segments_are_kept_whole_with_metadata():
    document = LoadedDocument(
        doc_type=DocumentType.PDF,
        segments=[Segment(text="Encryption at rest uses AES-256.", page=16, section="3.3 Security")],
        unit_count=1,
    )
    [chunk] = chunk_document(document, _settings())
    assert (chunk.chunk_id, chunk.page, chunk.section) == (0, 16, "3.3 Security")
    assert chunk.index_text == "3.3 Security\nEncryption at rest uses AES-256."


def test_long_segments_split_within_size_and_keep_page_and_section():
    sentence = "Access reviews are performed quarterly by the security team. "
    document = LoadedDocument(
        doc_type=DocumentType.PDF, segments=[Segment(text=sentence * 12, page=20, section="5.14")], unit_count=1
    )
    chunks = chunk_document(document, _settings())
    assert len(chunks) > 1
    assert all(len(c.text) <= 200 for c in chunks)
    assert all((c.page, c.section) == (20, "5.14") for c in chunks)
    assert [c.chunk_id for c in chunks] == list(range(len(chunks)))


def test_kb_rows_use_the_larger_row_limit():
    row = "question: Q\nanswer: " + "word " * 50  # ~260 chars: over PDF chunk size, under the KB row limit
    document = LoadedDocument(doc_type=DocumentType.JSON, segments=[Segment(text=row, source_id="r1")], unit_count=1)
    [chunk] = chunk_document(document, _settings())
    assert chunk.source_id == "r1"
    assert chunk.index_text == chunk.text.strip()
