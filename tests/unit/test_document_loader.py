import json
from pathlib import Path

import pytest

from app.core.config import Settings
from app.core.exceptions import DocumentParseError, DocumentTooLongError, EmptyDocumentError
from app.models import DocumentType
from app.services.document_loader import LoadedDocument, load_document

FIXTURES = Path(__file__).parent.parent / "fixtures"


@pytest.fixture(scope="module")
def nave_pdf() -> LoadedDocument:
    return load_document((FIXTURES / "nave_soc2.pdf").read_bytes(), DocumentType.PDF, Settings(_env_file=None))


class TestRealSoc2Pdf:
    def test_skips_toc_and_divider_pages_but_keeps_cover(self, nave_pdf):
        pages = {segment.page for segment in nave_pdf.segments}
        assert {3, 4, 5, 8, 13}.isdisjoint(pages)  # two ToC pages + "SECTION N" divider slides
        assert 1 in pages  # cover carries the audit period, which is answerable content
        assert nave_pdf.unit_count == 84

    def test_running_header_removed(self, nave_pdf):
        assert not any(s.text.startswith("A Type 2 Independent Service") for s in nave_pdf.segments)

    def test_third_party_table_is_labeled_with_its_section(self, nave_pdf):
        segment = next(s for s in nave_pdf.segments if "Third-Party Summary" in s.text)
        assert segment.page == 17
        assert segment.section == "3.4 Third Party Access:"
        assert "GCP Cloud hosting provider" in segment.text

    def test_early_sections_are_labeled_on_their_real_pages(self, nave_pdf):
        first_section_page = {}
        for segment in nave_pdf.segments:
            first_section_page.setdefault(segment.section, segment.page)
        assert first_section_page["Management’s Assertion"] == 6
        assert first_section_page["Inherent Limitations"] == 10
        assert first_section_page["Opinion"] == 11

    def test_section_changes_mid_page(self, nave_pdf):
        sections_on_14 = [s.section for s in nave_pdf.segments if s.page == 14]
        assert sections_on_14[0].startswith("DC 1")
        assert sections_on_14[-1].startswith("DC 2")


class TestPdfErrors:
    def test_corrupt_pdf(self, settings):
        with pytest.raises(DocumentParseError):
            load_document(b"%PDF-1.4 garbage that is not a pdf", DocumentType.PDF, settings)

    def test_page_limit(self):
        settings = Settings(_env_file=None, max_pdf_pages=10)
        with pytest.raises(DocumentTooLongError, match="84 pages"):
            load_document((FIXTURES / "nave_soc2.pdf").read_bytes(), DocumentType.PDF, settings)

    def test_pdf_without_text(self, settings):
        from pypdf import PdfWriter
        import io

        writer = PdfWriter()
        writer.add_blank_page(width=612, height=792)
        buffer = io.BytesIO()
        writer.write(buffer)
        with pytest.raises(EmptyDocumentError, match="OCR"):
            load_document(buffer.getvalue(), DocumentType.PDF, settings)


class TestJsonKnowledgeBase:
    def test_records_use_row_id_and_keep_all_fields(self, settings):
        document = load_document((FIXTURES / "nave_kb.json").read_bytes(), DocumentType.JSON, settings)
        first = document.segments[0]
        assert first.source_id == "53148c6413ac11f0a90176c5cf2df9da"
        assert first.page is None
        assert first.text.startswith("question: Where are your data centres located?")
        assert "answer: Our data centers are located in the US Central region" in first.text
        assert "confidence: high" in first.text
        assert "53148c6413ac11f0a90176c5cf2df9da" not in first.text

    @pytest.mark.parametrize(
        "payload",
        [
            {"question": {"0": "Q-a", "1": "Q-b"}, "answer": {"0": "A-a", "1": "A-b"}},  # pandas columns
            {"columns": ["question", "answer"], "data": [["Q-a", "A-a"], ["Q-b", "A-b"]]},  # pandas split
            {"items": [{"question": "Q-a", "answer": "A-a"}, {"question": "Q-b", "answer": "A-b"}]},
        ],
    )
    def test_common_tabular_shapes(self, settings, payload):
        document = load_document(json.dumps(payload).encode(), DocumentType.JSON, settings)
        assert [s.text for s in document.segments] == ["question: Q-a\nanswer: A-a", "question: Q-b\nanswer: A-b"]

    def test_index_orient_uses_keys_as_ids(self, settings):
        payload = {"r1": {"question": "Q", "answer": "A"}}
        document = load_document(json.dumps(payload).encode(), DocumentType.JSON, settings)
        assert document.segments[0].source_id == "r1"

    def test_page_shaped_json_keeps_page_numbers(self, settings):
        payload = [{"page": 3, "text": "Encryption at rest uses AES-256."}]
        document = load_document(json.dumps(payload).encode(), DocumentType.JSON, settings)
        assert document.segments[0].page == 3

    def test_list_of_strings(self, settings):
        document = load_document(b'["First fact.", "Second fact."]', DocumentType.JSON, settings)
        assert [s.source_id for s in document.segments] == ["0", "1"]

    @pytest.mark.parametrize(("raw", "error"), [(b"{bad", DocumentParseError), (b"42", DocumentParseError), (b"[]", EmptyDocumentError), (b'[{"id": "x"}]', EmptyDocumentError)])
    def test_invalid_json_documents(self, settings, raw, error):
        with pytest.raises(error):
            load_document(raw, DocumentType.JSON, settings)

    def test_row_limit(self):
        settings = Settings(_env_file=None, max_kb_rows=2)
        with pytest.raises(DocumentTooLongError):
            load_document(b'["a", "b", "c"]', DocumentType.JSON, settings)
