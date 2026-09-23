import json

import pytest

from app.core.exceptions import (
    EmptyDocumentError,
    InvalidFileTypeError,
    InvalidQuestionsError,
    TooManyQuestionsError,
)
from app.models import DocumentType
from app.services.validation import (
    detect_document_type,
    parse_questions,
    sniff_document_type,
    validate_questions_filename,
)


class TestParseQuestions:
    def test_valid_list_is_stripped(self, settings):
        data = json.dumps(["  Which cloud providers?  ", "Is MFA enforced?"]).encode()
        assert parse_questions(data, settings) == ["Which cloud providers?", "Is MFA enforced?"]

    def test_accepts_utf8_bom(self, settings):
        data = b"\xef\xbb\xbf" + json.dumps(["Q1"]).encode()
        assert parse_questions(data, settings) == ["Q1"]

    @pytest.mark.parametrize(
        ("raw", "fragment"),
        [
            (b"", "empty"),
            (b"   ", "empty"),
            (b"not json", "not valid JSON"),
            (b'{"questions": ["Q1"]}', "JSON array"),
            (b"[]", "empty"),
            (b'["ok", 42]', "index 1"),
            (b'["ok", "   "]', "index 1"),
            (b"\xff\xfe", "UTF-8"),
        ],
    )
    def test_rejects_malformed_input(self, settings, raw, fragment):
        with pytest.raises(InvalidQuestionsError, match=fragment):
            parse_questions(raw, settings)

    def test_rejects_too_many_questions(self, settings):
        data = json.dumps([f"Q{i}" for i in range(settings.max_questions + 1)]).encode()
        with pytest.raises(TooManyQuestionsError, match=str(settings.max_questions)):
            parse_questions(data, settings)

    def test_accepts_exactly_the_limit(self, settings):
        data = json.dumps([f"Q{i}" for i in range(settings.max_questions)]).encode()
        assert len(parse_questions(data, settings)) == settings.max_questions

    def test_rejects_overlong_question(self, settings):
        data = json.dumps(["x" * (settings.max_question_chars + 1)]).encode()
        with pytest.raises(InvalidQuestionsError, match="characters"):
            parse_questions(data, settings)


class TestDocumentType:
    def test_sniffs_pdf(self, minimal_pdf_bytes):
        assert sniff_document_type(minimal_pdf_bytes) is DocumentType.PDF

    def test_sniffs_json_with_leading_whitespace_and_bom(self):
        assert sniff_document_type(b"\xef\xbb\xbf  \n[{}]") is DocumentType.JSON

    def test_unknown_content(self):
        assert sniff_document_type(b"PK\x03\x04 zip bytes") is None

    def test_detects_by_content_without_extension(self, minimal_pdf_bytes):
        assert detect_document_type(None, minimal_pdf_bytes) is DocumentType.PDF

    def test_rejects_unsupported_extension(self, minimal_pdf_bytes):
        with pytest.raises(InvalidFileTypeError, match=".docx"):
            detect_document_type("report.docx", minimal_pdf_bytes)

    def test_rejects_extension_content_mismatch(self, kb_json_bytes):
        with pytest.raises(InvalidFileTypeError, match="content is JSON"):
            detect_document_type("report.pdf", kb_json_bytes)

    def test_rejects_non_document_content(self):
        with pytest.raises(InvalidFileTypeError, match="not a PDF or JSON"):
            detect_document_type("report.pdf", b"hello world")

    def test_rejects_empty_document(self):
        with pytest.raises(EmptyDocumentError):
            detect_document_type("report.pdf", b"  ")


class TestQuestionsFilename:
    @pytest.mark.parametrize("name", ["questions.json", "Q.JSON", None, "questions"])
    def test_allows_json_or_missing_extension(self, name):
        validate_questions_filename(name)

    def test_rejects_other_extensions(self):
        with pytest.raises(InvalidFileTypeError):
            validate_questions_filename("questions.csv")
