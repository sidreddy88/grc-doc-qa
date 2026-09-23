import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from app.deps import get_pipeline
from app.main import create_app
from app.models import NOT_FOUND_ANSWER, AnswerResult


def _files(questions: bytes, document: bytes, questions_name="questions.json", document_name="report.pdf"):
    return {
        "questions": (questions_name, questions, "application/json"),
        "document": (document_name, document, "application/octet-stream"),
    }


@pytest.fixture
def client(settings):
    return TestClient(create_app(settings))


class TestContract:
    def test_health(self, client):
        assert client.get("/health").json() == {"status": "ok"}

    def test_returns_one_result_per_question_in_order(self, client, questions_bytes, minimal_pdf_bytes):
        response = client.post("/qa", files=_files(questions_bytes, minimal_pdf_bytes))
        assert response.status_code == 200
        body = response.json()
        assert [r["question"] for r in body["results"]] == json.loads(questions_bytes)
        assert all(r["answer"] == NOT_FOUND_ANSWER for r in body["results"])

    def test_accepts_json_document(self, client, questions_bytes, kb_json_bytes):
        response = client.post("/qa", files=_files(questions_bytes, kb_json_bytes, document_name="kb.json"))
        assert response.status_code == 200


class TestErrors:
    def test_missing_document_field(self, client, questions_bytes):
        response = client.post("/qa", files={"questions": ("q.json", questions_bytes, "application/json")})
        assert response.status_code == 422
        error = response.json()["error"]
        assert error["code"] == "invalid_request"
        assert any("document" in d["field"] for d in error["details"])

    def test_unsupported_document_type(self, client, questions_bytes):
        response = client.post("/qa", files=_files(questions_bytes, b"hello", document_name="notes.txt"))
        assert response.status_code == 415
        assert response.json()["error"]["code"] == "invalid_file_type"

    def test_malformed_questions(self, client, minimal_pdf_bytes):
        response = client.post("/qa", files=_files(b"{not json", minimal_pdf_bytes))
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "invalid_questions"

    def test_too_many_questions(self, client, settings, minimal_pdf_bytes):
        questions = json.dumps([f"Q{i}" for i in range(settings.max_questions + 1)]).encode()
        response = client.post("/qa", files=_files(questions, minimal_pdf_bytes))
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "too_many_questions"

    def test_document_over_per_file_limit(self, settings, questions_bytes):
        small = settings.model_copy(update={"max_document_bytes": 1024})
        client = TestClient(create_app(small))
        response = client.post("/qa", files=_files(questions_bytes, b"%PDF-" + b"0" * 2048))
        assert response.status_code == 413
        assert response.json()["error"]["code"] in {"file_too_large", "request_too_large"}

    def test_request_body_over_total_limit_rejected_early(self, settings, questions_bytes):
        small = settings.model_copy(update={"max_document_bytes": 1024, "max_questions_bytes": 1024})
        client = TestClient(create_app(small))
        response = client.post("/qa", files=_files(questions_bytes, b"%PDF-" + b"0" * 200_000))
        assert response.status_code == 413
        assert response.json()["error"]["code"] == "request_too_large"

    def test_pipeline_timeout_returns_504(self, settings, questions_bytes, minimal_pdf_bytes):
        class SlowPipeline:
            async def answer(self, questions, document, doc_type) -> list[AnswerResult]:
                await asyncio.sleep(5)
                return []

        app = create_app(settings.model_copy(update={"request_timeout_s": 0.05}))
        app.dependency_overrides[get_pipeline] = SlowPipeline
        response = TestClient(app).post("/qa", files=_files(questions_bytes, minimal_pdf_bytes))
        assert response.status_code == 504
        assert response.json()["error"]["code"] == "processing_timeout"

    def test_unexpected_error_is_500_without_leaking_details(self, settings, questions_bytes, minimal_pdf_bytes):
        class BrokenPipeline:
            async def answer(self, questions, document, doc_type) -> list[AnswerResult]:
                raise RuntimeError("secret internal detail")

        app = create_app(settings)
        app.dependency_overrides[get_pipeline] = BrokenPipeline
        response = TestClient(app, raise_server_exceptions=False).post(
            "/qa", files=_files(questions_bytes, minimal_pdf_bytes)
        )
        assert response.status_code == 500
        assert "secret" not in response.text
        assert response.json()["error"]["code"] == "internal_error"
