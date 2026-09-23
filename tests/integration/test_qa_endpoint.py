"""End-to-end tests through the HTTP layer with the real pipeline; only the models and the LLM are faked."""

import asyncio
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.deps import get_pipeline
from app.main import create_app
from app.models import NOT_FOUND_ANSWER
from app.services.pipeline import QAPipeline
from app.services.llm import OpenAIStructuredLLM
from tests.fakes import ScriptedLLM, approve_all, grounded_synthesis, make_test_pipeline

FIXTURES = Path(__file__).parent.parent / "fixtures"


def _files(questions: bytes, document: bytes, questions_name="questions.json", document_name="report.pdf"):
    return {
        "questions": (questions_name, questions, "application/json"),
        "document": (document_name, document, "application/octet-stream"),
    }


def _pipeline(settings, llm) -> QAPipeline:
    return make_test_pipeline(settings, llm)


@pytest.fixture
def llm() -> ScriptedLLM:
    return ScriptedLLM({"_SynthesisOutput": [grounded_synthesis("Grounded answer.")], "_JudgeOutput": [approve_all]})


@pytest.fixture
def app(settings, llm):
    app = create_app(settings)
    pipeline = _pipeline(settings, llm)
    app.dependency_overrides[get_pipeline] = lambda: pipeline
    return app


@pytest.fixture
def client(app):
    return TestClient(app)


class TestAnswers:
    def test_health(self, client):
        assert client.get("/health").json() == {"status": "ok"}

    def test_real_soc2_pdf_end_to_end(self, client):
        questions = json.dumps(["Which cloud providers do you rely on?", "What is the capital of France?"]).encode()
        response = client.post("/qa", files=_files(questions, (FIXTURES / "nave_soc2.pdf").read_bytes()))
        assert response.status_code == 200
        body = response.json()
        cloud, france = body["results"]

        assert cloud["answer"] == "Grounded answer."
        citation = cloud["citations"][0]
        assert isinstance(citation["page"], int) and citation["excerpt"]
        assert "source_id" not in citation  # PDF citations use page numbers; None fields are omitted

        assert france["answer"] == NOT_FOUND_ANSWER and france["citations"] == []
        assert body["meta"]["questions"] == 2
        assert body["meta"]["usage"]["llm_calls"] == 2  # only the answerable question reached the LLM

    def test_json_knowledge_base_cites_row_ids(self, client):
        questions = json.dumps(["Where are your data centres located?"]).encode()
        response = client.post(
            "/qa", files=_files(questions, (FIXTURES / "nave_kb.json").read_bytes(), document_name="kb.json")
        )
        assert response.status_code == 200
        citation = response.json()["results"][0]["citations"][0]
        assert citation["source_id"] == "53148c6413ac11f0a90176c5cf2df9da"
        assert "page" not in citation

    def test_index_is_reused_across_requests(self, client):
        questions = json.dumps(["Where are your data centres located?"]).encode()
        document = (FIXTURES / "nave_kb.json").read_bytes()
        first = client.post("/qa", files=_files(questions, document, document_name="kb.json")).json()
        second = client.post("/qa", files=_files(questions, document, document_name="kb.json")).json()
        assert (first["meta"]["index_cache_hit"], second["meta"]["index_cache_hit"]) == (False, True)


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

    def test_corrupt_pdf_is_a_clear_422(self, client, questions_bytes, minimal_pdf_bytes):
        response = client.post("/qa", files=_files(questions_bytes, minimal_pdf_bytes))
        assert response.status_code == 422
        assert response.json()["error"]["code"] in {"document_parse_error", "empty_document"}

    def test_document_over_per_file_limit(self, settings, llm, questions_bytes):
        small = settings.model_copy(update={"max_document_bytes": 1024})
        app = create_app(small)
        app.dependency_overrides[get_pipeline] = lambda: _pipeline(small, llm)
        response = TestClient(app).post("/qa", files=_files(questions_bytes, b"%PDF-" + b"0" * 2048))
        assert response.status_code == 413
        assert response.json()["error"]["code"] in {"file_too_large", "request_too_large"}

    def test_request_body_over_total_limit_rejected_early(self, settings, questions_bytes):
        small = settings.model_copy(update={"max_document_bytes": 1024, "max_questions_bytes": 1024})
        response = TestClient(create_app(small)).post("/qa", files=_files(questions_bytes, b"%PDF-" + b"0" * 200_000))
        assert response.status_code == 413
        assert response.json()["error"]["code"] == "request_too_large"

    def test_pipeline_timeout_returns_504(self, settings, questions_bytes, minimal_pdf_bytes):
        class SlowPipeline:
            async def answer(self, questions, document, doc_type):
                await asyncio.sleep(5)

        app = create_app(settings.model_copy(update={"request_timeout_s": 0.05}))
        app.dependency_overrides[get_pipeline] = SlowPipeline
        response = TestClient(app).post("/qa", files=_files(questions_bytes, minimal_pdf_bytes))
        assert response.status_code == 504
        assert response.json()["error"]["code"] == "processing_timeout"

    def test_missing_api_key_fails_fast_with_503(self, settings):
        # The real LLM client, with no key configured: the request must fail before indexing starts.
        pipeline = _pipeline(settings, OpenAIStructuredLLM(settings))
        app = create_app(settings)
        app.dependency_overrides[get_pipeline] = lambda: pipeline
        questions = json.dumps(["Which cloud providers do you rely on?"]).encode()
        document = (FIXTURES / "nave_kb.json").read_bytes()
        response = TestClient(app).post("/qa", files=_files(questions, document, document_name="kb.json"))
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "service_misconfigured"
        assert len(pipeline._index_service._cache) == 0

    def test_unexpected_error_is_500_without_leaking_details(self, settings, questions_bytes, minimal_pdf_bytes):
        class BrokenPipeline:
            async def answer(self, questions, document, doc_type):
                raise RuntimeError("secret internal detail")

        app = create_app(settings)
        app.dependency_overrides[get_pipeline] = BrokenPipeline
        response = TestClient(app, raise_server_exceptions=False).post(
            "/qa", files=_files(questions_bytes, minimal_pdf_bytes)
        )
        assert response.status_code == 500
        assert "secret" not in response.text
        assert response.json()["error"]["code"] == "internal_error"


def test_ui_is_served_at_root_without_shadowing_the_api(client):
    page = client.get("/")
    assert page.status_code == 200 and "<form id=\"qa-form\">" in page.text
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/nope").json()["error"]["code"] == "not_found"
