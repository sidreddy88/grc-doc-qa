import json
import logging

from fastapi.testclient import TestClient

from app.core.logging import JsonFormatter, request_id_var
from app.deps import get_pipeline
from app.main import create_app


def _format(record_extra: dict, request_id: str | None = None) -> dict:
    record = logging.LogRecord("app.test", logging.INFO, __file__, 1, "question_answered", (), None)
    for key, value in record_extra.items():
        setattr(record, key, value)
    token = request_id_var.set(request_id)
    try:
        return json.loads(JsonFormatter().format(record))
    finally:
        request_id_var.reset(token)


def test_json_lines_include_extras_and_request_id():
    line = _format({"latency_ms": 12, "outcome": "answered"}, request_id="abc123")
    assert line["event"] == "question_answered"
    assert line["level"] == "INFO"
    assert (line["latency_ms"], line["outcome"], line["request_id"]) == (12, "answered", "abc123")
    assert "args" not in line and "msg" not in line


def test_request_id_generated_and_echoed(settings):
    client = TestClient(create_app(settings))
    response = client.get("/health")
    assert len(response.headers["x-request-id"]) == 32


def test_wellformed_inbound_request_id_is_honoured_and_bad_one_replaced(settings):
    client = TestClient(create_app(settings))
    assert client.get("/health", headers={"X-Request-ID": "trace-42"}).headers["x-request-id"] == "trace-42"
    assert client.get("/health", headers={"X-Request-ID": "bad id\n"}).headers["x-request-id"] != "bad id\n"


def test_error_bodies_carry_the_request_id(settings):
    app = create_app(settings)
    app.dependency_overrides[get_pipeline] = lambda: None
    client = TestClient(app)
    response = client.post("/qa", headers={"X-Request-ID": "req-7"})
    assert response.json()["error"]["request_id"] == "req-7"
