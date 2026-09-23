import json

import pytest

from app.core.config import Settings


@pytest.fixture
def settings() -> Settings:
    # _env_file=None keeps a developer's local .env (and real API key) out of the test run.
    return Settings(_env_file=None, openai_api_key=None)


@pytest.fixture
def questions_bytes() -> bytes:
    return json.dumps(["Which cloud providers do you rely on?", "Is data encrypted at rest?"]).encode()


@pytest.fixture
def minimal_pdf_bytes() -> bytes:
    return b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF\n"


@pytest.fixture
def kb_json_bytes() -> bytes:
    rows = [
        {
            "id": "row-1",
            "question": "Where are your data centres located?",
            "answer": "US Central region on Google Cloud Platform.",
            "rationale": "All customer data is physically stored in the USA.",
            "confidence": "high",
        }
    ]
    return json.dumps(rows).encode()
