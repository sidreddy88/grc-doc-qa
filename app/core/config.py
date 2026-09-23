from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openai_api_key: SecretStr | None = None
    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    max_document_bytes: int = Field(default=20 * 1024 * 1024, gt=0)
    max_questions_bytes: int = Field(default=256 * 1024, gt=0)
    max_questions: int = Field(default=50, gt=0)
    max_question_chars: int = Field(default=1000, gt=0)
    max_pdf_pages: int = Field(default=500, gt=0)
    max_kb_rows: int = Field(default=5000, gt=0)

    request_timeout_s: float = Field(default=120.0, gt=0)
    llm_timeout_s: float = Field(default=30.0, gt=0)
    llm_max_retries: int = Field(default=2, ge=0)
    max_concurrency: int = Field(default=5, gt=0)

    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
