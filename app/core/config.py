from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openai_api_key: SecretStr | None = None
    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_query_prefix: str = "Represent this sentence for searching relevant passages: "
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    max_document_bytes: int = Field(default=20 * 1024 * 1024, gt=0)
    max_questions_bytes: int = Field(default=256 * 1024, gt=0)
    max_questions: int = Field(default=50, gt=0)
    max_question_chars: int = Field(default=1000, gt=0)
    max_pdf_pages: int = Field(default=500, gt=0)
    max_kb_rows: int = Field(default=5000, gt=0)

    chunk_size_chars: int = Field(default=1000, gt=100)
    chunk_overlap_chars: int = Field(default=150, ge=0)
    # Knowledge-base rows are self-contained Q&A units, so they're only split when unusually long.
    kb_row_max_chars: int = Field(default=2500, gt=100)

    retrieval_candidates: int = Field(default=20, gt=0)
    fusion_vector_weight: float = Field(default=0.7, ge=0, le=1)
    rerank_candidates: int = Field(default=20, gt=0)
    top_k_default: int = Field(default=5, gt=0)
    top_k_explanatory: int = Field(default=8, gt=0)
    top_k_checklist_item: int = Field(default=3, gt=0)
    max_context_chunks: int = Field(default=12, gt=0)
    # Below this cross-encoder probability for the best chunk, skip the LLM and answer "Not found".
    # Calibrated on the sample SOC 2 report: unanswerable questions scored < 1e-4, answerable ones > 2e-3.
    min_relevance: float = Field(default=5e-4, ge=0, le=1)

    index_cache_entries: int = Field(default=16, gt=0)
    index_cache_ttl_s: float = Field(default=3600.0, gt=0)

    answer_cache_similarity: float = Field(default=0.97, gt=0, le=1)
    answer_cache_entries_per_document: int = Field(default=512, gt=0)
    answer_cache_ttl_s: float = Field(default=3600.0, gt=0)

    request_timeout_s: float = Field(default=120.0, gt=0)
    llm_timeout_s: float = Field(default=30.0, gt=0)
    llm_max_retries: int = Field(default=2, ge=0)
    max_concurrency: int = Field(default=5, gt=0)

    # Published gpt-4o-mini prices (USD per million tokens), used only for cost estimates in logs/meta.
    llm_input_usd_per_mtok: float = Field(default=0.15, ge=0)
    llm_output_usd_per_mtok: float = Field(default=0.60, ge=0)

    warm_up_models: bool = True
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
