from functools import lru_cache

from app.services.pipeline import QAPipeline


@lru_cache
def get_pipeline() -> QAPipeline:
    return QAPipeline()
