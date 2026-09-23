import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.qa import router as qa_router
from app.core.config import Settings, get_settings
from app.core.errors import register_error_handlers
from app.core.logging import RequestContextMiddleware, configure_logging
from app.core.middleware import BodySizeLimitMiddleware
from app.deps import build_pipeline

logger = logging.getLogger(__name__)

# Multipart framing and headers add a little on top of the two file limits.
_MULTIPART_OVERHEAD_BYTES = 64 * 1024
_STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if settings.openai_api_key is None:
            logger.warning("openai_api_key_missing")
        app.state.pipeline = build_pipeline(settings)
        if settings.warm_up_models:
            await asyncio.to_thread(app.state.pipeline.warm_up)
        yield

    app = FastAPI(
        title="grc-doc-qa",
        version="0.1.0",
        description="Answers questionnaire-style questions from a compliance document, with citations.",
        lifespan=lifespan,
    )
    app.add_middleware(
        BodySizeLimitMiddleware,
        max_body_bytes=settings.max_document_bytes + settings.max_questions_bytes + _MULTIPART_OVERHEAD_BYTES,
    )
    app.add_middleware(RequestContextMiddleware)  # added last = outermost, so every response gets an id
    app.dependency_overrides[get_settings] = lambda: settings
    register_error_handlers(app)
    app.include_router(qa_router)

    @app.get("/health", tags=["ops"])
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    # Mounted last so API routes take precedence; serves the minimal UI at "/".
    app.mount("/", StaticFiles(directory=_STATIC_DIR, html=True), name="ui")
    return app


app = create_app()
