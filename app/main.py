from fastapi import FastAPI

from app.api.qa import router as qa_router
from app.core.config import Settings, get_settings
from app.core.errors import register_error_handlers
from app.core.middleware import BodySizeLimitMiddleware

# Multipart framing and headers add a little on top of the two file limits.
_MULTIPART_OVERHEAD_BYTES = 64 * 1024


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    app = FastAPI(
        title="grc-doc-qa",
        version="0.1.0",
        description="Answers questionnaire-style questions from a compliance document, with citations.",
    )
    app.add_middleware(
        BodySizeLimitMiddleware,
        max_body_bytes=settings.max_document_bytes + settings.max_questions_bytes + _MULTIPART_OVERHEAD_BYTES,
    )
    app.dependency_overrides[get_settings] = lambda: settings
    register_error_handlers(app)
    app.include_router(qa_router)

    @app.get("/health", tags=["ops"])
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
