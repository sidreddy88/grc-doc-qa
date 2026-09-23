import logging

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core import exceptions as exc

logger = logging.getLogger(__name__)

# Services raise domain errors; only this module knows about HTTP status codes.
_STATUS_BY_ERROR: dict[type[exc.AppError], int] = {
    exc.InvalidFileTypeError: status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
    exc.FileTooLargeError: status.HTTP_413_CONTENT_TOO_LARGE,
    exc.InvalidQuestionsError: status.HTTP_422_UNPROCESSABLE_CONTENT,
    exc.TooManyQuestionsError: status.HTTP_422_UNPROCESSABLE_CONTENT,
    exc.DocumentParseError: status.HTTP_422_UNPROCESSABLE_CONTENT,
    exc.EmptyDocumentError: status.HTTP_422_UNPROCESSABLE_CONTENT,
    exc.DocumentTooLongError: status.HTTP_422_UNPROCESSABLE_CONTENT,
    exc.ProcessingTimeoutError: status.HTTP_504_GATEWAY_TIMEOUT,
    exc.UpstreamServiceError: status.HTTP_502_BAD_GATEWAY,
    exc.ConfigurationError: status.HTTP_503_SERVICE_UNAVAILABLE,
}


def _error_body(request: Request, code: str, message: str, details: list | None = None) -> dict:
    error: dict = {"code": code, "message": message}
    request_id = getattr(request.state, "request_id", None)
    if request_id:
        error["request_id"] = request_id
    if details:
        error["details"] = details
    return {"error": error}


def status_for(error: exc.AppError) -> int:
    for error_type in type(error).__mro__:
        if error_type in _STATUS_BY_ERROR:
            return _STATUS_BY_ERROR[error_type]
    return status.HTTP_500_INTERNAL_SERVER_ERROR


async def _handle_app_error(request: Request, error: exc.AppError) -> JSONResponse:
    status_code = status_for(error)
    log = logger.error if status_code >= 500 else logger.info
    log("request_failed", extra={"error_code": error.code, "status_code": status_code})
    return JSONResponse(status_code=status_code, content=_error_body(request, error.code, error.message))


async def _handle_request_validation(request: Request, error: RequestValidationError) -> JSONResponse:
    details = [
        {"field": ".".join(str(part) for part in item.get("loc", ())), "message": item.get("msg", "")}
        for item in error.errors()
    ]
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        content=_error_body(request, "invalid_request", "Request is missing or has malformed fields.", details),
    )


_CODE_BY_HTTP_STATUS = {404: "not_found", 405: "method_not_allowed", 413: "request_too_large"}


async def _handle_http_exception(request: Request, error: StarletteHTTPException) -> JSONResponse:
    code = _CODE_BY_HTTP_STATUS.get(error.status_code, "http_error")
    return JSONResponse(
        status_code=error.status_code,
        content=_error_body(request, code, str(error.detail)),
        headers=getattr(error, "headers", None),
    )


async def _handle_unexpected(request: Request, error: Exception) -> JSONResponse:
    logger.exception("unhandled_error")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=_error_body(request, "internal_error", "An unexpected error occurred."),
    )


def register_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(exc.AppError, _handle_app_error)
    app.add_exception_handler(RequestValidationError, _handle_request_validation)
    app.add_exception_handler(StarletteHTTPException, _handle_http_exception)
    app.add_exception_handler(Exception, _handle_unexpected)
