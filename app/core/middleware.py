from starlette.exceptions import HTTPException
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send


class RequestBodyTooLarge(HTTPException):
    def __init__(self, max_bytes: int) -> None:
        super().__init__(status_code=413, detail=f"Request body exceeds the {max_bytes // (1024 * 1024)} MB limit.")


class BodySizeLimitMiddleware:
    """Rejects oversized request bodies before Starlette spools them to disk.

    Checks Content-Length up front and also counts streamed bytes, so chunked
    uploads without a Content-Length header are bounded too.
    """

    def __init__(self, app: ASGIApp, max_body_bytes: int) -> None:
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        content_length = dict(scope["headers"]).get(b"content-length")
        if content_length is not None and content_length.isdigit() and int(content_length) > self.max_body_bytes:
            # Outside the exception-handling layer here, so respond directly rather than raising.
            error = RequestBodyTooLarge(self.max_body_bytes)
            response = JSONResponse(
                status_code=error.status_code,
                content={"error": {"code": "request_too_large", "message": error.detail}},
            )
            await response(scope, receive, send)
            return

        received = 0

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_body_bytes:
                    # HTTPException subclass so FastAPI's body parser re-raises it instead of wrapping it as a 400.
                    raise RequestBodyTooLarge(self.max_body_bytes)
            return message

        await self.app(scope, limited_receive, send)
