class AppError(Exception):
    """Base for errors that should reach the client with a stable code and message."""

    code = "internal_error"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class InvalidFileTypeError(AppError):
    code = "invalid_file_type"


class FileTooLargeError(AppError):
    code = "file_too_large"


class InvalidQuestionsError(AppError):
    code = "invalid_questions"


class TooManyQuestionsError(AppError):
    code = "too_many_questions"


class DocumentParseError(AppError):
    code = "document_parse_error"


class EmptyDocumentError(AppError):
    code = "empty_document"


class DocumentTooLongError(AppError):
    code = "document_too_long"


class ProcessingTimeoutError(AppError):
    code = "processing_timeout"


class UpstreamServiceError(AppError):
    code = "upstream_unavailable"


class ConfigurationError(AppError):
    code = "service_misconfigured"
