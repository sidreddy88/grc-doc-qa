from fastapi import UploadFile

from app.core.exceptions import FileTooLargeError

_READ_CHUNK_BYTES = 64 * 1024


def format_bytes(num_bytes: int) -> str:
    if num_bytes >= 1024 * 1024:
        return f"{num_bytes / (1024 * 1024):.0f} MB"
    return f"{num_bytes / 1024:.0f} KB"


async def read_upload_limited(upload: UploadFile, max_bytes: int, field: str) -> bytes:
    if upload.size is not None and upload.size > max_bytes:
        raise FileTooLargeError(f"The {field} file exceeds the {format_bytes(max_bytes)} limit.")

    buffer = bytearray()
    while chunk := await upload.read(_READ_CHUNK_BYTES):
        buffer.extend(chunk)
        if len(buffer) > max_bytes:
            raise FileTooLargeError(f"The {field} file exceeds the {format_bytes(max_bytes)} limit.")
    return bytes(buffer)
