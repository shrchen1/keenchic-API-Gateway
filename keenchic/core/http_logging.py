from __future__ import annotations

import json
import re
import tempfile
import time
import uuid
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from datetime import datetime, timezone
from typing import Any, BinaryIO
from urllib.parse import parse_qsl

import structlog
from starlette.datastructures import Headers, UploadFile
from starlette.formparsers import MultiPartParser
from starlette.types import ASGIApp, Message, Receive, Scope, Send

PAYLOAD_PREVIEW_LIMIT_BYTES = 256 * 1024
REQUEST_SPOOL_MAX_MEMORY_BYTES = 1024 * 1024
REDACTED = "[REDACTED]"
REDACTED_BINARY = "[REDACTED_BINARY]"
REDACTED_IMAGE = "[REDACTED_IMAGE]"

_REQUEST_HEADER_ALLOWLIST = {
    "content-type": "Content-Type",
    "content-length": "Content-Length",
    "x-inspection-name": "X-Inspection-Name",
    "user-agent": "User-Agent",
}
_RESPONSE_HEADER_ALLOWLIST = {
    "content-type": "Content-Type",
    "content-length": "Content-Length",
    "content-disposition": "Content-Disposition",
}
_SENSITIVE_KEYS = {
    "api_key",
    "apikey",
    "auth",
    "authorization",
    "client_secret",
    "credential",
    "credentials",
    "password",
    "passwd",
    "pwd",
    "secret",
    "token",
}
_SENSITIVE_SUFFIXES = (
    "_api_key",
    "_credential",
    "_credentials",
    "_password",
    "_passwd",
    "_secret",
    "_token",
)
_IMAGE_KEYS = {
    "base64_image",
    "b64_image",
    "diag_img",
    "diag_img_en",
    "image",
    "image_b64",
    "image_base64",
}

log = structlog.get_logger(__name__)


def payload_logging_is_enabled(log_level: str) -> bool:
    """Return whether DEBUG payload capture should be enabled."""
    return log_level.strip().upper() == "DEBUG"


def sanitize_payload(value: Any, field_name: str | None = None) -> Any:
    """Recursively remove credentials, image payloads, and raw binary values."""
    if field_name is not None:
        if _is_sensitive_key(field_name):
            return REDACTED
        if _is_image_key(field_name):
            return REDACTED_IMAGE

    if isinstance(value, Mapping):
        return {
            str(key): sanitize_payload(item, str(key)) for key, item in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [sanitize_payload(item, field_name) for item in value]
    if isinstance(value, (bytes, bytearray, memoryview)):
        return REDACTED_BINARY
    if isinstance(value, str) and value.lstrip().lower().startswith("data:image/"):
        return REDACTED_IMAGE
    return value


def payload_log_fields(payload: Any) -> dict[str, Any]:
    """Sanitize payload and bound its serialized log representation."""
    sanitized = sanitize_payload(payload)
    serialized = json.dumps(
        sanitized,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )
    serialized_bytes = serialized.encode("utf-8")
    original_size = len(serialized_bytes)
    if original_size <= PAYLOAD_PREVIEW_LIMIT_BYTES:
        return {"payload": sanitized}

    preview = serialized_bytes[:PAYLOAD_PREVIEW_LIMIT_BYTES].decode(
        "utf-8",
        errors="ignore",
    )
    return {
        "payload_preview": preview,
        "truncated": True,
        "original_size_bytes": original_size,
    }


class HttpLoggingMiddleware:
    """Log request lifecycle summaries and sanitized DEBUG payload details."""

    def __init__(
        self,
        app: ASGIApp,
        log_level_getter: Callable[[], str],
    ) -> None:
        self.app = app
        self.log_level_getter = log_level_getter

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = uuid.uuid4().hex[:12]
        path = scope.get("path", "")
        method = scope.get("method", "")
        request_headers = Headers(raw=scope.get("headers", []))
        inspection_name = request_headers.get("X-Inspection-Name")
        capture_payload = payload_logging_is_enabled(
            self.log_level_getter()
        ) and not _is_payload_excluded(path)

        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)
        start = time.monotonic()
        request_fields: dict[str, Any] = {
            "method": method,
            "path": path,
            "inspection_name": inspection_name,
            "received_at": datetime.now(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
        }
        request_capture_error_type: str | None = None
        request_spool: BinaryIO | None = None
        response_capture = _ResponseCapture(enabled=capture_payload)
        try:
            app_receive = receive
            if capture_payload:
                request_spool = tempfile.SpooledTemporaryFile(
                    max_size=REQUEST_SPOOL_MAX_MEMORY_BYTES,
                    mode="w+b",
                )
                body_complete = await _spool_request_body(receive, request_spool)
                try:
                    (
                        request_details,
                        request_capture_error_type,
                    ) = await self._capture_request_payload(
                        scope,
                        request_headers,
                        request_spool,
                    )
                    request_fields.update(request_details)
                except Exception as exc:
                    request_capture_error_type = type(exc).__name__
                    request_spool.seek(0)
                app_receive = _request_replay(
                    request_spool,
                    receive,
                    body_complete,
                )

            await self.app(
                scope,
                app_receive,
                response_capture.wrap_send(send),
            )
            response_fields, response_capture_error_type = (
                response_capture.finalize_fields()
            )
            log.info("http.request", **request_fields)
            if request_capture_error_type is not None:
                _log_capture_failure("request", request_capture_error_type)

            response_fields.update(
                {
                    "method": method,
                    "path": path,
                    "inspection_name": inspection_name,
                    "duration_ms": _duration_ms(start),
                }
            )
            if response_capture_error_type is not None:
                _log_capture_failure("response", response_capture_error_type)
            log.info("http.response", **response_fields)
        except Exception as exc:
            _, response_capture_error_type = response_capture.finalize_fields()
            log.info("http.request", **request_fields)
            if request_capture_error_type is not None:
                _log_capture_failure("request", request_capture_error_type)
            if response_capture_error_type is not None:
                _log_capture_failure("response", response_capture_error_type)
            log.error(
                "http.error",
                method=method,
                path=path,
                inspection_name=inspection_name,
                status_code=500,
                duration_ms=_duration_ms(start),
                error=str(exc),
            )
            raise
        finally:
            if request_spool is not None:
                request_spool.close()
            structlog.contextvars.clear_contextvars()

    async def _capture_request_payload(
        self,
        scope: Scope,
        headers: Headers,
        spool: BinaryIO,
    ) -> tuple[dict[str, Any], str | None]:
        body_size = _spool_size(spool)
        media_type = _media_type(headers.get("content-type"))
        event_fields: dict[str, Any] = {
            "query": sanitize_payload(_query_fields(scope)),
            "headers": _allowlisted_headers(headers, _REQUEST_HEADER_ALLOWLIST),
            "content_type": media_type or None,
            "body_size_bytes": body_size,
        }
        error_type: str | None = None

        try:
            payload = await _parse_request_payload(headers, spool, media_type)
            if payload is not None:
                event_fields.update(payload_log_fields(payload))
        except Exception as exc:
            error_type = type(exc).__name__
        finally:
            spool.seek(0)

        return event_fields, error_type


class _ResponseCapture:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self.status_code = 500
        self.headers = Headers()
        self.body_size_bytes = 0
        self.body = (
            tempfile.SpooledTemporaryFile(
                max_size=REQUEST_SPOOL_MAX_MEMORY_BYTES,
                mode="w+b",
            )
            if enabled
            else None
        )
        self._logged = False
        self._capture_json = False

    def wrap_send(self, send: Send) -> Send:
        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                self.status_code = message["status"]
                self.headers = Headers(raw=message.get("headers", []))
                self._capture_json = _is_json_media_type(
                    _media_type(self.headers.get("content-type"))
                )
            elif message["type"] == "http.response.body":
                chunk = message.get("body", b"")
                self.body_size_bytes += len(chunk)
                if self.enabled and self._capture_json and self.body is not None:
                    self.body.write(chunk)
            await send(message)

        return send_wrapper

    def finalize_fields(self) -> tuple[dict[str, Any], str | None]:
        if self._logged:
            return {"status_code": self.status_code}, None
        self._logged = True
        event_fields: dict[str, Any] = {"status_code": self.status_code}
        error_type: str | None = None
        try:
            if self.enabled:
                media_type = _media_type(self.headers.get("content-type"))
                event_fields.update(
                    {
                        "headers": _allowlisted_headers(
                            self.headers,
                            _RESPONSE_HEADER_ALLOWLIST,
                        ),
                        "content_type": media_type or None,
                        "body_size_bytes": self.body_size_bytes,
                    }
                )
                if self._capture_json and self.body is not None:
                    self.body.seek(0)
                    payload = json.loads(self.body.read().decode("utf-8"))
                    event_fields.update(payload_log_fields(payload))
        except Exception as exc:
            error_type = type(exc).__name__
        finally:
            if self.body is not None:
                self.body.close()
        return event_fields, error_type


async def _spool_request_body(receive: Receive, spool: BinaryIO) -> bool:
    while True:
        message = await receive()
        if message["type"] == "http.disconnect":
            spool.seek(0)
            return False
        if message["type"] != "http.request":
            continue
        spool.write(message.get("body", b""))
        if not message.get("more_body", False):
            spool.seek(0)
            return True


def _request_replay(
    spool: BinaryIO,
    receive: Receive,
    body_complete: bool,
) -> Receive:
    replay_complete = False

    async def replay() -> Message:
        nonlocal replay_complete
        if replay_complete:
            if not body_complete:
                return {"type": "http.disconnect"}
            return await receive()

        chunk = spool.read(64 * 1024)
        if chunk:
            spool_has_more = _spool_has_more(spool)
            replay_complete = not spool_has_more
            return {
                "type": "http.request",
                "body": chunk,
                "more_body": spool_has_more or not body_complete,
            }
        replay_complete = True
        if not body_complete:
            return {"type": "http.disconnect"}
        return {"type": "http.request", "body": b"", "more_body": False}

    return replay


async def _parse_request_payload(
    headers: Headers,
    spool: BinaryIO,
    media_type: str,
) -> Any | None:
    if not _spool_size(spool):
        return None
    if _is_json_media_type(media_type):
        spool.seek(0)
        return json.loads(spool.read().decode(_charset(headers.get("content-type"))))
    if media_type == "application/x-www-form-urlencoded":
        spool.seek(0)
        text = spool.read().decode(_charset(headers.get("content-type")))
        return _pairs_to_mapping(parse_qsl(text, keep_blank_values=True))
    if media_type == "multipart/form-data":
        return await _parse_multipart(headers, spool)
    return None


async def _parse_multipart(
    headers: Headers,
    spool: BinaryIO,
) -> dict[str, Any]:
    spool.seek(0)
    stream = _spool_stream(spool)
    form = await MultiPartParser(headers=headers, stream=stream).parse()
    fields: list[tuple[str, str]] = []
    files: list[dict[str, Any]] = []
    try:
        for field_name, value in form.multi_items():
            if isinstance(value, UploadFile):
                files.append(
                    {
                        "field_name": field_name,
                        "filename": value.filename,
                        "content_type": value.content_type,
                        "size_bytes": value.size,
                    }
                )
            else:
                fields.append((field_name, value))
    finally:
        await form.close()
    return {"fields": _pairs_to_mapping(fields), "files": files}


async def _spool_stream(spool: BinaryIO) -> AsyncIterator[bytes]:
    while True:
        chunk = spool.read(64 * 1024)
        if not chunk:
            break
        yield chunk


def _query_fields(scope: Scope) -> dict[str, Any]:
    raw_query = scope.get("query_string", b"").decode("utf-8", errors="replace")
    return _pairs_to_mapping(parse_qsl(raw_query, keep_blank_values=True))


def _pairs_to_mapping(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key not in result:
            result[key] = value
        elif isinstance(result[key], list):
            result[key].append(value)
        else:
            result[key] = [result[key], value]
    return result


def _allowlisted_headers(
    headers: Headers,
    allowlist: Mapping[str, str],
) -> dict[str, str]:
    return {
        display_name: headers[key]
        for key, display_name in allowlist.items()
        if key in headers
    }


def _media_type(content_type: str | None) -> str:
    if not content_type:
        return ""
    return content_type.split(";", 1)[0].strip().lower()


def _charset(content_type: str | None) -> str:
    if not content_type:
        return "utf-8"
    match = re.search(r"charset=([^;\s]+)", content_type, flags=re.IGNORECASE)
    return match.group(1).strip('"') if match else "utf-8"


def _is_json_media_type(media_type: str) -> bool:
    return media_type == "application/json" or media_type.endswith("+json")


def _is_payload_excluded(path: str) -> bool:
    return (
        path == "/openapi.json"
        or path == "/docs"
        or path.startswith("/docs/")
        or path == "/redoc"
        or path.startswith("/redoc/")
    )


def _normalize_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_")


def _is_sensitive_key(key: str) -> bool:
    normalized = _normalize_key(key)
    compact = normalized.replace("_", "")
    return (
        normalized in _SENSITIVE_KEYS
        or normalized.endswith(_SENSITIVE_SUFFIXES)
        or compact.endswith(
            (
                "apikey",
                "authorization",
                "credential",
                "credentials",
                "password",
                "passwd",
                "secret",
                "token",
            )
        )
    )


def _is_image_key(key: str) -> bool:
    normalized = _normalize_key(key)
    if normalized in _IMAGE_KEYS:
        return True
    compact = normalized.replace("_", "")
    if compact in {
        "base64image",
        "b64image",
        "diagimg",
        "diagimgen",
        "image",
        "imageb64",
        "imagebase64",
    }:
        return True
    return "base64" in compact and any(
        marker in compact for marker in ("image", "img", "photo", "picture")
    )


def _spool_size(spool: BinaryIO) -> int:
    position = spool.tell()
    spool.seek(0, 2)
    size = spool.tell()
    spool.seek(position)
    return size


def _spool_has_more(spool: BinaryIO) -> bool:
    position = spool.tell()
    spool.seek(0, 2)
    end = spool.tell()
    spool.seek(position)
    return position < end


def _duration_ms(start: float) -> float:
    return round((time.monotonic() - start) * 1000, 1)


def _log_capture_failure(phase: str, error_type: str) -> None:
    log.warning(
        "http.payload_capture_failed",
        phase=phase,
        error_type=error_type,
    )


__all__ = [
    "HttpLoggingMiddleware",
    "PAYLOAD_PREVIEW_LIMIT_BYTES",
    "payload_log_fields",
    "payload_logging_is_enabled",
    "sanitize_payload",
]
