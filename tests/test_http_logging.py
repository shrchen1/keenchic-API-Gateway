from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from unittest.mock import AsyncMock

import numpy as np
import pytest
import structlog
from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.testclient import TestClient
from starlette.types import Message, Receive, Scope, Send

import keenchic.api.router as standard_router_module
import keenchic.core.http_logging as http_logging
import main as main_module
from keenchic.api.router import router
from keenchic.api.taimide_router import taimide_router
from keenchic.core.config import settings
from keenchic.core.http_logging import (
    HttpLoggingMiddleware,
    REDACTED,
    REDACTED_IMAGE,
)
from keenchic.core.logging import configure_logging


def _make_app(log_level: str, log_format: str = "json") -> FastAPI:
    configure_logging(log_format, log_level)
    app = FastAPI()
    app.add_middleware(
        HttpLoggingMiddleware,
        log_level_getter=lambda: log_level,
    )

    @app.post("/echo")
    async def echo(request: Request) -> JSONResponse:
        return JSONResponse(await request.json())

    @app.post("/form")
    async def form(request: Request) -> JSONResponse:
        parsed = await request.form()
        return JSONResponse({"item_count": len(parsed.multi_items())})

    @app.post("/validated")
    async def validated(count: int = Body(embed=True)) -> JSONResponse:
        return JSONResponse({"count": count})

    @app.get("/health")
    async def health() -> JSONResponse:
        return JSONResponse({"status": "ok"})

    @app.get("/protected")
    async def protected() -> JSONResponse:
        raise HTTPException(status_code=401, detail="Unauthorized")

    @app.get("/fail")
    async def fail() -> JSONResponse:
        raise RuntimeError("intentional failure")

    @app.get("/stream")
    async def stream() -> StreamingResponse:
        async def chunks() -> AsyncIterator[bytes]:
            yield b"stream-binary-secret"

        return StreamingResponse(chunks(), media_type="application/octet-stream")

    @app.post("/binary")
    async def binary(request: Request) -> Response:
        body = await request.body()
        return Response(body, media_type="application/octet-stream")

    @app.get("/invalid-json")
    async def invalid_json() -> Response:
        return Response(b"not-json", media_type="application/json")

    return app


def _json_events(capsys: pytest.CaptureFixture[str]) -> list[dict[str, object]]:
    output = capsys.readouterr().out
    return [json.loads(line) for line in output.splitlines() if line.startswith("{")]


def _event(
    events: list[dict[str, object]],
    name: str,
    path: str | None = None,
) -> dict[str, object]:
    matches = [event for event in events if event.get("event") == name]
    if path is not None:
        matches = [event for event in matches if event.get("path") == path]
    assert len(matches) == 1
    return matches[0]


def test_debug_json_logs_sanitized_request_and_response_payloads(
    capsys: pytest.CaptureFixture[str],
) -> None:
    app = _make_app("DEBUG")
    client = TestClient(app)
    response = client.post(
        "/echo?token=query-secret&tag=first&tag=second",
        headers={
            "X-API-KEY": "header-secret",
            "X-Inspection-Name": "ocr/pill-count",
            "User-Agent": "payload-test",
        },
        json={
            "password": "body-secret",
            "nested": {"safe": "visible"},
            "diag_img": "image-secret",
            "preview": "data:image/png;base64,data-uri-secret",
        },
    )

    assert response.status_code == 200
    events = _json_events(capsys)
    request_event = _event(events, "http.request", "/echo")
    response_event = _event(events, "http.response", "/echo")

    assert request_event["query"] == {
        "token": REDACTED,
        "tag": ["first", "second"],
    }
    assert request_event["headers"] == {
        "Content-Type": "application/json",
        "Content-Length": request_event["headers"]["Content-Length"],
        "X-Inspection-Name": "ocr/pill-count",
        "User-Agent": "payload-test",
    }
    assert request_event["payload"]["password"] == REDACTED
    assert request_event["payload"]["diag_img"] == REDACTED_IMAGE
    assert request_event["payload"]["preview"] == REDACTED_IMAGE
    assert response_event["payload"]["password"] == REDACTED
    assert response_event["payload"]["diag_img"] == REDACTED_IMAGE
    assert response_event["method"] == "POST"
    assert response_event["inspection_name"] == "ocr/pill-count"
    correlated_events = [
        event
        for event in events
        if event.get("event") in {"http.request", "http.response"}
    ]
    assert len(correlated_events) == 2
    assert len({event["request_id"] for event in correlated_events}) == 1

    serialized_events = json.dumps(events, ensure_ascii=False)
    for secret in (
        "query-secret",
        "header-secret",
        "body-secret",
        "image-secret",
        "data-uri-secret",
    ):
        assert secret not in serialized_events


def test_debug_text_payload_events_are_single_line_and_sanitized(
    capsys: pytest.CaptureFixture[str],
) -> None:
    app = _make_app("DEBUG", "text")
    client = TestClient(app)

    response = client.post("/echo", json={"password": "text-secret"})

    assert response.status_code == 200
    output = capsys.readouterr().out
    request_lines = [line for line in output.splitlines() if "http.request" in line]
    response_lines = [line for line in output.splitlines() if "http.response" in line]
    assert len(request_lines) == 1
    assert len(response_lines) == 1
    assert "text-secret" not in output
    assert REDACTED in output


def test_info_logs_summaries_without_payload_events(
    capsys: pytest.CaptureFixture[str],
) -> None:
    app = _make_app("INFO")
    client = TestClient(app)

    response = client.post("/echo", json={"safe": "value"})

    assert response.status_code == 200
    events = _json_events(capsys)
    request_event = _event(events, "http.request", "/echo")
    response_event = _event(events, "http.response", "/echo")
    assert "payload" not in request_event
    assert "headers" not in request_event
    assert "payload" not in response_event
    assert "headers" not in response_event


def test_access_log_precedes_grouped_request_and_response_events(
    capsys: pytest.CaptureFixture[str],
) -> None:
    app = _make_app("DEBUG")

    async def access_log_wrapper(
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        async def send_with_access_log(message: Message) -> None:
            if message["type"] == "http.response.start":
                structlog.get_logger("uvicorn.access").info("uvicorn.access")
            await send(message)

        await app(scope, receive, send_with_access_log)

    client = TestClient(access_log_wrapper)
    assert client.get("/health").status_code == 200

    events = _json_events(capsys)
    lifecycle_events = [
        event["event"]
        for event in events
        if event.get("event") in {"uvicorn.access", "http.request", "http.response"}
    ]
    assert lifecycle_events == ["uvicorn.access", "http.request", "http.response"]


def test_debug_lifespan_warns_that_payload_logging_is_enabled(
    capsys: pytest.CaptureFixture[str],
    restore_settings: None,
) -> None:
    settings.LOG_FORMAT = "json"
    settings.LOG_LEVEL = "DEBUG"
    settings.KEENCHIC_EDITION = "standard"

    async def run_lifespan() -> None:
        async with main_module.lifespan(FastAPI()):
            pass

    asyncio.run(run_lifespan())

    events = _json_events(capsys)
    enabled_events = [
        event for event in events if event.get("event") == "payload_logging.enabled"
    ]
    assert enabled_events == [
        {
            "event": "payload_logging.enabled",
            "level": "warning",
            "output": "stdout",
            "preview_limit_bytes": 256 * 1024,
            "timestamp": enabled_events[0]["timestamp"],
        }
    ]


def test_debug_logs_health_and_error_statuses_but_excludes_docs(
    capsys: pytest.CaptureFixture[str],
) -> None:
    app = _make_app("DEBUG")
    client = TestClient(app, raise_server_exceptions=False)

    assert client.get("/health").status_code == 200
    assert client.get("/protected").status_code == 401
    assert client.post("/validated", json={"count": "invalid"}).status_code == 422
    assert client.get("/missing").status_code == 404
    assert client.get("/fail").status_code == 500
    assert client.get("/docs").status_code == 200
    assert client.get("/openapi.json").status_code == 200

    events = _json_events(capsys)
    request_events = {
        event["path"]: event for event in events if event.get("event") == "http.request"
    }
    assert {
        "/health",
        "/protected",
        "/validated",
        "/missing",
        "/fail",
        "/docs",
        "/openapi.json",
    } <= request_events.keys()
    assert "headers" in request_events["/health"]
    assert "headers" not in request_events["/docs"]
    assert "headers" not in request_events["/openapi.json"]

    response_statuses = {
        event["status_code"]
        for event in events
        if event.get("event") == "http.response"
    }
    assert {200, 401, 404, 422} <= response_statuses
    assert any(
        event.get("event") == "http.error" and event.get("status_code") == 500
        for event in events
    )


def test_form_and_multipart_logs_preserve_values_and_omit_file_bytes(
    capsys: pytest.CaptureFixture[str],
) -> None:
    app = _make_app("DEBUG")
    client = TestClient(app)

    form_response = client.post(
        "/form",
        content="tag=first&tag=second&password=form-secret",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    multipart_response = client.post(
        "/form",
        data={"note": "visible", "password": "multipart-secret"},
        files={"image": ("inspection.png", b"multipart-binary-secret", "image/png")},
    )

    assert form_response.json() == {"item_count": 3}
    assert multipart_response.json() == {"item_count": 3}
    events = _json_events(capsys)
    request_events = [event for event in events if event.get("event") == "http.request"]
    assert request_events[0]["payload"] == {
        "tag": ["first", "second"],
        "password": REDACTED,
    }
    multipart_payload = request_events[1]["payload"]
    assert multipart_payload["fields"] == {
        "note": "visible",
        "password": REDACTED,
    }
    assert multipart_payload["files"] == [
        {
            "field_name": "image",
            "filename": "inspection.png",
            "content_type": "image/png",
            "size_bytes": len(b"multipart-binary-secret"),
        }
    ]
    serialized_events = json.dumps(events)
    assert "form-secret" not in serialized_events
    assert "multipart-secret" not in serialized_events
    assert "multipart-binary-secret" not in serialized_events


def test_unknown_request_and_streaming_response_log_metadata_only(
    capsys: pytest.CaptureFixture[str],
) -> None:
    app = _make_app("DEBUG")
    client = TestClient(app)

    binary_response = client.post(
        "/binary",
        content=b"request-binary-secret",
        headers={"Content-Type": "application/x-custom"},
    )
    stream_response = client.get("/stream")

    assert binary_response.content == b"request-binary-secret"
    assert stream_response.content == b"stream-binary-secret"
    events = _json_events(capsys)
    binary_request = _event(events, "http.request", "/binary")
    stream_response_event = next(
        event
        for event in events
        if event.get("event") == "http.response"
        and event.get("content_type") == "application/octet-stream"
        and event.get("body_size_bytes") == len(b"stream-binary-secret")
    )
    assert "payload" not in binary_request
    assert "payload_preview" not in binary_request
    assert "payload" not in stream_response_event
    serialized_events = json.dumps(events)
    assert "request-binary-secret" not in serialized_events
    assert "stream-binary-secret" not in serialized_events


def test_capture_failures_do_not_change_request_or_response(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _make_app("DEBUG")
    client = TestClient(app)

    async def fail_request_capture(*args: object, **kwargs: object) -> object:
        raise ValueError("capture detail must not be logged")

    monkeypatch.setattr(http_logging, "_parse_request_payload", fail_request_capture)
    echo_response = client.post("/echo", json={"safe": "still-replayed"})
    invalid_json_response = client.get("/invalid-json")

    assert echo_response.json() == {"safe": "still-replayed"}
    assert invalid_json_response.status_code == 200
    assert invalid_json_response.content == b"not-json"
    events = _json_events(capsys)
    failures = [
        event for event in events if event.get("event") == "http.payload_capture_failed"
    ]
    assert {event["phase"] for event in failures} == {"request", "response"}
    assert {event["error_type"] for event in failures} == {
        "JSONDecodeError",
        "ValueError",
    }
    assert "capture detail must not be logged" not in json.dumps(events)


@pytest.fixture
def restore_settings() -> Iterator[None]:
    original = settings.model_dump()
    try:
        yield
    finally:
        for key, value in original.items():
            setattr(settings, key, value)


def test_standard_inspect_logs_file_metadata_and_sanitized_result(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    restore_settings: None,
) -> None:
    configure_logging("json", "DEBUG")
    settings.KEENCHIC_API_KEY = "standard-api-secret"
    app = FastAPI()
    app.add_middleware(HttpLoggingMiddleware, log_level_getter=lambda: "DEBUG")
    app.include_router(router)

    decode_upload = AsyncMock(return_value=np.zeros((1, 1, 3), dtype=np.uint8))
    run_inspection = AsyncMock(
        return_value={
            "result": 0,
            "pred_text": "42",
            "diag_img": "response-image-secret",
        }
    )
    monkeypatch.setattr(standard_router_module, "_decode_upload", decode_upload)
    monkeypatch.setattr(
        standard_router_module.inspection_manager, "run", run_inspection
    )

    client = TestClient(app)
    response = client.post(
        "/api/v1/inspect?include_diag=true",
        headers={
            "X-API-KEY": "standard-api-secret",
            "X-Inspection-Name": "ocr/pill-count",
        },
        files={
            "image": (
                "pill.png",
                b"standard-image-binary-secret",
                "image/png",
            )
        },
    )

    assert response.status_code == 200
    run_inspection.assert_awaited_once()
    events = _json_events(capsys)
    request_event = _event(events, "http.request", "/api/v1/inspect")
    response_event = _event(events, "http.response", "/api/v1/inspect")
    assert request_event["payload"]["files"] == [
        {
            "field_name": "image",
            "filename": "pill.png",
            "content_type": "image/png",
            "size_bytes": len(b"standard-image-binary-secret"),
        }
    ]
    assert response_event["payload"]["diag_img"] == REDACTED_IMAGE
    serialized_events = json.dumps(events)
    assert "standard-api-secret" not in serialized_events
    assert "standard-image-binary-secret" not in serialized_events
    assert "response-image-secret" not in serialized_events


def test_taimide_upload_and_download_log_metadata_without_binary(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    restore_settings: None,
) -> None:
    configure_logging("json", "DEBUG")
    template_dir = tmp_path / "templates"
    upload_dir = tmp_path / "uploads"
    template_dir.mkdir()
    (upload_dir / "photos").mkdir(parents=True)
    (upload_dir / "reports").mkdir()
    template_path = template_dir / "template.xlsx"
    template_path.write_bytes(b"download-binary-secret")
    settings.KEENCHIC_API_KEY = "taimide-api-secret"
    settings.KEENCHIC_TAIMIDE_TEMPLATE_DIR = str(template_dir)
    settings.KEENCHIC_TAIMIDE_UPLOAD_DIR = str(upload_dir)

    app = FastAPI()
    app.add_middleware(HttpLoggingMiddleware, log_level_getter=lambda: "DEBUG")
    app.include_router(taimide_router)
    client = TestClient(app)

    upload_response = client.post(
        "/api/taimide/v1/reports",
        headers={"X-API-KEY": "taimide-api-secret"},
        files={
            "file": (
                "檢測報告.xlsx",
                b"excel-upload-binary-secret",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        },
    )
    download_response = client.get(
        "/api/taimide/v1/templates/template.xlsx",
        headers={"X-API-KEY": "taimide-api-secret"},
    )

    assert upload_response.status_code == 201
    assert download_response.content == b"download-binary-secret"
    events = _json_events(capsys)
    upload_event = _event(
        events,
        "http.request",
        "/api/taimide/v1/reports",
    )
    assert upload_event["payload"]["files"][0] == {
        "field_name": "file",
        "filename": "檢測報告.xlsx",
        "content_type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "size_bytes": len(b"excel-upload-binary-secret"),
    }
    download_event = next(
        event
        for event in events
        if event.get("event") == "http.response"
        and event.get("content_type")
        == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    assert "payload" not in download_event
    assert download_event["headers"]["Content-Disposition"].startswith(
        "attachment; filename="
    )
    serialized_events = json.dumps(events, ensure_ascii=False)
    assert "taimide-api-secret" not in serialized_events
    assert "excel-upload-binary-secret" not in serialized_events
    assert "download-binary-secret" not in serialized_events
