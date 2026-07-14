from __future__ import annotations

from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.requests import Request

from keenchic.api.router import router
from keenchic.core.config import settings
from keenchic.core.http_logging import (
    HttpLoggingMiddleware,
    PAYLOAD_PREVIEW_LIMIT_BYTES,
    payload_logging_is_enabled,
)
from keenchic.core.logging import configure_logging

log = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging(settings.LOG_FORMAT, settings.LOG_LEVEL)
    log.info("app.startup", log_format=settings.LOG_FORMAT, log_level=settings.LOG_LEVEL)
    if payload_logging_is_enabled(settings.LOG_LEVEL):
        log.warning(
            "payload_logging.enabled",
            output="stdout",
            preview_limit_bytes=PAYLOAD_PREVIEW_LIMIT_BYTES,
        )
    
    if settings.KEENCHIC_EDITION == "taimide":
        template_dir_str = (settings.KEENCHIC_TAIMIDE_TEMPLATE_DIR or "").strip()
        if not template_dir_str:
            raise RuntimeError("KEENCHIC_TAIMIDE_TEMPLATE_DIR is required for taimide edition")
        
        from pathlib import Path
        template_dir = Path(template_dir_str)
        if not template_dir.is_dir():
            raise RuntimeError(f"KEENCHIC_TAIMIDE_TEMPLATE_DIR is not a valid directory: {template_dir_str}")
            
        upload_dir_str = (settings.KEENCHIC_TAIMIDE_UPLOAD_DIR or "").strip()
        if not upload_dir_str:
            raise RuntimeError("KEENCHIC_TAIMIDE_UPLOAD_DIR is required for taimide edition")
            
        upload_dir = Path(upload_dir_str)
        try:
            (upload_dir / "photos").mkdir(parents=True, exist_ok=True)
            (upload_dir / "reports").mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            raise RuntimeError(f"Failed to create Taimide upload directories: {exc}") from exc
            
        log.info(
            "taimide.startup_configured",
            template_dir=str(template_dir),
            upload_dir=str(upload_dir),
        )

    yield
    log.info("app.shutdown")


app = FastAPI(title="Keenchic Inspection API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    HttpLoggingMiddleware,
    log_level_getter=lambda: settings.LOG_LEVEL,
)


def _sanitize_error(err: dict) -> dict:
    result = {}
    for k, v in err.items():
        if k == "input":
            continue
        if k == "ctx" and isinstance(v, dict):
            result[k] = {ck: str(cv) if isinstance(cv, Exception) else cv for ck, cv in v.items()}
        else:
            result[k] = v
    return result


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": [_sanitize_error(e) for e in exc.errors()]})


app.include_router(router)

if settings.KEENCHIC_EDITION == "taimide":
    from keenchic.api.taimide_router import taimide_router

    app.include_router(taimide_router)

__all__ = ["app"]
