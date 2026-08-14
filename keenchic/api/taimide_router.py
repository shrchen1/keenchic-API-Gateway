"""Taimide-edition API router.

Provides endpoints exclusive to the taimide custom edition:

- ``GET /api/taimide/v1/templates`` — list available templates (.xlsx and .json)
- ``GET /api/taimide/v1/templates/{filename}`` — download a specific template file
- ``POST /api/taimide/v1/photos`` — upload a full photo (non-cropped)
- ``POST /api/taimide/v1/reports`` — upload a filled Excel report

This router is only mounted when ``KEENCHIC_EDITION=taimide``.
"""

from __future__ import annotations

import os
from pathlib import Path

import structlog
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from keenchic.api.deps import require_api_key
from keenchic.core.config import settings
from keenchic.core.file_saver import (
    generate_safe_filename,
    generate_taimide_report_filename,
    save_file,
)

log = structlog.get_logger(__name__)

taimide_router = APIRouter()

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
REPORT_SUBFOLDER_MAX_LENGTH = 100
XLSX_MEDIA_TYPE = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
JSON_MEDIA_TYPE = "application/json"
NO_STORE_HEADERS = {"Cache-Control": "no-store"}


def _get_template_dir() -> Path:
    """Resolve the template directory and handle errors."""
    raw = (settings.KEENCHIC_TAIMIDE_TEMPLATE_DIR or "").strip()
    if not raw:
        raise HTTPException(
            status_code=500,
            detail="Taimide template directory is not configured (KEENCHIC_TAIMIDE_TEMPLATE_DIR)",
        )
    path = Path(raw)
    try:
        resolved = path.resolve(strict=True)
        if not resolved.is_dir():
            raise HTTPException(
                status_code=500,
                detail="Taimide template path is not a directory",
            )
        return resolved
    except (FileNotFoundError, OSError) as exc:
        log.error("taimide.template_dir_invalid", error=str(exc))
        raise HTTPException(
            status_code=500,
            detail="Taimide template directory is not accessible",
        ) from exc


def _get_upload_dir() -> Path:
    """Resolve the Taimide upload base directory."""
    raw = (settings.KEENCHIC_TAIMIDE_UPLOAD_DIR or "").strip()
    if not raw:
        raise HTTPException(
            status_code=500,
            detail="Taimide upload directory is not configured (KEENCHIC_TAIMIDE_UPLOAD_DIR)",
        )
    return Path(raw).resolve()


def _normalize_report_subfolder(subfolder: str | None) -> str | None:
    """Normalize an optional client-chosen report group directory name."""
    if subfolder is None:
        return None

    normalized = subfolder.strip()
    if not normalized:
        return None

    is_safe = all(
        character.isalnum() or character in {"-", "_"}
        for character in normalized
    )
    if len(normalized) > REPORT_SUBFOLDER_MAX_LENGTH or not is_safe:
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid subfolder. Allowed: 1-100 Unicode alphanumeric "
                "characters, '-', '_'"
            ),
        )
    return normalized


def _resolve_report_directory(reports_dir: Path, subfolder: str | None) -> Path:
    """Create and resolve a real, direct child directory for a report group."""
    if subfolder is None:
        return reports_dir

    conflict_detail = "Report subfolder conflicts with an existing path"
    try:
        reports_dir.mkdir(parents=True, exist_ok=True)
        reports_root = reports_dir.resolve(strict=True)
    except OSError as exc:
        raise HTTPException(status_code=500, detail="Failed to save report") from exc

    candidate = reports_dir / subfolder
    if candidate.is_symlink():
        raise HTTPException(status_code=409, detail=conflict_detail)

    try:
        candidate.mkdir(exist_ok=True)
    except FileExistsError as exc:
        raise HTTPException(status_code=409, detail=conflict_detail) from exc
    except OSError as exc:
        raise HTTPException(status_code=500, detail="Failed to save report") from exc

    if candidate.is_symlink() or not candidate.is_dir():
        raise HTTPException(status_code=409, detail=conflict_detail)

    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise HTTPException(status_code=500, detail="Failed to save report") from exc
    if resolved.parent != reports_root:
        raise HTTPException(status_code=409, detail=conflict_detail)
    return resolved


@taimide_router.get(
    "/api/taimide/v1/templates",
    dependencies=[Depends(require_api_key)],
)
def list_templates() -> JSONResponse:
    """List available template files (.xlsx and .json) in the configured directory."""
    template_dir = _get_template_dir()
    files_list = []
    try:
        for entry in template_dir.iterdir():
            if entry.is_file() and not entry.is_symlink():
                if entry.suffix.lower() in {".xlsx", ".json"}:
                    try:
                        size = entry.stat().st_size
                        files_list.append({
                            "filename": entry.name,
                            "size_bytes": size,
                        })
                    except OSError:
                        continue
    except OSError as exc:
        log.error("taimide.list_templates_failed", error=str(exc))
        raise HTTPException(status_code=500, detail="Failed to list templates")

    # Sort by filename for consistent presentation
    files_list.sort(key=lambda x: x["filename"])
    return JSONResponse({"files": files_list}, headers=NO_STORE_HEADERS)


@taimide_router.get(
    "/api/taimide/v1/templates/{filename}",
    dependencies=[Depends(require_api_key)],
)
def download_template(filename: str) -> FileResponse:
    """Download a template file (.xlsx or .json) by filename with path traversal protection."""
    # Basic path traversal checks on the input string
    if "/" in filename or "\\" in filename or filename in {".", ".."}:
        raise HTTPException(status_code=400, detail="Invalid filename")

    template_dir = _get_template_dir()
    file_path = (template_dir / filename).resolve()

    # Verify that resolved path remains within the template directory
    try:
        if not file_path.is_relative_to(template_dir) or not file_path.is_file():
            raise HTTPException(status_code=404, detail="Template file not found")
    except (ValueError, FileNotFoundError):
        raise HTTPException(status_code=404, detail="Template file not found")

    ext = file_path.suffix.lower()
    if ext not in {".xlsx", ".json"}:
        raise HTTPException(status_code=400, detail="Invalid file type")

    media_type = XLSX_MEDIA_TYPE if ext == ".xlsx" else JSON_MEDIA_TYPE
    return FileResponse(
        file_path,
        media_type=media_type,
        filename=file_path.name,
        headers=NO_STORE_HEADERS,
    )


@taimide_router.post(
    "/api/taimide/v1/photos",
    dependencies=[Depends(require_api_key)],
    status_code=201,
)
async def upload_photo(
    file: UploadFile = File(..., description="Photo file to upload"),
) -> JSONResponse:
    """Upload a full photo and save it to the photos/ subdirectory."""
    orig_name = file.filename or "photo.jpg"
    ext = Path(orig_name).suffix.lower()
    if ext not in {".jpg", ".jpeg", ".png", ".webp"}:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file extension: {ext}. Allowed: .jpg, .jpeg, .png, .webp",
        )

    data = await file.read()
    if not data:
        raise HTTPException(status_code=422, detail="Uploaded file is empty")
    if len(data) > MAX_UPLOAD_BYTES:
        size_mb = len(data) / (1024 * 1024)
        limit_mb = MAX_UPLOAD_BYTES / (1024 * 1024)
        raise HTTPException(
            status_code=413,
            detail=f"File too large: {size_mb:.1f} MB exceeds {limit_mb:.0f} MB limit",
        )

    upload_dir = _get_upload_dir()
    photos_dir = os.path.join(upload_dir, "photos")

    try:
        filename = generate_safe_filename(orig_name, getattr(file, "content_type", None))
        save_file(data, photos_dir, filename)
    except Exception as exc:
        log.error("taimide.photo_upload_failed", filename=orig_name, error=str(exc))
        raise HTTPException(status_code=500, detail="Failed to save photo") from exc

    log.info("taimide.photo_saved", filename=filename, size_bytes=len(data))
    return JSONResponse(
        {
            "filename": filename,
            "size_bytes": len(data),
            "saved_to": "photos",
        },
        status_code=201,
    )


@taimide_router.post(
    "/api/taimide/v1/reports",
    dependencies=[Depends(require_api_key)],
    status_code=201,
)
async def upload_report(
    file: UploadFile = File(..., description="Excel report file to upload"),
    subfolder: str | None = Form(
        None,
        description="Optional client-chosen report group directory",
    ),
) -> JSONResponse:
    """Upload an Excel report and save it to the reports/ subdirectory."""
    normalized_subfolder = _normalize_report_subfolder(subfolder)

    # --- validate file ---------------------------------------------------
    orig_name = file.filename or "report.xlsx"
    ext = Path(orig_name).suffix.lower()
    if ext not in {".xlsx"}:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file extension: {ext}. Allowed: .xlsx",
        )

    data = await file.read()
    if not data:
        raise HTTPException(status_code=422, detail="Uploaded file is empty")
    if len(data) > MAX_UPLOAD_BYTES:
        size_mb = len(data) / (1024 * 1024)
        limit_mb = MAX_UPLOAD_BYTES / (1024 * 1024)
        raise HTTPException(
            status_code=413,
            detail=f"File too large: {size_mb:.1f} MB exceeds {limit_mb:.0f} MB limit",
        )

    # --- save ------------------------------------------------------------
    upload_dir = _get_upload_dir()
    reports_dir = upload_dir / "reports"

    try:
        destination_dir = _resolve_report_directory(
            reports_dir,
            normalized_subfolder,
        )
        filename = generate_taimide_report_filename(
            orig_name, getattr(file, "content_type", None),
        )
        save_file(data, str(destination_dir), filename)
    except HTTPException as exc:
        error_fields = {
            "filename": orig_name,
            "error": exc.detail,
        }
        if normalized_subfolder is not None:
            error_fields["subfolder"] = normalized_subfolder
        log.error("taimide.report_upload_failed", **error_fields)
        raise
    except Exception as exc:
        error_fields = {"filename": orig_name, "error": str(exc)}
        if normalized_subfolder is not None:
            error_fields["subfolder"] = normalized_subfolder
        log.error("taimide.report_upload_failed", **error_fields)
        raise HTTPException(status_code=500, detail="Failed to save report") from exc

    log_fields = {"filename": filename, "size_bytes": len(data)}
    if normalized_subfolder is not None:
        log_fields["subfolder"] = normalized_subfolder
    log.info("taimide.report_saved", **log_fields)

    response_content = {
        "filename": filename,
        "size_bytes": len(data),
        "saved_to": "reports",
    }
    if normalized_subfolder is not None:
        response_content["subfolder"] = normalized_subfolder
    return JSONResponse(
        response_content,
        status_code=201,
    )
