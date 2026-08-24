import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from keenchic.core.config import Settings, settings
from keenchic.core.file_saver import (
    generate_safe_filename,
    generate_taimide_report_filename,
    save_file,
)
import main as main_mod


@pytest.fixture
def temp_dirs():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        template_dir = tmp_path / "templates"
        template_dir.mkdir()
        upload_dir = tmp_path / "uploads"
        upload_dir.mkdir()
        yield template_dir, upload_dir


@pytest.fixture
def taimide_client(temp_dirs):
    template_dir, upload_dir = temp_dirs

    # Save original settings
    orig_api_key = settings.KEENCHIC_API_KEY
    orig_edition = settings.KEENCHIC_EDITION
    orig_template_dir = settings.KEENCHIC_TAIMIDE_TEMPLATE_DIR
    orig_upload_dir = settings.KEENCHIC_TAIMIDE_UPLOAD_DIR

    # Mutate settings directly so that deps.py and other modules see the values
    settings.KEENCHIC_API_KEY = "test-key"
    settings.KEENCHIC_EDITION = "taimide"
    settings.KEENCHIC_TAIMIDE_TEMPLATE_DIR = str(template_dir)
    settings.KEENCHIC_TAIMIDE_UPLOAD_DIR = str(upload_dir)

    # Re-create a fresh app configured for taimide edition
    app = FastAPI(title="Test Taimide App")
    from keenchic.api.router import router
    from keenchic.api.taimide_router import taimide_router

    app.include_router(router)
    app.include_router(taimide_router)

    # Pre-create subdirectories to avoid lifespan setup for simple client testing
    (upload_dir / "photos").mkdir(exist_ok=True)
    (upload_dir / "reports").mkdir(exist_ok=True)

    client = TestClient(app)
    try:
        yield client, template_dir, upload_dir
    finally:
        # Restore settings
        settings.KEENCHIC_API_KEY = orig_api_key
        settings.KEENCHIC_EDITION = orig_edition
        settings.KEENCHIC_TAIMIDE_TEMPLATE_DIR = orig_template_dir
        settings.KEENCHIC_TAIMIDE_UPLOAD_DIR = orig_upload_dir


# ---------------------------------------------------------------------------
# Tests: file_saver utility
# ---------------------------------------------------------------------------


def test_generate_safe_filename():
    fn1 = generate_safe_filename("user_file.xlsx")
    assert fn1.endswith(".xlsx")
    assert "user_file" in fn1

    fn2 = generate_safe_filename("raw_image", "image/png")
    assert fn2.endswith(".png")
    assert "raw_image" in fn2

    fn3 = generate_safe_filename("no_ext_no_mime")
    assert fn3.endswith(".jpg")
    assert "no_ext_no_mime" in fn3


def test_save_file_writes_data(temp_dirs):
    _, upload_dir = temp_dirs
    dest_dir = upload_dir / "sub"
    file_path = save_file(b"test-bytes", str(dest_dir), "output.txt")
    assert os.path.exists(file_path)
    assert Path(file_path).read_bytes() == b"test-bytes"


def test_generate_taimide_report_filename():
    # Test safe filename generation preserving extension and removing spaces/specials
    fn = generate_taimide_report_filename("final report.xlsx")
    assert fn == "finalreport.xlsx"

    # Test with Unicode/Chinese characters
    fn_zh = generate_taimide_report_filename("日常檢測報告.xlsx")
    assert fn_zh == "日常檢測報告.xlsx"

    # Test unsafe characters and path traversal
    fn_unsafe = generate_taimide_report_filename("../../unsafe/檢測*?名稱.xlsx")
    assert fn_unsafe == "檢測名稱.xlsx"


# ---------------------------------------------------------------------------
# Tests: taimide templates API
# ---------------------------------------------------------------------------


def test_list_templates(taimide_client):
    client, template_dir, _ = taimide_client

    # Write files in template dir
    (template_dir / "report.xlsx").write_bytes(b"xlsx-data")
    (template_dir / "metadata.json").write_bytes(b'{"meta": true}')
    (template_dir / "ignored.txt").write_bytes(b"txt-data")

    response = client.get(
        "/api/taimide/v1/templates",
        headers={"X-API-KEY": "test-key"},
    )
    assert response.status_code == 200
    res_json = response.json()
    assert "files" in res_json
    files = res_json["files"]
    assert len(files) == 2

    filenames = [f["filename"] for f in files]
    assert "report.xlsx" in filenames
    assert "metadata.json" in filenames
    assert "ignored.txt" not in filenames

    # Verify sizes are correct
    for f in files:
        if f["filename"] == "report.xlsx":
            assert f["size_bytes"] == len(b"xlsx-data")
        if f["filename"] == "metadata.json":
            assert f["size_bytes"] == len(b'{"meta": true}')


def test_list_templates_unauthorized(taimide_client):
    client, _, _ = taimide_client
    response = client.get("/api/taimide/v1/templates")
    assert response.status_code == 401


def test_download_template_xlsx(taimide_client):
    client, template_dir, _ = taimide_client
    (template_dir / "template.xlsx").write_bytes(b"xlsx-content")

    response = client.get(
        "/api/taimide/v1/templates/template.xlsx",
        headers={"X-API-KEY": "test-key"},
    )
    assert response.status_code == 200
    assert response.content == b"xlsx-content"
    assert response.headers["content-type"] == (
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    assert response.headers["cache-control"] == "no-store"


def test_download_template_json(taimide_client):
    client, template_dir, _ = taimide_client
    (template_dir / "template.json").write_bytes(b'{"json": true}')

    response = client.get(
        "/api/taimide/v1/templates/template.json",
        headers={"X-API-KEY": "test-key"},
    )
    assert response.status_code == 200
    assert response.content == b'{"json": true}'
    assert response.headers["content-type"] == "application/json"
    assert response.headers["cache-control"] == "no-store"


def test_download_template_not_found(taimide_client):
    client, _, _ = taimide_client
    response = client.get(
        "/api/taimide/v1/templates/nonexistent.xlsx",
        headers={"X-API-KEY": "test-key"},
    )
    assert response.status_code == 404


def test_download_template_invalid_type(taimide_client):
    client, template_dir, _ = taimide_client
    (template_dir / "dangerous.sh").write_bytes(b"rm -rf /")

    response = client.get(
        "/api/taimide/v1/templates/dangerous.sh",
        headers={"X-API-KEY": "test-key"},
    )
    assert response.status_code == 400


@pytest.mark.parametrize("bad_name", ["../etc/passwd", "..\\etc\\passwd", ".."])
def test_download_template_path_traversal(taimide_client, bad_name):
    client, _, _ = taimide_client
    response = client.get(
        f"/api/taimide/v1/templates/{bad_name}",
        headers={"X-API-KEY": "test-key"},
    )
    assert response.status_code in (400, 404)


# ---------------------------------------------------------------------------
# Tests: taimide photos upload API
# ---------------------------------------------------------------------------


def test_upload_photo_success(taimide_client):
    client, _, upload_dir = taimide_client
    photo_data = b"photo-bytes-content"

    response = client.post(
        "/api/taimide/v1/photos",
        headers={"X-API-KEY": "test-key"},
        files={"file": ("inspection_photo.png", photo_data, "image/png")},
    )
    assert response.status_code == 201
    res_json = response.json()
    assert "filename" in res_json
    assert res_json["size_bytes"] == len(photo_data)
    assert res_json["saved_to"] == "photos"

    # Confirm storage in photos subdir
    saved_file = upload_dir / "photos" / res_json["filename"]
    assert saved_file.is_file()
    assert saved_file.read_bytes() == photo_data


def test_upload_photo_invalid_ext(taimide_client):
    client, _, _ = taimide_client
    response = client.post(
        "/api/taimide/v1/photos",
        headers={"X-API-KEY": "test-key"},
        files={"file": ("data.xlsx", b"xlsx-content", "application/vnd.ms-excel")},
    )
    assert response.status_code == 400
    assert "Invalid file extension" in response.json()["detail"]


def test_upload_photo_empty(taimide_client):
    client, _, _ = taimide_client
    response = client.post(
        "/api/taimide/v1/photos",
        headers={"X-API-KEY": "test-key"},
        files={"file": ("photo.jpg", b"", "image/jpeg")},
    )
    assert response.status_code == 422
    assert "empty" in response.json()["detail"].lower()


def test_upload_photo_too_large(taimide_client):
    client, _, _ = taimide_client
    large_data = b"x" * (10 * 1024 * 1024 + 1)
    response = client.post(
        "/api/taimide/v1/photos",
        headers={"X-API-KEY": "test-key"},
        files={"file": ("photo.jpg", large_data, "image/jpeg")},
    )
    assert response.status_code == 413
    assert "exceeds" in response.json()["detail"]


# ---------------------------------------------------------------------------
# Tests: taimide reports upload API
# ---------------------------------------------------------------------------


def test_upload_report_success(taimide_client):
    client, _, upload_dir = taimide_client
    report_data = b"report-bytes-content"

    response = client.post(
        "/api/taimide/v1/reports",
        headers={"X-API-KEY": "test-key"},
        files={"file": ("日常檢測 報告.xlsx", report_data, "application/vnd.ms-excel")},
    )
    assert response.status_code == 201
    res_json = response.json()
    assert res_json["filename"] == "日常檢測報告.xlsx"
    assert res_json["size_bytes"] == len(report_data)
    assert res_json["saved_to"] == "reports"
    assert "subfolder" not in res_json

    saved_file = upload_dir / "reports" / res_json["filename"]
    assert saved_file.is_file()
    assert saved_file.read_bytes() == report_data


def test_upload_report_to_subfolder_creates_reuses_and_overwrites(taimide_client):
    client, _, upload_dir = taimide_client

    first_response = client.post(
        "/api/taimide/v1/reports",
        headers={"X-API-KEY": "test-key"},
        files={"file": ("report.xlsx", b"first", "application/vnd.ms-excel")},
        data={"subfolder": "  批號-LOT_001  "},
    )
    assert first_response.status_code == 201
    assert first_response.json() == {
        "filename": "report.xlsx",
        "size_bytes": len(b"first"),
        "saved_to": "reports",
        "subfolder": "批號-LOT_001",
    }

    saved_file = upload_dir / "reports" / "批號-LOT_001" / "report.xlsx"
    assert saved_file.read_bytes() == b"first"

    second_response = client.post(
        "/api/taimide/v1/reports",
        headers={"X-API-KEY": "test-key"},
        files={"file": ("report.xlsx", b"second", "application/vnd.ms-excel")},
        data={"subfolder": "批號-LOT_001"},
    )
    assert second_response.status_code == 201
    assert saved_file.read_bytes() == b"second"


def test_upload_report_same_filename_in_different_subfolders(taimide_client):
    client, _, upload_dir = taimide_client

    for subfolder, content in (("Instrument_A", b"a"), ("Instrument_B", b"b")):
        response = client.post(
            "/api/taimide/v1/reports",
            headers={"X-API-KEY": "test-key"},
            files={"file": ("report.xlsx", content, "application/vnd.ms-excel")},
            data={"subfolder": subfolder},
        )
        assert response.status_code == 201

    assert (upload_dir / "reports" / "Instrument_A" / "report.xlsx").read_bytes() == b"a"
    assert (upload_dir / "reports" / "Instrument_B" / "report.xlsx").read_bytes() == b"b"


@pytest.mark.parametrize("subfolder", ["", "   "])
def test_upload_report_blank_subfolder_preserves_existing_behavior(
    taimide_client,
    subfolder,
):
    client, _, upload_dir = taimide_client
    response = client.post(
        "/api/taimide/v1/reports",
        headers={"X-API-KEY": "test-key"},
        files={"file": ("report.xlsx", b"report", "application/vnd.ms-excel")},
        data={"subfolder": subfolder},
    )

    assert response.status_code == 201
    assert response.json() == {
        "filename": "report.xlsx",
        "size_bytes": len(b"report"),
        "saved_to": "reports",
    }
    assert (upload_dir / "reports" / "report.xlsx").read_bytes() == b"report"


@pytest.mark.parametrize(
    "subfolder",
    [
        "LOT 001",
        "../outside",
        "..\\outside",
        ".",
        "..",
        "LOT/001",
        "LOT.001",
        "批號!",
        "A" * 101,
    ],
)
def test_upload_report_invalid_subfolder_is_rejected(taimide_client, subfolder):
    client, _, upload_dir = taimide_client
    response = client.post(
        "/api/taimide/v1/reports",
        headers={"X-API-KEY": "test-key"},
        files={"file": ("report.xlsx", b"report", "application/vnd.ms-excel")},
        data={"subfolder": subfolder},
    )

    assert response.status_code == 400
    assert response.json()["detail"].startswith("Invalid subfolder")
    assert not (upload_dir / "outside").exists()


def test_upload_report_subfolder_conflicts_with_file(taimide_client):
    client, _, upload_dir = taimide_client
    conflict_path = upload_dir / "reports" / "LOT-001"
    conflict_path.write_bytes(b"not-a-directory")

    response = client.post(
        "/api/taimide/v1/reports",
        headers={"X-API-KEY": "test-key"},
        files={"file": ("report.xlsx", b"report", "application/vnd.ms-excel")},
        data={"subfolder": "LOT-001"},
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "Report subfolder conflicts with an existing path"


def test_upload_report_subfolder_rejects_symlink(taimide_client):
    client, _, upload_dir = taimide_client
    outside_dir = upload_dir / "outside"
    outside_dir.mkdir()
    subfolder_link = upload_dir / "reports" / "LOT-001"
    subfolder_link.symlink_to(outside_dir, target_is_directory=True)

    response = client.post(
        "/api/taimide/v1/reports",
        headers={"X-API-KEY": "test-key"},
        files={"file": ("report.xlsx", b"report", "application/vnd.ms-excel")},
        data={"subfolder": "LOT-001"},
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "Report subfolder conflicts with an existing path"
    assert not (outside_dir / "report.xlsx").exists()


def test_upload_report_invalid_ext(taimide_client):
    client, _, upload_dir = taimide_client
    response = client.post(
        "/api/taimide/v1/reports",
        headers={"X-API-KEY": "test-key"},
        files={"file": ("photo.png", b"png-content", "image/png")},
        data={"subfolder": "LOT-001"},
    )
    assert response.status_code == 400
    assert "Invalid file extension" in response.json()["detail"]
    assert not (upload_dir / "reports" / "LOT-001").exists()


def test_upload_report_empty(taimide_client):
    client, _, upload_dir = taimide_client
    response = client.post(
        "/api/taimide/v1/reports",
        headers={"X-API-KEY": "test-key"},
        files={"file": ("report.xlsx", b"", "application/vnd.ms-excel")},
        data={"subfolder": "LOT-001"},
    )
    assert response.status_code == 422
    assert "empty" in response.json()["detail"].lower()
    assert not (upload_dir / "reports" / "LOT-001").exists()


def test_upload_report_too_large(taimide_client):
    client, _, upload_dir = taimide_client
    large_data = b"x" * (10 * 1024 * 1024 + 1)
    response = client.post(
        "/api/taimide/v1/reports",
        headers={"X-API-KEY": "test-key"},
        files={"file": ("report.xlsx", large_data, "application/vnd.ms-excel")},
        data={"subfolder": "LOT-001"},
    )
    assert response.status_code == 413
    assert "exceeds" in response.json()["detail"]
    assert not (upload_dir / "reports" / "LOT-001").exists()

# ---------------------------------------------------------------------------
# Tests: taimide startup validation
# ---------------------------------------------------------------------------


def test_lifespan_taimide_valid(temp_dirs):
    import asyncio
    template_dir, upload_dir = temp_dirs

    custom_settings = Settings()
    custom_settings.KEENCHIC_EDITION = "taimide"
    custom_settings.KEENCHIC_TAIMIDE_TEMPLATE_DIR = str(template_dir)
    custom_settings.KEENCHIC_TAIMIDE_UPLOAD_DIR = str(upload_dir)

    # Ensure upload subdirs do not exist before startup
    photos_dir = upload_dir / "photos"
    reports_dir = upload_dir / "reports"
    if photos_dir.exists():
        shutil.rmtree(photos_dir)
    if reports_dir.exists():
        shutil.rmtree(reports_dir)

    async def run_lifespan():
        async with main_mod.lifespan(FastAPI()):
            pass

    with patch("main.settings", custom_settings):
        asyncio.run(run_lifespan())

    assert photos_dir.is_dir()
    assert reports_dir.is_dir()


def test_lifespan_taimide_missing_template_dir(temp_dirs):
    import asyncio
    _, upload_dir = temp_dirs

    custom_settings = Settings()
    custom_settings.KEENCHIC_EDITION = "taimide"
    custom_settings.KEENCHIC_TAIMIDE_TEMPLATE_DIR = ""
    custom_settings.KEENCHIC_TAIMIDE_UPLOAD_DIR = str(upload_dir)

    async def run_lifespan():
        async with main_mod.lifespan(FastAPI()):
            pass

    with patch("main.settings", custom_settings):
        with pytest.raises(RuntimeError) as exc_info:
            asyncio.run(run_lifespan())
        assert "KEENCHIC_TAIMIDE_TEMPLATE_DIR is required" in str(exc_info.value)


def test_lifespan_taimide_invalid_template_dir(temp_dirs):
    import asyncio
    _, upload_dir = temp_dirs

    custom_settings = Settings()
    custom_settings.KEENCHIC_EDITION = "taimide"
    custom_settings.KEENCHIC_TAIMIDE_TEMPLATE_DIR = "/nonexistent/path/here"
    custom_settings.KEENCHIC_TAIMIDE_UPLOAD_DIR = str(upload_dir)

    async def run_lifespan():
        async with main_mod.lifespan(FastAPI()):
            pass

    with patch("main.settings", custom_settings):
        with pytest.raises(RuntimeError) as exc_info:
            asyncio.run(run_lifespan())
        assert "is not a valid directory" in str(exc_info.value)


def test_lifespan_taimide_missing_upload_dir(temp_dirs):
    import asyncio
    template_dir, _ = temp_dirs

    custom_settings = Settings()
    custom_settings.KEENCHIC_EDITION = "taimide"
    custom_settings.KEENCHIC_TAIMIDE_TEMPLATE_DIR = str(template_dir)
    custom_settings.KEENCHIC_TAIMIDE_UPLOAD_DIR = ""

    async def run_lifespan():
        async with main_mod.lifespan(FastAPI()):
            pass

    with patch("main.settings", custom_settings):
        with pytest.raises(RuntimeError) as exc_info:
            asyncio.run(run_lifespan())
        assert "KEENCHIC_TAIMIDE_UPLOAD_DIR is required" in str(exc_info.value)



# ---------------------------------------------------------------------------
# Tests: edition isolation
# ---------------------------------------------------------------------------


def test_standard_edition_no_taimide_endpoint():
    # Save original settings
    orig_api_key = settings.KEENCHIC_API_KEY
    orig_edition = settings.KEENCHIC_EDITION

    # Mutate settings directly
    settings.KEENCHIC_API_KEY = "test-key"
    settings.KEENCHIC_EDITION = "standard"

    try:
        # Setup standard app (router included, but taimide_router not included)
        app = FastAPI()
        from keenchic.api.router import router

        app.include_router(router)
        if settings.KEENCHIC_EDITION == "taimide":
            from keenchic.api.taimide_router import taimide_router

            app.include_router(taimide_router)

        client = TestClient(app)

        # Standard health check should still work
        response = client.get("/health")
        assert response.status_code == 200

        # Taimide template list endpoint should return 404
        response = client.get(
            "/api/taimide/v1/templates",
            headers={"X-API-KEY": "test-key"},
        )
        assert response.status_code == 404

        # Taimide photos endpoint should return 404
        response = client.post(
            "/api/taimide/v1/photos",
            headers={"X-API-KEY": "test-key"},
            files={"file": ("photo.jpg", b"fake-data", "image/jpeg")},
        )
        assert response.status_code == 404
    finally:
        # Restore settings
        settings.KEENCHIC_API_KEY = orig_api_key
        settings.KEENCHIC_EDITION = orig_edition
