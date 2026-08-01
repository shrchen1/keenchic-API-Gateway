import cv2
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import Response

from keenchic.api.router import router
from keenchic.core.config import settings
from keenchic.core.inspection_manager import inspection_manager


@pytest.fixture
def grid_client(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[TestClient, list[dict[str, object]]]:
    app = FastAPI()
    app.include_router(router)
    monkeypatch.setattr(settings, "KEENCHIC_API_KEY", "test-key")

    calls: list[dict[str, object]] = []

    async def run(
        inspection_name: str, image: np.ndarray, **kwargs: object
    ) -> dict[str, object]:
        calls.append({"inspection_name": inspection_name, "image": image, **kwargs})
        return {
            "result": 0,
            "pred_text_L": [["0.21", "76.19", "0.05", "0.16"]],
        }

    monkeypatch.setattr(inspection_manager, "run", run)
    return TestClient(app), calls


def _image_upload() -> tuple[str, bytes, str]:
    ok, encoded = cv2.imencode(".png", np.zeros((8, 8, 3), dtype=np.uint8))
    assert ok
    return "table.png", encoded.tobytes(), "image/png"


def _post_grid(client: TestClient, **form: str) -> Response:
    return client.post(
        "/api/v1/inspect",
        headers={
            "X-API-KEY": "test-key",
            "X-Inspection-Name": "ocr/meter-table-grid",
        },
        files={"image": _image_upload()},
        data=form,
    )


def test_grid_route_returns_matrix_and_forwards_table_size(
    grid_client: tuple[TestClient, list[dict[str, object]]],
) -> None:
    client, calls = grid_client

    response = _post_grid(client, table_size="[4,4]")

    assert response.status_code == 200
    assert response.json() == {
        "result": 0,
        "pred_text_L": [["0.21", "76.19", "0.05", "0.16"]],
    }
    assert calls[0]["inspection_name"] == "ocr/meter-table-grid"
    assert calls[0]["table_size"] == "[4,4]"


def test_grid_route_requires_table_size(
    grid_client: tuple[TestClient, list[dict[str, object]]],
) -> None:
    client, calls = grid_client

    response = _post_grid(client)

    assert response.status_code == 422
    assert "table_size" in response.json()["detail"]
    assert calls == []


def test_grid_route_rejects_input_coords(
    grid_client: tuple[TestClient, list[dict[str, object]]],
) -> None:
    client, calls = grid_client

    response = _post_grid(client, table_size="4,4", input_coords="1,1")

    assert response.status_code == 422
    assert "input_coords" in response.json()["detail"]
    assert calls == []


@pytest.mark.parametrize("table_size", ["[9,4]", "[4,9]", "[1,1]", "invalid"])
def test_grid_route_rejects_unsupported_table_size(
    grid_client: tuple[TestClient, list[dict[str, object]]], table_size: str
) -> None:
    client, calls = grid_client

    response = _post_grid(client, table_size=table_size)

    assert response.status_code == 422
    assert "table_size" in response.json()["detail"]
    assert calls == []


def test_grid_route_rejects_other_inspection_fields(
    grid_client: tuple[TestClient, list[dict[str, object]]],
) -> None:
    client, calls = grid_client

    response = _post_grid(client, table_size="4,4", YMD_option="1")

    assert response.status_code == 422
    assert "YMD_option" in response.json()["detail"]
    assert calls == []


def test_grid_route_maps_runtime_failure_to_503(
    grid_client: tuple[TestClient, list[dict[str, object]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _ = grid_client

    async def fail(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise RuntimeError("backend unavailable")

    monkeypatch.setattr(inspection_manager, "run", fail)
    response = _post_grid(client, table_size="4,4")

    assert response.status_code == 503
    assert response.json()["detail"] == "backend unavailable"


def test_grid_route_maps_unexpected_failure_to_503(
    grid_client: tuple[TestClient, list[dict[str, object]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _ = grid_client

    async def fail(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise OSError("device lost")

    monkeypatch.setattr(inspection_manager, "run", fail)
    response = _post_grid(client, table_size="4,4")

    assert response.status_code == 503
    assert response.json()["detail"] == "device lost"


def test_cell_route_keeps_cell_fields_compatible(
    grid_client: tuple[TestClient, list[dict[str, object]]],
) -> None:
    client, calls = grid_client

    response = client.post(
        "/api/v1/inspect",
        headers={
            "X-API-KEY": "test-key",
            "X-Inspection-Name": "ocr/meter-table",
        },
        files={"image": _image_upload()},
        data={"input_coords": "1,1", "table_size": "2,2"},
    )

    assert response.status_code == 200
    assert calls[0]["inspection_name"] == "ocr/meter-table"
    assert calls[0]["input_coords"] == "1,1"
    assert calls[0]["table_size"] == "2,2"


def test_cell_route_keeps_unexpected_exception_behavior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = FastAPI()
    app.include_router(router)
    monkeypatch.setattr(settings, "KEENCHIC_API_KEY", "test-key")

    async def fail(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise OSError("device lost")

    monkeypatch.setattr(inspection_manager, "run", fail)
    client = TestClient(app, raise_server_exceptions=False)
    response = client.post(
        "/api/v1/inspect",
        headers={
            "X-API-KEY": "test-key",
            "X-Inspection-Name": "ocr/meter-table",
        },
        files={"image": _image_upload()},
        data={"input_coords": "1,1", "table_size": "2,2"},
    )

    assert response.status_code == 500
