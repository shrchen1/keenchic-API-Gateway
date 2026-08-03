import base64
import sys
import types
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from keenchic.inspections.adapters.ocr.meter_table_grid import MeterTableGridAdapter


def _loaded_adapter(proc: Callable[..., dict[str, Any]]) -> MeterTableGridAdapter:
    adapter = MeterTableGridAdapter()
    adapter._proc = proc
    adapter._detect_crop = object()
    adapter._model_crop = object()
    adapter._detect_num = object()
    adapter._model_num = object()
    return adapter


def test_grid_adapter_accepts_only_grid_fields() -> None:
    assert MeterTableGridAdapter.accepted_kwargs() == {"include_diag", "table_size"}


@pytest.mark.parametrize("table_size", ["[4,4]", "4,4"])
def test_grid_run_returns_submodule_matrix_unchanged(table_size: str) -> None:
    expected = [
        ["0.21", "76.19", "0.05", "0.16"],
        ["0.21", "76.19", "0.05", "0.16"],
        ["0.22", "77.27", "0.05", "0.17"],
        ["0.12", "75.00", "0.03", "0.09"],
    ]
    captured: dict[str, Any] = {}

    def proc(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"result": 0, "pred_text_L": expected}

    result = _loaded_adapter(proc).run(np.zeros((8, 8, 3), dtype=np.uint8), table_size=table_size)

    assert result == {"result": 0, "pred_text_L": expected}
    assert captured["detection_args"] == {"settings": {"table_size": [4, 4]}}


def test_grid_run_supports_single_cell_grid() -> None:
    expected = [["0.21"]]
    captured: dict[str, Any] = {}

    def proc(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"result": 0, "pred_text_L": expected}

    result = _loaded_adapter(proc).run(
        np.zeros((8, 8, 3), dtype=np.uint8), table_size="[1,1]"
    )

    assert result == {"result": 0, "pred_text_L": expected}
    assert captured["detection_args"] == {"settings": {"table_size": [1, 1]}}


def test_grid_run_preserves_unrecognized_cells_as_ng() -> None:
    expected = [["0.21", "N/G", "0.05", "0.16"]]
    adapter = _loaded_adapter(lambda **_: {"result": 0, "pred_text_L": expected})

    result = adapter.run(np.zeros((8, 8, 3), dtype=np.uint8), table_size="1,4")

    assert result == {"result": 0, "pred_text_L": expected}


def test_grid_run_maps_empty_submodule_result_to_detection_failure() -> None:
    adapter = _loaded_adapter(lambda **_: {"result": 2, "pred_text_L": []})

    result = adapter.run(np.zeros((8, 8, 3), dtype=np.uint8), table_size="4,4")

    assert result == {"result": 2, "pred_text_L": []}


def test_grid_run_includes_base64_diagnostic_png_when_requested() -> None:
    diagnostic = np.zeros((8, 8, 3), dtype=np.uint8)
    adapter = _loaded_adapter(
        lambda **_: {
            "result": 0,
            "pred_text_L": [["0.21", "76.19"]],
            "diag_img": diagnostic,
        }
    )

    result = adapter.run(
        np.zeros((8, 8, 3), dtype=np.uint8),
        table_size="1,2",
        include_diag=True,
    )

    assert result["result"] == 0
    assert result["pred_text_L"] == [["0.21", "76.19"]]
    assert base64.b64decode(result["diag_img"]).startswith(b"\x89PNG")


def test_grid_run_propagates_runtime_failure() -> None:
    def proc(**_: Any) -> dict[str, Any]:
        raise RuntimeError("inference failed")

    adapter = _loaded_adapter(proc)

    with pytest.raises(RuntimeError, match="inference failed"):
        adapter.run(np.zeros((8, 8, 3), dtype=np.uint8), table_size="4,4")


@pytest.mark.parametrize(
    "error", [TypeError("bad model output"), OSError("device lost")]
)
def test_grid_run_wraps_unexpected_inference_failure(error: Exception) -> None:
    def proc(**_: Any) -> dict[str, Any]:
        raise error

    adapter = _loaded_adapter(proc)

    with pytest.raises(RuntimeError, match="Grid inference failed"):
        adapter.run(np.zeros((8, 8, 3), dtype=np.uint8), table_size="4,4")


def test_grid_run_rejects_unloaded_adapter() -> None:
    with pytest.raises(RuntimeError, match="Models not loaded"):
        MeterTableGridAdapter().run(
            np.zeros((8, 8, 3), dtype=np.uint8), table_size="4,4"
        )


@pytest.mark.parametrize(
    "table_size",
    [
        None,
        "[9,4]",
        "[4,9]",
        "[0,4]",
        "[4,0]",
        "not-a-size",
        "[4.0,4]",
        "[true,4]",
    ],
)
def test_grid_run_rejects_invalid_table_size(table_size: str | None) -> None:
    adapter = _loaded_adapter(lambda **_: {"result": 0, "pred_text_L": []})

    with pytest.raises(ValueError):
        adapter.run(np.zeros((8, 8, 3), dtype=np.uint8), table_size=table_size)


def test_grid_run_rejects_cell_coordinate() -> None:
    adapter = _loaded_adapter(lambda **_: {"result": 0, "pred_text_L": []})

    with pytest.raises(ValueError, match="input_coords"):
        adapter.run(
            np.zeros((8, 8, 3), dtype=np.uint8),
            table_size="4,4",
            input_coords="1,1",
        )


def test_grid_openvino_imports_shared_detection_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detect_module = types.ModuleType("model_detect_openvino_512")
    detect_module.detect_smp = object()
    detect_module.detect_yolo12 = object()
    detect_module.get_smp_model = object()
    detect_module.get_yolo12_model = object()
    proc_module = types.ModuleType("procd_table_L")
    proc_module.proc = object()
    monkeypatch.setitem(sys.modules, "model_detect_openvino_512", detect_module)
    monkeypatch.setitem(sys.modules, "procd_table_L", proc_module)

    imports = MeterTableGridAdapter()._import_openvino()

    assert imports["backend"] == "openvino"
    assert imports["proc"] is proc_module.proc


def test_grid_tensorrt_imports_shared_detection_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeContext:
        def push(self) -> None:
            return None

        def pop(self) -> None:
            return None

    class FakeDevice:
        def make_context(self) -> FakeContext:
            return FakeContext()

    cuda_module = types.ModuleType("pycuda.driver")
    cuda_module.init = lambda: None
    cuda_module.Device = lambda _index: FakeDevice()
    pycuda_module = types.ModuleType("pycuda")
    pycuda_module.__path__ = []
    monkeypatch.setitem(sys.modules, "pycuda", pycuda_module)
    monkeypatch.setitem(sys.modules, "pycuda.driver", cuda_module)

    detect_module = types.ModuleType("model_detect_trt_512")
    detect_module.detect_smp = object()
    detect_module.detect_yolo12 = object()
    detect_module.get_smp_model = object()
    detect_module.get_yolo12_model = object()
    proc_module = types.ModuleType("procd_table_L")
    proc_module.proc = object()
    monkeypatch.setitem(sys.modules, "model_detect_trt_512", detect_module)
    monkeypatch.setitem(sys.modules, "procd_table_L", proc_module)

    imports = MeterTableGridAdapter()._import_trt()

    assert imports["backend"] == "tensorrt"
    assert imports["proc"] is proc_module.proc
