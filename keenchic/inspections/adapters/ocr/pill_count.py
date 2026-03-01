from __future__ import annotations

import base64
import os
import sys
from typing import Any, Optional

import numpy as np

from keenchic.inspections.base import InspectionAdapter
from keenchic.inspections.result_codes import InspectionResultCode

# Absolute path to pill_count_st package inside the submodule.
_SUBMODULE_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "ocr", "pill_count_st")
)


def _ensure_submodule_on_path() -> None:
    if _SUBMODULE_DIR not in sys.path:
        sys.path.insert(0, _SUBMODULE_DIR)
    # procd_pill.py imports streamlit at module level but never uses it in proc().
    # Mock it to avoid ImportError in non-UI gateway environments.
    if "streamlit" not in sys.modules:
        try:
            import streamlit  # noqa: F401
        except ImportError:
            from unittest.mock import MagicMock
            sys.modules["streamlit"] = MagicMock()


def _b64_png(img: np.ndarray) -> str:
    import cv2

    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise ValueError("Failed to encode diagnostic image")
    return base64.b64encode(buf.tobytes()).decode("ascii")


class PillCountAdapter(InspectionAdapter):
    """Adapter wrapping the pill_count_st inference engine.

    Performs instance segmentation to count pills in an image.
    Supports OpenVINO (CPU) and TensorRT (GPU) backends.
    """

    def __init__(self) -> None:
        self._proc: Any = None
        self._detect_crop: Any = None
        self._get_crop_model: Any = None
        self._model_crop: Any = None
        self._cuda_context: Any = None
        self._backend_active: str | None = None

    # ------------------------------------------------------------------
    # InspectionAdapter interface
    # ------------------------------------------------------------------

    def load_models(self, backend: str) -> None:
        _ensure_submodule_on_path()

        desired = backend.strip().lower()
        try_order = (
            ["tensorrt", "openvino"] if desired in ("auto", "tensorrt") else ["openvino", "tensorrt"]
        )

        last_exc: Optional[Exception] = None
        for choice in try_order:
            try:
                imports = self._import_trt() if choice == "tensorrt" else self._import_openvino()
                self._activate(imports)
                last_exc = None
                break
            except Exception as exc:
                print(f"PillCountAdapter: skipping {choice} backend: {exc}")
                last_exc = exc

        if last_exc is not None:
            raise RuntimeError(f"All backends failed for pill-count: {last_exc}") from last_exc

        try:
            self._model_crop = self._get_crop_model()
        except Exception as exc:
            self.unload_models()
            raise RuntimeError(f"Model weight loading failed: {exc}") from exc

    def unload_models(self) -> None:
        self._model_crop = None
        self._proc = None
        self._detect_crop = None
        self._get_crop_model = None  # type: ignore[assignment]
        if self._cuda_context is not None:
            try:
                self._cuda_context.pop()
            except Exception:
                pass
            self._cuda_context = None
        self._backend_active = None

    def run(self, image: np.ndarray, **kwargs) -> dict:
        if self._proc is None:
            raise RuntimeError("Models not loaded — call load_models() first")

        include_diag: bool = bool(kwargs.get("include_diag", False))
        debug: bool = bool(kwargs.get("debug", False))

        cuda_pushed = False
        if self._cuda_context is not None and self._backend_active == "tensorrt":
            self._cuda_context.push()
            cuda_pushed = True

        try:
            result = self._proc(
                image=image,
                detection_args={"settings": {}},
                models=[self._detect_crop, self._model_crop],
                debug=debug,
            )
        finally:
            if cuda_pushed:
                self._cuda_context.pop()

        return self._build_payload(result, include_diag)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _import_openvino(self) -> dict:
        from model_openvino_yolo import (  # type: ignore[import]
            detect_yolov12Seg_512,
            get_model_yolov12Seg_512,
        )
        from procd_pill import proc  # type: ignore[import]

        return {
            "backend": "openvino",
            "get_crop_model": get_model_yolov12Seg_512,
            "detect_crop": detect_yolov12Seg_512,
            "proc": proc,
            "cuda_context": None,
        }

    def _import_trt(self) -> dict:
        import pycuda.driver as cuda  # type: ignore[import]

        cuda.init()
        ctx = cuda.Device(0).make_context()

        from model_trt_yolo import (  # type: ignore[import]
            detect_yolov12Seg_512,
            get_model_yolov12Seg_512,
        )
        from procd_pill import proc  # type: ignore[import]

        return {
            "backend": "tensorrt",
            "get_crop_model": get_model_yolov12Seg_512,
            "detect_crop": detect_yolov12Seg_512,
            "proc": proc,
            "cuda_context": ctx,
        }

    def _activate(self, imports: dict) -> None:
        self._detect_crop = imports["detect_crop"]
        self._get_crop_model = imports["get_crop_model"]
        self._proc = imports["proc"]
        self._cuda_context = imports["cuda_context"]
        self._backend_active = imports["backend"]

    def _build_payload(self, result: dict, include_diag: bool) -> dict:
        payload: dict = {
            "result": int(result.get("result", InspectionResultCode.DETECTION_FAILED)),
            "pill_counts": int(result.get("pill_counts", 0)),
        }
        if include_diag:
            for key in ("diag_img", "diag_img_en"):
                if result.get(key) is not None:
                    try:
                        payload[key] = _b64_png(result[key])
                    except Exception:
                        payload[key] = None
        return payload
