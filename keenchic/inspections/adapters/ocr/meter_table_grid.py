from __future__ import annotations

import json
import re
from typing import Any

import numpy as np

from keenchic.inspections.adapters.ocr.meter_table import MeterTableAdapter, _b64_png
from keenchic.inspections.result_codes import InspectionResultCode

GRID_INSPECTION_NAME = "ocr/meter-table-grid"
_SUPPORTED_ROW_COUNTS = frozenset((*range(1, 9), 15))
_MAX_COLUMN_COUNT = 8


def _parse_grid_table_size(raw: str | None) -> list[int]:
    if raw is None:
        raise ValueError(f"table_size is required for '{GRID_INSPECTION_NAME}'")

    try:
        rows, columns = _parse_strict_grid_coords(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(
            "invalid table_size: expected '[rows,columns]' or 'rows,columns'"
        ) from exc

    if rows not in _SUPPORTED_ROW_COUNTS:
        raise ValueError(
            f"invalid table_size: rows must be 1-8 or 15 for '{GRID_INSPECTION_NAME}'"
        )
    if columns < 1 or columns > _MAX_COLUMN_COUNT:
        raise ValueError(
            "invalid table_size: columns must be between 1 and 8 for "
            f"'{GRID_INSPECTION_NAME}'"
        )
    if rows == 1 and columns == 1:
        raise ValueError(
            "table_size '[1,1]' belongs to 'ocr/meter-table', not "
            f"'{GRID_INSPECTION_NAME}'"
        )
    return [rows, columns]


def _parse_strict_grid_coords(raw: str) -> tuple[int, int]:
    """Parse exactly two integer coordinates for the Grid contract."""
    value = raw.strip()
    if value.startswith("["):
        parsed = json.loads(value)
        if (
            not isinstance(parsed, list)
            or len(parsed) != 2
            or any(
                isinstance(item, bool) or not isinstance(item, int)
                for item in parsed
            )
        ):
            raise ValueError("table_size must contain two integers")
        return parsed[0], parsed[1]

    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 2 or any(not re.fullmatch(r"[+-]?\d+", part) for part in parts):
        raise ValueError("table_size must contain two integers")
    return int(parts[0]), int(parts[1])


class MeterTableGridAdapter(MeterTableAdapter):
    """Adapter for returning every detected meter-table cell."""

    @classmethod
    def accepted_kwargs(cls) -> set[str]:
        return {"include_diag", "table_size"}

    @classmethod
    def validate_kwargs(cls, kwargs: dict[str, Any]) -> None:
        _parse_grid_table_size(kwargs.get("table_size"))

    def run(self, image: np.ndarray, **kwargs: Any) -> dict[str, Any]:
        if "input_coords" in kwargs:
            raise ValueError(f"input_coords is not accepted by '{GRID_INSPECTION_NAME}'")

        table_size = _parse_grid_table_size(kwargs.get("table_size"))
        include_diag = bool(kwargs.get("include_diag", False))
        if self._proc is None:
            raise RuntimeError("Models not loaded — call load_models() first")

        try:
            result = self._invoke_proc(
                image=image,
                detection_args={"settings": {"table_size": table_size}},
            )
        except Exception as exc:
            raise RuntimeError(f"Grid inference failed: {exc}") from exc

        pred_text_L = result.get("pred_text_L", [])
        payload: dict[str, Any] = {
            "result": (
                InspectionResultCode.SUCCESS
                if pred_text_L
                else InspectionResultCode.DETECTION_FAILED
            ),
            "pred_text_L": pred_text_L,
        }
        if include_diag and result.get("diag_img") is not None:
            try:
                payload["diag_img"] = _b64_png(result["diag_img"])
            except Exception:
                payload["diag_img"] = None
        return payload

    def _import_openvino(self) -> dict[str, Any]:
        from model_detect_openvino_512 import (  # type: ignore[import]
            detect_smp,
            detect_yolo12,
            get_smp_model,
            get_yolo12_model,
        )
        from procd_table_L import proc  # type: ignore[import]

        return {
            "backend": "openvino",
            "get_crop_model": get_smp_model,
            "detect_crop": detect_smp,
            "get_num_model": get_yolo12_model,
            "detect_num": detect_yolo12,
            "proc": proc,
            "cuda_context": None,
        }

    def _import_trt(self) -> dict[str, Any]:
        import pycuda.driver as cuda  # type: ignore[import]

        cuda.init()
        ctx = cuda.Device(0).make_context()

        from model_detect_trt_512 import (  # type: ignore[import]
            detect_smp,
            detect_yolo12,
            get_smp_model,
            get_yolo12_model,
        )
        from procd_table_L import proc  # type: ignore[import]

        return {
            "backend": "tensorrt",
            "get_crop_model": get_smp_model,
            "detect_crop": detect_smp,
            "get_num_model": get_yolo12_model,
            "detect_num": detect_yolo12,
            "proc": proc,
            "cuda_context": ctx,
        }
