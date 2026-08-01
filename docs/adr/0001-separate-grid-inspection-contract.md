# Separate Cell and Grid Inspection Contracts

Grid Inspection will use the existing `POST /api/v1/inspect` route with `X-Inspection-Name: ocr/meter-table-grid`, while Cell Inspection keeps `ocr/meter-table` and its current request and response contracts unchanged. This preserves existing clients and the shared inspection routing architecture; adding an output mode to Cell Inspection would give one inspection identity multiple response shapes, while adding another HTTP route would duplicate routing behavior.

Grid Inspection is a shared OCR capability rather than a Taimide-only route. It is available in any edition whose wheel includes the meter-table adapter and submodule files; Taimide-specific file-management endpoints remain in `taimide_router.py`.

Grid Inspection supports the same backends as Cell Inspection: GPU and AUTO prefer TensorRT with OpenVINO fallback, while CPU and OpenVINO use OpenVINO directly. Both backends expose the same Grid Result contract.
