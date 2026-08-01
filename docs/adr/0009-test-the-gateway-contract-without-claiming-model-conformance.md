# Test the Gateway Contract Without Claiming Model Conformance

Grid Inspection completion requires adapter unit tests for validation, unchanged matrix pass-through, `"N/G"`, empty results, technical exceptions, and diagnostics; HTTP tests for routing, 422 and 503 boundaries, and Cell Inspection compatibility; and build tests covering `procd_table_L` plus OpenVINO and TensorRT imports in applicable editions. Mocked OCR tests verify the Gateway contract only and must not be presented as evidence that the real model conforms to the upstream assumptions in ADR 0008.
