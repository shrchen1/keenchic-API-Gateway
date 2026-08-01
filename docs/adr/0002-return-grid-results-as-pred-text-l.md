# Return Grid Results as `pred_text_L`

A successful Grid Inspection will return the OCR submodule's `pred_text_L` as a row-major list of lists whose dimensions correspond to the requested `table_size`. The API will preserve the submodule's cell text without renaming the field or converting the successful result into another representation, so a `table_size` of `[4, 4]` maps to four row lists containing four cell readings each.

The API depends on the submodule contract to place `"N/G"` at every unrecognized or padded cell, return an empty list when the image cannot be inspected, and crop results when `table_size` is smaller than the detected table. API Gateway will not reconstruct positions, pad cells, or crop the matrix.

Grid Inspection supports the shared `include_diag` query parameter, defaulting to `false`. When requested and available, `diag_img` is returned as a Base64-encoded PNG; an encoding failure produces `diag_img=null` without changing the OCR result.

The normal Grid Inspection payload contains only `result` and `pred_text_L`; `include_diag=true` additionally includes `diag_img`. An image-level failure returns `result=2` with `pred_text_L=[]`. The payload does not include Cell Inspection's `pred_text`, echo `table_size`, or add a separate message field.
