# Require `table_size` for Grid Inspection

Grid Inspection requires `table_size` in every request and has no default dimensions. Missing or malformed dimensions will return HTTP 422 before inference because the value defines the expected `pred_text_L` shape and controls the submodule's padding and cropping behavior; silently defaulting to `[2, 2]` could return a plausible but incorrectly cropped result.

Valid dimensions use a row count from 1 through 8 or exactly 15 and a column count from 1 through 8. The `[1, 1]` size is rejected for Grid Inspection because that request belongs to Cell Inspection. Unsupported dimensions return HTTP 422 before inference. The column limit also prevents excessive `"N/G"` padding from consuming unbounded CPU and memory.

As a multipart form field, `table_size` accepts either `"[rows,columns]"` or `"rows,columns"`; both forms are parsed into the same two-integer dimensions before being passed to the submodule.

Grid Inspection does not accept `input_coords` because it always returns the complete Grid Result. Supplying that Cell Inspection field returns HTTP 422 through the adapter field whitelist rather than being silently ignored.
