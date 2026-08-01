# Extend the Unified Inspect Response for Grid Results

The shared `InspectResponse` schema will add `pred_text_L: list[list[str]]` with an empty-list default while keeping `pred_text: str` unchanged. Cell Inspection responses continue to contain `pred_text`, and Grid Inspection responses contain `pred_text_L`; widening `pred_text` to a union would weaken existing client contracts, while returning an undocumented field would make OpenAPI and generated clients inaccurate.
