# Separate Grid Results from HTTP Errors

Grid Inspection returns HTTP 200 with `result=0` for a non-empty `pred_text_L` and HTTP 200 with `result=2` plus `pred_text_L=[]` when inference completes but no Grid Result can be produced. Missing or invalid inputs return HTTP 422, while model loading, backend, and unexpected runtime failures return HTTP 503. Technical failures must not be represented as image-level detection failures; existing API-key authentication behavior remains unchanged.
