# Trust the Grid Submodule Contract

Grid Inspection design and implementation will treat the submodule author's stated behavior as the upstream contract: non-empty `pred_text_L` is a `table_size`-shaped row-major matrix, unrecognized or padded cells contain `"N/G"`, smaller requested dimensions crop the result, and an uninspectable image returns an empty list. API Gateway will not compensate for differences observed in the current `5a5d6fc` implementation; real-model conformance remains unverified rather than blocking the Gateway work.
