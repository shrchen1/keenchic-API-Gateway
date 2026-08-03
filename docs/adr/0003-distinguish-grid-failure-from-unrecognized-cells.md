# Distinguish Grid Failure from Unrecognized Cells

Grid Inspection will return `result=0` whenever `pred_text_L` is a non-empty matrix, including matrices containing one or more `"N/G"` cells. It will return `result=2` only when the submodule returns `pred_text_L=[]`, meaning no Grid Result could be produced. This lets clients retain successfully recognized cells while treating `"N/G"` as a cell-level outcome rather than an image-level failure.
