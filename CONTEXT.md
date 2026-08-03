# Inspection

This context defines the inspection concepts exposed by the API service.

## Meter Table OCR

**Meter Table**:
A meter display whose readings are arranged into a known number of rows and columns.
_Avoid_: Table image, meter grid

**Cell Inspection**:
An inspection that extracts the reading at one requested position in a Meter Table.
_Avoid_: Single-table inspection, legacy inspection

**Grid Inspection**:
An inspection that receives one Meter Table image and its expected dimensions, then extracts every cell reading as a rectangular grid.
_Avoid_: Batch inspection, full-image inspection, list inspection

**Grid Result**:
A row-major rectangular matrix in which each outer element represents one Meter Table row and each inner element represents that row's cell readings from left to right.
_Avoid_: Flat result, cell list, batch result

**Unrecognized Cell**:
An expected Meter Table position for which Grid Inspection cannot extract a reading.
_Avoid_: Missing cell, empty cell, unavailable cell

**Grid Inspection Failure**:
A Grid Inspection outcome in which no Grid Result can be produced from the image.
_Avoid_: Partial failure, Unrecognized Cell
