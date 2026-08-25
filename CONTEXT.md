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

## Report Uploads

**Report Group**:
A client-chosen grouping of related Excel inspection reports, typically reports for the same batch produced by different instruments.
_Avoid_: Subfolder, batch folder

## Jetson Deployment

**Build Jetson**:
A Jetson Orin device that receives the complete release inputs and produces a Jetson Release Wheel.
_Avoid_: Target Jetson, build server

**Target Jetson**:
A supported Jetson Orin production device that installs release artifacts and does not retain the application source repository.
_Avoid_: Build Jetson, development Jetson

**Jetson Release Wheel**:
The platform-specific application package produced on a Build Jetson for installation on a Target Jetson.
_Avoid_: Portable Jetson wheel, source bundle

**System GPU Stack**:
The CUDA and TensorRT platform software provisioned independently on a Target Jetson and excluded from the Jetson Release Wheel.
_Avoid_: Bundled CUDA, wheel CUDA

**GPU Python Binding**:
The PyCUDA package installed separately from the Jetson Release Wheel and linked to the System GPU Stack.
_Avoid_: Bundled PyCUDA, system CUDA

**Supported Orin Runtime Profile**:
The approved Jetson Orin hardware and system-software combination that defines binary compatibility for a Jetson Release Wheel and its model engines.
_Avoid_: Any Orin, generic aarch64 environment

**Taimide Algorithm Set**:
The Cell Inspection and Grid Inspection capabilities included in a Taimide Jetson Release Wheel.
_Avoid_: All registered algorithms, Taimide routes

**Taimide Jetson Profile**:
The release configuration that combines the Taimide Algorithm Set with the Supported Orin Runtime Profile and requires TensorRT execution.
_Avoid_: Taimide edition, generic Jetson build

**Approved TensorRT Engine**:
A prebuilt model engine admitted to the formal release input set for the Supported Orin Runtime Profile; matching candidates are selected by their filename version and may be explicitly rolled back.
_Avoid_: Build output, mtime-selected engine, bundled CUDA

**Release Version**:
The calendar-based identity assigned to one published Jetson Release Wheel revision, with an increasing numeric revision for additional releases on the same day.
_Avoid_: Build timestamp, model filename, Git revision
