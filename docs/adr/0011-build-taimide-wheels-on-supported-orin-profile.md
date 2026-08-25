# Build Taimide wheels on one supported Orin profile

The formal Taimide Jetson release is built on a separate Build Jetson that
matches the Target Jetson runtime: Jetson Orin compute capability 8.7, L4T
R36.4.4, Python 3.10, CUDA 12.6, TensorRT 10.3.x, and PyCUDA 2026.1. The build
script rejects mismatched environments instead of claiming broader Jetson
compatibility.

The release contains only `ocr/meter-table` and `ocr/meter-table-grid`. It
packages one prebuilt head engine and one prebuilt yolo engine, selected by the
latest `YYYYMMDD` in each filename unless an explicit dated rollback is given.
The build neither creates TensorRT engines nor uses file modification time as a
release decision.

CUDA and TensorRT remain part of the system GPU stack. PyCUDA is an exact wheel
dependency but is provisioned separately because it must link to that stack; it
is not embedded in the application wheel. Other Python dependencies may be
installed from binary PyPI wheels under the generated constraints.

Each calendar-versioned release is immutable and includes the application
wheel, an embedded and sidecar build manifest, target constraints, and
checksums. The build records source and input hashes without requiring a clean
Git checkout, then proves the packaged engines can deserialize and the wheel
can be installed and invoked from a temporary environment outside the source
repository. This supports source-free Target Jetsons while keeping the exact
release inputs auditable.
