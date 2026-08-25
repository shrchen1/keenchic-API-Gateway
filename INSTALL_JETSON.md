# Keenchic API Gateway - Taimide Jetson Orin Release Guide

This guide covers producing a release on a Build Jetson and installing it on a
separate Target Jetson. Both devices must match the supported NVIDIA Jetson
Orin production profile. The release wheel contains only the Taimide Cell and
Grid inspection algorithms; the Target Jetson does not need a source
repository.

---

## Prerequisites

| Item | Requirement |
|------|-------------|
| Hardware | NVIDIA Jetson Orin, compute capability 8.7 |
| OS | L4T R36.4.4 (Ubuntu 22.04 based) |
| Python | 3.10 |
| System GPU stack | CUDA 12.6, TensorRT 10.3.x |
| Separately provisioned binding | PyCUDA 2026.1, built for the target system GPU stack |

## Build the Release on the Build Jetson

The Build Jetson must contain the complete source repository, approved
prebuilt TensorRT engines, and the build virtual environment at
`/home/nvidia/venv_st`. The build does not create TensorRT engines.

`--release-version` is optional. When it is omitted, the script uses the Build
Jetson's local system date and selects the next revision after the highest
same-day release already present in `dist/`.

For a release with an externally assigned version, set it explicitly. Replace
the placeholder before running the commands:

```bash
RELEASE_VERSION="<release-version>"
```

Release versions use `YYYY.M.D`. Additional releases on the same day use
`.1`, `.2`, and so on; `.0` is not used. A release directory is immutable, so
an explicitly selected version is rejected if `dist/${RELEASE_VERSION}`
already exists.

The profile independently selects the newest approved head and yolo engines by
the `YYYYMMDD` date in each filename, not by modification time. For an explicit
model rollback, add one or both overrides to the build command:

```bash
--engine-date head=20260821 --engine-date yolo=20260823
```

Build and run all mandatory verification gates:

```bash
cd /home/nvidia/keenchic-API-Gateway

/home/nvidia/venv_st/bin/python -B build_wheel.py \
  --profile taimide-jetson \
  --release-version "${RELEASE_VERSION}"
```

To let the script select the version automatically instead, omit the version
option:

```bash
cd /home/nvidia/keenchic-API-Gateway
/home/nvidia/venv_st/bin/python -B build_wheel.py --profile taimide-jetson
```

The selected version and final release location are printed in the build
output. Use that version when copying and installing the artifacts.

The command must exit with status `0` and end with:

```text
Wheel built and verified: keenchic_api_gateway-<version>+taimide-cp310-cp310-linux_aarch64.whl
```

The build automatically verifies:

- the fixed Orin/L4T/Python/CUDA/TensorRT/PyCUDA profile;
- deserialization of the selected head and yolo TensorRT engines;
- wheel version, platform tag, dependency metadata, and embedded manifest;
- that only the selected engines are packaged;
- that both Cell and Grid adapters are Cython `.so` files with no matching
  `.py` source in the wheel;
- installation into a temporary source-free virtual environment;
- `keenchic-serve --help` and deserialization of engines from site-packages.

Verify the generated release files:

```bash
cd "/home/nvidia/keenchic-API-Gateway/dist/${RELEASE_VERSION}"
sha256sum --check SHA256SUMS
```

Every entry must report `OK`.

## Files to Copy to the Target Jetson

Copy all four files from one release directory to the Jetson (via USB, SCP,
or any method):

```text
keenchic_api_gateway-<release-version>+taimide-cp310-cp310-linux_aarch64.whl
build-manifest.json
target-constraints.txt
SHA256SUMS
```

For the remaining examples, set the same release version and place the four
files in one directory on the Target Jetson:

```bash
RELEASE_VERSION="<release-version>"
RELEASE_DIR="/home/nvidia/release-${RELEASE_VERSION}"
WHEEL="${RELEASE_DIR}/keenchic_api_gateway-${RELEASE_VERSION}+taimide-cp310-cp310-linux_aarch64.whl"
```

Verify the artifact set before installing:

```bash
cd "${RELEASE_DIR}"
sha256sum --check SHA256SUMS
```

---

## Install on the Target Jetson

### Step 1: Create a Virtual Environment

Open a terminal on the Jetson and run:

```bash
cd /home/nvidia
python3 -m venv --system-site-packages keenchic-env
```

> **Why `--system-site-packages`?**
> JetPack pre-installs TensorRT and other GPU packages into the
> system Python. This flag allows the virtual environment to access those
> packages. Without it, the application will fail because these packages
> are not available on PyPI for aarch64.

### Step 2: Activate the Virtual Environment

```bash
source /home/nvidia/keenchic-env/bin/activate
```

After activation, your terminal prompt will change to show `(keenchic-env)`:

```
(keenchic-env) nvidia@jetson:~$
```

### Step 3: Verify the GPU Runtime

Run the following command to confirm JetPack packages are accessible:

```bash
python3 - <<'PY'
from importlib.metadata import version

import pycuda.driver as cuda
import tensorrt

cuda.init()
device = cuda.Device(0)
assert device.compute_capability() == (8, 7)
assert tensorrt.__version__.startswith("10.3.")
assert version("pycuda") == "2026.1"
print(device.name(), device.compute_capability(), tensorrt.__version__, version("pycuda"))
PY
```

Expected output:

```
Orin (8, 7) 10.3.x 2026.1
```

If you see an `ImportError`, check that your JetPack installation is complete.

PyCUDA is deliberately not bundled in the application wheel. The Target
Jetson administrator must provision PyCUDA 2026.1 against the installed system
CUDA stack before installing the application.

### Step 4: Install and Inspect the Wheel

```bash
cd "${RELEASE_DIR}"
pip install \
  --only-binary=:all: \
  --constraint target-constraints.txt \
  "${WHEEL}"
```

pip will install remaining dependencies from binary wheels under the exact
constraints captured on the Build Jetson. CUDA, TensorRT, and PyCUDA remain
external to the application wheel.

To verify the installation succeeded:

```bash
pip show keenchic-api-gateway
```

You should see `Name: keenchic-api-gateway` and
`Version: <release-version>+taimide`.

Confirm both protected adapters load from compiled extensions:

```bash
cd /tmp
python3 - <<'PY'
import importlib
from pathlib import Path

for name in (
    "keenchic.inspections.adapters.ocr.meter_table",
    "keenchic.inspections.adapters.ocr.meter_table_grid",
):
    module = importlib.import_module(name)
    path = Path(module.__file__).resolve()
    assert path.suffix == ".so", path
    print(path)
PY
```

### Step 5: Configure Taimide Runtime Directories and API Key

The gateway requires an API key to authenticate incoming requests. Set it as
an environment variable:

```bash
export KEENCHIC_API_KEY="your-api-key-here"
export KEENCHIC_EDITION="taimide"
export KEENCHIC_TAIMIDE_TEMPLATE_DIR="/home/nvidia/keenchic-data/templates"
export KEENCHIC_TAIMIDE_UPLOAD_DIR="/home/nvidia/keenchic-data/uploads"

mkdir -p \
  "${KEENCHIC_TAIMIDE_TEMPLATE_DIR}" \
  "${KEENCHIC_TAIMIDE_UPLOAD_DIR}"
```

Replace `your-api-key-here` with the actual API key provided to you.
Copy the approved paired `.xlsx` and `.json` templates into
`KEENCHIC_TAIMIDE_TEMPLATE_DIR`. Templates and uploads are external runtime
data and are not included in the wheel.

### Step 6: Start the Service

```bash
keenchic-serve --backend gpu --edition taimide --port 8000
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--backend` | Inference backend: `gpu` (TensorRT), `cpu`, `auto` | `gpu` |
| `--host` | Listen address | `0.0.0.0` (all interfaces) |
| `--port` | Listen port | `8000` |

When the service starts successfully, you will see output similar to:

```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

### Step 7: Test the Service

From another terminal (or another machine on the same network), send a health
check request:

```bash
curl http://<JETSON_IP>:8000/health
```

Replace `<JETSON_IP>` with the Jetson's IP address (use `hostname -I` to find it).
The response must report `edition=taimide` and `backend_config=GPU`. Model
loading is lazy, so `loaded_inspection` may initially be `null`.

Complete deployment acceptance with real Cell and Grid inference requests.
For example, Grid inspection requires a real image and `table_size`:

```bash
curl -sS -X POST http://<JETSON_IP>:8000/api/v1/inspect \
  -H "X-API-KEY: ${KEENCHIC_API_KEY}" \
  -H "X-Inspection-Name: ocr/meter-table-grid" \
  -F "image=@/path/to/table.PNG" \
  -F "table_size=[4,4]"
```

A successful response has HTTP 200, `result=0`, and a `pred_text_L` matrix
matching the requested table dimensions. After inference, `/health` should
report `backend=tensorrt` and the loaded inspection name.

---

## Running as a Background Service (Optional)

To keep the gateway running after you close the terminal, you can use `systemd`.

### Create the Environment File

Keep the API key out of the unit file:

```bash
sudo install -d -m 0750 /etc/keenchic
sudo tee /etc/keenchic/gateway.env > /dev/null <<'EOF'
KEENCHIC_API_KEY=your-api-key-here
KEENCHIC_EDITION=taimide
KEENCHIC_BACKEND=GPU
KEENCHIC_TAIMIDE_TEMPLATE_DIR=/home/nvidia/keenchic-data/templates
KEENCHIC_TAIMIDE_UPLOAD_DIR=/home/nvidia/keenchic-data/uploads
LOG_LEVEL=INFO
LOG_FORMAT=text
EOF
sudo chmod 600 /etc/keenchic/gateway.env
```

Replace `your-api-key-here` before starting the service.

### Create the Service File

```bash
sudo tee /etc/systemd/system/keenchic.service > /dev/null << 'EOF'
[Unit]
Description=Keenchic API Gateway
Wants=network-online.target
After=network-online.target

[Service]
Type=simple
User=nvidia
WorkingDirectory=/home/nvidia
EnvironmentFile=/etc/keenchic/gateway.env
ExecStart=/home/nvidia/keenchic-env/bin/keenchic-serve --backend gpu --edition taimide --host 0.0.0.0 --port 8000
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
```

### Enable and Start

```bash
sudo systemctl daemon-reload
sudo systemctl enable keenchic        # auto-start on boot
sudo systemctl start keenchic         # start now
```

### Check Status

```bash
sudo systemctl status keenchic
```

### View Logs

```bash
journalctl -u keenchic -f
```

### Stop the Service

```bash
sudo systemctl stop keenchic
```

---

## Troubleshooting

### `ImportError: No module named 'tensorrt'`

Virtual environment was created without `--system-site-packages`. Recreate it:

```bash
deactivate 2>/dev/null || true
mv /home/nvidia/keenchic-env /home/nvidia/keenchic-env-without-system-packages
python3 -m venv --system-site-packages /home/nvidia/keenchic-env
source /home/nvidia/keenchic-env/bin/activate

RELEASE_VERSION="<release-version>"
RELEASE_DIR="/home/nvidia/release-${RELEASE_VERSION}"
WHEEL="${RELEASE_DIR}/keenchic_api_gateway-${RELEASE_VERSION}+taimide-cp310-cp310-linux_aarch64.whl"

pip install \
  --only-binary=:all: \
  --constraint "${RELEASE_DIR}/target-constraints.txt" \
  "${WHEEL}"
```

The previous environment remains at
`/home/nvidia/keenchic-env-without-system-packages` for recovery.

### `No matching distribution found for pycuda==2026.1`

PyCUDA must be provisioned separately for the Target Jetson's CUDA stack. Do
not remove `--only-binary=:all:` and do not let pip build PyCUDA implicitly as
part of the application installation. Ask the Target Jetson administrator to
install the approved PyCUDA 2026.1 wheel, verify it with Step 3, and then retry
the application installation.

### `Connection refused` when testing with curl

1. Check the service is running: `sudo systemctl status keenchic`
2. Check the port is not blocked by firewall: `sudo ufw allow 8000`
3. Check the Jetson's IP: `hostname -I`

### `401 Unauthorized`

The request is missing the `X-API-KEY` header or the key does not match.
Include it in your curl command:

```bash
curl -H "X-API-KEY: your-api-key-here" http://<JETSON_IP>:8000/api/v1/inspect ...
```

### Service crashes on first request

The first inference request loads the TensorRT model into GPU memory, which
may take 10-30 seconds. If the Jetson runs out of memory, try closing other
GPU-intensive applications first.

---

## Environment Variables Reference

| Variable | Description | Default |
|----------|-------------|---------|
| `KEENCHIC_API_KEY` | API key for request authentication (required) | `""` |
| `KEENCHIC_BACKEND` | Inference backend: `GPU`, `CPU`, `AUTO` | `GPU` |
| `KEENCHIC_UPLOAD_DIR` | Directory to save uploaded images (optional) | not set |
| `KEENCHIC_EDITION` | Edition: `standard` or `taimide` | `standard` |
| `KEENCHIC_TAIMIDE_TEMPLATE_DIR` | Absolute path containing templates (required for `taimide`) | not set |
| `KEENCHIC_TAIMIDE_UPLOAD_DIR` | Base directory for Taimide uploads (required for `taimide`) | not set |
| `LOG_FORMAT` | Log format: `text` or `json` | `text` |
| `LOG_LEVEL` | Log level: `DEBUG`, `INFO`, `WARNING`, `ERROR` | `INFO` |
