import os
import uuid
from datetime import datetime, timezone


def generate_safe_filename(original_name: str, content_type: str | None = None) -> str:
    """Generate a safe, unique filename using the strategy:
    YYYYMMDD-HHMMSS-mmm-<uuid8>-<safe_name>.<ext>
    """
    base = os.path.basename(original_name or "")
    name, ext = os.path.splitext(base)
    if not ext and content_type:
        ext = (
            ("." + content_type.split("/", 1)[1].lower())
            if content_type.startswith("image/")
            else ".jpg"
        )
    if not ext:
        ext = ".jpg"

    # Ensure the extension starts with a dot
    if not ext.startswith("."):
        ext = "." + ext

    dt = datetime.now(timezone.utc)
    ts = dt.strftime("%Y%m%d-%H%M%S") + f"-{int(dt.microsecond / 1000):03d}"
    safe = (
        "".join(c for c in name if c.isalnum() or c in ("-", "_")).strip()[:50]
    ) or "upload"
    return f"{ts}-{uuid.uuid4().hex[:8]}-{safe}{ext}"


def save_file(data: bytes, directory: str, filename: str) -> str:
    """Write bytes to directory/filename, creating the directory if it doesn't exist.
    Returns the absolute path to the saved file.
    """
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, filename)
    with open(filepath, "wb") as f:
        f.write(data)
    return os.path.abspath(filepath)


def generate_safe_filename_with_batch(
    original_name: str,
    batch_number: str,
    content_type: str | None = None,
) -> str:
    """Generate a safe filename that includes a batch number.

    Format: YYYYMMDD-HHMMSS-<batch_number>-<uuid8>-<safe_name>.<ext>

    Unlike ``generate_safe_filename``, this variant drops the millisecond
    component and inserts *batch_number* between the timestamp and the UUID.
    """
    base = os.path.basename(original_name or "")
    name, ext = os.path.splitext(base)
    if not ext and content_type:
        ext = (
            ("." + content_type.split("/", 1)[1].lower())
            if content_type.startswith("image/")
            else ".xlsx"
        )
    if not ext:
        ext = ".xlsx"

    # Ensure the extension starts with a dot
    if not ext.startswith("."):
        ext = "." + ext

    dt = datetime.now(timezone.utc)
    ts = dt.strftime("%Y%m%d-%H%M%S")
    safe = (
        "".join(c for c in name if c.isalnum() or c in ("-", "_")).strip()[:50]
    ) or "upload"
    return f"{ts}-{batch_number}-{uuid.uuid4().hex[:8]}-{safe}{ext}"

