from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Required: API key for authenticating incoming requests (X-API-KEY header)
    KEENCHIC_API_KEY: str = ""

    # Backend selection: GPU (TensorRT, fallback to OV), CPU/openvino, AUTO
    KEENCHIC_BACKEND: str = "GPU"

    # Optional: directory to persist raw uploaded images to disk
    KEENCHIC_UPLOAD_DIR: str | None = None

    # Edition: "standard" (default) or "taimide"
    KEENCHIC_EDITION: str = "standard"

    # [Taimide] Directory containing paired .xlsx and .json template files for download
    KEENCHIC_TAIMIDE_TEMPLATE_DIR: str | None = None

    # [Taimide] Base directory for Taimide uploads (auto-creates photos/ and reports/ subdirs)
    KEENCHIC_TAIMIDE_UPLOAD_DIR: str | None = None

    # Logging: format ("text" or "json") and level ("DEBUG"/"INFO"/"WARNING"/"ERROR")
    LOG_FORMAT: str = "text"
    LOG_LEVEL: str = "INFO"


settings = Settings()


def initialize_settings(
    env_file: str | None = None,
    backend: str | None = None,
    edition: str | None = None,
    log_level: str | None = None,
    log_format: str | None = None,
) -> None:
    """Dynamically update the global settings instance from CLI arguments and custom env file."""
    import os

    # 1. Update os.environ first to ensure pydantic reads them correctly on any reload
    if backend:
        os.environ["KEENCHIC_BACKEND"] = backend.upper()
    if edition:
        os.environ["KEENCHIC_EDITION"] = edition
    if log_level:
        os.environ["LOG_LEVEL"] = log_level.upper()
    if log_format:
        os.environ["LOG_FORMAT"] = log_format.lower()

    # 2. Re-instantiate Settings with the new configuration
    kwargs = {}
    if env_file:
        kwargs["_env_file"] = env_file

    new_settings = Settings(**kwargs)

    # 3. In-place update the global settings instance to maintain reference equality.
    # Use model_dump() + setattr to avoid corrupting pydantic v2 internal metadata
    # (e.g. __fields_set__, __pydantic_fields_set__) that __dict__.update() would overwrite.
    for key, value in new_settings.model_dump().items():
        setattr(settings, key, value)

