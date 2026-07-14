from keenchic.core.http_logging import (
    PAYLOAD_PREVIEW_LIMIT_BYTES,
    REDACTED,
    REDACTED_BINARY,
    REDACTED_IMAGE,
    payload_log_fields,
    sanitize_payload,
)


def test_sanitize_payload_redacts_nested_credentials_and_images() -> None:
    payload = {
        "Password": "password-secret",
        "nested": {
            "accessToken": "token-secret",
            "client_secret": "client-secret",
            "safe": "visible",
        },
        "diagImg": "base64-secret",
        "preview": "data:image/png;base64,data-uri-secret",
        "items": [{"API-KEY": "api-key-secret"}],
        "binary": b"binary-secret",
    }

    sanitized = sanitize_payload(payload)

    assert sanitized == {
        "Password": REDACTED,
        "nested": {
            "accessToken": REDACTED,
            "client_secret": REDACTED,
            "safe": "visible",
        },
        "diagImg": REDACTED_IMAGE,
        "preview": REDACTED_IMAGE,
        "items": [{"API-KEY": REDACTED}],
        "binary": REDACTED_BINARY,
    }


def test_sanitize_payload_preserves_repeated_values_and_file_metadata() -> None:
    payload = {
        "fields": {"tag": ["first", "second"]},
        "files": [
            {
                "field_name": "image",
                "filename": "inspection.png",
                "content_type": "image/png",
                "size_bytes": 123,
            }
        ],
    }

    assert sanitize_payload(payload) == payload


def test_payload_log_fields_truncates_sanitized_serialized_preview() -> None:
    fields = payload_log_fields(
        {
            "password": "must-not-appear",
            "content": "x" * (PAYLOAD_PREVIEW_LIMIT_BYTES + 1024),
        }
    )

    assert fields["truncated"] is True
    assert fields["original_size_bytes"] > PAYLOAD_PREVIEW_LIMIT_BYTES
    assert len(fields["payload_preview"].encode("utf-8")) <= PAYLOAD_PREVIEW_LIMIT_BYTES
    assert "must-not-appear" not in fields["payload_preview"]
    assert REDACTED in fields["payload_preview"]
