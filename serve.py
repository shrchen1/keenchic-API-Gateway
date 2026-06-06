import argparse
import uvicorn

from keenchic.core.config import initialize_settings, settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Start Keenchic API Gateway")
    parser.add_argument(
        "--backend",
        choices=["gpu", "cpu", "auto"],
        default=None,
        help=(
            "Inference backend: gpu (TensorRT with OV fallback), "
            "cpu (OpenVINO only), auto. "
            "Overrides KEENCHIC_BACKEND env var."
        ),
    )
    parser.add_argument(
        "--edition",
        choices=["standard", "taimide"],
        default=None,
        help="Edition: standard (default) or taimide. Overrides KEENCHIC_EDITION env var.",
    )
    parser.add_argument(
        "--env-file",
        default=None,
        help="Path to custom env configuration file. Overrides default .env.",
    )
    parser.add_argument(
        "--log-level",
        choices=["debug", "info", "warning", "error"],
        default=None,
        help="Logging level. Overrides LOG_LEVEL env var.",
    )
    parser.add_argument(
        "--log-format",
        choices=["text", "json"],
        default=None,
        help="Logging format. Overrides LOG_FORMAT env var.",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    # Initialize settings dynamically
    initialize_settings(
        env_file=args.env_file,
        backend=args.backend,
        edition=args.edition,
        log_level=args.log_level,
        log_format=args.log_format,
    )

    # Print startup configuration banner.
    # Skip in json log mode to avoid polluting structured log output (e.g. Loki, Datadog).
    if settings.LOG_FORMAT.lower() != "json":
        print("=" * 60)
        print("KEENCHIC API GATEWAY CONFIGURATION")
        print("-" * 60)
        print(f"Active Edition  : {settings.KEENCHIC_EDITION}")
        print(f"Backend         : {settings.KEENCHIC_BACKEND}")
        print(f"Upload Dir      : {settings.KEENCHIC_UPLOAD_DIR or '(none)'}")
        print(f"Taimide Template: {settings.KEENCHIC_TAIMIDE_TEMPLATE_DIR or '(none)'}")
        print(f"Taimide Uploads : {settings.KEENCHIC_TAIMIDE_UPLOAD_DIR or '(none)'}")
        print(f"Log Level       : {settings.LOG_LEVEL}")
        print(f"Log Format      : {settings.LOG_FORMAT}")
        print(f"Listen Address  : http://{args.host}:{args.port}")
        print("=" * 60)

    # Enforce workers=1 due to machine learning singleton structure
    uvicorn.run("main:app", host=args.host, port=args.port, workers=1)



if __name__ == "__main__":
    main()
