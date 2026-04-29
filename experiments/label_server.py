"""CLI entry point for the labelling + experiment-launcher web app.

Usage (from project root)::

    python -m experiments.label_server [--port 8000]

Binds to 127.0.0.1 by default. If you're on the VM over VS Code
Remote-SSH the port auto-forwards - open
``http://localhost:8000/`` in your local browser.

The server is workspace-scoped: it manages every run directory under
``outputs/`` and can also launch new detection runs as child processes
(``python -m experiments.run_cleaning_detection`` under the hood).
"""

from __future__ import annotations

import argparse
import getpass
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_dotenv(path: Path) -> None:
    """Tiny zero-dep ``.env`` loader.

    The web-verifier needs ``TAVILY_API_KEY`` (and ``LLMClient`` reads
    ``AZURE_OPENAI_*``) at request time. We load ``<repo>/.env`` here so
    operators can run ``python -m experiments.label_server`` without
    having to remember ``set -a; source .env`` first. We do NOT
    overwrite values already in the environment - shell exports win,
    matching the convention of every popular dotenv library.
    """
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        # Strip matching surrounding quotes (common in .env files).
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv(PROJECT_ROOT / ".env")

import uvicorn  # noqa: E402

from src.labeling.app import create_app  # noqa: E402


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="experiments.label_server",
        description=(
            "Launch the data-cleaning labelling + experiment-launcher "
            "web app for this project."
        ),
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
        help="Project root (contains outputs/, data/, config/). Default: repo root.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--annotator",
        default=None,
        help="Default annotator name recorded on every label. Defaults to $USER.",
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=("critical", "error", "warning", "info", "debug"),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    annotator = args.annotator or getpass.getuser() or "anonymous"
    app = create_app(args.project_root, annotator=annotator)

    print(
        f"[label_server] project_root={args.project_root}  annotator={annotator}\n"
        f"[label_server] UI:  http://{args.host}:{args.port}/\n"
        f"[label_server] API: http://{args.host}:{args.port}/api/docs",
        flush=True,
    )

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        access_log=False,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
