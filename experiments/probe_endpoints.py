"""Probe (endpoint, model) availability for the GPT-5 family.

Walks every endpoint in :data:`src.core.llm_client.DEFAULT_ENDPOINTS`
(or a ``--endpoints`` override) and tries each requested model
exactly once. Prints a Markdown matrix showing which combinations are
alive, which return 404, and which produced a different error.

Side effect: combos that 404 during the probe are recorded inside the
:class:`LLMClient` instance so any follow-up calls in the same process
already know to skip them.

Usage
-----
    # Probe the default endpoint pool against gpt-5.2 / 5.4 / 5.5.
    python -m experiments.probe_endpoints

    # Custom model list / endpoint subset.
    python -m experiments.probe_endpoints --models gpt-5.2,gpt-5.5
    python -m experiments.probe_endpoints --endpoints https://a.../,https://b.../

    # Verbose mode shows the underlying exception class for non-OK
    # cells.
    python -m experiments.probe_endpoints -v
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.llm_client import DEFAULT_ENDPOINTS, LLMClient  # noqa: E402

DEFAULT_MODELS = ["gpt-5.2", "gpt-5.4", "gpt-5.5"]


def _short_endpoint(url: str) -> str:
    """Strip scheme and the well-known azure suffixes for compact tables."""
    short = url.replace("https://", "").replace("http://", "").rstrip("/")
    for suffix in (
        ".openai.azure.com", ".cognitiveservices.azure.com",
    ):
        if short.endswith(suffix):
            short = short[: -len(suffix)]
            break
    return short


def _format_cell(status: str) -> str:
    """Compact human-readable status code for the matrix."""
    if status == "ok":
        return "OK"
    if status == "404":
        return "404"
    return status  # exception class name


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        default=",".join(DEFAULT_MODELS),
        help=(
            "Comma-separated list of models to probe. "
            f"Default: {','.join(DEFAULT_MODELS)}"
        ),
    )
    parser.add_argument(
        "--endpoints",
        default=None,
        help=(
            "Comma-separated endpoint URLs. "
            "Default: src.core.llm_client.DEFAULT_ENDPOINTS."
        ),
    )
    parser.add_argument(
        "--prompt",
        default="Reply with exactly the word: pong.",
        help="Probe prompt. Keep short to keep the test cheap.",
    )
    parser.add_argument(
        "--reasoning-effort",
        default="low",
        help=(
            "Reasoning effort for Responses-API models. The GPT-5 family "
            "accepts {none, low, medium, high, xhigh}; ``low`` keeps the "
            "probe cheap. (``minimal`` is rejected by gpt-5.2 as of "
            "2026-05.)"
        ),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Per-call timeout (seconds). Lower for faster sweeps.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Show DEBUG logging including the underlying error per failure.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if args.endpoints:
        endpoints = [e.strip() for e in args.endpoints.split(",") if e.strip()]
    else:
        endpoints = list(DEFAULT_ENDPOINTS)

    print(
        f"Probing {len(endpoints)} endpoint(s) x {len(models)} model(s) "
        f"= {len(endpoints) * len(models)} combinations.",
        file=sys.stderr,
    )

    client = LLMClient(
        endpoints=endpoints,
        max_retries=1,
        backoff_base_seconds=0.0,
        request_timeout_seconds=args.timeout,
    )
    matrix = client.probe_availability(
        models,
        prompt=args.prompt,
        reasoning_effort=args.reasoning_effort,
    )

    # ------------------------------------------------------------------
    # Markdown matrix
    # ------------------------------------------------------------------
    header = "| Endpoint | " + " | ".join(models) + " |"
    sep = "|---|" + "|".join(["---"] * len(models)) + "|"
    print()
    print(header)
    print(sep)
    for ep in endpoints:
        cells = [_format_cell(matrix[ep].get(m, "?")) for m in models]
        print(f"| {_short_endpoint(ep)} | " + " | ".join(cells) + " |")

    # ------------------------------------------------------------------
    # Summary line
    # ------------------------------------------------------------------
    counts: dict[str, int] = {"ok": 0, "404": 0, "other": 0}
    for ep in endpoints:
        for m in models:
            s = matrix[ep].get(m, "?")
            if s == "ok":
                counts["ok"] += 1
            elif s == "404":
                counts["404"] += 1
            else:
                counts["other"] += 1
    total = len(endpoints) * len(models)
    print(
        f"\n{counts['ok']} alive / {counts['404']} unavailable (404) / "
        f"{counts['other']} other / {total} probed."
    )

    # Per-model live endpoint counts.
    print("\nLive endpoints by model:")
    for m in models:
        live = [ep for ep in endpoints if matrix[ep].get(m) == "ok"]
        print(f"  {m}: {len(live)} / {len(endpoints)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
