"""Shared building blocks for Azure OpenAI RFT CLIs.

This module hosts the bits that any RFT-task CLI in this repo needs:
Azure client construction, the on-disk ``state.json`` schema, the
file-import waiter, the fine-tune capability pre-flight, generic
``upload`` / ``status`` / ``cancel`` / ``models`` handlers, and a few
small argparse helpers.

Task-specific scripts (`experiments.demo_rft` for Countdown,
`experiments.typo_rft` for typo / typo_v3) own their own ``prepare``
and ``submit`` steps -- everything else is delegated here.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from azure.identity import AzureCliCredential, get_bearer_token_provider
from openai import AzureOpenAI

# Make ``src.core.llm_client`` importable when invoked as a module from
# the repo root.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.core.llm_client import DEFAULT_ENDPOINTS  # noqa: E402

logger = logging.getLogger("rft_common")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default Azure OpenAI API version. RFT requires a preview API version;
# bump this when Microsoft promotes new versions.
DEFAULT_API_VERSION = "2025-04-01-preview"

STATE_FILE = "state.json"  # remembers uploaded file ids and job id

# Statuses returned by ``client.files.retrieve``. ``processed`` is the
# only state in which a file can be referenced by ``fine_tuning.jobs.create``.
_FILE_READY = "processed"
_FILE_TERMINAL_FAILURE = {"error", "deleting", "deleted"}

_TERMINAL_STATUSES = {"succeeded", "failed", "cancelled"}


# ---------------------------------------------------------------------------
# Endpoint resolution + client
# ---------------------------------------------------------------------------


def resolve_default_endpoint() -> str:
    """Pick the configured RFT endpoint using the same precedence as ``LLMClient``.

    Order:
      1. ``AZURE_OPENAI_ENDPOINT`` (single-endpoint override).
      2. First entry of ``AZURE_OPENAI_ENDPOINTS`` (pool, comma-separated).
    """
    single = os.environ.get("AZURE_OPENAI_ENDPOINT")
    if single:
        return single.strip()
    pool = os.environ.get("AZURE_OPENAI_ENDPOINTS")
    if pool:
        for ep in pool.split(","):
            ep = ep.strip()
            if ep:
                return ep
    if DEFAULT_ENDPOINTS:
        return DEFAULT_ENDPOINTS[0]
    raise ValueError(
        "Azure OpenAI endpoint is not configured; set AZURE_OPENAI_ENDPOINT, "
        "AZURE_OPENAI_ENDPOINTS, or pass --endpoint"
    )


def build_client(endpoint: str, api_version: str) -> AzureOpenAI:
    """Construct an AzureOpenAI client, preferring Entra over API keys."""
    api_key = os.environ.get("AZURE_OPENAI_KEY")
    if api_key:
        logger.info("Using API key auth (AZURE_OPENAI_KEY is set).")
        return AzureOpenAI(
            api_version=api_version,
            azure_endpoint=endpoint,
            api_key=api_key,
        )
    logger.info("Using AzureCliCredential (run `az login` if this fails).")
    token_provider = get_bearer_token_provider(
        AzureCliCredential(),
        "https://cognitiveservices.azure.com/.default",
    )
    return AzureOpenAI(
        api_version=api_version,
        azure_endpoint=endpoint,
        azure_ad_token_provider=token_provider,
    )


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------


@dataclass
class DemoState:
    """On-disk record of what we've created so commands can chain."""

    train_file_id: str | None = None
    valid_file_id: str | None = None
    job_id: str | None = None

    @classmethod
    def load(cls, output_dir: Path) -> "DemoState":
        path = output_dir / STATE_FILE
        if not path.exists():
            return cls()
        return cls(**json.loads(path.read_text()))

    def save(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / STATE_FILE).write_text(json.dumps(self.__dict__, indent=2))


# ---------------------------------------------------------------------------
# File upload waiter
# ---------------------------------------------------------------------------


def wait_for_file_ready(
    client: AzureOpenAI,
    file_id: str,
    *,
    timeout_seconds: float = 600.0,
    poll_interval_seconds: float = 5.0,
) -> None:
    """Block until an uploaded file finishes importing.

    Azure OpenAI accepts the upload immediately (HTTP 201) but performs
    server-side schema validation asynchronously. ``fine_tuning.jobs.create``
    fails with ``invalidPayload: The specified file reference must point to
    a completed file import.`` if the file is still ``pending``. We poll the
    file resource until ``status == "processed"`` (success) or a terminal
    failure status (in which case we surface the server's error message).
    """
    deadline = time.monotonic() + timeout_seconds
    last_status: str | None = None
    while True:
        f = client.files.retrieve(file_id)
        status = getattr(f, "status", None)
        if status != last_status:
            logger.info("File %s status=%s", file_id, status)
            last_status = status
        if status == _FILE_READY:
            return
        if status in _FILE_TERMINAL_FAILURE:
            details = getattr(f, "status_details", None) or getattr(f, "error", None)
            raise RuntimeError(
                f"File {file_id} import failed: status={status!r} details={details!r}"
            )
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"File {file_id} not ready after {timeout_seconds:.0f}s "
                f"(last status={status!r})."
            )
        time.sleep(poll_interval_seconds)


# ---------------------------------------------------------------------------
# Model capability pre-flight
# ---------------------------------------------------------------------------


def _model_capabilities(model: Any) -> dict[str, Any]:
    """Pull the ``capabilities`` dict off an SDK model record.

    Azure OpenAI surfaces ``capabilities`` and ``lifecycle_status`` as
    non-standard fields, which land in ``__pydantic_extra__`` on the SDK
    model object. Returns an empty dict if the endpoint doesn't supply
    them (e.g. real OpenAI endpoints).
    """
    extra = getattr(model, "__pydantic_extra__", None) or {}
    caps = extra.get("capabilities") or {}
    return caps if isinstance(caps, dict) else {}


def list_fine_tunable_models(client: AzureOpenAI) -> list[Any]:
    """Return every model the endpoint advertises with ``fine_tune=True``."""
    page = client.models.list()
    return [m for m in page.data if _model_capabilities(m).get("fine_tune")]


def assert_model_fine_tunable(
    client: AzureOpenAI, model_id: str, *, skip: bool
) -> None:
    """Fail fast if the chosen base model isn't fine-tunable on this resource.

    Azure's ``fine_tuning.jobs.create`` returns an opaque HTTP 400 with
    message ``"The specified base model does not support fine-tuning."``
    when the model isn't enabled for RFT on the resource. We can detect
    this up-front by checking ``capabilities.fine_tune`` from
    ``models.list``, which is much easier to act on than the post-upload
    failure.
    """
    if skip:
        return
    try:
        candidates = list_fine_tunable_models(client)
    except Exception as exc:  # noqa: BLE001 - any error here is non-fatal
        logger.warning(
            "Could not pre-verify fine-tune capability (continuing): %s", exc
        )
        return
    if any(m.id == model_id for m in candidates):
        return
    available = sorted(m.id for m in candidates) or ["<none>"]
    raise RuntimeError(
        f"Base model {model_id!r} is not fine-tunable on this endpoint. "
        f"Models with capabilities.fine_tune=true here: {available}. "
        f"Either pick one of those via --base-model, point --endpoint at "
        f"a resource that has the desired model enabled for RFT, or pass "
        f"--skip-model-check to bypass this pre-flight (the underlying "
        f"jobs.create call will then fail with the original HTTP 400)."
    )


# ---------------------------------------------------------------------------
# Generic upload / status / cancel / models handlers
# ---------------------------------------------------------------------------


def upload_files(
    args: argparse.Namespace,
    train_filename: str,
    valid_filename: str,
) -> None:
    """Upload train + valid JSONL files referenced by ``state.json``.

    Files must already be present in ``args.output_dir`` under the
    provided filenames (the per-task ``prepare`` step is responsible
    for staging them there).
    """
    client = build_client(args.endpoint, args.api_version)
    state = DemoState.load(args.output_dir)

    for attr, filename in [
        ("train_file_id", train_filename),
        ("valid_file_id", valid_filename),
    ]:
        path = args.output_dir / filename
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run `prepare` first or pass --output-dir."
            )
        with path.open("rb") as fh:
            resp = client.files.create(file=fh, purpose="fine-tune")
        setattr(state, attr, resp.id)
        logger.info("Uploaded %s -> %s", filename, resp.id)
        # Save the id eagerly so a slow import + Ctrl-C doesn't lose it.
        state.save(args.output_dir)
        wait_for_file_ready(client, resp.id)


def _print_events(client: AzureOpenAI, job_id: str, after: str | None) -> str | None:
    """Stream new fine-tuning events; return the most recent event id."""
    # ``after`` is exclusive: only events newer than this id are returned.
    page = client.fine_tuning.jobs.list_events(
        fine_tuning_job_id=job_id, limit=20, after=after
    )
    # The SDK returns events newest-first; we want chronological order.
    events = list(reversed(list(page.data)))
    for ev in events:
        ts = time.strftime("%H:%M:%S", time.localtime(ev.created_at))
        print(f"  [{ts}] {ev.level:<7} {ev.message}")
    return events[-1].id if events else after


def cmd_status(args: argparse.Namespace) -> None:
    state = DemoState.load(args.output_dir)
    job_id = args.job_id or state.job_id
    if not job_id:
        raise RuntimeError("No job id known. Pass --job-id or run `submit` first.")

    client = build_client(args.endpoint, args.api_version)
    after_event: str | None = None

    for tick in itertools.count():
        job = client.fine_tuning.jobs.retrieve(job_id)
        print(
            f"\n[{tick:>3}] status={job.status} "
            f"trained_tokens={getattr(job, 'trained_tokens', None)} "
            f"fine_tuned_model={getattr(job, 'fine_tuned_model', None)}"
        )
        after_event = _print_events(client, job_id, after_event)

        if job.status in _TERMINAL_STATUSES or not args.follow:
            break
        time.sleep(args.poll_interval)


def cmd_cancel(args: argparse.Namespace) -> None:
    state = DemoState.load(args.output_dir)
    job_id = args.job_id or state.job_id
    if not job_id:
        raise RuntimeError("No job id known. Pass --job-id or run `submit` first.")
    client = build_client(args.endpoint, args.api_version)
    job = client.fine_tuning.jobs.cancel(job_id)
    logger.info("Cancelled %s (status=%s)", job_id, job.status)


def cmd_models(args: argparse.Namespace) -> None:
    """List models the configured endpoint exposes.

    Useful for verifying that a model name you intend to pass to
    ``--base-model`` or ``--grader-model`` actually resolves on this
    resource (Azure surfaces an opaque HTTP 400 otherwise).

    Default view filters to models with ``capabilities.fine_tune=true``
    so you can immediately see which base models the endpoint is
    actually willing to fine-tune. Pass ``--all`` to also see models
    that are inference-only or otherwise unrelated.
    """
    client = build_client(args.endpoint, args.api_version)
    page = client.models.list()
    rows = list(page.data)

    if not args.all:
        rows = [m for m in rows if _model_capabilities(m).get("fine_tune")]

    rows.sort(key=lambda m: m.id)

    if not rows:
        if not args.all:
            print(
                "No models on this endpoint advertise capabilities.fine_tune=true. "
                "Either this resource doesn't have any fine-tuning capacity "
                "enabled, or use --all to inspect what is present."
            )
        else:
            print("Endpoint returned no models at all.")
        return

    width = max((len(m.id) for m in rows), default=10)
    header = f"{'model id':<{width}}  {'lifecycle':<22}  {'fine_tune':<9}  other capabilities"
    print(header)
    print("-" * len(header))
    for m in rows:
        extra = getattr(m, "__pydantic_extra__", None) or {}
        caps = _model_capabilities(m)
        lifecycle = extra.get("lifecycle_status", "") or ""
        ft = "true" if caps.get("fine_tune") else "false"
        other = [
            k for k in ("chat_completion", "completion", "embeddings", "inference")
            if caps.get(k)
        ]
        print(f"{m.id:<{width}}  {lifecycle:<22}  {ft:<9}  {', '.join(other)}")


# ---------------------------------------------------------------------------
# Argparse helpers
# ---------------------------------------------------------------------------


def add_common_args(p: argparse.ArgumentParser, default_output_dir: Path) -> None:
    p.add_argument(
        "--endpoint",
        default=None,
        help="Azure OpenAI resource endpoint, e.g. https://<resource>.openai.azure.com/. "
        "Defaults to $AZURE_OPENAI_ENDPOINT or the first entry of "
        "$AZURE_OPENAI_ENDPOINTS.",
    )
    p.add_argument(
        "--api-version",
        default=DEFAULT_API_VERSION,
        help=f"Azure OpenAI REST API version (default {DEFAULT_API_VERSION}).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
        help=f"Where to store JSONL data and state.json (default {default_output_dir}).",
    )


def add_status_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--job-id", default=None, help="Defaults to the last submitted job.")
    p.add_argument("--follow", action="store_true", help="Poll until terminal status.")
    p.add_argument(
        "--poll-interval", type=int, default=30,
        help="Seconds between polls when --follow.",
    )


def add_models_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--all",
        action="store_true",
        help="Show every model the endpoint returns, not just those with "
        "capabilities.fine_tune=true.",
    )


def add_cancel_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--job-id", default=None)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def resolve_endpoint_if_needed(args: argparse.Namespace, *, network_command: bool) -> None:
    """Fill in ``args.endpoint`` for network-touching commands."""
    if network_command and not args.endpoint:
        args.endpoint = resolve_default_endpoint()
        logger.info("Using endpoint %s (auto-resolved).", args.endpoint)
