"""End-to-end Reinforcement Fine-Tuning (RFT) demo on Azure OpenAI.

Mirrors the workflow described in
https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/reinforcement-fine-tuning
and the Microsoft Foundry "RFT_Countdown" reference demo, but packaged as a
single self-contained Python script that fits the conventions of this repo
(Entra ID auth by default, ``AzureOpenAI`` client, no external dataset).

Task
----
A miniature "Countdown" arithmetic puzzle:

* The user is given a ``target`` integer and a list of ``nums``.
* The model must combine **every** number exactly once with +, -, *, / to
  hit the target (or get as close as possible) and return JSON of the
  form ``{"expression": "...", "result": "..."}``.

This is a textbook RFT use case (many valid answers, easy to grade by
running the expression, hard to learn by imitation) and works with any of
the supported RFT base models (``o4-mini``, ``gpt-5``).

What the script does
--------------------
1. ``prepare`` - synthesizes a tiny RFT-formatted training + validation
   JSONL (no external download).
2. ``upload``  - uploads both files to Azure OpenAI with
   ``purpose="fine-tune"``.
3. ``submit``  - submits an RFT job with a model-based grader
   (``score_model``) and a strict JSON ``response_format``.
4. ``status``  - polls a running job and prints the most recent events.
5. ``cancel``  - cancels a running job.
6. ``all``     - runs prepare -> upload -> submit and then tails status
   until the job leaves the running state.

Authentication
--------------
By default the script uses ``AzureCliCredential`` (run ``az login`` once),
which matches the rest of this repo. To use an API key instead, set
``AZURE_OPENAI_KEY``.

Endpoint selection
------------------
If ``--endpoint`` is omitted, the script reuses the same configuration as
:mod:`src.core.llm_client`:

1. ``$AZURE_OPENAI_ENDPOINT`` (single-endpoint override) - if set.
2. The first entry in ``$AZURE_OPENAI_ENDPOINTS`` (comma-separated pool) -
   shared with ``LLMClient``.
3. The first entry in :data:`src.core.llm_client.DEFAULT_ENDPOINTS`.

Fine-tuning binds the job to one resource, so we just pick the first
endpoint instead of round-robining across the pool. Use ``--endpoint`` to
pin a specific resource if your RFT-enabled deployment lives somewhere
other than the pool's first entry.

Usage
-----
::

    # One-shot end-to-end demo (small, cheap), endpoint auto-picked from
    # the same env vars LLMClient uses:
    python -m experiments.demo_rft all

    # Or step-by-step:
    python -m experiments.demo_rft prepare
    python -m experiments.demo_rft upload
    python -m experiments.demo_rft submit
    python -m experiments.demo_rft status --job-id ftjob-xxxxxxxx

Notes
-----
* RFT requires a separately approved Azure OpenAI resource; submission
  will fail with HTTP 400 / "model not available" until your tenant has
  access to the chosen base model for fine-tuning.
* Azure pauses RFT jobs at $5,000 of cumulative training + grading cost;
  the defaults below are set deliberately small.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from azure.identity import AzureCliCredential, get_bearer_token_provider
from openai import AzureOpenAI

# Make ``src.core.llm_client`` importable when this file is run as a module
# from the repo root (matches the pattern used by other experiment scripts,
# e.g. ``experiments/probe_endpoints.py``).
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.core.llm_client import DEFAULT_ENDPOINTS  # noqa: E402
from RFT._common import _typo_task

logger = logging.getLogger("demo_rft")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_OUTPUT_DIR = Path("outputs/rft_demo")
TRAIN_FILE = "countdown_train.jsonl"
VALID_FILE = "countdown_valid.jsonl"
STATE_FILE = "state.json"  # remembers uploaded file ids and job id

# RFT-supported base model. Use ``demo_rft models`` to confirm which
# fine-tunable models the configured endpoint exposes; override via
# ``--base-model`` if you want a different one. ``o4-mini-2025-04-16``
# is the documented GA RFT model and is enabled on this repo's endpoint
# pool. ``gpt-5-2025-08-07`` is the dated form for gpt-5 RFT but is
# still preview-gated per-tenant.
DEFAULT_BASE_MODEL = "o4-mini-2025-04-16"

# Model used by the ``score_model`` grader. The grader model does NOT
# need to be deployed in your resource - the training service supplies
# it - but it must be one of the model names Microsoft lists as a
# supported grader (currently ``gpt-4o-2024-08-06`` and the
# ``o3-mini-2025-01-31`` reasoning model).
DEFAULT_GRADER_MODEL = "o3-mini-2025-01-31"

# Default Azure OpenAI API version. RFT requires a preview API version;
# bump this when Microsoft promotes new versions.
DEFAULT_API_VERSION = "2025-04-01-preview"


SYSTEM_INSTRUCTION = (
    "You are an expert in arithmetic problem solving. Given a target "
    "integer and a list of numbers, combine ALL of the numbers exactly "
    "once using +, -, *, or / to reach the target. Use parentheses to "
    "control evaluation order. Return ONLY a JSON object with the keys "
    '"expression" and "result", both as strings, e.g. '
    '{"expression": "(75 * 3) + (50 * 2) - (8 + 7)", "result": "310"}.'
)


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------


def _generate_puzzle(rng: random.Random) -> tuple[int, list[int]]:
    """Build a (target, nums) pair that is *solvable* by construction.

    We pick 4 small positive integers and a random operator tree, then
    evaluate it to derive the target. This guarantees the puzzle is
    reachable without giving the model the answer.
    """
    while True:
        nums = [rng.randint(1, 25) for _ in range(4)]
        ops = [rng.choice(["+", "-", "*"]) for _ in range(3)]
        # Random left-leaning parenthesisation:
        expr = f"(({nums[0]} {ops[0]} {nums[1]}) {ops[1]} {nums[2]}) {ops[2]} {nums[3]}"
        try:
            target = eval(expr)  # noqa: S307 - controlled input
        except Exception:
            continue
        if isinstance(target, int) and 1 <= target <= 1000:
            return target, nums


def _build_record(target: int, nums: list[int]) -> dict[str, Any]:
    """Shape one JSONL record exactly as the RFT API expects.

    Requirements (from the Microsoft Learn page):
      * ``messages`` is a chat-completions-style list.
      * The LAST message must have ``role == "user"`` (no assistant
        ground-truth - RFT learns from the grader, not labels).
      * Any extra top-level fields (here ``target`` and ``nums``) are
        passed through to the grader via the ``item.*`` namespace.
    """
    user_msg = (
        f"Target: {target}\n"
        f"Numbers: {nums}\n"
        "Return JSON with keys 'expression' and 'result'."
    )
    return {
        "messages": [
            {"role": "developer", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": user_msg},
        ],
        # Grader-visible fields. The grader template references
        # ``{{ item.target }}`` / ``{{ item.nums }}``.
        "target": str(target),
        "nums": str(nums),
    }


def cmd_prepare(args: argparse.Namespace) -> None:
    """Stage train + validation JSONL files for the chosen task.

    For ``--task countdown`` (the original demo) we synthesise the
    puzzle records on the fly. For ``--task typo`` the JSONL files
    are produced by ``experiments.synth_typo_llm`` and are merely
    *staged* here -- copied (and shape-validated) into the demo state
    directory under the names expected by ``upload`` / ``submit``.
    Keeping prepare as a thin staging step (rather than re-running
    synthesis) lets you iterate on the trained job without re-paying
    the synth LLM bill.
    """
    task = getattr(args, "task", "countdown")
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if task == "typo":
        _prepare_typo(args, output_dir)
        return

    rng = random.Random(args.seed)
    train_path = output_dir / TRAIN_FILE
    valid_path = output_dir / VALID_FILE

    for path, n in [(train_path, args.train_size), (valid_path, args.valid_size)]:
        with path.open("w", encoding="utf-8") as fh:
            for _ in range(n):
                target, nums = _generate_puzzle(rng)
                fh.write(json.dumps(_build_record(target, nums)) + "\n")
        logger.info("Wrote %d records -> %s", n, path)

    # Preview one record so the operator can sanity-check the shape.
    with train_path.open(encoding="utf-8") as fh:
        sample = json.loads(fh.readline())
    print("\nSample training record:")
    print(json.dumps(sample, indent=2))


def _prepare_typo(args: argparse.Namespace, output_dir: Path) -> None:
    """Validate + stage typo train/valid JSONL into ``output_dir``.

    Caller must supply ``--train-jsonl`` and ``--valid-jsonl``; we
    refuse to silently regenerate from data/ because that would mask
    a misconfigured synth run.
    """
    if not args.train_jsonl or not args.valid_jsonl:
        raise RuntimeError(
            "--task typo requires --train-jsonl and --valid-jsonl. "
            "Run experiments.synth_typo_llm first."
        )
    train_src = Path(args.train_jsonl)
    valid_src = Path(args.valid_jsonl)
    if not train_src.exists() or not valid_src.exists():
        raise FileNotFoundError(
            f"Could not find one or both of {train_src}, {valid_src}."
        )

    train_dst = output_dir / _typo_task.TYPO_TRAIN_FILE
    valid_dst = output_dir / _typo_task.TYPO_VALID_FILE

    for src, dst in [(train_src, train_dst), (valid_src, valid_dst)]:
        n = _copy_and_validate_typo_jsonl(src, dst)
        logger.info("Staged %d typo records: %s -> %s", n, src, dst)

    with train_dst.open(encoding="utf-8") as fh:
        sample = json.loads(fh.readline())
    print("\nSample staged record:")
    print(json.dumps(sample, indent=2))


def _copy_and_validate_typo_jsonl(src: Path, dst: Path) -> int:
    """Copy JSONL src -> dst, validating each row matches the RFT shape.

    The RFT API requires ``messages`` ending in a user turn AND every
    top-level non-``messages`` field to be a JSON string (so the grader
    template can do textual substitution). A misconfigured synth would
    otherwise surface only as an opaque HTTP 400 mid-upload.
    """
    count = 0
    with src.open(encoding="utf-8") as in_fh, dst.open("w", encoding="utf-8") as out_fh:
        for ln, line in enumerate(in_fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{src}:{ln} is not valid JSON: {exc}"
                ) from exc
            msgs = rec.get("messages")
            if not isinstance(msgs, list) or not msgs:
                raise ValueError(
                    f"{src}:{ln} missing non-empty 'messages' array."
                )
            if msgs[-1].get("role") != "user":
                raise ValueError(
                    f"{src}:{ln} last message must be role=user."
                )
            for key in ("gold_label", "gold_correction", "candidate_value",
                        "column", "source"):
                if key not in rec:
                    raise ValueError(
                        f"{src}:{ln} missing top-level grader field {key!r}."
                    )
            out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1
    if count == 0:
        raise ValueError(f"{src} contained no records.")
    return count


# ---------------------------------------------------------------------------
# Azure client + state persistence
# ---------------------------------------------------------------------------


def _build_client(endpoint: str, api_version: str) -> AzureOpenAI:
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
# Upload
# ---------------------------------------------------------------------------


# Statuses returned by ``client.files.retrieve``. ``processed`` is the
# only state in which a file can be referenced by ``fine_tuning.jobs.create``;
# anything else either means it's still importing or has failed validation.
_FILE_READY = "processed"
_FILE_TERMINAL_FAILURE = {"error", "deleting", "deleted"}


def _wait_for_file_ready(
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


def _task_filenames(task: str) -> tuple[str, str]:
    """Resolve (train_filename, valid_filename) for the chosen task."""
    if task == "typo":
        return _typo_task.TYPO_TRAIN_FILE, _typo_task.TYPO_VALID_FILE
    return TRAIN_FILE, VALID_FILE


def cmd_upload(args: argparse.Namespace) -> None:
    client = _build_client(args.endpoint, args.api_version)
    state = DemoState.load(args.output_dir)

    train_fn, valid_fn = _task_filenames(getattr(args, "task", "countdown"))
    for attr, filename in [("train_file_id", train_fn), ("valid_file_id", valid_fn)]:
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
        _wait_for_file_ready(client, resp.id)


# ---------------------------------------------------------------------------
# Submit
# ---------------------------------------------------------------------------


def _build_grader(grader_model: str) -> dict[str, Any]:
    """Score-model grader: returns 0-5 based on correctness.

    The grader receives the original ``target`` and ``nums`` via the
    ``item.*`` namespace, and the model's response via
    ``sample.output_text``. Template substitution is performed by the
    training service (see the "Graders" section of the RFT docs).
    """
    return {
        "name": "countdown_grader",
        "type": "score_model",
        "model": grader_model,
        "input": [
            {
                "role": "developer",
                "content": (
                    "You are a mathematical evaluator. You are given:\n"
                    "  - target: a string representation of an integer\n"
                    "  - numbers: a string representation of a Python list of integers\n"
                    "  - output: the model's JSON output containing 'expression' "
                    "and 'result' fields\n"
                    "\n"
                    "Score the output on a 0-5 scale:\n"
                    "  5: Valid expression; uses every number exactly once; "
                    "     evaluates to exactly the target.\n"
                    "  4: Valid; uses every number once; off by 1.\n"
                    "  3: Valid; uses every number once; off by 2-5.\n"
                    "  2: Valid; uses every number once; off by more than 5.\n"
                    "  1: Minor issues - e.g., number reused or missing.\n"
                    "  0: Invalid expression, wrong numbers, or unparseable JSON.\n"
                    "\n"
                    "Respond ONLY with:\n"
                    "Score: <0-5>\n"
                    "Reasoning: <one short sentence>"
                ),
            },
            {
                "role": "user",
                # NOTE: do NOT put whitespace inside the {{ ... }} braces.
                # The Azure RFT grader parser does not strip whitespace from
                # variable names, so ``{{ item.target }}`` is read as the
                # variable named " item.target " (with leading/trailing
                # spaces) and rejected with HTTP 400:
                #   "Variable  item.target  is not allowed.
                #    Only 'sample' and 'item' namespaces are allowed".
                # Use ``{{item.target}}`` (no spaces) instead.
                "content": (
                    '{"target": {{item.target}}, '
                    '"numbers": {{item.nums}}, '
                    '"output": {{sample.output_text}}}'
                ),
            },
        ],
        "pass_threshold": 5,
        "range": [0.0, 5.0],
    }


def _build_response_format() -> dict[str, Any]:
    """Strict JSON schema for the model's output (forces parseable JSON)."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "countdown_answer",
            "schema": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The arithmetic expression using all input numbers.",
                    },
                    "result": {
                        "type": "string",
                        "description": "The evaluated numeric result, as a string.",
                    },
                },
                "required": ["expression", "result"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }


def _assert_model_fine_tunable(
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
        candidates = _list_fine_tunable_models(client)
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


def cmd_submit(args: argparse.Namespace) -> None:
    state = DemoState.load(args.output_dir)
    if not (state.train_file_id and state.valid_file_id):
        raise RuntimeError("Train/valid file IDs missing. Run `upload` first.")

    client = _build_client(args.endpoint, args.api_version)

    # Pre-flight: fail fast with an actionable message if the chosen base
    # model isn't fine-tunable on this resource (saves an Azure round-trip
    # plus the opaque HTTP 400 'does not support fine-tuning').
    _assert_model_fine_tunable(
        client, args.base_model, skip=getattr(args, "skip_model_check", False)
    )

    # Defensive: if the operator runs `submit` directly (or `upload` was
    # interrupted before its polling completed), make sure both files are
    # actually importable before issuing the job-create request, which would
    # otherwise fail with HTTP 400 "must point to a completed file import".
    _wait_for_file_ready(client, state.train_file_id)
    _wait_for_file_ready(client, state.valid_file_id)

    task = getattr(args, "task", "countdown")
    if task == "typo":
        grader = _typo_task.build_grader(args.grader_model)
        response_format = _typo_task.build_response_format()
    else:
        grader = _build_grader(args.grader_model)
        response_format = _build_response_format()

    method: dict[str, Any] = {
        "type": "reinforcement",
        "reinforcement": {
            "grader": grader,
            "response_format": response_format,
            "hyperparameters": {
                "n_epochs": args.n_epochs,
                "batch_size": args.batch_size,
                "eval_interval": args.eval_interval,
                "eval_samples": args.eval_samples,
                "reasoning_effort": args.reasoning_effort,
                "learning_rate_multiplier": args.learning_rate_multiplier,
                "compute_multiplier": args.compute_multiplier,
            },
        },
    }

    job = client.fine_tuning.jobs.create(
        model=args.base_model,
        training_file=state.train_file_id,
        validation_file=state.valid_file_id,
        method=method,
        suffix=args.suffix,
    )
    state.job_id = job.id
    state.save(args.output_dir)
    logger.info("Submitted RFT job: %s (status=%s)", job.id, job.status)
    print(f"\nJob ID: {job.id}")
    print(f"Watch with: python -m experiments.demo_rft status --job-id {job.id}")


# ---------------------------------------------------------------------------
# Status / cancel
# ---------------------------------------------------------------------------


_TERMINAL_STATUSES = {"succeeded", "failed", "cancelled"}


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

    client = _build_client(args.endpoint, args.api_version)
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
    client = _build_client(args.endpoint, args.api_version)
    job = client.fine_tuning.jobs.cancel(job_id)
    logger.info("Cancelled %s (status=%s)", job_id, job.status)


# ---------------------------------------------------------------------------
# Models - list what the endpoint exposes
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


def _list_fine_tunable_models(client: AzureOpenAI) -> list[Any]:
    """Return every model the endpoint advertises with ``fine_tune=True``."""
    page = client.models.list()
    return [m for m in page.data if _model_capabilities(m).get("fine_tune")]


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
    client = _build_client(args.endpoint, args.api_version)
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
# "all" - prepare + upload + submit + follow
# ---------------------------------------------------------------------------


def cmd_all(args: argparse.Namespace) -> None:
    cmd_prepare(args)
    cmd_upload(args)
    cmd_submit(args)
    # Reuse the same follow logic to tail the job. ``cmd_status`` reads
    # ``args.job_id`` (and falls back to ``state.job_id`` when None), but
    # the ``all`` subparser doesn't define that flag, so prime it here.
    args.job_id = None
    args.follow = True
    cmd_status(args)


# ---------------------------------------------------------------------------
# calibrate-grader - simulate the typo grader locally
# ---------------------------------------------------------------------------


def cmd_calibrate_grader(args: argparse.Namespace) -> None:
    """Run the typo grader against eval-split rows BEFORE training.

    For each sampled eval row we synthesise two trainee replies -- one
    that matches the gold (should score ~5) and one that contradicts
    it (should score <=2) -- then ask the configured grader model to
    score each. Reports mean scores per response type and the
    "agreement rate" (% of samples where correct > incorrect). A run
    with agreement below ~85% indicates the grader prompt itself is
    miscalibrated and would teach the wrong signal during training.

    We invoke the grader via :mod:`src.core.llm_client` rather than
    deploying it to the configured endpoint, because the chosen grader
    model (``o3-mini-2025-01-31``) is not necessarily available as a
    user-deployable model on every Azure OpenAI resource. ``LLMClient``
    handles per-endpoint 404 routing so the grader can be tested on
    any endpoint in the pool that does carry the model.
    """
    from src.core.llm_client import LLMClient  # local import: optional dep

    eval_path = Path(args.eval_jsonl)
    if not eval_path.exists():
        raise FileNotFoundError(
            f"Eval JSONL not found: {eval_path}. Run "
            f"RFT.typo.v1.scripts.split_typo_benchmark first."
        )
    rows = [
        json.loads(line)
        for line in eval_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rng = random.Random(args.seed)
    if args.sample_size and args.sample_size < len(rows):
        rows = rng.sample(rows, args.sample_size)
    logger.info("Calibrating grader on %d eval rows.", len(rows))

    client = LLMClient()
    developer_msg = _typo_task.grader_developer_message()

    correct_scores: list[float] = []
    incorrect_scores: list[float] = []
    disagreements: list[dict[str, Any]] = []
    parse_failures = 0

    for idx, gold in enumerate(rows):
        # Benchmark uses label='error'; trainee + grader vocab is
        # 'typo'. Normalise once so all downstream helpers see the
        # trainee word.
        norm_gold = dict(gold)
        if norm_gold.get("label") == "error":
            norm_gold["label"] = "typo"
        scores = {}
        for kind, synthesiser in [
            ("correct", _typo_task.synthesize_correct_response),
            ("incorrect", _typo_task.synthesize_incorrect_response),
        ]:
            fake_reply = synthesiser(norm_gold)
            reply_text = json.dumps(fake_reply, ensure_ascii=False)
            user_msg = _typo_task.render_grader_user_message(
                column=norm_gold.get("column", ""),
                candidate_value=norm_gold.get("original_value", ""),
                gold_label=norm_gold.get("label", ""),
                gold_correction=norm_gold.get("corrected_value"),
                model_output_text=reply_text,
            )
            full_prompt = developer_msg + "\n\n" + user_msg
            resp = client.query(
                args.grader_model,
                full_prompt,
                reasoning_effort=args.reasoning_effort,
            )
            if not resp.ok:
                logger.warning(
                    "Row %d %s reply: grader call failed (%s).",
                    idx, kind, resp.error,
                )
                scores[kind] = None
                continue
            score = _typo_task.parse_grader_score(resp.text)
            if score is None:
                parse_failures += 1
                logger.warning(
                    "Row %d %s: could not parse Score: from grader text: %r",
                    idx, kind, resp.text[:200],
                )
                scores[kind] = None
                continue
            scores[kind] = score

        if scores.get("correct") is not None:
            correct_scores.append(scores["correct"])
        if scores.get("incorrect") is not None:
            incorrect_scores.append(scores["incorrect"])
        if scores.get("correct") is not None and scores.get("incorrect") is not None:
            if scores["correct"] <= scores["incorrect"]:
                disagreements.append({
                    "row_index": idx,
                    "source": gold.get("source"),
                    "column": gold.get("column"),
                    "original_value": gold.get("original_value"),
                    "label": gold.get("label"),
                    "score_correct": scores["correct"],
                    "score_incorrect": scores["incorrect"],
                })

    def _mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else float("nan")

    n_pairs = min(len(correct_scores), len(incorrect_scores))
    agreement = (
        (n_pairs - len(disagreements)) / n_pairs if n_pairs else float("nan")
    )
    summary = {
        "grader_model": args.grader_model,
        "n_rows_sampled": len(rows),
        "n_correct_scored": len(correct_scores),
        "n_incorrect_scored": len(incorrect_scores),
        "n_parse_failures": parse_failures,
        "mean_score_correct": _mean(correct_scores),
        "mean_score_incorrect": _mean(incorrect_scores),
        "agreement_rate": agreement,
        "n_disagreements": len(disagreements),
    }
    print(json.dumps(summary, indent=2))

    if args.report_path:
        report = {
            "summary": summary,
            "disagreements": disagreements,
        }
        Path(args.report_path).write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("Wrote grader calibration report to %s", args.report_path)

    threshold = args.min_agreement
    if not (agreement >= threshold):  # NaN-safe (NaN >= x is False)
        logger.warning(
            "Grader agreement %.3f below threshold %.3f -- refine the "
            "grader prompt before training.",
            agreement, threshold,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _resolve_default_endpoint() -> str:
    """Pick the default RFT endpoint using the same precedence as ``LLMClient``.

    Order:
      1. ``AZURE_OPENAI_ENDPOINT`` (single-endpoint override).
      2. First entry of ``AZURE_OPENAI_ENDPOINTS`` (pool, comma-separated).
      3. First entry of :data:`src.core.llm_client.DEFAULT_ENDPOINTS`.
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
    return DEFAULT_ENDPOINTS[0]


def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--endpoint",
        default=None,
        help="Azure OpenAI resource endpoint, e.g. https://<resource>.openai.azure.com/. "
        "Defaults (in order): $AZURE_OPENAI_ENDPOINT, first of "
        "$AZURE_OPENAI_ENDPOINTS, first of src.core.llm_client.DEFAULT_ENDPOINTS.",
    )
    p.add_argument(
        "--api-version",
        default=DEFAULT_API_VERSION,
        help=f"Azure OpenAI REST API version (default {DEFAULT_API_VERSION}).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Where to store JSONL data and state.json (default {DEFAULT_OUTPUT_DIR}).",
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    # prepare
    sp = sub.add_parser("prepare", help="Generate synthetic train/valid JSONL files.")
    _add_common_args(sp)
    sp.add_argument("--task", choices=["countdown", "typo"], default="countdown",
                    help="Which task to prepare. 'countdown' generates synthetic "
                    "puzzles; 'typo' stages the JSONLs produced by "
                    "experiments.synth_typo_llm.")
    sp.add_argument("--train-size", type=int, default=100,
                    help="countdown only -- number of puzzles to synthesise.")
    sp.add_argument("--valid-size", type=int, default=50,
                    help="countdown only -- number of puzzles to synthesise.")
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--train-jsonl", default=None,
                    help="typo only -- path to train JSONL produced by synth.")
    sp.add_argument("--valid-jsonl", default=None,
                    help="typo only -- path to valid JSONL produced by synth.")
    sp.set_defaults(func=cmd_prepare)

    # upload
    sp = sub.add_parser("upload", help="Upload train/valid files to Azure OpenAI.")
    _add_common_args(sp)
    sp.add_argument("--task", choices=["countdown", "typo"], default="countdown")
    sp.set_defaults(func=cmd_upload)

    # submit
    sp = sub.add_parser("submit", help="Submit the RFT fine-tuning job.")
    _add_common_args(sp)
    sp.add_argument("--task", choices=["countdown", "typo"], default="countdown")
    sp.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    sp.add_argument("--grader-model", default=DEFAULT_GRADER_MODEL)
    sp.add_argument("--suffix", default="countdown-rft-demo")
    sp.add_argument("--n-epochs", type=int, default=1)
    sp.add_argument("--batch-size", type=int, default=4)
    sp.add_argument("--eval-interval", type=int, default=5)
    sp.add_argument("--eval-samples", type=int, default=2)
    sp.add_argument(
        "--reasoning-effort",
        choices=["low", "medium", "high"],
        default="medium",
    )
    sp.add_argument("--learning-rate-multiplier", type=float, default=1.0)
    sp.add_argument("--compute-multiplier", type=float, default=1.0)
    sp.add_argument(
        "--skip-model-check",
        action="store_true",
        help="Skip the capabilities.fine_tune pre-flight check.",
    )
    sp.set_defaults(func=cmd_submit)

    # status
    sp = sub.add_parser("status", help="Poll a fine-tuning job.")
    _add_common_args(sp)
    sp.add_argument("--job-id", default=None, help="Defaults to the last submitted job.")
    sp.add_argument("--follow", action="store_true", help="Poll until terminal status.")
    sp.add_argument("--poll-interval", type=int, default=30, help="Seconds between polls when --follow.")
    sp.set_defaults(func=cmd_status)

    # cancel
    sp = sub.add_parser("cancel", help="Cancel a running fine-tuning job.")
    _add_common_args(sp)
    sp.add_argument("--job-id", default=None)
    sp.set_defaults(func=cmd_cancel)

    # models
    sp = sub.add_parser(
        "models",
        help="List fine-tunable models on the endpoint (use --all for every model).",
    )
    _add_common_args(sp)
    sp.add_argument(
        "--all",
        action="store_true",
        help="Show every model the endpoint returns, not just those with "
        "capabilities.fine_tune=true.",
    )
    sp.set_defaults(func=cmd_models)

    # all
    sp = sub.add_parser("all", help="Run prepare -> upload -> submit -> follow.")
    _add_common_args(sp)
    sp.add_argument("--task", choices=["countdown", "typo"], default="countdown")
    sp.add_argument("--train-size", type=int, default=100)
    sp.add_argument("--valid-size", type=int, default=50)
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--train-jsonl", default=None,
                    help="typo only -- path to train JSONL produced by synth.")
    sp.add_argument("--valid-jsonl", default=None,
                    help="typo only -- path to valid JSONL produced by synth.")
    sp.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    sp.add_argument("--grader-model", default=DEFAULT_GRADER_MODEL)
    sp.add_argument("--suffix", default="countdown-rft-demo")
    sp.add_argument("--n-epochs", type=int, default=1)
    sp.add_argument("--batch-size", type=int, default=4)
    sp.add_argument("--eval-interval", type=int, default=5)
    sp.add_argument("--eval-samples", type=int, default=2)
    sp.add_argument(
        "--reasoning-effort",
        choices=["low", "medium", "high"],
        default="medium",
    )
    sp.add_argument("--learning-rate-multiplier", type=float, default=1.0)
    sp.add_argument("--compute-multiplier", type=float, default=1.0)
    sp.add_argument("--poll-interval", type=int, default=30)
    sp.add_argument(
        "--skip-model-check",
        action="store_true",
        help="Skip the capabilities.fine_tune pre-flight check.",
    )
    sp.set_defaults(func=cmd_all)

    # calibrate-grader (typo task only)
    sp = sub.add_parser(
        "calibrate-grader",
        help="Smoke-test the typo grader against the eval split BEFORE training.",
    )
    _add_common_args(sp)
    sp.add_argument(
        "--eval-jsonl",
        default="outputs/typo_rft/benchmark_split/eval.jsonl",
        help="Path to the eval JSONL from experiments.split_typo_benchmark.",
    )
    sp.add_argument(
        "--grader-model",
        # Use a chat model deployed on the pool by default so this
        # subcommand is runnable without a private o3-mini deployment.
        # The ACTUAL training grader (configured in submit/all) defaults
        # to o3-mini-2025-01-31 -- pass --grader-model o3-mini-... here
        # when your endpoint exposes it to test the production grader.
        default="gpt-5.2,gpt-5.4,gpt-5.5",
        help="Grader model (single name or comma-separated pool). "
        "LLMClient routes around endpoints that lack the model.",
    )
    sp.add_argument("--sample-size", type=int, default=20,
                    help="Number of eval rows to sample for calibration.")
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--reasoning-effort",
                    choices=["low", "medium", "high"], default="medium")
    sp.add_argument("--min-agreement", type=float, default=0.85,
                    help="Warning threshold for correct-vs-incorrect score ordering.")
    sp.add_argument("--report-path", default=None,
                    help="Optional JSON path to dump full per-row scores + disagreements.")
    sp.set_defaults(func=cmd_calibrate_grader)

    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = build_parser()
    args = parser.parse_args(argv)

    # Resolve endpoint default for any network-touching command. ``prepare``
    # is local-only, so we leave it unset there to avoid a misleading log line.
    if args.command != "prepare" and not args.endpoint:
        args.endpoint = _resolve_default_endpoint()
        logger.info("Using endpoint %s (auto-resolved).", args.endpoint)

    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
