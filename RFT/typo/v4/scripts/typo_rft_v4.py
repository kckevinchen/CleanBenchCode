"""Typo v4 RFT CLI for Azure OpenAI.

Self-contained launcher for the v4 column-audit typo task. Reuses
the shared building blocks in ``experiments._rft_common`` (Azure
client, state file, upload waiter, model pre-flight, status / cancel
handlers) but wires the v4-specific grader, response_format, and
train/valid file shapes.

Subcommands
-----------
1. ``prepare`` -- validate + stage train/valid JSONL produced by
   ``RFT/typo/v4/scripts/synth_typo_llm_v4.py``.
2. ``upload``  -- upload both files to Azure OpenAI with
   ``purpose="fine-tune"``.
3. ``submit``  -- submit an RFT job with the v4 grader + response_format.
4. ``status``  -- poll a running job and print events.
5. ``cancel``  -- cancel a running job.
6. ``models``  -- list fine-tunable models on the endpoint.
7. ``all``     -- prepare -> upload -> submit -> follow.

Unlike v3, v4 uses a ``score_model`` grader (not a Python grader), so
the v3 "Job failed in postprocessing" failure mode that plagued the
Python-grader path is sidestepped entirely.

Usage
-----
::

    # End-to-end:
    python RFT/typo/v4/scripts/typo_rft_v4.py all \\
        --train-jsonl RFT/typo/v4/data/train.jsonl \\
        --valid-jsonl RFT/typo/v4/data/valid.jsonl \\
        --output-dir outputs/rft_typo_v4 \\
        --suffix typo-rft-v4
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from RFT._common import _rft_common
from RFT.typo.v4 import _task as _typo_v4_task  # noqa: E402
from RFT._common._rft_common import (  # noqa: E402
    DemoState,
    add_cancel_args,
    add_common_args,
    add_models_args,
    add_status_args,
    assert_model_fine_tunable,
    build_client,
    cmd_cancel,
    cmd_models,
    cmd_status,
    configure_logging,
    resolve_endpoint_if_needed,
    upload_files,
    wait_for_file_ready,
)

logger = logging.getLogger("typo_rft_v4")


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_OUTPUT_DIR = Path("outputs/rft_typo_v4")

# Same base model as v3 -- ``o4-mini-2025-04-16`` is the documented GA
# RFT model enabled on this repo's endpoint pool.
DEFAULT_BASE_MODEL = "o4-mini-2025-04-16"

# v4 grader model must be on the score_model whitelist.
DEFAULT_GRADER_MODEL = _typo_v4_task.TYPO_V4_DEFAULT_GRADER_MODEL

DEFAULT_SUFFIX = "typo-rft-v4"

TRAIN_FILE = _typo_v4_task.TYPO_V4_TRAIN_FILE
VALID_FILE = _typo_v4_task.TYPO_V4_VALID_FILE


# Required item.* fields the v4 grader template references. The RFT
# service only verifies that referenced template variables exist on
# every row, so a missing field here surfaces post-upload as an
# opaque HTTP 400 -- much harder to debug than a local AssertionError.
_V4_REQUIRED_FIELDS = (
    "column", "source", "gold_typos",
    "ambiguous_values", "column_values_json",
)


# ---------------------------------------------------------------------------
# Prepare
# ---------------------------------------------------------------------------


def _copy_and_validate(src: Path, dst: Path) -> int:
    """Copy JSONL src -> dst, validating each row matches the v4 RFT shape."""
    count = 0
    with src.open(encoding="utf-8") as in_fh, \
            dst.open("w", encoding="utf-8") as out_fh:
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
            for key in _V4_REQUIRED_FIELDS:
                if key not in rec:
                    raise ValueError(
                        f"{src}:{ln} missing top-level grader field "
                        f"{key!r}."
                    )
                if not isinstance(rec[key], str):
                    raise ValueError(
                        f"{src}:{ln} field {key!r} must be a JSON string "
                        f"(got {type(rec[key]).__name__}). Re-run "
                        f"build_record so it gets json.dumps'd."
                    )
            out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1
    if count == 0:
        raise ValueError(f"{src} contained no records.")
    return count


def cmd_prepare(args: argparse.Namespace) -> None:
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.train_jsonl or not args.valid_jsonl:
        raise RuntimeError(
            "v4 requires --train-jsonl and --valid-jsonl. Run "
            "RFT/typo/v4/scripts/synth_typo_llm_v4.py first."
        )
    train_src = Path(args.train_jsonl)
    valid_src = Path(args.valid_jsonl)
    if not train_src.exists() or not valid_src.exists():
        raise FileNotFoundError(
            f"Could not find one or both of {train_src}, {valid_src}."
        )

    train_dst = output_dir / TRAIN_FILE
    valid_dst = output_dir / VALID_FILE

    for src, dst in [(train_src, train_dst), (valid_src, valid_dst)]:
        n = _copy_and_validate(src, dst)
        logger.info("Staged %d v4 records: %s -> %s", n, src, dst)

    with train_dst.open(encoding="utf-8") as fh:
        sample = json.loads(fh.readline())
    print("\nSample staged record (truncated):")
    sample["messages"][0]["content"] = (
        sample["messages"][0]["content"][:200] + "... (truncated)"
    )
    print(json.dumps(sample, indent=2, ensure_ascii=False)[:2000])


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------


def cmd_upload(args: argparse.Namespace) -> None:
    upload_files(args, TRAIN_FILE, VALID_FILE)


# ---------------------------------------------------------------------------
# Submit
# ---------------------------------------------------------------------------


def cmd_submit(args: argparse.Namespace) -> None:
    state = DemoState.load(args.output_dir)
    if not (state.train_file_id and state.valid_file_id):
        raise RuntimeError(
            "Train/valid file IDs missing. Run `upload` first."
        )

    client = build_client(args.endpoint, args.api_version)
    assert_model_fine_tunable(
        client, args.base_model,
        skip=getattr(args, "skip_model_check", False),
    )
    wait_for_file_ready(client, state.train_file_id)
    wait_for_file_ready(client, state.valid_file_id)

    grader = _typo_v4_task.build_grader(grader_model=args.grader_model)
    response_format = _typo_v4_task.build_response_format()

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
    logger.info("Submitted v4 RFT job: %s (status=%s)", job.id, job.status)
    print(f"\nJob ID: {job.id}")
    print(
        f"Watch with: python {Path(__file__).name} status "
        f"--output-dir {args.output_dir} --follow"
    )


# ---------------------------------------------------------------------------
# all -- prepare + upload + submit + follow
# ---------------------------------------------------------------------------


def cmd_all(args: argparse.Namespace) -> None:
    cmd_prepare(args)
    cmd_upload(args)
    cmd_submit(args)
    args.job_id = None
    args.follow = True
    cmd_status(args)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _add_train_valid_jsonl_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--train-jsonl",
        default="RFT/typo/v4/data/train.jsonl",
        help="Path to v4 train JSONL (default: RFT/typo/v4/data/train.jsonl).",
    )
    p.add_argument(
        "--valid-jsonl",
        default="RFT/typo/v4/data/valid.jsonl",
        help="Path to v4 valid JSONL (default: RFT/typo/v4/data/valid.jsonl).",
    )


def _add_submit_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    p.add_argument(
        "--grader-model", default=DEFAULT_GRADER_MODEL,
        choices=list(_typo_v4_task.TYPO_V4_GRADER_MODEL_WHITELIST),
    )
    p.add_argument("--suffix", default=DEFAULT_SUFFIX)
    p.add_argument("--n-epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--eval-interval", type=int, default=5)
    p.add_argument("--eval-samples", type=int, default=2)
    p.add_argument(
        "--reasoning-effort",
        choices=["low", "medium", "high"],
        default="medium",
    )
    p.add_argument("--learning-rate-multiplier", type=float, default=1.0)
    p.add_argument("--compute-multiplier", type=float, default=1.0)
    p.add_argument(
        "--skip-model-check",
        action="store_true",
        help="Skip the capabilities.fine_tune pre-flight check.",
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser(
        "prepare",
        help="Validate + stage v4 train/valid JSONL into output-dir.",
    )
    add_common_args(sp, DEFAULT_OUTPUT_DIR)
    _add_train_valid_jsonl_args(sp)
    sp.set_defaults(func=cmd_prepare)

    sp = sub.add_parser(
        "upload", help="Upload train/valid files to Azure OpenAI.",
    )
    add_common_args(sp, DEFAULT_OUTPUT_DIR)
    sp.set_defaults(func=cmd_upload)

    sp = sub.add_parser("submit", help="Submit the v4 RFT fine-tuning job.")
    add_common_args(sp, DEFAULT_OUTPUT_DIR)
    _add_submit_args(sp)
    sp.set_defaults(func=cmd_submit)

    sp = sub.add_parser("status", help="Poll a fine-tuning job.")
    add_common_args(sp, DEFAULT_OUTPUT_DIR)
    add_status_args(sp)
    sp.set_defaults(func=cmd_status)

    sp = sub.add_parser("cancel", help="Cancel a running fine-tuning job.")
    add_common_args(sp, DEFAULT_OUTPUT_DIR)
    add_cancel_args(sp)
    sp.set_defaults(func=cmd_cancel)

    sp = sub.add_parser(
        "models",
        help="List fine-tunable models on the endpoint.",
    )
    add_common_args(sp, DEFAULT_OUTPUT_DIR)
    add_models_args(sp)
    sp.set_defaults(func=cmd_models)

    sp = sub.add_parser(
        "all", help="Run prepare -> upload -> submit -> follow.",
    )
    add_common_args(sp, DEFAULT_OUTPUT_DIR)
    _add_train_valid_jsonl_args(sp)
    _add_submit_args(sp)
    sp.add_argument("--poll-interval", type=int, default=30)
    sp.set_defaults(func=cmd_all)

    return p


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    parser = build_parser()
    args = parser.parse_args(argv)
    resolve_endpoint_if_needed(args, network_command=args.command != "prepare")
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
