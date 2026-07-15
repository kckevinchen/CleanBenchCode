"""Typo v7 RFT CLI for Azure OpenAI.

Thin wrapper around v5's CLI that uses the v7 grader variants
(``RFT.typo.v7._grader_variants``) and v7 data files.

Three reward variants:
    r1 -- additive    : 2.5 * T * C * (P + R)
    r2 -- harmonic    : 5   * T * C * 2PR / (P + R)
    r3 -- weighted    : 5   * T * C * (0.6P + 0.4R)

Hyperparameters: n_epochs=1, lr_mult=1.0, compute_mult=1.0 (the
"baseline" recipe that was v6's best FT result).

Usage::

    python RFT/typo/v7/scripts/typo_rft_v7.py all \\
        --reward-variant r1 \\
        --output-dir outputs/rft_typo_v7_r1 \\
        --suffix v7-r1 \\
        --endpoint <endpoint-name>
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
from RFT.typo.v5 import _task as _v5  # noqa: E402
from RFT.typo.v7._grader_variants import (  # noqa: E402
    V7_VARIANTS,
    build_grader_v7,
)

logger = logging.getLogger("typo_rft_v7")


DEFAULT_OUTPUT_DIR = Path("outputs/rft_typo_v7")
DEFAULT_BASE_MODEL = "o4-mini-2025-04-16"
DEFAULT_GRADER_MODEL = _v5.TYPO_V5_DEFAULT_GRADER_MODEL
DEFAULT_SUFFIX = "typo-rft-v7"

TRAIN_FILE = _v5.TYPO_V5_TRAIN_FILE
VALID_FILE = _v5.TYPO_V5_VALID_FILE

# v7 grader uses ALL of: precision, recall, trivial, column_cap.
# Recall sub-grader needs ``gold_typos`` and ``ambiguous_values`` on
# every row -- those were optional in v5/v6 (gold sub-grader was
# unwired) but are mandatory in v7.
_V7_REQUIRED_FIELDS = (
    "column", "source", "column_values_json",
    "gold_typos", "ambiguous_values",
)


def _copy_and_validate(src: Path, dst: Path) -> int:
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
                raise ValueError(f"{src}:{ln} invalid JSON: {exc}") from exc
            msgs = rec.get("messages")
            if not isinstance(msgs, list) or not msgs:
                raise ValueError(f"{src}:{ln} missing non-empty 'messages'.")
            if msgs[-1].get("role") != "user":
                raise ValueError(f"{src}:{ln} last message must be role=user.")
            for key in _V7_REQUIRED_FIELDS:
                if key not in rec:
                    raise ValueError(
                        f"{src}:{ln} missing top-level grader field {key!r}."
                    )
                if not isinstance(rec[key], str):
                    raise ValueError(
                        f"{src}:{ln} field {key!r} must be a JSON string "
                        f"(got {type(rec[key]).__name__})."
                    )
            out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1
    if count == 0:
        raise ValueError(f"{src} contained no records.")
    return count


def cmd_prepare(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_src = Path(args.train_jsonl)
    valid_src = Path(args.valid_jsonl)
    if not train_src.exists() or not valid_src.exists():
        raise FileNotFoundError(f"Missing {train_src} or {valid_src}.")
    for src, dst in [
        (train_src, args.output_dir / TRAIN_FILE),
        (valid_src, args.output_dir / VALID_FILE),
    ]:
        n = _copy_and_validate(src, dst)
        logger.info("Staged %d v7 records: %s -> %s", n, src, dst)


def cmd_upload(args: argparse.Namespace) -> None:
    upload_files(args, TRAIN_FILE, VALID_FILE)


def cmd_submit(args: argparse.Namespace) -> None:
    state = DemoState.load(args.output_dir)
    if not (state.train_file_id and state.valid_file_id):
        raise RuntimeError("Train/valid file IDs missing. Run `upload` first.")

    client = build_client(args.endpoint, args.api_version)
    assert_model_fine_tunable(
        client, args.base_model,
        skip=getattr(args, "skip_model_check", False),
    )
    wait_for_file_ready(client, state.train_file_id)
    wait_for_file_ready(client, state.valid_file_id)

    grader = build_grader_v7(args.reward_variant, grader_model=args.grader_model)
    response_format = _v5.build_response_format()

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
    if getattr(args, "seed", None) is not None:
        method["reinforcement"]["hyperparameters"]["seed"] = args.seed

    job = client.fine_tuning.jobs.create(
        model=args.base_model,
        training_file=state.train_file_id,
        validation_file=state.valid_file_id,
        method=method,
        suffix=args.suffix,
    )
    state.job_id = job.id
    state.save(args.output_dir)
    logger.info(
        "Submitted v7-%s RFT job: %s (status=%s)",
        args.reward_variant, job.id, job.status,
    )
    print(f"\nJob ID: {job.id}")


def cmd_all(args: argparse.Namespace) -> None:
    cmd_prepare(args)
    cmd_upload(args)
    cmd_submit(args)
    args.job_id = None
    args.follow = True
    cmd_status(args)


def _add_train_valid_jsonl_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--train-jsonl", default="RFT/typo/v7/data/train.jsonl")
    p.add_argument("--valid-jsonl", default="RFT/typo/v7/data/valid.jsonl")


def _add_submit_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--reward-variant",
        choices=list(V7_VARIANTS),
        required=True,
        help="Which v7 reward shape to use (r1=additive, r2=harmonic, "
             "r3=weighted).",
    )
    p.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    p.add_argument(
        "--grader-model", default=DEFAULT_GRADER_MODEL,
        choices=list(_v5.TYPO_V5_GRADER_MODEL_WHITELIST),
    )
    p.add_argument("--suffix", default=DEFAULT_SUFFIX)
    p.add_argument("--n-epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--eval-interval", type=int, default=5)
    p.add_argument("--eval-samples", type=int, default=2)
    p.add_argument("--reasoning-effort", choices=["low","medium","high"], default="medium")
    p.add_argument("--learning-rate-multiplier", type=float, default=1.0)
    p.add_argument("--compute-multiplier", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--skip-model-check", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("prepare")
    add_common_args(sp, DEFAULT_OUTPUT_DIR)
    _add_train_valid_jsonl_args(sp)
    sp.set_defaults(func=cmd_prepare)

    sp = sub.add_parser("upload")
    add_common_args(sp, DEFAULT_OUTPUT_DIR)
    sp.set_defaults(func=cmd_upload)

    sp = sub.add_parser("submit")
    add_common_args(sp, DEFAULT_OUTPUT_DIR)
    _add_submit_args(sp)
    sp.set_defaults(func=cmd_submit)

    sp = sub.add_parser("status")
    add_common_args(sp, DEFAULT_OUTPUT_DIR)
    add_status_args(sp)
    sp.set_defaults(func=cmd_status)

    sp = sub.add_parser("cancel")
    add_common_args(sp, DEFAULT_OUTPUT_DIR)
    add_cancel_args(sp)
    sp.set_defaults(func=cmd_cancel)

    sp = sub.add_parser("models")
    add_common_args(sp, DEFAULT_OUTPUT_DIR)
    add_models_args(sp)
    sp.set_defaults(func=cmd_models)

    sp = sub.add_parser("all")
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
