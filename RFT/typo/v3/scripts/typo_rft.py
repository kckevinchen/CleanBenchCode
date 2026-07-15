"""Typo / typo_v3 Reinforcement Fine-Tuning (RFT) CLI on Azure OpenAI.

Mirrors the lifecycle of :mod:`experiments.demo_rft` (which stays scoped to
the Countdown reference demo) but specialises every step for the typo
tasks defined in :mod:`experiments._typo_task` (v2, per-cell) and
:mod:`experiments._typo_v3_task` (v3, per-column).

Subcommands
-----------
1. ``prepare``           - validate + stage train/valid JSONL files produced
                           by ``experiments.synth_typo_llm`` (v2) or
                           ``experiments.synth_typo_llm_v3`` (v3).
2. ``upload``            - upload both files to Azure OpenAI with
                           ``purpose="fine-tune"``.
3. ``submit``            - submit an RFT job with the task-specific grader
                           and JSON ``response_format``.
4. ``status``            - poll a running job and print events.
5. ``cancel``            - cancel a running job.
6. ``models``            - list fine-tunable models on the endpoint.
7. ``all``               - prepare -> upload -> submit and tail status.
8. ``calibrate-grader``  - smoke-test the v2 score-model grader against
                           eval-split rows BEFORE training (v2 only; v3
                           uses a deterministic Python grader and needs
                           no calibration).

Authentication and endpoint selection are inherited unchanged from
:mod:`experiments._rft_common`.

Usage
-----
::

    # End-to-end v3 fine-tune:
    python -m experiments.typo_rft prepare --task typo_v3 \\
        --train-jsonl RFT/typo/v3/data/train.jsonl \\
        --valid-jsonl RFT/typo/v3/data/valid.jsonl \\
        --output-dir outputs/rft_typo_v3
    python -m experiments.typo_rft upload --task typo_v3 \\
        --output-dir outputs/rft_typo_v3
    python -m experiments.typo_rft submit --task typo_v3 \\
        --output-dir outputs/rft_typo_v3 --suffix typo-rft-v3

    # Or all at once:
    python -m experiments.typo_rft all --task typo_v3 \\
        --train-jsonl RFT/typo/v3/data/train.jsonl \\
        --valid-jsonl RFT/typo/v3/data/valid.jsonl \\
        --output-dir outputs/rft_typo_v3 --suffix typo-rft-v3
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any

# Make ``src.core.llm_client`` importable when invoked as a module from
# the repo root.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from RFT._common import _rft_common, _typo_task, _typo_v3_task
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

logger = logging.getLogger("typo_rft")


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_OUTPUT_DIR = Path("outputs/rft_typo")

# RFT-supported base model. ``o4-mini-2025-04-16`` is the documented GA
# RFT model and is enabled on this repo's endpoint pool.
DEFAULT_BASE_MODEL = "o4-mini-2025-04-16"

# Model used by the v2 ``score_model`` grader. v3's grader is a Python
# grader and ignores this flag.
DEFAULT_GRADER_MODEL = "o3-mini-2025-01-31"

DEFAULT_SUFFIX = "typo-rft"


# ---------------------------------------------------------------------------
# Task plumbing
# ---------------------------------------------------------------------------

TASK_CHOICES = ("typo", "typo_v3")


def _task_filenames(task: str) -> tuple[str, str]:
    """Resolve (train_filename, valid_filename) for the chosen task."""
    if task == "typo_v3":
        return (
            _typo_v3_task.TYPO_V3_TRAIN_FILE,
            _typo_v3_task.TYPO_V3_VALID_FILE,
        )
    return _typo_task.TYPO_TRAIN_FILE, _typo_task.TYPO_VALID_FILE


# v2 vs v3 required item.* fields. The RFT service only verifies that
# referenced template variables exist; getting the wrong shape here
# surfaces post-upload as an opaque HTTP 400 -- much harder to debug
# than a local AssertionError.
_TYPO_V2_REQUIRED_FIELDS = (
    "gold_label", "gold_correction", "candidate_value", "column", "source",
)
_TYPO_V3_REQUIRED_FIELDS = (
    "column", "source", "gold_typos", "ambiguous_values",
)


def _copy_and_validate_typo_jsonl(
    src: Path, dst: Path, *, task: str,
) -> int:
    """Copy JSONL src -> dst, validating each row matches the RFT shape.

    The RFT API requires ``messages`` ending in a user turn AND every
    top-level non-``messages`` field to be a JSON string (so the grader
    template can do textual substitution). A misconfigured synth would
    otherwise surface only as an opaque HTTP 400 mid-upload.

    ``task`` selects which top-level fields are required -- v2's
    per-cell shape vs v3's per-column shape.
    """
    required_fields = (
        _TYPO_V3_REQUIRED_FIELDS if task == "typo_v3"
        else _TYPO_V2_REQUIRED_FIELDS
    )
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
            for key in required_fields:
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
# Prepare
# ---------------------------------------------------------------------------


def cmd_prepare(args: argparse.Namespace) -> None:
    """Validate + stage typo train/valid JSONL into ``output_dir``.

    Caller must supply ``--train-jsonl`` and ``--valid-jsonl``; we
    refuse to silently regenerate from ``data/`` because that would
    mask a misconfigured synth run.

    Works for both v2 (per-cell) and v3 (per-column) variants -- the
    only difference is which set of item.* fields the validator
    expects and which filenames the upload step will look for.
    """
    task = args.task
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.train_jsonl or not args.valid_jsonl:
        raise RuntimeError(
            f"--task {task} requires --train-jsonl and --valid-jsonl. "
            f"Run the corresponding synth script first."
        )
    train_src = Path(args.train_jsonl)
    valid_src = Path(args.valid_jsonl)
    if not train_src.exists() or not valid_src.exists():
        raise FileNotFoundError(
            f"Could not find one or both of {train_src}, {valid_src}."
        )

    train_fn, valid_fn = _task_filenames(task)
    train_dst = output_dir / train_fn
    valid_dst = output_dir / valid_fn

    for src, dst in [(train_src, train_dst), (valid_src, valid_dst)]:
        n = _copy_and_validate_typo_jsonl(src, dst, task=task)
        logger.info("Staged %d %s records: %s -> %s", n, task, src, dst)

    with train_dst.open(encoding="utf-8") as fh:
        sample = json.loads(fh.readline())
    print("\nSample staged record:")
    print(json.dumps(sample, indent=2))


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------


def cmd_upload(args: argparse.Namespace) -> None:
    train_fn, valid_fn = _task_filenames(args.task)
    upload_files(args, train_fn, valid_fn)


# ---------------------------------------------------------------------------
# Submit
# ---------------------------------------------------------------------------


def cmd_submit(args: argparse.Namespace) -> None:
    state = DemoState.load(args.output_dir)
    if not (state.train_file_id and state.valid_file_id):
        raise RuntimeError("Train/valid file IDs missing. Run `upload` first.")

    client = build_client(args.endpoint, args.api_version)

    # Pre-flight: fail fast with an actionable message if the chosen base
    # model isn't fine-tunable on this resource.
    assert_model_fine_tunable(
        client, args.base_model, skip=getattr(args, "skip_model_check", False)
    )

    # Defensive: if the operator runs `submit` directly (or `upload` was
    # interrupted before its polling completed), make sure both files are
    # actually importable before issuing the job-create request.
    wait_for_file_ready(client, state.train_file_id)
    wait_for_file_ready(client, state.valid_file_id)

    task = args.task
    if task == "typo_v3":
        # v3 uses a Python grader (set-F1) that does not need a grader
        # model; the parameter is accepted-but-ignored for API parity
        # with v2's build_grader signature.
        grader = _typo_v3_task.build_grader(args.grader_model)
        response_format = _typo_v3_task.build_response_format()
    else:
        grader = _typo_task.build_grader(args.grader_model)
        response_format = _typo_task.build_response_format()

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
    print(f"Watch with: python -m experiments.typo_rft status --job-id {job.id}")


# ---------------------------------------------------------------------------
# all - prepare + upload + submit + follow
# ---------------------------------------------------------------------------


def cmd_all(args: argparse.Namespace) -> None:
    cmd_prepare(args)
    cmd_upload(args)
    cmd_submit(args)
    # Reuse the shared follow logic to tail the job. ``cmd_status`` reads
    # ``args.job_id`` (and falls back to ``state.job_id`` when None), but
    # the ``all`` subparser doesn't define that flag, so prime it here.
    args.job_id = None
    args.follow = True
    cmd_status(args)


# ---------------------------------------------------------------------------
# calibrate-grader (v2 only)
# ---------------------------------------------------------------------------


def cmd_calibrate_grader(args: argparse.Namespace) -> None:
    """Run the v2 typo grader against eval-split rows BEFORE training.

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

    v3 uses a deterministic Python grader and does not need this step.
    """
    from src.core.llm_client import LLMClient  # local import: optional dep

    eval_path = Path(args.eval_jsonl)
    if not eval_path.exists():
        raise FileNotFoundError(
            f"Eval JSONL not found: {eval_path}. Run "
            f"RFT.typo.v3.scripts.split_typo_benchmark_v3 first."
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


def _add_task_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--task",
        choices=TASK_CHOICES,
        default="typo_v3",
        help="Which typo task variant to run (default typo_v3).",
    )


def _add_train_valid_jsonl_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--train-jsonl", default=None,
        help="Path to train JSONL produced by the synth step.",
    )
    p.add_argument(
        "--valid-jsonl", default=None,
        help="Path to valid JSONL produced by the synth step.",
    )


def _add_submit_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    p.add_argument("--grader-model", default=DEFAULT_GRADER_MODEL)
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

    # prepare
    sp = sub.add_parser(
        "prepare",
        help="Validate + stage train/valid JSONL files into output-dir.",
    )
    add_common_args(sp, DEFAULT_OUTPUT_DIR)
    _add_task_arg(sp)
    _add_train_valid_jsonl_args(sp)
    sp.set_defaults(func=cmd_prepare)

    # upload
    sp = sub.add_parser("upload", help="Upload train/valid files to Azure OpenAI.")
    add_common_args(sp, DEFAULT_OUTPUT_DIR)
    _add_task_arg(sp)
    sp.set_defaults(func=cmd_upload)

    # submit
    sp = sub.add_parser("submit", help="Submit the RFT fine-tuning job.")
    add_common_args(sp, DEFAULT_OUTPUT_DIR)
    _add_task_arg(sp)
    _add_submit_args(sp)
    sp.set_defaults(func=cmd_submit)

    # status
    sp = sub.add_parser("status", help="Poll a fine-tuning job.")
    add_common_args(sp, DEFAULT_OUTPUT_DIR)
    add_status_args(sp)
    sp.set_defaults(func=cmd_status)

    # cancel
    sp = sub.add_parser("cancel", help="Cancel a running fine-tuning job.")
    add_common_args(sp, DEFAULT_OUTPUT_DIR)
    add_cancel_args(sp)
    sp.set_defaults(func=cmd_cancel)

    # models
    sp = sub.add_parser(
        "models",
        help="List fine-tunable models on the endpoint (use --all for every model).",
    )
    add_common_args(sp, DEFAULT_OUTPUT_DIR)
    add_models_args(sp)
    sp.set_defaults(func=cmd_models)

    # all
    sp = sub.add_parser("all", help="Run prepare -> upload -> submit -> follow.")
    add_common_args(sp, DEFAULT_OUTPUT_DIR)
    _add_task_arg(sp)
    _add_train_valid_jsonl_args(sp)
    _add_submit_args(sp)
    sp.add_argument("--poll-interval", type=int, default=30)
    sp.set_defaults(func=cmd_all)

    # calibrate-grader (v2 only)
    sp = sub.add_parser(
        "calibrate-grader",
        help="Smoke-test the v2 typo grader against the eval split BEFORE training.",
    )
    add_common_args(sp, DEFAULT_OUTPUT_DIR)
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
    configure_logging()
    parser = build_parser()
    args = parser.parse_args(argv)

    # ``prepare`` is local-only, so leave endpoint unset there to avoid a
    # misleading log line.
    resolve_endpoint_if_needed(args, network_command=args.command != "prepare")

    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
