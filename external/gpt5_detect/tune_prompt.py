"""Leakage-gated three-agent prompt tuning for the GPT-5.5 cell detector.

Agents have deliberately asymmetric access:

* prompt writer: current prompt + judge-approved generic feedback only;
* evaluator: development tables, labels, predictions, and metrics;
* judge: evaluator feedback + the same development dossier;
* no agent receives test identifiers, tables, labels, predictions, or metrics.

The selected prompt is locked by SHA-256 before the target detector is run on
the untouched test split. An optional all-table run is a non-primary artifact.

Examples:
    python external/gpt5_detect/tune_prompt.py --dry-run
    python external/gpt5_detect/tune_prompt.py --rounds 5 --run-dir outputs/prompt_tuning/run1
    python external/gpt5_detect/tune_prompt.py --status --run-dir outputs/prompt_tuning/run1
    python external/gpt5_detect/tune_prompt.py --resume --run-dir outputs/prompt_tuning/run1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import traceback
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
REIN_DIR = SCRIPT_DIR.parent / "rein"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REIN_DIR))
sys.path.insert(0, str(REPO_ROOT))

import common as C  # noqa: E402
import prompt as BASE_PROMPT  # noqa: E402
import tuning as T  # noqa: E402

DETECTOR_SCRIPT = SCRIPT_DIR / "run.py"
SCORER_SCRIPT = REIN_DIR / "score.py"

PROMPT_WRITER_SYSTEM = """\
You are the PROMPT WRITER in a leakage-controlled prompt-optimization workflow.

You may see only a current system prompt and generic suggestions approved by a
separate judge. You never see evaluation tables, values, column names, source
identifiers, labels, predictions, examples, metrics, or test data.

Write a complete replacement system prompt for a model that detects four
cell-level data-quality errors: typo, inconsistency, semantic_incompatible, and
numeric_outlier. Preserve a strict JSON object with an `issues` array and the
fields row, column, error_type, observed, explanation, corrected_value, and
confidence. Keep instructions general across arbitrary tables. Do not invent
dataset-specific examples, literal values, column names, formats, counts, or
domains. Do not mention a benchmark, dataset, tuning process, feedback, or score.

Return one JSON object and nothing else:
{
  "system_prompt": "<complete replacement prompt>",
  "change_summary": ["<generic change>", "..."]
}
"""

EVALUATOR_SYSTEM = """\
You are the EVALUATOR in a leakage-controlled prompt-optimization workflow.

You may inspect a development-only dossier containing aggregate metrics and
sampled table contexts, labels, predictions, false positives, false negatives,
and true positives. Diagnose prompt-level causes and propose generic
improvements. Your response will be sent to a leakage judge and, only if it
passes, to a prompt writer that has no data access.

STRICT OUTPUT POLICY:
- Never mention or quote any source, table, file, identifier, row index, column
  name, cell value, domain entity, literal pattern, exact format, example,
  benchmark/corpus/dataset name, metric value, count, percentage, or split detail.
- Do not reproduce or paraphrase a sampled record.
- Use only high-level guidance that applies to arbitrary tabular data.
- You may refer generically to precision, recall, rows, columns, peer values,
  error-type definitions, confidence, and the JSON output contract.
- Give actionable changes, not a rewritten prompt.

Return one JSON object and nothing else:
{
  "summary": "<generic diagnosis without data details>",
  "suggestions": [
    {
      "category": "precision|recall|taxonomy|reasoning|output_contract",
      "change": "<generic prompt change>",
      "rationale": "<generic reason>"
    }
  ]
}
"""

JUDGE_SYSTEM = """\
You are the JUDGE in a leakage-controlled prompt-optimization workflow.

Compare evaluator feedback against the supplied development-only dossier and
the deterministic scanner report. Reject feedback if it contains or implies
any benchmark reference, source/table/file identifier, row or column location,
column name, cell value, domain-specific fact, literal pattern or exact format,
example derived from the data, metric/count/percentage, or other information
that could encode the development data. Also reject advice tailored to a
particular sampled schema or domain. Generic advice about precision, recall,
peer-value reasoning, the four public error types, confidence, and the JSON
contract is allowed.

The deterministic scanner is a mandatory lower bound: if it reports a
violation, you must reject. Do not quote protected content in your response.
Return generic categories only.

Return one JSON object and nothing else:
{
  "passed": true | false,
  "violations": [
    "benchmark_reference|data_identifier|data_value|domain_specificity|exact_format|example|location|metric"
  ]
}
Use an empty violations list when passed.
"""

INITIAL_WRITER_GUIDANCE = {
    "summary": "Create the first robust, domain-independent candidate.",
    "suggestions": [
        {
            "category": "reasoning",
            "change": (
                "Clarify evidence requirements and the distinction between local anomalies "
                "and legitimate variation while retaining broad type coverage."
            ),
            "rationale": (
                "The initial candidate should balance recall with explicit safeguards "
                "against unsupported flags."
            ),
        }
    ],
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _extract_json(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(
            r"^```(?:json)?\s*|\s*```$",
            "",
            stripped,
            flags=re.IGNORECASE,
        ).strip()
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        start, end = stripped.find("{"), stripped.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            value = json.loads(stripped[start : end + 1])
        except json.JSONDecodeError:
            return None
    return value if isinstance(value, dict) else None


def _transition(state_path: Path, state: dict[str, Any], status: str, **updates: Any) -> None:
    state["status"] = status
    state["updated_at"] = utc_now()
    state.update(updates)
    state.setdefault("events", []).append(
        {"at": state["updated_at"], "status": status, "round": state.get("current_round")}
    )
    T.atomic_write_json(state_path, state)


def _print_status(run_dir: Path) -> int:
    state_path = run_dir / "state.json"
    if not state_path.exists():
        raise SystemExit(f"No tuning state found at {state_path}")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    summary = {
        "run_dir": str(run_dir),
        "status": state.get("status"),
        "current_round": state.get("current_round"),
        "completed_rounds": [
            {
                "round": value["round"],
                "precision": value["metrics"]["precision"],
                "recall": value["metrics"]["recall"],
                "f1": value["metrics"]["f1"],
            }
            for value in state.get("rounds", [])
        ],
        "best_round": state.get("best_round"),
        "locked_prompt_sha256": state.get("locked_prompt_sha256"),
        "final_test_metrics": state.get("final_test_metrics"),
        "failure": state.get("failure"),
    }
    print(json.dumps(summary, indent=2))
    return 0


def _run_command(command: list[str], log_path: Path, metadata_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    T.atomic_write_json(metadata_path, {"command": command, "started_at": utc_now()})
    environment = dict(os.environ)
    environment["PYTHONUNBUFFERED"] = "1"
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update({"finished_at": utc_now(), "returncode": completed.returncode})
    T.atomic_write_json(metadata_path, metadata)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {completed.returncode}; see {log_path}"
        )


def _call_json_agent(
    client: Any,
    *,
    role: str,
    model: str,
    reasoning_effort: str,
    system_prompt: str,
    user_payload: dict[str, Any],
    artifact_dir: Path,
    validator: Callable[[Any], list[str]],
    attempts: int,
) -> dict[str, Any]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    prior_errors: list[str] = []
    for attempt in range(1, attempts + 1):
        request = {
            "role": role,
            "input": user_payload,
            "response_validation": prior_errors,
        }
        response = client.query(
            model,
            json.dumps(request, ensure_ascii=False),
            system_prompt=system_prompt,
            reasoning_effort=reasoning_effort,
        )
        prefix = artifact_dir / f"attempt_{attempt:02d}"
        prefix.with_suffix(".txt").write_text(response.text or "", encoding="utf-8")
        T.atomic_write_json(
            prefix.with_suffix(".meta.json"),
            {
                "role": role,
                "model": response.model or model,
                "endpoint": response.endpoint,
                "attempts": response.attempts,
                "ok": response.ok,
                "error": response.error,
            },
        )
        if not response.ok:
            prior_errors = ["agent call failed; return the required JSON on retry"]
            continue
        parsed = _extract_json(response.text)
        errors = validator(parsed)
        if parsed is not None:
            T.atomic_write_json(prefix.with_suffix(".json"), parsed)
        T.atomic_write_json(prefix.with_suffix(".validation.json"), {"errors": errors})
        if not errors and parsed is not None:
            T.atomic_write_json(artifact_dir / "accepted.json", parsed)
            return parsed
        prior_errors = errors or ["response was not a JSON object"]
    raise RuntimeError(f"{role} failed validation after {attempts} attempts")


def _writer_validator(
    previous_prompt: str,
    dossier: dict[str, Any] | None,
) -> Callable[[Any], list[str]]:
    def validate(payload: Any) -> list[str]:
        if not isinstance(payload, dict):
            return ["writer response must be a JSON object"]
        prompt = payload.get("system_prompt")
        if not isinstance(prompt, str):
            return ["system_prompt must be a string"]
        summary = payload.get("change_summary")
        errors = []
        if not isinstance(summary, list) or not all(
            isinstance(value, str) and value.strip() for value in summary
        ):
            errors.append("change_summary must be a list of non-empty generic strings")
        errors.extend(
            T.validate_candidate_prompt(
                prompt,
                previous_prompt=previous_prompt,
                dossier=dossier,
            )
        )
        return errors

    return validate


def _judge_categories(scan: dict[str, Any], judge: dict[str, Any]) -> list[str]:
    mapping = {
        "benchmark_reference": "benchmark_reference",
        "location_reference": "location",
        "file_reference": "data_identifier",
        "hash_identifier": "data_identifier",
        "numeric_literal": "metric",
        "example": "example",
        "identifier": "data_identifier",
        "header": "data_identifier",
        "cell_value": "data_value",
    }
    categories = {
        mapping.get(item["category"], "domain_specificity")
        for item in scan.get("violations", [])
    }
    categories.update(judge.get("violations", []))
    categories.discard("none")
    return sorted(categories)


def _approved_feedback(
    client: Any,
    *,
    dossier: dict[str, Any],
    round_dir: Path,
    evaluator_model: str,
    judge_model: str,
    reasoning_effort: str,
    agent_attempts: int,
    feedback_attempts: int,
) -> dict[str, Any]:
    rejection_categories: list[str] = []
    for feedback_attempt in range(1, feedback_attempts + 1):
        attempt_dir = round_dir / "feedback_gate" / f"feedback_{feedback_attempt:02d}"
        feedback = _call_json_agent(
            client,
            role="evaluator",
            model=evaluator_model,
            reasoning_effort=reasoning_effort,
            system_prompt=EVALUATOR_SYSTEM,
            user_payload={
                "development_dossier": dossier,
                "previous_rejection_categories": rejection_categories,
            },
            artifact_dir=attempt_dir / "evaluator",
            validator=T.validate_evaluator_feedback,
            attempts=agent_attempts,
        )
        scan = T.scan_for_development_leak(feedback, dossier)
        T.atomic_write_json(attempt_dir / "deterministic_scan.json", scan)
        judge = _call_json_agent(
            client,
            role="judge",
            model=judge_model,
            reasoning_effort=reasoning_effort,
            system_prompt=JUDGE_SYSTEM,
            user_payload={
                "development_dossier": dossier,
                "evaluator_feedback": feedback,
                "deterministic_scan": scan,
            },
            artifact_dir=attempt_dir / "judge",
            validator=T.validate_judge_verdict,
            attempts=agent_attempts,
        )
        passed = bool(scan["passed"] and judge["passed"])
        rejection_categories = _judge_categories(scan, judge)
        gate = {
            "passed": passed,
            "feedback_attempt": feedback_attempt,
            "rejection_categories": rejection_categories,
        }
        T.atomic_write_json(attempt_dir / "gate.json", gate)
        if passed:
            approved = {
                "summary": feedback["summary"],
                "suggestions": feedback["suggestions"],
                "gate": {
                    "feedback_attempt": feedback_attempt,
                    "deterministic_scan_passed": True,
                    "judge_passed": True,
                },
            }
            T.atomic_write_json(round_dir / "approved_feedback.json", approved)
            return approved
    raise RuntimeError(
        f"No evaluator feedback passed the leakage gate after {feedback_attempts} attempts"
    )


def _run_detector_stage(
    *,
    stage_dir: Path,
    stage_name: str,
    prompt_path: Path,
    datasets: list[str],
    benchmark_sha256: str,
    ds_root: Path,
    neutral_file: Path,
    cache_dir: Path,
    args: argparse.Namespace,
) -> tuple[Path, Path, Path]:
    stage_dir.mkdir(parents=True, exist_ok=True)
    dataset_list_path = stage_dir / "datasets.json"
    T.atomic_write_json(
        dataset_list_path,
        {"datasets": datasets, "benchmark_sha256": benchmark_sha256},
    )
    tag = re.sub(r"[^A-Za-z0-9_.-]", "_", stage_name)
    predictions_path = stage_dir / f"predictions_{tag}.jsonl"
    issues_path = stage_dir / f"issues_{tag}.jsonl"
    score_path = stage_dir / "score.json"
    detector_marker = stage_dir / ".detector_complete"

    if not detector_marker.exists():
        command = [
            sys.executable,
            str(DETECTOR_SCRIPT),
            "--datasets-dir",
            str(ds_root),
            "--results-dir",
            str(stage_dir),
            "--model",
            args.model,
            "--chunk-size",
            str(args.chunk_size),
            "--reasoning-effort",
            args.reasoning_effort,
            "--min-confidence",
            args.min_confidence,
            "--concurrency",
            str(args.concurrency),
            "--dataset-list",
            str(dataset_list_path),
            "--system-prompt-file",
            str(prompt_path),
            "--tag",
            tag,
            "--cache-dir",
            str(cache_dir),
        ]
        if args.no_target_cache:
            command.append("--no-cache")
        _run_command(
            command,
            stage_dir / "detector.log",
            stage_dir / "detector_command.json",
        )
        detector_marker.write_text(utc_now(), encoding="utf-8")

    if not predictions_path.exists() or not issues_path.exists():
        raise RuntimeError(f"Detector stage {stage_name} did not produce required artifacts")

    if not score_path.exists():
        command = [
            sys.executable,
            str(SCORER_SCRIPT),
            "--predictions",
            str(predictions_path),
            "--datasets-dir",
            str(ds_root),
            "--neutral-file",
            str(neutral_file),
            "--out",
            str(score_path),
        ]
        _run_command(
            command,
            stage_dir / "score.log",
            stage_dir / "score_command.json",
        )
    T.score_summary(score_path)
    return predictions_path, issues_path, score_path


def _state_round(
    state: dict[str, Any],
    round_number: int,
    metrics: dict[str, Any],
    prompt_path: Path,
    predictions_path: Path,
    score_path: Path,
) -> None:
    record = {
        "round": round_number,
        "prompt_path": str(prompt_path),
        "prompt_sha256": T.file_sha256(prompt_path),
        "predictions_path": str(predictions_path),
        "predictions_sha256": T.file_sha256(predictions_path),
        "score_path": str(score_path),
        "score_sha256": T.file_sha256(score_path),
        "metrics": metrics,
    }
    existing = [
        value for value in state.setdefault("rounds", []) if value["round"] != round_number
    ]
    existing.append(record)
    state["rounds"] = sorted(existing, key=lambda value: value["round"])


def _verify_round_artifacts(state: dict[str, Any], run_dir: Path) -> None:
    for record in state.get("rounds", []):
        for path_key, hash_key in (
            ("prompt_path", "prompt_sha256"),
            ("predictions_path", "predictions_sha256"),
            ("score_path", "score_sha256"),
        ):
            path = Path(record[path_key]).resolve()
            try:
                path.relative_to(run_dir)
            except ValueError as exc:
                raise ValueError(f"round artifact escapes run directory: {path}") from exc
            if not path.exists() or T.file_sha256(path) != record[hash_key]:
                raise ValueError(f"round artifact changed after scoring: {path}")


def _implementation_fingerprint() -> str:
    digest = hashlib.sha256()
    for path in (
        Path(__file__),
        SCRIPT_DIR / "tuning.py",
        DETECTOR_SCRIPT,
        SCRIPT_DIR / "prompt.py",
        SCORER_SCRIPT,
    ):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _config(
    args: argparse.Namespace,
    ds_root: Path,
    neutral_file: Path,
) -> dict[str, Any]:
    return {
        "rounds": args.rounds,
        "dev_fraction": args.dev_fraction,
        "split_seed": args.split_seed,
        "model": args.model,
        "writer_model": args.writer_model or args.agent_model,
        "evaluator_model": args.evaluator_model or args.agent_model,
        "judge_model": args.judge_model or args.agent_model,
        "chunk_size": args.chunk_size,
        "reasoning_effort": args.reasoning_effort,
        "agent_reasoning_effort": args.agent_reasoning_effort,
        "min_confidence": args.min_confidence,
        "concurrency": args.concurrency,
        "sample_limit": args.sample_limit,
        "agent_attempts": args.agent_attempts,
        "feedback_attempts": args.feedback_attempts,
        "run_full": args.run_full,
        "no_target_cache": args.no_target_cache,
        "datasets_dir": str(ds_root),
        "neutral_file": str(neutral_file),
        "neutral_file_sha256": T.file_sha256(neutral_file),
        "implementation_sha256": _implementation_fingerprint(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--dev-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=20260713)
    parser.add_argument("--model", default="gpt-5.5", help="target detector model")
    parser.add_argument("--agent-model", default="gpt-5.5")
    parser.add_argument("--writer-model", default="")
    parser.add_argument("--evaluator-model", default="")
    parser.add_argument("--judge-model", default="")
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--agent-reasoning-effort", default="medium")
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--min-confidence", choices=["low", "medium", "high"], default="low")
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--sample-limit", type=int, default=6)
    parser.add_argument("--agent-attempts", type=int, default=3)
    parser.add_argument("--feedback-attempts", type=int, default=3)
    parser.add_argument("--datasets-dir", default=str(C.DATASETS_DIR))
    parser.add_argument(
        "--neutral-file",
        default=str(C.DEFAULT_BENCHMARK_DIR / "neutral_cells.jsonl"),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-full", action="store_true")
    parser.add_argument("--no-target-cache", action="store_true")
    args = parser.parse_args()

    if args.rounds < 2:
        parser.error("--rounds must be at least 2")
    if args.chunk_size < 1 or args.concurrency < 1:
        parser.error("--chunk-size and --concurrency must be positive")
    if args.status and not args.run_dir:
        parser.error("--status requires --run-dir")

    run_dir = (
        Path(args.run_dir).resolve()
        if args.run_dir
        else (
            REPO_ROOT
            / "outputs"
            / "prompt_tuning"
            / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        )
    )
    if args.status:
        return _print_status(run_dir)

    ds_root = Path(args.datasets_dir).resolve()
    neutral_file = Path(args.neutral_file).resolve()
    state_path = run_dir / "state.json"
    if not neutral_file.exists():
        parser.error(f"neutral file does not exist: {neutral_file}")
    config = _config(args, ds_root, neutral_file)

    if state_path.exists():
        if not args.resume:
            parser.error(f"{state_path} exists; use --resume")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("config") != config:
            parser.error("resume configuration does not match the saved run")
        split = json.loads((run_dir / "split.json").read_text(encoding="utf-8"))
        T.validate_split(split, ds_root)
        T.write_split_files(run_dir, split)
        _verify_round_artifacts(state, run_dir)
        state.pop("failure", None)
        _transition(state_path, state, "resuming")
    else:
        if args.resume:
            parser.error(f"cannot resume: {state_path} does not exist")
        run_dir.mkdir(parents=True, exist_ok=False)
        split = T.create_dataset_split(
            ds_root,
            dev_fraction=args.dev_fraction,
            seed=args.split_seed,
        )
        T.write_split_files(run_dir, split)
        state = {
            "schema_version": 1,
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "status": "initialized",
            "config": config,
            "split_summary": split["summary"],
            "rounds": [],
            "events": [],
        }
        _transition(state_path, state, "initialized")

    estimated_calls = T.estimated_dev_calls(
        ds_root,
        split["development"],
        args.chunk_size,
    )
    estimate = {
        "target_calls_per_round": estimated_calls,
        "target_calls_for_tuning": estimated_calls * args.rounds,
        "development_tables": len(split["development"]),
        "test_tables": len(split["test"]),
    }
    T.atomic_write_json(run_dir / "call_estimate.json", estimate)
    if args.dry_run:
        _transition(state_path, state, "dry_run_ready", call_estimate=estimate)
        print(json.dumps({"run_dir": str(run_dir), "split": split["summary"], **estimate}, indent=2))
        return 0

    from src.core.llm_client import LLMClient

    client = LLMClient()
    writer_model = args.writer_model or args.agent_model
    evaluator_model = args.evaluator_model or args.agent_model
    judge_model = args.judge_model or args.agent_model
    cache_dir = run_dir / "target_cache"
    base_prompt = BASE_PROMPT.build_system_prompt()
    rounds_root = run_dir / "rounds"

    try:
        for round_number in range(args.rounds):
            round_dir = rounds_root / f"round_{round_number:02d}"
            round_dir.mkdir(parents=True, exist_ok=True)
            prompt_path = round_dir / "prompt.txt"
            _transition(
                state_path,
                state,
                "writing_prompt" if not prompt_path.exists() else "running_candidate",
                current_round=round_number,
            )

            if not prompt_path.exists():
                if round_number == 0:
                    previous_prompt = base_prompt
                    feedback = INITIAL_WRITER_GUIDANCE
                    dossier = None
                else:
                    previous_prompt = (
                        rounds_root / f"round_{round_number - 1:02d}" / "prompt.txt"
                    ).read_text(encoding="utf-8")
                    feedback = json.loads(
                        (
                            rounds_root
                            / f"round_{round_number - 1:02d}"
                            / "approved_feedback.json"
                        ).read_text(encoding="utf-8")
                    )
                    dossier = json.loads(
                        (
                            rounds_root
                            / f"round_{round_number - 1:02d}"
                            / "evaluation_dossier.json"
                        ).read_text(encoding="utf-8")
                    )
                writer = _call_json_agent(
                    client,
                    role="prompt_writer",
                    model=writer_model,
                    reasoning_effort=args.agent_reasoning_effort,
                    system_prompt=PROMPT_WRITER_SYSTEM,
                    user_payload={
                        "current_system_prompt": previous_prompt,
                        "approved_generic_feedback": feedback,
                    },
                    artifact_dir=round_dir / "writer",
                    validator=_writer_validator(previous_prompt, dossier),
                    attempts=args.agent_attempts,
                )
                prompt_path.write_text(writer["system_prompt"].strip() + "\n", encoding="utf-8")

            _transition(
                state_path,
                state,
                "running_candidate",
                current_round=round_number,
            )
            predictions_path, issues_path, score_path = _run_detector_stage(
                stage_dir=round_dir / "development",
                stage_name=f"agentic_r{round_number:02d}",
                prompt_path=prompt_path,
                datasets=split["development"],
                benchmark_sha256=split["benchmark_sha256"],
                ds_root=ds_root,
                neutral_file=neutral_file,
                cache_dir=cache_dir,
                args=args,
            )
            metrics = T.score_summary(score_path)
            _state_round(
                state,
                round_number,
                metrics,
                prompt_path,
                predictions_path,
                score_path,
            )
            _transition(
                state_path,
                state,
                "candidate_scored",
                current_round=round_number,
            )

            if round_number == args.rounds - 1:
                continue
            dossier_path = round_dir / "evaluation_dossier.json"
            if dossier_path.exists():
                dossier = json.loads(dossier_path.read_text(encoding="utf-8"))
            else:
                dossier = T.build_evaluation_dossier(
                    ds_root,
                    split["development"],
                    predictions_path,
                    issues_path,
                    score_path,
                    sample_limit=args.sample_limit,
                    sample_seed=f"{split['seed']}:{round_number}",
                )
                T.atomic_write_json(dossier_path, dossier)
            _transition(
                state_path,
                state,
                "evaluating_candidate",
                current_round=round_number,
            )
            if not (round_dir / "approved_feedback.json").exists():
                _approved_feedback(
                    client,
                    dossier=dossier,
                    round_dir=round_dir,
                    evaluator_model=evaluator_model,
                    judge_model=judge_model,
                    reasoning_effort=args.agent_reasoning_effort,
                    agent_attempts=args.agent_attempts,
                    feedback_attempts=args.feedback_attempts,
                )
            _transition(
                state_path,
                state,
                "feedback_approved",
                current_round=round_number,
            )

        _verify_round_artifacts(state, run_dir)
        best = T.select_best_round(state["rounds"])
        best_prompt_path = Path(best["prompt_path"])
        if T.file_sha256(best_prompt_path) != best["prompt_sha256"]:
            raise ValueError("best prompt bytes no longer match their development score")
        locked_prompt_path = run_dir / "locked_prompt.txt"
        shutil.copyfile(best_prompt_path, locked_prompt_path)
        locked_sha = T.file_sha256(locked_prompt_path)
        T.atomic_write_json(
            run_dir / "prompt_lock.json",
            {
                "locked_at": utc_now(),
                "best_round": best["round"],
                "development_metrics": best["metrics"],
                "prompt_sha256": locked_sha,
                "test_access_before_lock": False,
            },
        )
        _transition(
            state_path,
            state,
            "prompt_locked",
            current_round=None,
            best_round=best["round"],
            locked_prompt_sha256=locked_sha,
        )

        _transition(state_path, state, "running_final_test")
        _, _, test_score_path = _run_detector_stage(
            stage_dir=run_dir / "final_test",
            stage_name="agentic_test",
            prompt_path=locked_prompt_path,
            datasets=split["test"],
            benchmark_sha256=split["benchmark_sha256"],
            ds_root=ds_root,
            neutral_file=neutral_file,
            cache_dir=cache_dir,
            args=args,
        )
        test_metrics = T.score_summary(test_score_path)
        _transition(
            state_path,
            state,
            "final_test_complete",
            final_test_metrics=test_metrics,
        )

        if args.run_full:
            all_path = run_dir / "all_datasets.json"
            T.atomic_write_json(
                all_path,
                {"datasets": sorted(split["development"] + split["test"])},
            )
            _transition(state_path, state, "running_optional_full")
            _, _, full_score_path = _run_detector_stage(
                stage_dir=run_dir / "optional_full",
                stage_name="agentic_full",
                prompt_path=locked_prompt_path,
                datasets=sorted(split["development"] + split["test"]),
                benchmark_sha256=split["benchmark_sha256"],
                ds_root=ds_root,
                neutral_file=neutral_file,
                cache_dir=cache_dir,
                args=args,
            )
            state["optional_full_metrics"] = T.score_summary(full_score_path)

        _transition(state_path, state, "completed", current_round=None)
        print(
            json.dumps(
                {
                    "run_dir": str(run_dir),
                    "best_round": best["round"],
                    "development_metrics": best["metrics"],
                    "final_test_metrics": test_metrics,
                    "locked_prompt_sha256": locked_sha,
                },
                indent=2,
            )
        )
        return 0
    except Exception as exc:
        _transition(
            state_path,
            state,
            "failed",
            failure={"type": type(exc).__name__, "message": str(exc)},
        )
        traceback.print_exc()
        raise


if __name__ == "__main__":
    raise SystemExit(main())
