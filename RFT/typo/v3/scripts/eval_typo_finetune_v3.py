"""Evaluate a v3 (column-audit) typo-RFT fine-tuned model.

Pipeline (mirrors :mod:`experiments.eval_typo_finetune` but per
column instead of per cell):

1. Load the v3 eval split written by
   :mod:`experiments.split_typo_benchmark_v3`. Each row is one
   ``(source, column)`` audit with a ready-to-send
   ``messages=[developer, user]`` payload and JSON-encoded
   ``gold_typos`` / ``ambiguous_values`` item.* fields.
2. Hit the configured Azure OpenAI deployment via the Responses API.
3. Parse the trainee's strict JSON reply
   (``{"typos": [{value, correction, flavour, rationale}]}``).
4. Score each column with
   :func:`experiments._typo_v3_task.compute_column_f1` (same logic
   the RFT grader uses online).
5. Produce TWO reports:

   * **Primary**: per-column macro/micro precision/recall/F1 across
     all eval columns.
   * **Per-cell projection**: project the column-level predictions
     onto the v2 cell-level eval split (each v2 row gets a "typo"
     prediction if its ``original_value`` is in the predicted set
     for its column, else "clean"). This lets us compare the v3
     deployment to a v2 deployment on the same axis v2 was trained
     for, head-to-head.

Usage
-----
::

    python -m experiments.eval_typo_finetune_v3 \\
        --eval-jsonl outputs/typo_rft/benchmark_split_v3/eval.jsonl \\
        --v2-eval-jsonl RFT/typo/v2/data/eval.jsonl \\
        --output-dir outputs/typo_rft/eval_v3_ft \\
        --deployment o4-mini-2025-04-16-typo-rft-v3
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from azure.identity import AzureCliCredential, get_bearer_token_provider
from openai import APIError, AzureOpenAI, BadRequestError

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from RFT._common import _typo_v3_task
from src.core.llm_client import configured_endpoint_pool

logger = logging.getLogger("eval_typo_finetune_v3")


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_EVAL = Path("outputs/typo_rft/benchmark_split_v3/eval.jsonl")
DEFAULT_V2_EVAL = Path("RFT/typo/v2/data/eval.jsonl")
DEFAULT_OUTPUT_DIR = Path("outputs/typo_rft/eval_v3_ft")
_CONFIGURED_ENDPOINTS = configured_endpoint_pool()
DEFAULT_ENDPOINT = _CONFIGURED_ENDPOINTS[0] if _CONFIGURED_ENDPOINTS else ""
DEFAULT_DEPLOYMENT = "o4-mini-2025-04-16"  # base model; override at run-time
DEFAULT_API_VERSION = "2025-03-01-preview"


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


def build_client(endpoint: str, api_version: str) -> AzureOpenAI:
    """Mirror v2 auth: AzureCliCredential + bearer token provider."""
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
# Trainee output parsing
# ---------------------------------------------------------------------------


def _parse_typos_response(text: str) -> list[dict[str, Any]]:
    """Parse the strict JSON reply. Tolerate ```json fences.

    Returns the typos list (possibly empty) or ``[]`` on any failure.
    The caller distinguishes "parse failure" from "empty list" by
    checking whether ``raw_text`` was non-empty and not parseable.
    """
    if not text:
        return []
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```\s*$", "", s)
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        return []
    if not isinstance(obj, dict):
        return []
    typos = obj.get("typos")
    if not isinstance(typos, list):
        return []
    out: list[dict[str, Any]] = []
    for t in typos:
        if isinstance(t, dict):
            out.append(t)
    return out


# ---------------------------------------------------------------------------
# Per-column prediction record
# ---------------------------------------------------------------------------


@dataclass
class ColumnPrediction:
    """One column's prediction. Serialised to predictions.jsonl."""

    row_index: int
    source: str
    column: str
    gold_typos: list[dict[str, str]]
    ambiguous_values: list[str]
    raw_text: str | None
    parsed_typos: list[dict[str, Any]]
    parse_failed: bool
    error: str | None
    # Per-column metrics, computed once and cached so summary.json
    # doesn't need to recompute on reload.
    col_metrics: dict[str, Any] = field(default_factory=dict)


def _predict_column(
    client: AzureOpenAI,
    deployment: str,
    row_index: int,
    eval_row: dict[str, Any],
    *,
    reasoning_effort: str = "medium",
) -> ColumnPrediction:
    """Send one column's audit prompt to the deployment.

    The eval row already carries ``messages=[developer, user]`` from
    :func:`_typo_v3_task.build_record`, so we just forward it.
    """
    gold_typos = json.loads(eval_row.get("gold_typos", "[]") or "[]")
    ambiguous = json.loads(eval_row.get("ambiguous_values", "[]") or "[]")
    raw_text: str | None = None
    error: str | None = None
    parsed: list[dict[str, Any]] = []
    parse_failed = False

    try:
        # Pass the trainee messages as-is. The v3 deployment was
        # trained with the strict-JSON response_format, so we don't
        # need to set ``text.format`` here.
        resp = client.responses.create(
            model=deployment,
            input=eval_row["messages"],
            reasoning={"effort": reasoning_effort},
        )
        raw_text = resp.output_text
        parsed = _parse_typos_response(raw_text)
        if raw_text and not parsed:
            # Non-empty reply that didn't parse as the schema.
            # We still treat it as no-typos for scoring, but flag it.
            parse_failed = True
    except BadRequestError as exc:
        error = f"bad_request: {exc}"
    except APIError as exc:
        error = f"api_error: {exc}"
    except Exception as exc:  # network / SDK glitches
        error = f"{type(exc).__name__}: {exc}"

    col_metrics = _typo_v3_task.compute_column_f1(
        pred_typos=parsed,
        gold_typos=gold_typos,
        ambiguous_values=ambiguous,
    )

    return ColumnPrediction(
        row_index=row_index,
        source=eval_row.get("source", "") or "",
        column=eval_row.get("column", "") or "",
        gold_typos=gold_typos,
        ambiguous_values=ambiguous,
        raw_text=raw_text,
        parsed_typos=parsed,
        parse_failed=parse_failed,
        error=error,
        col_metrics=col_metrics,
    )


# ---------------------------------------------------------------------------
# Resume support
# ---------------------------------------------------------------------------


def _load_existing_predictions(path: Path) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    out: dict[int, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        idx = rec.get("row_index")
        if isinstance(idx, int):
            out[idx] = rec
    return out


# ---------------------------------------------------------------------------
# Metrics: per-column primary
# ---------------------------------------------------------------------------


def _compute_column_metrics(
    predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate per-column F1 into macro / micro / coverage stats."""
    n_total = len(predictions)
    n_parse_failed = sum(1 for p in predictions if p.get("parse_failed"))
    n_api_errors = sum(1 for p in predictions if p.get("error"))

    # Macro: average F1/P/R across columns.
    sum_f1 = sum(p["col_metrics"].get("f1", 0.0) for p in predictions)
    sum_p = sum(p["col_metrics"].get("precision", 0.0) for p in predictions)
    sum_r = sum(p["col_metrics"].get("recall", 0.0) for p in predictions)

    # Micro: aggregate TP/FP/FN across all columns.
    tp = sum(p["col_metrics"].get("tp", 0) for p in predictions)
    fp = sum(p["col_metrics"].get("fp", 0) for p in predictions)
    fn = sum(p["col_metrics"].get("fn", 0) for p in predictions)
    micro_p = tp / (tp + fp) if (tp + fp) else 0.0
    micro_r = tp / (tp + fn) if (tp + fn) else 0.0
    micro_f1 = (
        2 * micro_p * micro_r / (micro_p + micro_r)
        if (micro_p + micro_r) else 0.0
    )

    # Column type buckets: how does the model do on each kind of column?
    by_kind: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "n_columns": 0, "sum_f1": 0.0, "tp": 0, "fp": 0, "fn": 0,
    })
    for p in predictions:
        gold = p.get("gold_typos") or []
        ambig = p.get("ambiguous_values") or []
        if gold:
            kind = "has_typo"
        elif ambig:
            kind = "ambiguous_only"
        else:
            kind = "clean"
        b = by_kind[kind]
        b["n_columns"] += 1
        b["sum_f1"] += p["col_metrics"].get("f1", 0.0)
        b["tp"] += p["col_metrics"].get("tp", 0)
        b["fp"] += p["col_metrics"].get("fp", 0)
        b["fn"] += p["col_metrics"].get("fn", 0)
    by_kind_out = {}
    for kind, b in by_kind.items():
        n = b["n_columns"] or 1
        by_kind_out[kind] = {
            "n_columns": b["n_columns"],
            "macro_f1": b["sum_f1"] / n,
            "tp": b["tp"], "fp": b["fp"], "fn": b["fn"],
        }

    return {
        "n_columns": n_total,
        "n_parse_failed": n_parse_failed,
        "n_api_errors": n_api_errors,
        "macro_precision": sum_p / n_total if n_total else 0.0,
        "macro_recall": sum_r / n_total if n_total else 0.0,
        "macro_f1": sum_f1 / n_total if n_total else 0.0,
        "micro_precision": micro_p,
        "micro_recall": micro_r,
        "micro_f1": micro_f1,
        "micro_tp": tp, "micro_fp": fp, "micro_fn": fn,
        "by_kind": by_kind_out,
    }


# ---------------------------------------------------------------------------
# Per-cell projection (for v2 comparability)
# ---------------------------------------------------------------------------


def _project_to_cells(
    predictions: list[dict[str, Any]],
    v2_eval_path: Path,
) -> dict[str, Any] | None:
    """For each v2 cell row, project the column's predicted typo set.

    A cell is "predicted typo" iff its ``original_value`` is in the
    column's predicted typo-value set. Compute v2-style binary
    precision/recall/F1 against the cell's gold ``label``.

    Ambiguous cells follow v2's rule: any binary answer is accepted.

    Returns ``None`` if ``v2_eval_path`` doesn't exist or is empty.
    """
    if not v2_eval_path.exists():
        logger.warning(
            "v2 eval not found at %s; skipping per-cell projection.",
            v2_eval_path,
        )
        return None
    pred_index: dict[tuple[str, str], set[str]] = {}
    for p in predictions:
        key = (p.get("source", ""), p.get("column", ""))
        values = {
            t.get("value") for t in (p.get("parsed_typos") or [])
            if isinstance(t.get("value"), str)
        }
        pred_index[key] = values

    n_total = 0
    n_predicted = {}
    confusion: dict[str, Counter] = defaultdict(Counter)
    not_covered = 0
    for line in v2_eval_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        cell = json.loads(line)
        src = cell.get("source", "")
        col = cell.get("column", "")
        gold = cell.get("label")
        if gold == "error":
            gold = "typo"
        key = (src, col)
        if key not in pred_index:
            not_covered += 1
            continue
        original = cell.get("original_value", "") or ""
        pred = "typo" if original in pred_index[key] else "clean"
        n_total += 1
        n_predicted[pred] = n_predicted.get(pred, 0) + 1
        confusion[gold][pred] += 1

    # v2-style binary P/R/F1: TP/FP/FN on {typo, clean}, ambiguous
    # gold is treated as correct on either binary answer.
    per_class: dict[str, dict[str, float]] = {}
    for label in ("typo", "clean"):
        tp = confusion.get(label, Counter()).get(label, 0)
        fn = sum(
            v for k, v in confusion.get(label, Counter()).items()
            if k != label and k in ("typo", "clean")
        )
        fp = 0
        for g in ("typo", "clean"):
            if g == label:
                continue
            fp += confusion.get(g, Counter()).get(label, 0)
        p_ = tp / (tp + fp) if (tp + fp) else 0.0
        r_ = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * p_ * r_ / (p_ + r_)) if (p_ + r_) else 0.0
        per_class[label] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": p_, "recall": r_, "f1": f1,
        }

    # Overall label accuracy: gold==pred OR gold=='ambiguous'.
    correct = 0
    for g, row in confusion.items():
        if g == "ambiguous":
            correct += sum(row.values())
        else:
            correct += row.get(g, 0)

    return {
        "n_cells_projected": n_total,
        "n_cells_uncovered": not_covered,
        "pred_split": n_predicted,
        "label_accuracy": correct / n_total if n_total else 0.0,
        "per_class": per_class,
        "confusion_matrix": {
            g: dict(c) for g, c in confusion.items()
        },
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _serialise(p: ColumnPrediction) -> dict[str, Any]:
    """Convert dataclass -> JSON-serialisable dict."""
    d = asdict(p)
    return d


def run_eval_v3(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"
    summary_path = output_dir / "summary.json"

    eval_rows = [
        json.loads(line)
        for line in Path(args.eval_jsonl).read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    if args.limit:
        eval_rows = eval_rows[: args.limit]
    logger.info(
        "Loaded %d v3 eval columns from %s", len(eval_rows), args.eval_jsonl,
    )

    existing = _load_existing_predictions(predictions_path)
    todo: list[tuple[int, dict[str, Any]]] = []
    for idx, row in enumerate(eval_rows):
        if idx in existing and not args.refresh:
            continue
        todo.append((idx, row))
    logger.info(
        "%d columns already have predictions; %d remain.",
        len(existing), len(todo),
    )

    client = build_client(args.endpoint, args.api_version)
    logger.info(
        "Targeting endpoint=%s deployment=%s api_version=%s",
        args.endpoint, args.deployment, args.api_version,
    )

    write_lock = threading.Lock()
    progress = {"done": 0, "errors": 0}

    def _job(idx: int, row: dict[str, Any]) -> ColumnPrediction:
        pred = _predict_column(
            client, args.deployment, idx, row,
            reasoning_effort=args.reasoning_effort,
        )
        with write_lock:
            with predictions_path.open("a", encoding="utf-8") as fh:
                fh.write(
                    json.dumps(_serialise(pred), ensure_ascii=False) + "\n"
                )
            progress["done"] += 1
            if pred.error:
                progress["errors"] += 1
            if progress["done"] % 10 == 0:
                logger.info(
                    "%d / %d done (%d errors).",
                    progress["done"], len(todo), progress["errors"],
                )
        return pred

    if todo:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = [pool.submit(_job, idx, row) for idx, row in todo]
            for fut in as_completed(futures):
                try:
                    fut.result()
                except Exception as exc:  # noqa: BLE001
                    logger.exception("Worker raised: %s", exc)
                    progress["errors"] += 1

    # Reload all predictions for metric computation.
    final = _load_existing_predictions(predictions_path)
    flat = [final[i] for i in sorted(final)]
    metrics: dict[str, Any] = _compute_column_metrics(flat)
    metrics["endpoint"] = args.endpoint
    metrics["deployment"] = args.deployment
    metrics["api_version"] = args.api_version
    metrics["reasoning_effort"] = args.reasoning_effort
    metrics["eval_jsonl"] = str(args.eval_jsonl)
    metrics["n_predictions_written"] = len(final)

    if args.v2_eval_jsonl:
        proj = _project_to_cells(flat, Path(args.v2_eval_jsonl))
        if proj is not None:
            metrics["per_cell_projection"] = proj

    summary_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    logger.info("Wrote summary -> %s", summary_path)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return metrics


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--eval-jsonl", type=Path, default=DEFAULT_EVAL)
    p.add_argument(
        "--v2-eval-jsonl", type=Path, default=DEFAULT_V2_EVAL,
        help="If set, also compute a per-cell projection over the v2 "
        "eval split (one cell-level binary P/R/F1 reading, for "
        "v2-vs-v3 head-to-head).",
    )
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--endpoint", default=DEFAULT_ENDPOINT, required=not bool(DEFAULT_ENDPOINT))
    p.add_argument("--deployment", default=DEFAULT_DEPLOYMENT)
    p.add_argument("--api-version", default=DEFAULT_API_VERSION)
    p.add_argument(
        "--reasoning-effort",
        choices=["low", "medium", "high"], default="medium",
    )
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--refresh", action="store_true")
    args = p.parse_args(argv)
    run_eval_v3(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
