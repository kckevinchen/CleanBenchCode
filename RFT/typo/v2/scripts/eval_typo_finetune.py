"""Evaluate a typo-RFT fine-tuned model against the held-out benchmark eval split.

v2 changes (binary task):

* The trainee emits one of two labels: ``typo`` or ``clean``. Any other
  label (including ``ambiguous``) is treated as a parse failure.
* The benchmark eval split still carries a third gold label
  ``ambiguous``. When ``gold_label == 'ambiguous'`` we count the row as
  correct regardless of whether the trainee said ``typo`` or ``clean``
  -- the rulebook itself says those cells are unresolvable from
  cell+header alone, so we don't penalise either choice.
* Per-class precision / recall / F1 are reported on the BINARY axis
  (typo, clean only); 'ambiguous' is reported separately as a coverage
  rate (how often the trainee returned a valid binary label).

For each row in ``outputs/typo_rft/benchmark_split_v2/eval.jsonl`` we:

1. Reconstruct sibling values for the same column by reading the
   source CSV under ``--corpus-root`` (so the model sees the same
   prompt shape it saw during training).
2. Build the trainee prompt via ``experiments._typo_task.build_user_message``
   and ``experiments._typo_task.TYPO_SYSTEM_INSTRUCTION``.
3. Invoke the configured deployment via the Azure OpenAI Responses API
   (matches the training-time inference contract).
4. Parse the structured JSON response and record the prediction.
5. Aggregate metrics: confusion matrix, binary per-class P/R/F1,
   correction exact-match (typo rows only), per-flavour breakdown,
   ambiguous-gold credit accounting.

Concurrency is handled via ``ThreadPoolExecutor`` (default 8 workers).
Per-row predictions are appended to ``predictions.jsonl`` as they
complete; re-running the script will skip rows already present so a
crash or Ctrl+C is recoverable.

Typical use
-----------
::

    python -m experiments.eval_typo_finetune \\
        --eval-jsonl outputs/typo_rft/benchmark_split_v2/eval.jsonl \\
        --endpoint  <azure-openai-endpoint> \
        --deployment o4-mini-2025-04-16-typo-rft-v2 \\
        --api-version 2025-03-01-preview \\
        --output-dir outputs/typo_rft/eval_v2_ft \\
        --concurrency 8

To benchmark the base model for a head-to-head comparison, run again
with a non-fine-tuned deployment, pointing ``--compare-to`` at the
previous predictions JSONL.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from azure.identity import AzureCliCredential, get_bearer_token_provider
from openai import APIError, AzureOpenAI, BadRequestError

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from RFT._common import _typo_task
from src.core.llm_client import configured_endpoint_pool

logger = logging.getLogger("eval_typo_finetune")


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_EVAL = Path("outputs/typo_rft/benchmark_split_v2/eval.jsonl")
DEFAULT_OUTPUT_DIR = Path("outputs/typo_rft/eval_v2_ft")
_CONFIGURED_ENDPOINTS = configured_endpoint_pool()
DEFAULT_ENDPOINT = _CONFIGURED_ENDPOINTS[0] if _CONFIGURED_ENDPOINTS else ""
# v2: pass an explicit --deployment at run-time. The v1 deployment
# (``o4-mini-2025-04-16-typo-rft-v1``) is still available; the v2
# deployment name will be assigned by Azure once fine-tuning is done.
DEFAULT_DEPLOYMENT = "o4-mini-2025-04-16-typo-rft-v1"
# NOTE: the Azure OpenAI Responses API is gated to ``2025-03-01-preview``
# and later (older api-versions return HTTP 400 "Azure OpenAI Responses
# API is enabled only for api-version 2025-03-01-preview and later").
# The deploy-doc snippet showed ``2024-12-01-preview`` for chat
# completions; we pin a newer version here so ``responses.create`` works.
DEFAULT_API_VERSION = "2025-03-01-preview"
DEFAULT_CORPUS_ROOT = Path("data/source_inconsistency_bing21")

# Hard cap on siblings included in the prompt -- matches the training
# default in ``_typo_task.build_user_message``.
SIBLING_CAP = 20
# Cap on rows we scan from a source CSV to find sibling values. Most
# benchmark CSVs are wide+short; 2000 rows is a generous ceiling.
MAX_CSV_ROWS = 2000


# ---------------------------------------------------------------------------
# Corpus indexing
# ---------------------------------------------------------------------------


def index_corpus(corpus_root: Path) -> dict[str, Path]:
    """Build ``{filename -> absolute_path}`` map by walking the corpus.

    The benchmark stores CSVs at ``<corpus>/<xlsx-hash>.xlsx/<file>.csv``
    so a flat name->path index is the easiest way to look them up by
    the ``source`` field of an eval row.
    """
    if not corpus_root.exists():
        logger.warning("Corpus root %s does not exist.", corpus_root)
        return {}
    out: dict[str, Path] = {}
    for dirpath, _, filenames in os.walk(corpus_root):
        for name in filenames:
            if not name.endswith(".csv"):
                continue
            # First write wins; the corpus shouldn't contain duplicates
            # but we don't want to clobber a good entry with a bad one.
            out.setdefault(name, Path(dirpath) / name)
    logger.info("Indexed %d CSVs under %s", len(out), corpus_root)
    return out


def fetch_sibling_values(
    csv_path: Path, column: str, candidate: str, *, max_n: int = SIBLING_CAP,
) -> list[str]:
    """Read the named column from ``csv_path`` and return distinct
    non-empty values that differ from ``candidate``.

    On any I/O or decode error we return an empty list rather than
    failing the whole eval row -- the model simply gets no sibling
    context for that one row.
    """
    try:
        with csv_path.open(encoding="utf-8", errors="replace", newline="") as fh:
            reader = csv.reader(fh)
            try:
                header = next(reader)
            except StopIteration:
                return []
            if column not in header:
                # Header may include stray whitespace; do a forgiving lookup.
                stripped = [h.strip() for h in header]
                if column.strip() not in stripped:
                    return []
                ci = stripped.index(column.strip())
            else:
                ci = header.index(column)
            seen: dict[str, None] = {}
            for row_idx, row in enumerate(reader):
                if row_idx >= MAX_CSV_ROWS:
                    break
                if ci >= len(row):
                    continue
                v = (row[ci] or "").strip()
                if not v or v == candidate:
                    continue
                seen[v] = None
                if len(seen) >= max_n:
                    break
        return list(seen.keys())
    except (OSError, UnicodeError, csv.Error) as exc:
        logger.debug("Could not read siblings from %s: %s", csv_path, exc)
        return []


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def build_client(endpoint: str, api_version: str) -> AzureOpenAI:
    """Build an Azure OpenAI client using ``AzureCliCredential``.

    Matches the auth pattern the rest of this codebase uses (see
    ``src.core.llm_client``).
    """
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
class Prediction:
    """One model prediction; serialised to predictions.jsonl."""

    row_index: int
    source: str
    column: str
    original_value: str
    gold_label: str  # normalised: error -> typo
    gold_correction: str | None
    raw_text: str | None
    parsed_label: str | None
    parsed_correction: str | None
    parsed_flavour: str | None
    parsed_rationale: str | None
    error: str | None


def _normalise_label(label: str | None) -> str | None:
    """Benchmark vocab 'error' -> trainee vocab 'typo'."""
    if label == "error":
        return "typo"
    return label


def _parse_response(text: str) -> dict[str, Any] | None:
    """Parse the trainee's JSON reply. Tolerate ```json fences."""
    if not text:
        return None
    s = text.strip()
    if s.startswith("```"):
        import re
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```\s*$", "", s)
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return None


def predict_row(
    client: AzureOpenAI,
    deployment: str,
    row_index: int,
    gold: dict[str, Any],
    sibling_values: list[str],
    *,
    reasoning_effort: str = "medium",
) -> Prediction:
    """Run one eval row through the model and return a Prediction."""
    user_msg = _typo_task.build_user_message(
        column=gold.get("column", "") or "",
        source=gold.get("source", "") or "",
        candidate_value=gold.get("original_value", "") or "",
        sibling_values=sibling_values,
        max_siblings=SIBLING_CAP,
    )

    raw_text: str | None = None
    error: str | None = None
    parsed: dict[str, Any] | None = None
    try:
        # The trainee was trained to emit strict JSON via the developer
        # message; we deliberately do *not* pass a ``text.format`` schema
        # here -- that would force the shape and mask the model's actual
        # ability to produce well-formed JSON on its own (which is what
        # we're evaluating).
        resp = client.responses.create(
            model=deployment,
            input=[
                {"role": "developer", "content": _typo_task.TYPO_SYSTEM_INSTRUCTION},
                {"role": "user", "content": user_msg},
            ],
            reasoning={"effort": reasoning_effort},
        )
        raw_text = resp.output_text
        parsed = _parse_response(raw_text)
        if parsed is None:
            error = "json_decode_failed"
    except BadRequestError as exc:
        error = f"bad_request: {exc}"
    except APIError as exc:
        error = f"api_error: {exc}"
    except Exception as exc:  # network / SDK glitches
        error = f"{type(exc).__name__}: {exc}"

    parsed_label = parsed.get("label") if isinstance(parsed, dict) else None
    parsed_correction = parsed.get("correction") if isinstance(parsed, dict) else None
    parsed_flavour = parsed.get("flavour") if isinstance(parsed, dict) else None
    parsed_rationale = parsed.get("rationale") if isinstance(parsed, dict) else None

    return Prediction(
        row_index=row_index,
        source=gold.get("source", "") or "",
        column=gold.get("column", "") or "",
        original_value=gold.get("original_value", "") or "",
        gold_label=_normalise_label(gold.get("label")) or "",
        gold_correction=gold.get("corrected_value"),
        raw_text=raw_text,
        parsed_label=parsed_label,
        parsed_correction=parsed_correction,
        parsed_flavour=parsed_flavour,
        parsed_rationale=parsed_rationale,
        error=error,
    )


# ---------------------------------------------------------------------------
# Resume support
# ---------------------------------------------------------------------------


def load_existing_predictions(path: Path) -> dict[int, dict[str, Any]]:
    """Return ``{row_index: record}`` for predictions already written."""
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
# Metrics
# ---------------------------------------------------------------------------


_GOLD_LABELS = ("typo", "clean", "ambiguous")
_OUTPUT_LABELS = ("typo", "clean")


def compute_metrics(
    predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate metrics from a flat list of prediction dicts (v2 binary).

    A row is correct when ANY of the following holds:
      * ``parsed_label == gold_label``
      * ``gold_label == 'ambiguous'`` AND ``parsed_label in {typo, clean}``
        (rulebook: ambiguous cells are unresolvable, so either binary
        answer is accepted).

    Per-class P/R/F1 is reported on the BINARY axis (typo, clean only).
    Rows with ``gold_label == 'ambiguous'`` are EXCLUDED from the
    binary P/R/F1 numerators/denominators -- including them would
    inflate one class's recall without a principled justification.
    Their handling is reported separately as ``ambiguous_coverage``.
    """
    confusion: dict[str, Counter] = {g: Counter() for g in _GOLD_LABELS}
    confusion["__other__"] = Counter()

    total = 0
    correct = 0
    parse_failures = 0
    api_errors = 0

    # Correction match: among rows where gold AND prediction agree on
    # 'typo', does the predicted correction string-equal the gold one?
    correction_attempts = 0
    correction_exact = 0

    # Per-flavour: infer the gold flavour from (original -> gold_correction)
    # for typo rows; compare to predicted flavour string.
    flavour_total: Counter = Counter()
    flavour_correct: Counter = Counter()

    # Ambiguous-gold coverage: of the ambiguous rows, how many got a
    # valid binary label (vs a parse failure)? And how do the binary
    # answers split between typo and clean?
    ambiguous_total = 0
    ambiguous_binary_valid = 0
    ambiguous_pred_split: Counter = Counter()

    for rec in predictions:
        total += 1
        gold = rec.get("gold_label") or ""
        pred = rec.get("parsed_label")
        # v2: any prediction outside the binary output space is a
        # parse failure (the model is supposed to commit).
        if pred is not None and pred not in _OUTPUT_LABELS:
            pred = None
            if not rec.get("error"):
                # Schema-violating label (e.g. the model emitted
                # 'ambiguous'); count as parse failure for accounting.
                parse_failures += 1
        if rec.get("error"):
            if rec["error"] == "json_decode_failed":
                parse_failures += 1
            else:
                api_errors += 1
        if pred is None:
            confusion.setdefault(gold, Counter())["__missing__"] += 1
            if gold == "ambiguous":
                ambiguous_total += 1
            continue
        if gold in confusion:
            confusion[gold][pred] += 1
        else:
            confusion["__other__"][pred] += 1
        if gold == "ambiguous":
            ambiguous_total += 1
            ambiguous_binary_valid += 1
            ambiguous_pred_split[pred] += 1
            # v2: ambiguous gold is correct on either binary answer.
            correct += 1
        elif pred == gold:
            correct += 1
        if gold == "typo" and pred == "typo":
            correction_attempts += 1
            gold_corr = (rec.get("gold_correction") or "").strip()
            pred_corr = (rec.get("parsed_correction") or "").strip()
            if gold_corr and pred_corr == gold_corr:
                correction_exact += 1
            flav = _typo_task._infer_flavour(
                rec.get("original_value") or "",
                rec.get("gold_correction") or "",
            )
            flavour_total[flav] += 1
            if rec.get("parsed_flavour") == flav:
                flavour_correct[flav] += 1

    # Per-class P/R/F1 on the BINARY axis only. Ambiguous-gold rows do
    # not enter the typo/clean P/R/F1 numerators -- they are reported
    # via ``ambiguous_coverage`` below.
    per_class: dict[str, dict[str, float]] = {}
    for label in _OUTPUT_LABELS:
        tp = confusion.get(label, Counter()).get(label, 0)
        # FN: gold==label but pred is the OTHER binary label
        # (parse failures are excluded; ambiguous gold is excluded).
        fn = sum(
            v for k, v in confusion.get(label, Counter()).items()
            if k != label and k in _OUTPUT_LABELS
        )
        # FP: pred==label from the OTHER binary gold class only.
        # Ambiguous-gold predictions of `label` are NOT false positives
        # under v2 -- the rulebook says either binary answer is
        # acceptable on ambiguous cells.
        fp = 0
        for g in _OUTPUT_LABELS:
            if g == label:
                continue
            fp += confusion.get(g, Counter()).get(label, 0)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        per_class[label] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1,
        }

    flavour_breakdown = {
        flav: {
            "n_gold": flavour_total[flav],
            "n_correct_flavour": flavour_correct[flav],
            "accuracy": (
                flavour_correct[flav] / flavour_total[flav]
                if flavour_total[flav] else float("nan")
            ),
        }
        for flav in sorted(flavour_total)
    }

    ambiguous_coverage = {
        "n_gold_ambiguous": ambiguous_total,
        "n_binary_valid": ambiguous_binary_valid,
        "binary_valid_rate": (
            ambiguous_binary_valid / ambiguous_total
            if ambiguous_total else float("nan")
        ),
        "pred_split": dict(ambiguous_pred_split),
    }

    return {
        "n_rows": total,
        "n_correct": correct,
        "label_accuracy": correct / total if total else float("nan"),
        "n_parse_failures": parse_failures,
        "n_api_errors": api_errors,
        "confusion_matrix": {
            g: dict(confusion.get(g, Counter())) for g in _GOLD_LABELS
        },
        "per_class": per_class,
        "ambiguous_coverage": ambiguous_coverage,
        "correction_attempts": correction_attempts,
        "correction_exact": correction_exact,
        "correction_exact_match_rate": (
            correction_exact / correction_attempts
            if correction_attempts else float("nan")
        ),
        "flavour_breakdown": flavour_breakdown,
    }


# ---------------------------------------------------------------------------
# Head-to-head with a prior run
# ---------------------------------------------------------------------------


def compare_runs(
    new_preds: list[dict[str, Any]], other_path: Path,
) -> dict[str, Any] | None:
    """If ``other_path`` exists, build a per-row delta vs the new run."""
    if not other_path.exists():
        return None
    other = load_existing_predictions(other_path)
    if not other:
        return None
    new_by_idx = {r.get("row_index"): r for r in new_preds}
    common = sorted(set(new_by_idx) & set(other))
    if not common:
        return None

    new_correct = 0
    other_correct = 0
    only_new_correct = 0
    only_other_correct = 0
    both_correct = 0
    both_wrong = 0
    for idx in common:
        a = new_by_idx[idx]
        b = other[idx]
        if a.get("gold_label") != b.get("gold_label"):
            # Different normalisation upstream; skip.
            continue
        gold = a.get("gold_label")
        # v2 scoring: a row is correct when label matches OR gold is
        # 'ambiguous' AND the prediction is a valid binary label.
        a_label = a.get("parsed_label")
        b_label = b.get("parsed_label")
        a_ok = (
            a_label == gold
            or (gold == "ambiguous" and a_label in _OUTPUT_LABELS)
        )
        b_ok = (
            b_label == gold
            or (gold == "ambiguous" and b_label in _OUTPUT_LABELS)
        )
        new_correct += int(a_ok)
        other_correct += int(b_ok)
        if a_ok and b_ok:
            both_correct += 1
        elif a_ok and not b_ok:
            only_new_correct += 1
        elif b_ok and not a_ok:
            only_other_correct += 1
        else:
            both_wrong += 1

    return {
        "compared_against": str(other_path),
        "n_common_rows": len(common),
        "new_accuracy": new_correct / len(common),
        "other_accuracy": other_correct / len(common),
        "delta": (new_correct - other_correct) / len(common),
        "both_correct": both_correct,
        "only_new_correct": only_new_correct,
        "only_other_correct": only_other_correct,
        "both_wrong": both_wrong,
    }


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def _serialise_prediction(p: Prediction) -> dict[str, Any]:
    """Convert a Prediction dataclass to a JSON-serialisable dict."""
    return {
        "row_index": p.row_index,
        "source": p.source,
        "column": p.column,
        "original_value": p.original_value,
        "gold_label": p.gold_label,
        "gold_correction": p.gold_correction,
        "parsed_label": p.parsed_label,
        "parsed_correction": p.parsed_correction,
        "parsed_flavour": p.parsed_flavour,
        "parsed_rationale": p.parsed_rationale,
        "raw_text": p.raw_text,
        "error": p.error,
    }


def run_eval(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"
    summary_path = output_dir / "summary.json"

    eval_rows = [
        json.loads(line)
        for line in Path(args.eval_jsonl).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.limit:
        eval_rows = eval_rows[: args.limit]
    logger.info("Loaded %d eval rows from %s", len(eval_rows), args.eval_jsonl)

    corpus_index = index_corpus(Path(args.corpus_root))
    n_with_corpus = sum(1 for r in eval_rows if r.get("source") in corpus_index)
    logger.info(
        "%d / %d eval rows have a matching CSV in the corpus index.",
        n_with_corpus, len(eval_rows),
    )

    existing = load_existing_predictions(predictions_path)
    todo: list[tuple[int, dict[str, Any]]] = []
    for idx, row in enumerate(eval_rows):
        if idx in existing and not args.refresh:
            continue
        todo.append((idx, row))
    logger.info(
        "%d rows already have predictions; %d remain to run.",
        len(existing), len(todo),
    )

    client = build_client(args.endpoint, args.api_version)
    logger.info(
        "Targeting endpoint=%s deployment=%s api_version=%s",
        args.endpoint, args.deployment, args.api_version,
    )

    write_lock = threading.Lock()
    progress = {"done": 0, "errors": 0}

    def _job(idx: int, row: dict[str, Any]) -> Prediction:
        sibs: list[str] = []
        src = row.get("source")
        if src and src in corpus_index:
            sibs = fetch_sibling_values(
                corpus_index[src],
                row.get("column", ""),
                row.get("original_value", "") or "",
            )
        pred = predict_row(
            client,
            args.deployment,
            idx,
            row,
            sibs,
            reasoning_effort=args.reasoning_effort,
        )
        with write_lock:
            with predictions_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(_serialise_prediction(pred), ensure_ascii=False) + "\n")
            progress["done"] += 1
            if pred.error:
                progress["errors"] += 1
            if progress["done"] % 25 == 0:
                logger.info(
                    "%d / %d done (%d errors so far).",
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

    # Reload all predictions (existing + new) for metric computation.
    final = load_existing_predictions(predictions_path)
    flat = [final[i] for i in sorted(final)]
    metrics = compute_metrics(flat)
    metrics["endpoint"] = args.endpoint
    metrics["deployment"] = args.deployment
    metrics["api_version"] = args.api_version
    metrics["reasoning_effort"] = args.reasoning_effort
    metrics["eval_jsonl"] = str(args.eval_jsonl)
    metrics["n_predictions_written"] = len(final)

    if args.compare_to:
        delta = compare_runs(flat, Path(args.compare_to))
        if delta is not None:
            metrics["head_to_head"] = delta

    summary_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Wrote summary -> %s", summary_path)

    # Print a brief human-readable digest to stdout.
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
    p.add_argument("--corpus-root", type=Path, default=DEFAULT_CORPUS_ROOT)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--endpoint", default=DEFAULT_ENDPOINT, required=not bool(DEFAULT_ENDPOINT))
    p.add_argument("--deployment", default=DEFAULT_DEPLOYMENT)
    p.add_argument("--api-version", default=DEFAULT_API_VERSION)
    p.add_argument(
        "--reasoning-effort", choices=["low", "medium", "high"], default="medium",
    )
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument(
        "--limit", type=int, default=None,
        help="Process only the first N rows (useful for smoke tests).",
    )
    p.add_argument(
        "--refresh", action="store_true",
        help="Re-predict rows that already have an entry in predictions.jsonl.",
    )
    p.add_argument(
        "--compare-to", type=Path, default=None,
        help="Path to another predictions.jsonl (e.g. baseline run) to "
        "compute a head-to-head delta against.",
    )
    args = p.parse_args(argv)
    run_eval(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
