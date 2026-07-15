"""Calibrate the v5 grader against the human-labelled typo benchmark.

This script tests the v5 typo grader as a stand-alone rulebook
detector BEFORE we wire it into the RFT training loop. The grader is
asked to judge ONE (value, correction) candidate at a time against
the v5 rulebook (no gold set, gold-free mode) and emit one of five
verdicts:

    valid_extra | ambiguous_extra | trivial_fp | entity_fp | plain_fp

We then compare the verdict to the human ground-truth label.

Two test modes (run both by default):

  Test A -- Recall.
      For every benchmark row with label='error' that is NOT a
      trivial-typo correction (per
      RFT.typo.v5._task.is_v5_trivial_correction), feed
      (value=original_value, correction=corrected_value) to the
      grader. Expected verdict: valid_extra. Anything else is a
      grader miss (the rulebook would NOT have rewarded a real
      typo during training).

  Test B -- Precision.
      For every benchmark row with label='clean' whose
      llm_suggested_value looks like a plausible single-value
      correction, feed (value=original_value,
      correction=llm_suggested_value) to the grader. Expected
      verdict: trivial_fp | entity_fp | plain_fp. valid_extra is
      a grader false alarm (the rulebook would have endorsed a
      bad fix during training).

Rows with label='ambiguous' are EXCLUDED from both modes (they
neither count for nor against the model in the v5 grader rubric).

Outputs (under --output-dir):
  verdicts.jsonl         -- one row per call with the candidate,
                            ground truth, grader verdict, reasoning.
  summary.json           -- confusion matrix + precision/recall/F1
                            of the grader-as-classifier.
  disagreements.jsonl    -- rows where the grader disagreed with
                            the human label (for manual review).

Usage
-----
::

    python -m RFT.typo.v5.eval.calibrate_grader_v5 \\
        --benchmark benchmark/benchmark_typo_20260512T213238Z_gpt5-family.jsonl \\
        --csv-root data/source_inconsistency_bing21 \\
        --rulebook RFT/typo/v5/prompts/rulebook.md \\
        --grader-prompt RFT/typo/v5/grader/grader_prompt_calibration.md \\
        --output-dir outputs/typo_rft/calibrate_grader_v5 \\
        --deployment gpt-4o-2024-08-06 \\
        --concurrency 12
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import logging
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from RFT.typo.v5._task import is_v5_trivial_correction  # noqa: E402
from src.core.llm_client import LLMClient  # noqa: E402


logger = logging.getLogger("calibrate_grader_v5")


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_BENCHMARK = Path(
    "benchmark/benchmark_typo_20260512T213238Z_gpt5-family.jsonl"
)
DEFAULT_CSV_ROOT = Path("data/source_inconsistency_bing21")
DEFAULT_RULEBOOK = Path("RFT/typo/v5/prompts/rulebook.md")
DEFAULT_GRADER_PROMPT = Path(
    "RFT/typo/v5/grader/grader_prompt_calibration.md"
)
DEFAULT_OUTPUT_DIR = Path("outputs/typo_rft/calibrate_grader_v5")
# Default model spec. Comma-separated for round-robin rotation across
# the GPT-5 family via ``src.core.llm_client.LLMClient`` (auth +
# endpoint pool managed there).
DEFAULT_MODEL = "gpt-5.2,gpt-5.4,gpt-5.5"

# Cap how many distinct values we show the grader for column context.
# 200 keeps the prompt small and matches the size used elsewhere in
# the v3/v5 pipeline.
COLUMN_VALUES_CAP = 200

# Set of valid verdict labels the grader may emit.
VALID_VERDICTS = {
    "valid_extra",
    "ambiguous_extra",
    "trivial_fp",
    "entity_fp",
    "plain_fp",
}

FP_VERDICTS = {"trivial_fp", "entity_fp", "plain_fp"}


# ---------------------------------------------------------------------------
# Column-values cache (load each CSV column at most once)
# ---------------------------------------------------------------------------


def _build_csv_index(csv_root: Path) -> dict[str, Path]:
    return {p.name: p for p in csv_root.rglob("*.csv")}


_column_values_cache: dict[tuple[str, str], list[tuple[str, int]]] = {}
_column_values_lock = threading.Lock()


def get_column_values(
    csv_index: dict[str, Path],
    source: str,
    column: str,
) -> list[tuple[str, int]]:
    """Return distinct-value (value, count) list for (source, column).

    Cached across calls. Returns [] when the source CSV cannot be
    located or the column is missing -- the grader still runs with
    an empty column_values list (degraded context, logged once).
    """
    key = (source, column)
    with _column_values_lock:
        if key in _column_values_cache:
            return _column_values_cache[key]

    counts: collections.Counter[str] = collections.Counter()
    path = csv_index.get(source)
    if path is None:
        logger.warning("CSV not found for source=%s", source)
        with _column_values_lock:
            _column_values_cache[key] = []
        return []
    try:
        with path.open(encoding="utf-8", errors="replace") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames is None or column not in reader.fieldnames:
                logger.warning(
                    "Column %r not in %s", column, source,
                )
                result: list[tuple[str, int]] = []
            else:
                for row in reader:
                    val = row.get(column)
                    if val is None:
                        continue
                    val = str(val)
                    if not val or val.strip() == "":
                        continue
                    counts[val] += 1
                result = counts.most_common(COLUMN_VALUES_CAP)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed reading %s: %s", path, exc)
        result = []

    with _column_values_lock:
        _column_values_cache[key] = result
    return result


# ---------------------------------------------------------------------------
# Benchmark loading + filtering
# ---------------------------------------------------------------------------


# Heuristic: an llm_suggested_value is a usable "correction" only when
# it looks like a single concise value, not a free-text instruction
# the model emitted in lieu of an actual fix.
_BAD_SUGG_PREFIXES = (
    "replace", "use ", "the ", "this ", "consider", "perhaps",
    "likely", "should", "either", "no ", "verify", "check",
    "confirm", "maybe", "may be", "possibly", "possible ", "n/a",
    "unclear", "uncertain", "if ", "appears", "looks like",
    "intended", "probably", "ambiguous",
    "change ", "set ", "expand", "rename", "remove ", "add ",
    "include", "fix ", "split ", "concatenate", "format",
    "swap ", "drop ", "keep ", "standardise", "standardize",
    "normalise", "normalize", "leave ", "either:", "make ",
    "convert ", "ensure ",
)


def _looks_like_instruction(v: str) -> bool:
    """Detect quoted-instruction-style suggestions like
    'Change \"X\" to \"Y\"'. These cannot be used as a candidate
    correction string."""
    if v.count('"') >= 2 and len(v) > 30:
        return True
    if v.count("'") >= 2 and len(v) > 30:
        return True
    return False


def _is_usable_correction(value: str | None) -> bool:
    if not isinstance(value, str):
        return False
    v = value.strip()
    if not v:
        return False
    if len(v) > 200:
        return False
    if "\n" in v:
        return False
    low = v.casefold()
    for p in _BAD_SUGG_PREFIXES:
        if low.startswith(p):
            return False
    if _looks_like_instruction(v):
        return False
    # Heuristic: real corrections rarely contain quoted explanations
    # like '... (or, if abbreviation is intended ...)'.
    if "(" in v and ")" in v and len(v) > 60:
        return False
    return True


def load_benchmark_items(
    benchmark_path: Path,
    *,
    include_recall: bool,
    include_precision: bool,
    dedupe: bool = True,
) -> list[dict[str, Any]]:
    """Read the benchmark and emit one calibration item per kept row.

    With ``dedupe=True`` (the default), duplicate
    ``(source, column, value, correction)`` keys are collapsed to a
    single item -- the benchmark is per-cell, so the same typo often
    appears in many rows of the same column and would otherwise eat
    grader budget.

    Each item carries:
      mode:           'recall' | 'precision'
      ground_truth:   'typo' | 'clean'
      source, row, column, value, correction, flavour, label, note
      occurrence_count: how many benchmark rows the item represents.
    """
    raw_items: list[dict[str, Any]] = []
    stats = collections.Counter()
    with benchmark_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            label = d.get("label")
            value = d.get("original_value")
            if not isinstance(value, str) or not value:
                stats["dropped_no_value"] += 1
                continue
            if label == "ambiguous":
                stats["dropped_ambiguous"] += 1
                continue
            if label == "error":
                if not include_recall:
                    continue
                corr = d.get("corrected_value")
                if not isinstance(corr, str) or not corr:
                    stats["dropped_error_no_correction"] += 1
                    continue
                if is_v5_trivial_correction(value, corr):
                    stats["dropped_error_trivial"] += 1
                    continue
                raw_items.append({
                    "mode": "recall",
                    "ground_truth": "typo",
                    "source": d.get("source", ""),
                    "row": d.get("row"),
                    "column": d.get("column", ""),
                    "value": value,
                    "correction": corr,
                    "flavour": "spelling",
                    "label": label,
                    "note": d.get("note"),
                })
                stats["kept_recall"] += 1
            elif label == "clean":
                if not include_precision:
                    continue
                sugg = d.get("llm_suggested_value")
                if not _is_usable_correction(sugg):
                    stats["dropped_clean_unusable_suggestion"] += 1
                    continue
                raw_items.append({
                    "mode": "precision",
                    "ground_truth": "clean",
                    "source": d.get("source", ""),
                    "row": d.get("row"),
                    "column": d.get("column", ""),
                    "value": value,
                    "correction": sugg,
                    "flavour": "spelling",
                    "label": label,
                    "note": d.get("note"),
                })
                stats["kept_precision"] += 1
            else:
                stats[f"dropped_label_{label}"] += 1

    if not dedupe:
        logger.info("Benchmark load stats: %s", dict(stats))
        for it in raw_items:
            it["occurrence_count"] = 1
        return raw_items

    # Dedupe by (mode, source, column, value, correction). Keep the
    # first row's metadata; track occurrence count for later weighting.
    seen: dict[tuple, dict[str, Any]] = {}
    for it in raw_items:
        key = (
            it["mode"], it["source"], it["column"],
            it["value"], it["correction"],
        )
        if key in seen:
            seen[key]["occurrence_count"] += 1
            stats["deduped_collapsed"] += 1
        else:
            it = dict(it)
            it["occurrence_count"] = 1
            seen[key] = it
    items = list(seen.values())
    stats["after_dedupe_recall"] = sum(
        1 for i in items if i["mode"] == "recall"
    )
    stats["after_dedupe_precision"] = sum(
        1 for i in items if i["mode"] == "precision"
    )
    logger.info("Benchmark load stats: %s", dict(stats))
    return items


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------


def load_developer_prompt(
    grader_prompt_path: Path,
    rulebook_path: Path,
) -> str:
    """Extract the developer message from the calibration grader prompt
    and substitute the rulebook placeholder."""
    grader_md = grader_prompt_path.read_text(encoding="utf-8")
    # The developer message is the first fenced code block.
    m = re.search(r"```\n(.*?)\n```", grader_md, flags=re.DOTALL)
    if m is None:
        raise ValueError(
            f"No fenced developer message in {grader_prompt_path}"
        )
    dev = m.group(1)
    rulebook = rulebook_path.read_text(encoding="utf-8")
    if "<<<RULEBOOK>>>" not in dev:
        raise ValueError(
            f"Grader prompt missing <<<RULEBOOK>>> placeholder"
        )
    return dev.replace("<<<RULEBOOK>>>", rulebook)


def build_user_message(
    column: str,
    column_values: list[tuple[str, int]],
    candidate: dict[str, Any],
) -> str:
    cv_payload = [
        {"value": v, "count": c} for v, c in column_values
    ]
    cand_payload = {
        "value": candidate["value"],
        "correction": candidate["correction"],
        "flavour": candidate.get("flavour", "spelling"),
        "rationale": candidate.get(
            "rationale", "candidate under adjudication"
        ),
    }
    return (
        f"Column: {column}\n"
        f"Column values (JSON, capped): "
        f"{json.dumps(cv_payload, ensure_ascii=False)}\n\n"
        f"Candidate (JSON): "
        f"{json.dumps(cand_payload, ensure_ascii=False)}"
    )


# ---------------------------------------------------------------------------
# Grader response parsing
# ---------------------------------------------------------------------------


_VERDICT_RE = re.compile(
    r"^\s*Verdict\s*:\s*([A-Za-z_]+)\s*$", flags=re.MULTILINE,
)
_REASONING_RE = re.compile(
    r"^\s*Reasoning\s*:\s*(.+?)\s*$", flags=re.MULTILINE,
)


def parse_verdict(text: str) -> tuple[str | None, str | None]:
    if not isinstance(text, str) or not text:
        return None, None
    m = _VERDICT_RE.search(text)
    verdict = m.group(1).strip().lower() if m else None
    if verdict not in VALID_VERDICTS:
        verdict = None
    r = _REASONING_RE.search(text)
    reasoning = r.group(1).strip() if r else None
    return verdict, reasoning


# ---------------------------------------------------------------------------
# Per-item grader call
# ---------------------------------------------------------------------------


def call_grader(
    client: LLMClient,
    model: str,
    developer_message: str,
    user_message: str,
    *,
    reasoning_effort: str = "low",
) -> tuple[str | None, str | None, str | None]:
    """Call the grader once. Returns (verdict, reasoning, raw_text).

    On any API error returns (None, None, raw_or_None) with the error
    logged. Uses ``LLMClient`` for endpoint rotation, retries, and
    GPT-5-family round-robin.
    """
    resp = client.query(
        model=model,
        prompt=user_message,
        system_prompt=developer_message,
        reasoning_effort=reasoning_effort,
        reasoning_summary="concise",
    )
    raw = resp.text or None
    if not resp.ok:
        logger.warning(
            "Grader error after %d attempt(s): %s",
            resp.attempts, resp.error,
        )
        return None, None, raw
    verdict, reasoning = parse_verdict(raw or "")
    return verdict, reasoning, raw


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _classify_outcome(mode: str, verdict: str | None) -> str:
    """Map (mode, verdict) -> outcome bucket for confusion matrix.

    Outcomes:
      recall_hit        mode=recall   verdict=valid_extra (correct)
      recall_ambiguous  mode=recall   verdict=ambiguous_extra (drop)
      recall_miss       mode=recall   verdict in FP buckets (wrong)
      recall_invalid    mode=recall   verdict could not be parsed
      precision_hit     mode=precision verdict in FP buckets (correct)
      precision_ambiguous mode=precision verdict=ambiguous_extra (drop)
      precision_miss    mode=precision verdict=valid_extra (wrong)
      precision_invalid mode=precision verdict could not be parsed
    """
    if verdict is None:
        return f"{mode}_invalid"
    if mode == "recall":
        if verdict == "valid_extra":
            return "recall_hit"
        if verdict == "ambiguous_extra":
            return "recall_ambiguous"
        return "recall_miss"
    if mode == "precision":
        if verdict in FP_VERDICTS:
            return "precision_hit"
        if verdict == "ambiguous_extra":
            return "precision_ambiguous"
        return "precision_miss"
    return "unknown"


def _process_item(
    client: LLMClient,
    model: str,
    developer_message: str,
    csv_index: dict[str, Path],
    item: dict[str, Any],
    *,
    reasoning_effort: str = "low",
) -> dict[str, Any]:
    column_values = get_column_values(
        csv_index, item.get("source", ""), item.get("column", ""),
    )
    user_msg = build_user_message(
        item.get("column", ""), column_values, item,
    )
    verdict, reasoning, raw = call_grader(
        client, model, developer_message, user_msg,
        reasoning_effort=reasoning_effort,
    )
    outcome = _classify_outcome(item["mode"], verdict)
    return {
        **item,
        "verdict": verdict,
        "reasoning": reasoning,
        "raw_grader_text": raw,
        "outcome": outcome,
    }


def summarise(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-item records into precision/recall metrics."""
    outcomes = collections.Counter(r["outcome"] for r in records)
    by_verdict = collections.Counter(
        (r["mode"], r.get("verdict") or "<invalid>") for r in records
    )

    # Recall test: % of human typos the grader classified as valid_extra,
    # ambiguous excluded from denominator.
    rec_hit = outcomes.get("recall_hit", 0)
    rec_miss = outcomes.get("recall_miss", 0)
    rec_amb = outcomes.get("recall_ambiguous", 0)
    rec_invalid = outcomes.get("recall_invalid", 0)
    rec_denom = rec_hit + rec_miss + rec_invalid
    grader_recall = rec_hit / rec_denom if rec_denom else None

    # Precision test: % of human-clean cells the grader classified as
    # an FP bucket (correctly rejected), ambiguous excluded.
    prec_hit = outcomes.get("precision_hit", 0)
    prec_miss = outcomes.get("precision_miss", 0)
    prec_amb = outcomes.get("precision_ambiguous", 0)
    prec_invalid = outcomes.get("precision_invalid", 0)
    prec_denom = prec_hit + prec_miss + prec_invalid
    grader_specificity = prec_hit / prec_denom if prec_denom else None

    # Classifier P/R/F1 treating valid_extra=positive, FP=negative.
    tp = rec_hit            # human=typo, grader=valid_extra
    fn = rec_miss           # human=typo, grader=FP
    fp = prec_miss          # human=clean, grader=valid_extra
    tn = prec_hit           # human=clean, grader=FP
    classifier_p = tp / (tp + fp) if (tp + fp) else None
    classifier_r = tp / (tp + fn) if (tp + fn) else None
    classifier_f1 = (
        2 * classifier_p * classifier_r / (classifier_p + classifier_r)
        if classifier_p and classifier_r else None
    )

    # Breakdown of grader-miss verdicts on recall test (which FP bucket
    # did it wrongly assign?) -- useful for diagnosing rulebook gaps.
    miss_bucket_breakdown = collections.Counter(
        r.get("verdict") for r in records
        if r["outcome"] == "recall_miss"
    )

    return {
        "n_records": len(records),
        "outcomes": dict(outcomes),
        "by_verdict": {f"{m}__{v}": c for (m, v), c in by_verdict.items()},
        "grader_recall_on_typos": grader_recall,
        "grader_specificity_on_clean": grader_specificity,
        "classifier": {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": classifier_p,
            "recall": classifier_r,
            "f1": classifier_f1,
        },
        "recall_miss_breakdown": dict(miss_bucket_breakdown),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    verdicts_path = output_dir / "verdicts.jsonl"
    summary_path = output_dir / "summary.json"
    disagreements_path = output_dir / "disagreements.jsonl"

    include_recall = args.mode in {"recall", "both"}
    include_precision = args.mode in {"precision", "both"}

    items = load_benchmark_items(
        Path(args.benchmark),
        include_recall=include_recall,
        include_precision=include_precision,
        dedupe=not args.no_dedupe,
    )
    if args.limit and args.limit > 0:
        # Stratified limit: take up to limit/2 from each mode so the
        # caller can get a fast smoke test that exercises both halves.
        per_mode = max(1, args.limit // 2)
        rec = [i for i in items if i["mode"] == "recall"][:per_mode]
        prec = [i for i in items if i["mode"] == "precision"][:per_mode]
        items = rec + prec
    logger.info(
        "Loaded %d calibration items (recall=%d, precision=%d)",
        len(items),
        sum(1 for i in items if i["mode"] == "recall"),
        sum(1 for i in items if i["mode"] == "precision"),
    )

    csv_index = _build_csv_index(Path(args.csv_root))
    logger.info("Indexed %d CSVs under %s", len(csv_index), args.csv_root)

    developer_message = load_developer_prompt(
        Path(args.grader_prompt), Path(args.rulebook),
    )
    logger.info(
        "Developer message: %d chars (rulebook spliced in)",
        len(developer_message),
    )

    client = LLMClient()

    records: list[dict[str, Any]] = []
    records_lock = threading.Lock()
    fh_lock = threading.Lock()
    # Truncate output file at start; we stream as we go for crash safety.
    verdicts_path.write_text("", encoding="utf-8")

    def worker(it: dict[str, Any]) -> dict[str, Any]:
        rec = _process_item(
            client, args.model, developer_message, csv_index, it,
            reasoning_effort=args.reasoning_effort,
        )
        with records_lock:
            records.append(rec)
        with fh_lock:
            with verdicts_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec

    n_done = 0
    n_total = len(items)
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(worker, it) for it in items]
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as exc:  # noqa: BLE001
                logger.error("Worker error: %s", exc)
            n_done += 1
            if n_done % 25 == 0 or n_done == n_total:
                logger.info("Progress: %d / %d", n_done, n_total)

    summary = summarise(records)
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    # Disagreements: recall_miss + precision_miss + *_invalid.
    disagreement_outcomes = {
        "recall_miss", "precision_miss",
        "recall_invalid", "precision_invalid",
    }
    with disagreements_path.open("w", encoding="utf-8") as fh:
        for r in records:
            if r["outcome"] in disagreement_outcomes:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    logger.info("Summary: %s", json.dumps(summary, indent=2))
    logger.info("Wrote %s", verdicts_path)
    logger.info("Wrote %s", summary_path)
    logger.info("Wrote %s", disagreements_path)
    return summary


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmark", default=str(DEFAULT_BENCHMARK))
    p.add_argument("--csv-root", default=str(DEFAULT_CSV_ROOT))
    p.add_argument("--rulebook", default=str(DEFAULT_RULEBOOK))
    p.add_argument("--grader-prompt", default=str(DEFAULT_GRADER_PROMPT))
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=(
            "Model spec for src.core.llm_client.LLMClient. Single "
            "name (e.g. 'gpt-5.2') or comma-separated rotation pool "
            "(e.g. 'gpt-5.2,gpt-5.4,gpt-5.5')."
        ),
    )
    p.add_argument(
        "--reasoning-effort", default="low",
        choices=["none", "low", "medium", "high", "xhigh"],
    )
    p.add_argument(
        "--mode",
        choices=["recall", "precision", "both"],
        default="both",
        help=(
            "Which calibration tests to run. 'recall' uses human-"
            "labelled typos; 'precision' uses human-labelled clean "
            "cells with usable llm_suggested_value as the candidate "
            "correction; 'both' runs both."
        ),
    )
    p.add_argument(
        "--limit", type=int, default=0,
        help="If >0, cap total items (stratified across modes).",
    )
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument(
        "--no-dedupe", action="store_true",
        help=(
            "Disable per-(source,column,value,correction) deduplication. "
            "Default is to dedupe -- the benchmark is per-cell so the "
            "same typo case otherwise appears many times."
        ),
    )
    return p


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
