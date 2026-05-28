"""Single-pass conservative-prompt precision baseline for single-column
error detection.

Walks ``--source-dir`` for ``*.xlsx/*.csv`` paths (or a single CSV),
loops over each requested single-column error type, and for every
``(csv, column, error_type)`` triple makes ONE LLM call against a
precision-focused conservative prompt. The prompt asks the model to
emit cell-level errors only when it is unambiguously confident; the
goal is to push precision as high as possible (recall is secondary).
Cell-level predictions are written per error type to
``predictions_<error_type>.jsonl`` so they can be scored against the
labeled benchmark JSONL files in ``benchmark/``.

Output is a self-contained run dir under ``--output-dir`` with:
  * ``predictions_<error_type>.jsonl`` -- one record per flagged cell.
  * ``cache.jsonl`` -- raw LLM responses keyed by prompt hash (so
    re-runs resume without paying LLM calls again).
  * ``summary.json`` -- config + per-type cell + column counts.
  * ``RUN_STATUS.json`` -- in_progress / completed / failed.

The runner does NOT score predictions itself; that's
:mod:`experiments.score_precision_baseline` against the benchmark
JSONLs.

Example::

    python -m experiments.run_precision_baseline \\
        --source-dir data/source_inconsistency_bing21 \\
        --error-types typo,inconsistency,numeric_outlier,semantic_incompatible \\
        --concurrency 20
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import logging
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.llm_client import LLMClient  # noqa: E402

logger = logging.getLogger("run_precision_baseline")

# How many distinct values to show the LLM per column. Above this we
# rank by frequency and truncate (rare singleton values are most likely
# to be typos / inconsistency anchors anyway).
COLUMN_CONTEXT_LIMIT = 80

# Prompt version stamp: bump whenever any precision_baseline_*.yaml
# changes so cached results are invalidated cleanly.
PROMPT_VERSION = "2026-05-18-v3.4"

# Single-column error types we know how to evaluate against the
# benchmark JSONLs in ``benchmark/``.
DEFAULT_ERROR_TYPES = (
    "typo",
    "inconsistency",
    "numeric_outlier",
    "semantic_incompatible",
)

# Maps error_type -> prompt YAML path (project-relative).
PROMPT_PATHS = {
    et: f"prompts/precision_baseline_{et}.yaml"
    for et in DEFAULT_ERROR_TYPES
}


# ---------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------


@dataclass
class PromptCfg:
    system: str
    user_template: str
    model: str
    reasoning_effort: str
    error_type: str


def load_prompt_cfg(path: Path, error_type: str) -> PromptCfg:
    doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return PromptCfg(
        system=str(doc.get("system") or "").strip(),
        user_template=str(doc.get("user_template") or ""),
        model=str(doc.get("model") or "gpt-5.2"),
        reasoning_effort=str(doc.get("reasoning_effort") or "medium"),
        error_type=error_type,
    )


_ZERO_WIDTH_CHARS = "​‌‍⁠﻿­"


def _sentinel_encode(s: str) -> str:
    """Replace runs of whitespace inside ``s`` with visible sentinels.

    Same convention as ``prompts/error_types.yaml`` (0% CSP):
      regular space inside a multi-space run -> "␣"
      single space at a word boundary        -> kept as-is
      tab                                    -> "→"
      newline                                -> "↵"
      zero-width chars                       -> "·"

    The point is to keep the LLM able to detect whitespace defects
    (the sentinel run shows the same shape) while removing literal
    long whitespace runs from the rendered prompt. Long literal
    runs and embedded newlines look like prompt-injection artifacts
    to Azure's jailbreak detector.
    """
    if not s:
        return s
    out = []
    for ch in s:
        if ch == "\t":
            out.append("→")
        elif ch == "\n" or ch == "\r":
            out.append("↵")
        elif ch in _ZERO_WIDTH_CHARS:
            out.append("·")
        else:
            out.append(ch)
    encoded = "".join(out)
    # Collapse runs of 2+ regular spaces into "␣"-sentinels (one per
    # space) so a "double space" still looks like a double sentinel.
    import re as _re
    encoded = _re.sub(r" {2,}", lambda m: "␣" * len(m.group(0)), encoded)
    return encoded


def render_prompt(
    cfg: PromptCfg,
    *,
    source: str,
    column: str,
    df: pd.DataFrame,
) -> Tuple[str, int, bool]:
    """Return ``(prompt_text, distinct_count, truncated)``.

    The LLM only sees the (header, distinct-value list with counts)
    payload -- no other columns, no row-level context. This keeps the
    baseline blind to anything beyond the column itself.

    Column header and value strings are sentinel-encoded (see
    ``_sentinel_encode``) to keep the LLM able to spot whitespace
    defects while avoiding the long literal whitespace runs that
    Azure's jailbreak filter flags.
    """
    vc = df[column].astype(str).value_counts(dropna=False)
    total = int(len(vc))
    truncated = total > COLUMN_CONTEXT_LIMIT
    rows: List[str] = []
    for v, n in vc.head(COLUMN_CONTEXT_LIMIT).items():
        encoded = _sentinel_encode(str(v))
        rows.append(f"  - {json.dumps(encoded, ensure_ascii=False)} (count={int(n)})")
    value_list = "\n".join(rows) if rows else "  (column is empty)"
    truncated_hint = (
        f" (showing top {COLUMN_CONTEXT_LIMIT} by frequency)"
        if truncated else ""
    )
    prompt = cfg.user_template.format(
        source=source,
        column=_sentinel_encode(column),
        value_list=value_list,
        distinct_count=total,
        truncated_hint=truncated_hint,
    )
    return prompt, total, truncated


# ---------------------------------------------------------------------
# LLM response parsing
# ---------------------------------------------------------------------

_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.+?)```", re.DOTALL | re.IGNORECASE)
_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_errors(text: str) -> Optional[List[Dict[str, Any]]]:
    """Parse ``{"errors": [...]}`` payloads. Returns ``None`` on shape
    failure; an empty list on a well-formed empty payload."""
    if not text:
        return None
    t = text.strip()
    fence = _CODE_FENCE_RE.search(t)
    if fence:
        t = fence.group(1).strip()
    if not t.startswith("{"):
        m = _JSON_OBJ_RE.search(t)
        if m:
            t = m.group(0)
    try:
        payload = json.loads(t)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    raw = payload.get("errors")
    if raw is None:
        return None
    if not isinstance(raw, list):
        return None
    out: List[Dict[str, Any]] = []
    for f in raw:
        if not isinstance(f, dict):
            continue
        value = f.get("value")
        if not isinstance(value, str):
            continue
        correction = f.get("correction")
        if correction is not None and not isinstance(correction, str):
            correction = None
        rationale = str(f.get("rationale") or "")
        out.append({
            "value": value,
            "correction": correction,
            "rationale": rationale,
        })
    return out


# ---------------------------------------------------------------------
# Cache (resume on rerun)
# ---------------------------------------------------------------------


def cache_key(
    *,
    system_prompt: str,
    prompt: str,
    model: str,
    reasoning_effort: str,
    error_type: str,
) -> str:
    payload = {
        "prompt_version": PROMPT_VERSION,
        "model": model,
        "reasoning_effort": reasoning_effort,
        "error_type": error_type,
        "system": system_prompt,
        "prompt": prompt,
    }
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def load_cache(path: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = rec.get("key")
            if isinstance(key, str):
                out[key] = rec
    return out


def append_cache(path: Path, rec: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False))
        fh.write("\n")


# ---------------------------------------------------------------------
# CSV resolution + load
# ---------------------------------------------------------------------


def resolve_csv_paths(source_dir: Path) -> List[Path]:
    """Walk ``source_dir`` for ``*.xlsx/*.csv`` paths. Returns absolute
    paths sorted by basename for deterministic iteration."""
    paths = []
    for p in source_dir.rglob("*.csv"):
        if p.is_file():
            paths.append(p)
    return sorted(paths, key=lambda p: p.name)


def load_csv(path: Path) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(
            path, dtype=str, keep_default_na=False, low_memory=False,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("failed to load %s: %s", path, exc)
        return None


# ---------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------


@dataclass
class Job:
    source: str
    csv_path: Path
    column: str
    error_type: str
    cfg: PromptCfg


@dataclass
class JobResult:
    source: str
    column: str
    error_type: str
    distinct_count: int
    truncated: bool
    parsed_errors: Optional[List[Dict[str, Any]]]
    raw_text: str
    error: Optional[str]
    predictions: List[Dict[str, Any]] = field(default_factory=list)
    elapsed_s: float = 0.0
    cached: bool = False


def expand_errors_to_predictions(
    *,
    source: str,
    column: str,
    error_type: str,
    df: pd.DataFrame,
    findings: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Convert per-column findings into per-row predictions.

    Each finding's ``value`` is matched verbatim against the column;
    every matching row becomes one cell-level prediction record. If the
    LLM emits a value that does not appear in the column verbatim, the
    prediction is dropped (silent: it is a recall miss, not a false
    positive, since no cell was flagged in our universe).
    """
    if not findings:
        return []
    col = df[column].astype(str)
    # The LLM saw sentinel-encoded values, so it reports sentinel-encoded
    # values in `f["value"]`. Match against the encoded column, but emit
    # the original (un-encoded) raw cell value in `original_value` so
    # downstream artifacts stay consumer-friendly.
    encoded_col = col.map(_sentinel_encode)
    out: List[Dict[str, Any]] = []
    for f in findings:
        value = f.get("value") or ""
        correction = f.get("correction")
        rationale = f.get("rationale") or ""
        mask = encoded_col == value
        rows = list(mask[mask].index.astype(int))
        for r in rows:
            out.append({
                "source": source,
                "row": int(r),
                "column": column,
                "original_value": col.iloc[r],
                "predicted_correction": correction,
                "error_type": error_type,
                "rationale": rationale,
            })
    return out


def process_job(
    job: Job,
    df: pd.DataFrame,
    *,
    llm: LLMClient,
    cache: Dict[str, Dict[str, Any]],
    cache_path: Path,
    cache_lock: threading.Lock,
) -> JobResult:
    t0 = time.time()
    cfg = job.cfg
    prompt, distinct_count, truncated = render_prompt(
        cfg, source=job.source, column=job.column, df=df,
    )
    key = cache_key(
        system_prompt=cfg.system,
        prompt=prompt,
        model=cfg.model,
        reasoning_effort=cfg.reasoning_effort,
        error_type=job.error_type,
    )
    cached_rec = cache.get(key)
    if cached_rec is not None:
        raw_text = str(cached_rec.get("raw") or "")
        parsed = parse_errors(raw_text)
        predictions = expand_errors_to_predictions(
            source=job.source, column=job.column, error_type=job.error_type,
            df=df, findings=parsed or [],
        )
        return JobResult(
            source=job.source, column=job.column, error_type=job.error_type,
            distinct_count=distinct_count, truncated=truncated,
            parsed_errors=parsed, raw_text=raw_text, error=None,
            predictions=predictions, elapsed_s=time.time() - t0, cached=True,
        )

    try:
        resp = llm.query(
            cfg.model, prompt,
            reasoning_effort=cfg.reasoning_effort,
            reasoning_summary="auto",
            system_prompt=cfg.system,
        )
        raw_text = resp.text or ""
        err = resp.error
    except Exception as exc:  # noqa: BLE001
        raw_text = ""
        err = f"call_failed: {exc}"

    parsed = parse_errors(raw_text) if raw_text else None

    rec = {
        "key": key,
        "source": job.source,
        "column": job.column,
        "error_type": job.error_type,
        "prompt_version": PROMPT_VERSION,
        "model": cfg.model,
        "reasoning_effort": cfg.reasoning_effort,
        "raw": raw_text,
        "error": err,
        "ts": _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    with cache_lock:
        cache[key] = rec
        append_cache(cache_path, rec)

    predictions = expand_errors_to_predictions(
        source=job.source, column=job.column, error_type=job.error_type,
        df=df, findings=parsed or [],
    )
    return JobResult(
        source=job.source, column=job.column, error_type=job.error_type,
        distinct_count=distinct_count, truncated=truncated,
        parsed_errors=parsed, raw_text=raw_text, error=err,
        predictions=predictions, elapsed_s=time.time() - t0, cached=False,
    )


# ---------------------------------------------------------------------
# Run-status
# ---------------------------------------------------------------------


def _write_run_status(run_dir: Path, status: str, *, detail: Optional[str] = None) -> None:
    """Mirror run_cleaning_detection's RUN_STATUS.json so the labeling
    UI picks the run up. Best-effort; never raises."""
    try:
        payload = {
            "status": status,
            "updated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        }
        if detail:
            payload["detail"] = detail[:500]
        (run_dir / "RUN_STATUS.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8",
        )
    except Exception:  # noqa: BLE001
        logger.debug("could not write RUN_STATUS.json", exc_info=True)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--source-dir", type=Path, required=True,
        help="Directory tree of CSV files (e.g. data/source_inconsistency_bing21).",
    )
    p.add_argument(
        "--error-types", type=str, default=",".join(DEFAULT_ERROR_TYPES),
        help=("Comma-separated single-column error type keys to run "
              "(default: typo,inconsistency,numeric_outlier,"
              "semantic_incompatible)."),
    )
    p.add_argument(
        "--output-dir", type=Path, default=None,
        help="Run dir parent. Default: outputs/precision_baseline_<UTC>.",
    )
    p.add_argument(
        "--run-name", type=str, default=None,
        help="Override the run dir basename (under --output-dir parent).",
    )
    p.add_argument(
        "--max-sources", type=int, default=None,
        help="Cap on the number of CSVs processed (debug / smoke).",
    )
    p.add_argument(
        "--sources", type=str, default=None,
        help="Comma-separated list of CSV basenames to process exclusively.",
    )
    p.add_argument(
        "--max-columns-per-source", type=int, default=None,
        help="Cap on columns per CSV (debug / smoke).",
    )
    p.add_argument(
        "--concurrency", type=int, default=20,
        help="Number of concurrent LLM calls.",
    )
    p.add_argument(
        "--model", type=str, default=None,
        help="Override the model field of every prompt YAML.",
    )
    p.add_argument(
        "--reasoning-effort", type=str, default=None,
        help="Override the reasoning_effort field of every prompt YAML.",
    )
    p.add_argument(
        "--log-level", type=str, default="INFO",
    )
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )
    for noisy in ("httpx", "httpcore", "openai", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    source_dir = args.source_dir.expanduser().resolve()
    if not source_dir.is_dir():
        logger.error("source-dir not found: %s", source_dir)
        return 2

    error_types = [s.strip() for s in args.error_types.split(",") if s.strip()]
    bad = [et for et in error_types if et not in PROMPT_PATHS]
    if bad:
        logger.error("unknown error types: %s; allowed: %s",
                     bad, sorted(PROMPT_PATHS))
        return 2

    # Load all prompts up-front so we fail fast if any is malformed.
    prompts: Dict[str, PromptCfg] = {}
    for et in error_types:
        path = PROJECT_ROOT / PROMPT_PATHS[et]
        cfg = load_prompt_cfg(path, et)
        if args.model:
            cfg = PromptCfg(
                system=cfg.system, user_template=cfg.user_template,
                model=args.model,
                reasoning_effort=args.reasoning_effort or cfg.reasoning_effort,
                error_type=et,
            )
        elif args.reasoning_effort:
            cfg = PromptCfg(
                system=cfg.system, user_template=cfg.user_template,
                model=cfg.model, reasoning_effort=args.reasoning_effort,
                error_type=et,
            )
        prompts[et] = cfg

    # Pick a representative model string for the run-dir name.
    rep_model = "+".join(sorted({cfg.model for cfg in prompts.values()}))
    safe_model = rep_model.replace("/", "_")[:60]

    # Output dir.
    out_root = (
        args.output_dir.resolve() if args.output_dir
        else PROJECT_ROOT / "outputs"
    )
    if args.run_name:
        run_dir = out_root / args.run_name
    else:
        ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = out_root / f"precision_baseline_{ts}_{safe_model}"
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_run_status(run_dir, "in_progress")
    logger.info("Run dir: %s", run_dir)

    cache_path = run_dir / "cache.jsonl"
    cache = load_cache(cache_path)
    cache_lock = threading.Lock()
    if cache:
        logger.info("Loaded %d cached call(s) from %s", len(cache), cache_path)

    # Resolve CSVs.
    csv_paths = resolve_csv_paths(source_dir)
    if args.sources:
        wanted = {s.strip() for s in args.sources.split(",") if s.strip()}
        csv_paths = [p for p in csv_paths if p.name in wanted]
    if args.max_sources is not None:
        csv_paths = csv_paths[: int(args.max_sources)]
    logger.info("Processing %d CSV(s)", len(csv_paths))

    # Build jobs.
    jobs: List[Job] = []
    df_cache: Dict[str, pd.DataFrame] = {}
    for p in csv_paths:
        src = p.name
        df = load_csv(p)
        if df is None or df.empty:
            continue
        df_cache[src] = df
        cols = list(df.columns)
        if args.max_columns_per_source is not None:
            cols = cols[: int(args.max_columns_per_source)]
        for col in cols:
            for et in error_types:
                jobs.append(Job(
                    source=src, csv_path=p, column=col,
                    error_type=et, cfg=prompts[et],
                ))
    logger.info("Built %d job(s) total (%d csvs × %d types × avg cols)",
                len(jobs), len(df_cache), len(error_types))

    try:
        llm = LLMClient()

        # Per-error-type prediction file handles (write streaming).
        pred_paths = {
            et: run_dir / f"predictions_{et}.jsonl" for et in error_types
        }
        # Truncate any prior partial prediction file from a previous run.
        # (Cache is preserved separately; predictions are re-derived from
        # cache anyway.)
        for path in pred_paths.values():
            path.write_text("", encoding="utf-8")
        pred_locks = {et: threading.Lock() for et in error_types}

        n_done = 0
        n_errors = 0
        n_cached = 0
        per_type_pred_counts: Dict[str, int] = {et: 0 for et in error_types}
        per_type_call_counts: Dict[str, int] = {et: 0 for et in error_types}
        t_start = time.time()

        def _run_one(job: Job) -> JobResult:
            df = df_cache[job.source]
            return process_job(
                job, df,
                llm=llm, cache=cache, cache_path=cache_path,
                cache_lock=cache_lock,
            )

        with ThreadPoolExecutor(max_workers=int(args.concurrency)) as pool:
            futs = {pool.submit(_run_one, j): j for j in jobs}
            for fut in as_completed(futs):
                job = futs[fut]
                try:
                    res = fut.result()
                except Exception as exc:  # noqa: BLE001
                    logger.exception(
                        "job failed: src=%s col=%s et=%s: %s",
                        job.source, job.column, job.error_type, exc,
                    )
                    n_errors += 1
                    n_done += 1
                    continue
                n_done += 1
                per_type_call_counts[res.error_type] += 1
                if res.cached:
                    n_cached += 1
                if res.error:
                    n_errors += 1
                # Persist predictions per type (append).
                if res.predictions:
                    path = pred_paths[res.error_type]
                    lock = pred_locks[res.error_type]
                    with lock:
                        with path.open("a", encoding="utf-8") as fh:
                            for rec in res.predictions:
                                fh.write(json.dumps(rec, ensure_ascii=False))
                                fh.write("\n")
                    per_type_pred_counts[res.error_type] += len(res.predictions)
                if n_done % 200 == 0 or n_done == len(jobs):
                    elapsed = time.time() - t_start
                    rate = n_done / elapsed if elapsed > 0 else 0
                    logger.info(
                        "Progress: %d/%d jobs (%.1f/s, %d cached, %d errors); "
                        "predictions so far: %s",
                        n_done, len(jobs), rate, n_cached, n_errors,
                        {et: per_type_pred_counts[et] for et in error_types},
                    )

        # Final summary.
        summary = {
            "mode": "single_pass_precision_baseline",
            "kind": "precision_baseline",
            "ts": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "source_dir": str(source_dir),
            "csv_count": len(df_cache),
            "csv_paths": [str(p) for p in csv_paths if p.name in df_cache],
            "config": {
                "error_types": error_types,
                "models": {et: prompts[et].model for et in error_types},
                "reasoning_effort": {
                    et: prompts[et].reasoning_effort for et in error_types
                },
                "concurrency": int(args.concurrency),
                "prompt_version": PROMPT_VERSION,
                "column_context_limit": COLUMN_CONTEXT_LIMIT,
            },
            "stats": {
                "n_jobs": len(jobs),
                "n_completed": n_done,
                "n_cached": n_cached,
                "n_errors": n_errors,
                "per_type_calls": per_type_call_counts,
                "per_type_predictions": per_type_pred_counts,
                "elapsed_s": time.time() - t_start,
            },
            "prediction_files": {
                et: pred_paths[et].name for et in error_types
            },
        }
        (run_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        _write_run_status(run_dir, "completed")
        logger.info(
            "Done. %d jobs (%d cached) -> predictions: %s",
            n_done, n_cached, per_type_pred_counts,
        )
        logger.info("Run dir: %s", run_dir)
        return 0
    except Exception as exc:  # noqa: BLE001
        logger.exception("run failed")
        _write_run_status(run_dir, "failed", detail=str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
