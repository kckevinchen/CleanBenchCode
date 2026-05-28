"""Single-pass GPT-5.5 typo labelling for ``data/curated_bing21_100``.

For every (source CSV, column) in the dataset, this script makes ONE
call to GPT-5.5 using the v5 rulebook + per-column distinct-value
listing (same prompt as ``RFT/typo_v5/_task.build_user_message``)
and writes the predictions as a detection-run directory under
``outputs/`` that the labelling frontend can open directly.

Output layout
-------------
::

    outputs/<ts>_gpt-5.5_curated_bing21_100/
        summary.json        -- minimal config + counts the runs list reads
        issues.json         -- per-finding records (frontend canonical input)
        issues.jsonl        -- same data as JSONL (for downstream tooling)
        clusters.json       -- one cluster per finding
        clusters.jsonl
        RUN_STATUS.json
        raw_responses/<src>__<col>.json   -- LLM payloads for audit

Each LLM call is one "report" indexed by a global counter. For every
typo entry the model returns, we expand it into one issue (and one
cluster) covering all rows in the column whose value matches verbatim.

Usage
-----
::

    python -m experiments.label_curated_bing21_100_gpt55 \\
        --dataset data/curated_bing21_100 \\
        --model gpt-5.5 \\
        --workers 8

To evaluate a different rulebook / system prompt, point
``--system-prompt-file`` at the new prompt text file. The default is
the v5 rulebook bundled in ``RFT/typo_v5/_task.py``; any plain-text
file (e.g. ``RFT/typo_v4/prompts/rulebook.md``,
``docs/typo_prompt_bundle/system_prompt.txt``, or a hand-written
ablation) is accepted -- the user-message format and JSON response
schema are unchanged across prompts so the rest of the pipeline is
prompt-agnostic.

By default the script is idempotent: if the target output directory
already has a ``raw_responses/<source>__<column>.json`` for a given
(source, column) and the response parsed cleanly, that pair is
skipped on re-run. Pass ``--refresh`` to discard the cache.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from azure.identity import AzureCliCredential, get_bearer_token_provider
from openai import AzureOpenAI

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from RFT.typo_v5 import _task as t  # noqa: E402

logger = logging.getLogger("label_curated_bing21_100_gpt55")


# Endpoint pool that carries gpt-5.5.
DEFAULT_ENDPOINTS = (
    "https://dmxaz-test-aoai-eu2.openai.azure.com/",
)
DEFAULT_API_VERSION = "2025-04-01-preview"
DEFAULT_MODEL = "gpt-5.5"
DEFAULT_DATASET = "data/curated_bing21_100"
DEFAULT_REASONING_EFFORT = "medium"


# ---------------------------------------------------------------------------
# Client setup
# ---------------------------------------------------------------------------


def _make_client(endpoint: str, api_version: str) -> AzureOpenAI:
    credential = AzureCliCredential()
    token_provider = get_bearer_token_provider(
        credential, "https://cognitiveservices.azure.com/.default",
    )
    return AzureOpenAI(
        api_version=api_version,
        azure_endpoint=endpoint,
        azure_ad_token_provider=token_provider,
        timeout=240.0,
    )


def _chat_rf_to_responses_text(chat_rf: dict[str, Any]) -> dict[str, Any]:
    """Flatten chat-style json_schema into Responses-API text.format."""
    inner = chat_rf.get("json_schema", {})
    return {
        "type": "json_schema",
        "name": inner.get("name", "output"),
        "schema": inner.get("schema", {}),
        "strict": inner.get("strict", True),
    }


def _extract_text(resp: Any) -> str:
    txt = getattr(resp, "output_text", None)
    if txt:
        return txt
    out = getattr(resp, "output", None) or []
    parts: list[str] = []
    for item in out:
        for c in getattr(item, "content", None) or []:
            tval = getattr(c, "text", None)
            if tval:
                parts.append(tval)
    return "".join(parts)


def _parse_json_obj(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```\s*$", "", s)
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Dataset iteration
# ---------------------------------------------------------------------------


def _load_csv(path: Path) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[])
    except UnicodeDecodeError:
        try:
            return pd.read_csv(
                path, dtype=str, keep_default_na=False, na_values=[],
                encoding="latin-1",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("CSV load failed for %s: %s", path, exc)
            return None
    except Exception as exc:  # noqa: BLE001
        logger.warning("CSV load failed for %s: %s", path, exc)
        return None


def _discover_csvs(dataset_dir: Path) -> list[Path]:
    paths = sorted(p for p in dataset_dir.rglob("*.csv") if p.is_file())
    # Basename-uniqueness check so the (source, row, column) join is
    # well-defined downstream.
    seen: dict[str, Path] = {}
    for p in paths:
        if p.name in seen and seen[p.name] != p:
            raise ValueError(
                f"Duplicate CSV basename {p.name!r} under {dataset_dir} "
                f"({seen[p.name]} vs {p})."
            )
        seen[p.name] = p
    return paths


# ---------------------------------------------------------------------------
# Single (source, column) LLM call
# ---------------------------------------------------------------------------


@dataclass
class ColumnTask:
    csv_path: Path
    column: str
    column_idx: int
    values_with_counts: list[tuple[str, int]]
    # df.index -> value, so we can recover affected rows for each
    # predicted typo value.
    row_to_value: dict[int, str]
    # Per-row dict of every column's value -- used to populate
    # ``full_row_values`` on the representative issue row.
    full_rows: dict[int, dict[str, str]]


@dataclass
class ColumnResult:
    task: ColumnTask
    raw_text: str | None
    error: str | None
    typos: list[dict[str, Any]] = field(default_factory=list)
    latency_s: float = 0.0


def _build_column_tasks(csv_path: Path) -> list[ColumnTask]:
    df = _load_csv(csv_path)
    if df is None or df.empty:
        return []
    df = df.fillna("")
    tasks: list[ColumnTask] = []
    full_rows: dict[int, dict[str, str]] = {
        int(i): {str(c): str(row[c]) for c in df.columns}
        for i, row in df.iterrows()
    }
    for col_idx, col in enumerate(df.columns):
        series = df[col].astype(str)
        non_empty = series[series != ""]
        if non_empty.empty:
            continue
        # Need at least 2 distinct values for a typo to make sense
        # (otherwise nothing to compare against).
        distinct = non_empty.value_counts()
        if len(distinct) < 2:
            continue
        values_with_counts = [
            (str(v), int(c)) for v, c in distinct.items()
        ]
        row_to_value = {int(i): str(v) for i, v in series.items() if v != ""}
        tasks.append(
            ColumnTask(
                csv_path=csv_path,
                column=str(col),
                column_idx=col_idx,
                values_with_counts=values_with_counts,
                row_to_value=row_to_value,
                full_rows=full_rows,
            )
        )
    return tasks


def _call_one(
    clients: list[AzureOpenAI],
    cursor: list[int],
    model: str,
    task: ColumnTask,
    system_instruction: str,
    text_format: dict[str, Any],
    reasoning_effort: str,
    cache_dir: Path,
    refresh: bool,
) -> ColumnResult:
    cache_path = cache_dir / f"{task.csv_path.name}__{_safe_col(task.column)}.json"
    if not refresh and cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            return ColumnResult(
                task=task,
                raw_text=cached.get("raw_text"),
                error=cached.get("error"),
                typos=cached.get("typos") or [],
                latency_s=float(cached.get("latency_s") or 0.0),
            )
        except Exception:
            pass

    user_msg = t.build_user_message(
        column=task.column,
        source=task.csv_path.name,
        values_with_counts=task.values_with_counts,
    )
    idx = cursor[0] % len(clients)
    cursor[0] += 1
    client = clients[idx]

    t0 = time.time()
    raw_text = None
    err = None
    try:
        resp = client.responses.create(
            model=model,
            instructions=system_instruction,
            input=user_msg,
            text={"format": text_format},
            reasoning={"effort": reasoning_effort},
        )
        raw_text = _extract_text(resp)
    except Exception as exc:  # noqa: BLE001
        err = f"{type(exc).__name__}: {exc}"
    latency = time.time() - t0

    typos: list[dict[str, Any]] = []
    parsed = _parse_json_obj(raw_text) if raw_text else None
    if isinstance(parsed, dict):
        for entry in parsed.get("typos", []) or []:
            if not isinstance(entry, dict):
                continue
            val = entry.get("value")
            corr = entry.get("correction")
            if not isinstance(val, str) or not val:
                continue
            typos.append({
                "value": val,
                "correction": corr if isinstance(corr, str) else "",
                "flavour": entry.get("flavour") or "spelling",
                "rationale": entry.get("rationale") or "",
            })

    result = ColumnResult(
        task=task, raw_text=raw_text, error=err,
        typos=typos, latency_s=latency,
    )
    try:
        cache_path.write_text(json.dumps({
            "source": task.csv_path.name,
            "column": task.column,
            "raw_text": raw_text,
            "error": err,
            "typos": typos,
            "latency_s": latency,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as exc:
        logger.warning("cache write failed for %s: %s", cache_path, exc)
    return result


def _safe_col(col: str) -> str:
    """Filesystem-safe column name for cache filenames."""
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", col)
    return safe[:120] or "col"


# ---------------------------------------------------------------------------
# Detection-run record assembly
# ---------------------------------------------------------------------------


def _build_records(
    results: Iterable[ColumnResult],
    model: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    issues: list[dict[str, Any]] = []
    clusters: list[dict[str, Any]] = []
    report_index = 0
    for r in results:
        # Each (source, column) LLM call is one "report" - assign a
        # global report index even when no typos came back so the
        # numbering is reproducible across runs.
        rid = report_index
        report_index += 1
        if not r.typos:
            continue
        task = r.task
        source = task.csv_path.name
        for i_idx, entry in enumerate(r.typos):
            val = entry["value"]
            affected = sorted(
                row for row, v in task.row_to_value.items() if v == val
            )
            if not affected:
                # Model invented a value that isn't in the column.
                # Skip rather than emit a phantom finding.
                continue
            rep_row = affected[0]
            issue_id = f"r{rid:04d}-i{i_idx:04d}"
            explanation = (
                f"typo:{entry['flavour']} - {entry['rationale']}".strip()
            )
            issue = {
                "id": issue_id,
                "report_index": rid,
                "chunk_start_row": 0,
                "chunk_end_row": len(task.row_to_value),
                "source": source,
                "model": model,
                "prompt": "error_types.typo",
                "error_type": "typo",
                "error_type_name": "Typo",
                "error_type_category": "Single-Column / Typo",
                "row": rep_row,
                "columns": [task.column],
                "affected_rows": affected,
                "category": "single-column / typo",
                "observed": val,
                "explanation": explanation,
                "suggested_fix": entry["correction"],
                "row_values": {task.column: val},
                "affected_rows_total": len(affected),
                "full_row_values": task.full_rows.get(rep_row, {}),
            }
            issues.append(issue)

            cluster = {
                "cluster_id": (
                    f"{source}__r{rep_row:05d}-c{task.column_idx:03d}"
                ),
                "row": rep_row,
                "source": source,
                "columns": [task.column],
                "row_values": task.full_rows.get(rep_row, {}),
                "error_types": ["typo"],
                "error_type_categories": ["Single-Column / Typo"],
                "evidence_count": 1,
                "detections": [{
                    "issue_id": issue_id,
                    "error_type": "typo",
                    "error_type_name": "Typo",
                    "error_type_category": "Single-Column / Typo",
                    "observed": val,
                    "explanation": explanation,
                    "suggested_fix": entry["correction"],
                    "columns": [task.column],
                    "affected_rows": affected,
                    "affected_rows_total": len(affected),
                    "model": model,
                }],
                "synthesised": False,
                "auto_synthesised": False,
                "likely_false_positive": False,
                "primary_error_type": "typo",
                "confidence": "medium",
            }
            clusters.append(cluster)
    return issues, clusters


def _build_summary(
    csv_paths: list[Path],
    dataset_dir: Path,
    model: str,
    issues: list[dict[str, Any]],
    clusters: list[dict[str, Any]],
    n_llm_calls: int,
) -> dict[str, Any]:
    by_source: dict[str, int] = {}
    for it in issues:
        by_source[it["source"]] = by_source.get(it["source"], 0) + 1
    return {
        "chunks": len(csv_paths),
        "llm_calls": n_llm_calls,
        "chunks_with_errors": 0,
        "chunks_with_parse_errors": 0,
        "total_issues": len(issues),
        "issues_by_category": {"single-column / typo": len(issues)},
        "issues_by_error_type": {"typo": len(issues)},
        "issues_by_error_type_category": {"Single-Column / Typo": len(issues)},
        "issues_by_source": by_source,
        "clusters_by_source": {
            s: sum(1 for c in clusters if c["source"] == s)
            for s in {c["source"] for c in clusters}
        },
        "clusters_total": len(clusters),
        "clusters_multi_type": 0,
        "clusters_single_type": len(clusters),
        "clusters_synthesised": 0,
        "clusters_auto_synthesised": 0,
        "clusters_synthesis_failed": 0,
        "clusters_likely_false_positives": 0,
        "clusters_by_primary_error_type": {"typo": len(clusters)} if clusters else {},
        "clusters_by_confidence": {"medium": len(clusters)} if clusters else {},
        "source_count": len(csv_paths),
        "mode": "single_pass",
        "model": model,
        "config": {
            "mode": "single_pass",
            "model": model,
            "chunk_size": None,
            "max_chunks": None,
            "column_mode": "single",
            "active_columns": None,
            "csv_path": str(dataset_dir.resolve()),
            "csv_glob": "*.csv",
            "csv_paths": [str(p.resolve()) for p in csv_paths],
            "source_count": len(csv_paths),
            "normalize_input": False,
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default=DEFAULT_DATASET, type=Path)
    p.add_argument(
        "--out-root", default="outputs", type=Path,
        help="Parent directory; a timestamped sub-dir is created.",
    )
    p.add_argument(
        "--tag", default="curated_bing21_100",
        help="Suffix appended to the run id (after the model name).",
    )
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument(
        "--reasoning-effort", default=DEFAULT_REASONING_EFFORT,
        choices=["none", "minimal", "low", "medium", "high", "xhigh"],
    )
    p.add_argument("--workers", type=int, default=8)
    p.add_argument(
        "--limit", type=int, default=None,
        help="Process at most N CSV files (for smoke runs).",
    )
    p.add_argument(
        "--limit-columns", type=int, default=None,
        help="Process at most N columns total (for smoke runs).",
    )
    p.add_argument(
        "--endpoints", default=",".join(DEFAULT_ENDPOINTS),
        help="Comma-separated Azure OpenAI endpoints carrying the model.",
    )
    p.add_argument(
        "--api-version", default=DEFAULT_API_VERSION,
    )
    p.add_argument(
        "--refresh", action="store_true",
        help="Ignore cached raw_responses/* and recall the LLM.",
    )
    p.add_argument(
        "--system-prompt-file", type=Path, default=None,
        help=(
            "Path to a plain-text file whose contents replace the "
            "default v5 rulebook system instruction. The user-message "
            "format and JSON response schema are unchanged so any "
            "rulebook-style prompt is a drop-in swap. Default: the "
            "v5 rulebook bundled in RFT/typo_v5/_task.py."
        ),
    )
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    dataset_dir = (
        args.dataset if args.dataset.is_absolute()
        else _PROJECT_ROOT / args.dataset
    )
    if not dataset_dir.is_dir():
        raise SystemExit(f"dataset dir not found: {dataset_dir}")

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{ts}_{args.model}_{args.tag}"
    run_dir = (
        args.out_root if args.out_root.is_absolute()
        else _PROJECT_ROOT / args.out_root
    ) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = run_dir / "raw_responses"
    cache_dir.mkdir(exist_ok=True)
    (run_dir / "RUN_STATUS.json").write_text(json.dumps({
        "status": "running",
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }, indent=2), encoding="utf-8")

    csv_paths = _discover_csvs(dataset_dir)
    if args.limit is not None:
        csv_paths = csv_paths[:args.limit]
    logger.info("Discovered %d CSV file(s) under %s", len(csv_paths), dataset_dir)

    tasks: list[ColumnTask] = []
    for cp in csv_paths:
        tasks.extend(_build_column_tasks(cp))
    if args.limit_columns is not None:
        tasks = tasks[:args.limit_columns]
    logger.info("Built %d (source, column) task(s)", len(tasks))

    endpoints = [e.strip() for e in args.endpoints.split(",") if e.strip()]
    clients = [_make_client(e, args.api_version) for e in endpoints]
    text_format = _chat_rf_to_responses_text(t.build_response_format())
    cursor = [0]

    if args.system_prompt_file is not None:
        spf = (
            args.system_prompt_file if args.system_prompt_file.is_absolute()
            else _PROJECT_ROOT / args.system_prompt_file
        )
        if not spf.is_file():
            raise SystemExit(f"--system-prompt-file not found: {spf}")
        system_instruction = spf.read_text(encoding="utf-8")
        prompt_source = str(spf)
        logger.info(
            "Using system prompt from %s (%d chars)",
            spf, len(system_instruction),
        )
    else:
        system_instruction = t.TYPO_V5_SYSTEM_INSTRUCTION
        prompt_source = "RFT.typo_v5._task.TYPO_V5_SYSTEM_INSTRUCTION"
        logger.info(
            "Using default v5 rulebook system prompt (%d chars)",
            len(system_instruction),
        )

    results: list[ColumnResult] = []
    total_typos = 0
    n_errors = 0
    t_start = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(
                _call_one, clients, cursor, args.model, task,
                system_instruction, text_format, args.reasoning_effort,
                cache_dir, args.refresh,
            ): task
            for task in tasks
        }
        for done_idx, fut in enumerate(as_completed(futs), start=1):
            r = fut.result()
            results.append(r)
            total_typos += len(r.typos)
            if r.error:
                n_errors += 1
            if done_idx % 25 == 0 or done_idx == len(futs):
                elapsed = time.time() - t_start
                logger.info(
                    "progress %d/%d  typos=%d  errors=%d  elapsed=%.1fs",
                    done_idx, len(futs), total_typos, n_errors, elapsed,
                )

    # Sort for stable report indexing (source, column_idx).
    results.sort(key=lambda r: (r.task.csv_path.name, r.task.column_idx))

    issues, clusters = _build_records(results, args.model)
    summary = _build_summary(
        csv_paths, dataset_dir, args.model, issues, clusters,
        n_llm_calls=len(tasks),
    )
    summary["config"]["system_prompt_source"] = prompt_source
    summary["config"]["reasoning_effort"] = args.reasoning_effort

    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    (run_dir / "issues.json").write_text(
        json.dumps(issues, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    with (run_dir / "issues.jsonl").open("w", encoding="utf-8") as fh:
        for it in issues:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")
    (run_dir / "clusters.json").write_text(
        json.dumps(clusters, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    with (run_dir / "clusters.jsonl").open("w", encoding="utf-8") as fh:
        for c in clusters:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")
    (run_dir / "RUN_STATUS.json").write_text(json.dumps({
        "status": "completed",
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }, indent=2), encoding="utf-8")

    logger.info(
        "DONE: %d typo issue(s), %d cluster(s) over %d column(s) "
        "(LLM errors: %d). Run dir: %s",
        len(issues), len(clusters), len(tasks), n_errors, run_dir,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
