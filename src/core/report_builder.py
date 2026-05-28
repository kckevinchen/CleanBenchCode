"""Post-processing helpers: render multiple report formats from ChunkReports.

Produced artifacts (see :func:`build_run_artifacts` for the full set):

* ``chunk_reports.json``      - pretty indented JSON (human readable).
* ``chunk_reports.jsonl``     - one chunk report per line (one per
  (chunk, error_type) pair for the per-type detector), slim (no raw_text).
* ``issues.json`` / ``issues.jsonl`` - one *raw* detection per line / per
  entry, with chunk metadata + error_type + the source CSV cell values
  for context.
* ``clusters.json`` / ``clusters.jsonl`` - Phase-A (+ optional Phase-B)
  cluster records, one per underlying defect. This is the dashboard's
  default view.
* ``summary.json``            - aggregates by category / error type
  / columns / rows plus cluster-level counters.
* ``report.md``               - human-readable Markdown summary.
* ``report.html``             - self-contained interactive dashboard
  (charts + filterable tables, clusters-first) with issues and clusters
  embedded inline.
* ``raw_responses/chunk_NNNN__<error_type>.txt`` - raw LLM text +
  reasoning summary per (chunk, error_type), kept separately so the primary
  JSON stays easy to read.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import pandas as pd

from .aggregator import Cluster, summarise_clusters
from .columns import canonicalize_columns
from .detector import ChunkReport, summarise_reports

logger = logging.getLogger(__name__)


_SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _normalize_corrected_value(
    raw: Any, columns: Sequence[str],
) -> Optional[Dict[str, Optional[str]]]:
    """Coerce the LLM's ``corrected_value`` field into a stable shape.

    The prompt asks for a ``{column: literal}`` mapping (or ``null``)
    but real LLM output sometimes wanders: a bare string for a
    single-column finding, a list aligned with ``columns``, sentinel
    strings like ``"null"``, etc. Normalise to either
    ``{column: literal_or_None, ...}`` (one key per emitted column,
    extras dropped) or ``None`` when no usable literal was supplied.
    Pure row-level findings (``columns == []``) always return
    ``None``.
    """
    cols = [c for c in (columns or []) if isinstance(c, str)]
    if not cols:
        return None
    if raw in (None, "", "null", "None"):
        return None

    out: Dict[str, Optional[str]] = {c: None for c in cols}
    if isinstance(raw, Mapping):
        for col in cols:
            val = raw.get(col)
            if val in (None, "", "null", "None"):
                out[col] = None
            else:
                s = str(val).strip()
                out[col] = s or None
    elif isinstance(raw, (list, tuple)):
        for col, val in zip(cols, raw):
            if val in (None, "", "null", "None"):
                out[col] = None
            else:
                s = str(val).strip()
                out[col] = s or None
    else:
        # Bare scalar (string / number): apply to first column only -
        # only safe for single-column findings, but harmless for the
        # multi-column case (later columns stay None).
        s = str(raw).strip()
        if s and s.lower() != "null":
            out[cols[0]] = s

    if not any(v for v in out.values()):
        return None
    return out


# ---------------------------------------------------------------------------
# Issue flattening
# ---------------------------------------------------------------------------


def _cell_to_text(value: Any, max_cell_chars: int) -> str:
    """Coerce ``value`` to a display string.

    Handles the awkward case where ``series[col]`` returns a Series
    instead of a scalar (happens when the source frame has duplicate
    column labels - a possibility even after whitespace normalisation).
    For a Series we pick the first non-null element and truncate. This
    keeps the helper total: it never raises on malformed input.
    """
    if isinstance(value, pd.Series):
        if value.empty:
            value = ""
        else:
            non_na = value.dropna()
            value = non_na.iloc[0] if len(non_na) else ""
    try:
        text = "" if pd.isna(value) else str(value)
    except (TypeError, ValueError):
        text = str(value) if value is not None else ""
    if len(text) > max_cell_chars:
        text = text[: max_cell_chars - 1] + "\u2026"
    return text


def _row_series(
    frame: Optional[pd.DataFrame], row_index: Any
) -> Optional[pd.Series]:
    """Look up a row by its INDEX label, falling back to positional lookup.

    The LLM is shown ``__row__`` values from ``chunk.frame.index`` (the
    original CSV row numbers), which survive perturbations such as row
    shuffling. Looking those numbers up via ``iloc`` would be wrong on a
    shuffled frame: position 42 in a shuffled frame is NOT the row that
    was originally row 42. Use ``.loc`` first, and only fall back to
    positional lookup when the label is missing AND the row index is
    within ``range(len(frame))`` (covers older callers that pass tiny
    frames with a default RangeIndex where label == position).
    """
    if frame is None or row_index is None:
        return None
    try:
        if row_index in frame.index:
            series = frame.loc[row_index]
            # Duplicate index labels would yield a DataFrame; pick first.
            if isinstance(series, pd.DataFrame):
                if series.empty:
                    return None
                series = series.iloc[0]
            return series
        if isinstance(row_index, int) and 0 <= row_index < len(frame):
            return frame.iloc[row_index]
    except Exception as exc:
        logger.debug(
            "row lookup failed for row=%r in frame of len %d: %s",
            row_index,
            0 if frame is None else len(frame),
            exc,
        )
    return None


def _issue_row_context(
    frame: Optional[pd.DataFrame],
    row_index: int,
    columns: Sequence[str],
    max_cell_chars: int = 240,
) -> Dict[str, str]:
    """Return a {column: value} dict for the referenced source row."""
    series = _row_series(frame, row_index)
    if series is None:
        return {}
    out: Dict[str, str] = {}
    for col in columns or []:
        if col in series.index:
            out[col] = _cell_to_text(series[col], max_cell_chars)
    return out


def _full_row_values(
    frame: Optional[pd.DataFrame],
    row_index: int,
    max_cell_chars: int = 240,
) -> Dict[str, str]:
    """Return ALL column values for the referenced source row (for Phase B)."""
    series = _row_series(frame, row_index)
    if series is None:
        return {}
    out: Dict[str, str] = {}
    # Iterate unique labels so duplicate columns collapse to one entry
    # (we pick the first non-null value via `_cell_to_text`).
    seen: set = set()
    for col in series.index:
        key = str(col)
        if key in seen:
            continue
        seen.add(key)
        out[key] = _cell_to_text(series[col], max_cell_chars)
    return out


def flatten_issues(
    reports: Iterable[ChunkReport],
    source_frames: Optional[Mapping[str, pd.DataFrame]] = None,
    *,
    include_full_row: bool = False,
) -> List[Dict[str, Any]]:
    """Flatten per-report issue lists into a single list of enriched records.

    ``source_frames`` is a mapping from CSV basename (e.g. ``"sample_100.csv"``)
    to its loaded DataFrame, so issues coming from different source files
    look up their row context in the right frame. For backward
    compatibility a single :class:`pandas.DataFrame` is also accepted and
    used for every report.

    When ``include_full_row`` is true, each record also gets an extra
    ``full_row_values`` dict holding every column of the referenced source
    row. This is what the Phase-B cluster synthesiser consumes (one prompt
    per cluster, so the full row is needed).
    """
    if isinstance(source_frames, pd.DataFrame):
        # Backward compat: a single frame applies to every report.
        single_frame = source_frames
        frame_lookup: Mapping[str, pd.DataFrame] = {}
    else:
        single_frame = None
        frame_lookup = source_frames or {}

    def _frame_for(source_name: str) -> Optional[pd.DataFrame]:
        if single_frame is not None:
            return single_frame
        return frame_lookup.get(source_name)

    flat: List[Dict[str, Any]] = []
    # Aggregate quality counters so the demo runner can surface a
    # one-line summary at the end ("LLM emitted 1342 issues; 17
    # referenced row indices that don't exist in the source frame").
    _empty_row_context = 0
    _total_dropped_oob = 0

    for report_idx, report in enumerate(reports):
        source_name = Path(report.source).name
        frame = _frame_for(source_name)
        for issue_idx, issue in enumerate(report.issues):
            row = issue.get("row")
            columns = issue.get("columns") or []
            if not isinstance(columns, list):
                columns = [str(columns)]

            # Canonicalize columns against the actual frame headers.
            # The LLM occasionally returns shortened / whitespace-mangled
            # versions of multi-line headers (e.g. ``"District"`` for
            # ``"District\nCode"``); without this resolution two issues
            # flagging the SAME cell can carry different column strings,
            # which prevents the aggregator's intersection-based
            # clustering from merging them and produces apparent
            # "duplicate clusters on the same cell" in the labeling UI.
            if frame is not None and columns:
                canonical = canonicalize_columns(columns, list(frame.columns))
                if canonical:
                    columns = canonical

            # Prefer the report's stamped metadata; fall back to the raw
            # issue payload, then to "unknown".
            error_type = (
                issue.get("error_type") or report.error_type or "unknown"
            )
            error_type_name = (
                issue.get("error_type_name") or report.error_type_name or error_type
            )
            error_type_category = (
                issue.get("error_type_category")
                or report.error_type_category
                or str(issue.get("category", "unknown"))
            )

            row_values = (
                _issue_row_context(frame, row, columns)
                if isinstance(row, int)
                else {}
            )
            if isinstance(row, int) and frame is not None and not row_values:
                # Either the LLM hallucinated a row number, or none of the
                # requested ``columns`` exist in the frame. Both are worth
                # surfacing once at debug level so they can be diagnosed
                # against the run's chunk_reports.jsonl.
                _empty_row_context += 1
                logger.debug(
                    "row context empty: source=%s row=%s columns=%s "
                    "frame_index_in=%s",
                    source_name,
                    row,
                    columns,
                    row in frame.index,
                )
            # Optional group-level listing of all rows sharing this defect
            # (used by inconsistency in the simple taxonomy, where the
            # text / pattern sub-patterns naturally produce group-level
            # findings). Normalise to a list of ints and drop the
            # ones that don't reference an actual row in the source frame
            # (defends against LLM hallucinations like "row 9999" in a
            # 100-row CSV - those would otherwise propagate into clusters
            # and the dashboard).
            affected_rows_raw = issue.get("affected_rows") or []
            if not isinstance(affected_rows_raw, list):
                affected_rows_raw = [affected_rows_raw]
            affected_rows: List[int] = []
            dropped_oob = 0
            for r in affected_rows_raw:
                try:
                    r_int = int(r)
                except (TypeError, ValueError):
                    continue
                if frame is not None:
                    if r_int in frame.index or (0 <= r_int < len(frame)):
                        affected_rows.append(r_int)
                    else:
                        dropped_oob += 1
                else:
                    affected_rows.append(r_int)
            if dropped_oob:
                _total_dropped_oob += dropped_oob
                logger.debug(
                    "dropped %d out-of-bounds affected_rows entries: "
                    "source=%s row=%s",
                    dropped_oob,
                    source_name,
                    row,
                )

            record: Dict[str, Any] = {
                "id": f"r{report_idx:04d}-i{issue_idx:04d}",
                "report_index": report_idx,
                "chunk_start_row": report.start_row,
                "chunk_end_row": report.end_row,
                "source": source_name,
                "model": report.model,
                "prompt": report.prompt_name,
                "error_type": error_type,
                "error_type_name": error_type_name,
                "error_type_category": error_type_category,
                "row": row,
                "columns": columns,
                "affected_rows": affected_rows,
                "category": str(issue.get("category", error_type_category)).lower(),
                "observed": issue.get("observed", ""),
                "explanation": issue.get("explanation", ""),
                "suggested_fix": issue.get("suggested_fix"),
                "corrected_value": _normalize_corrected_value(
                    issue.get("corrected_value"), columns,
                ),
                "row_values": row_values,
            }
            try:
                art = issue.get("affected_rows_total")
                if art is not None:
                    record["affected_rows_total"] = int(art)
            except (TypeError, ValueError):
                pass
            if include_full_row and isinstance(row, int):
                record["full_row_values"] = _full_row_values(frame, row)
            flat.append(record)
    if _empty_row_context or _total_dropped_oob:
        logger.warning(
            "report_builder data-quality issues: %d issues had empty "
            "row context (LLM referenced unknown row/columns); %d "
            "out-of-bounds affected_rows entries dropped. Run with "
            "log level DEBUG for per-issue detail.",
            _empty_row_context, _total_dropped_oob,
        )
    return flat


# ---------------------------------------------------------------------------
# File writers
# ---------------------------------------------------------------------------


def write_json(path: Path, payload: Any, *, indent: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=indent, ensure_ascii=False)


def write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_reports_pretty(reports: Sequence[ChunkReport], path: Path) -> None:
    """Write a pretty-printed JSON array of chunk reports (no raw_text)."""
    payload = [r.as_dict(include_raw=False) for r in reports]
    write_json(path, payload)


def _safe_slug(value: str) -> str:
    cleaned = _SAFE_FILENAME_RE.sub("_", value).strip("_")
    return cleaned or "report"


def write_raw_response_sidecars(reports: Sequence[ChunkReport], directory: Path) -> int:
    """Persist raw LLM text and reasoning summaries as individual .txt files.

    Files are named ``NNNN__<error_type>.txt`` when an error type is present,
    otherwise ``NNNN.txt``. The numeric prefix preserves emit order.
    """
    directory.mkdir(parents=True, exist_ok=True)
    written = 0
    for idx, report in enumerate(reports):
        slug_parts: List[str] = [f"{idx:04d}"]
        if report.error_type:
            slug_parts.append(_safe_slug(report.error_type))
        filename = "__".join(slug_parts) + ".txt"
        path = directory / filename
        blocks: List[str] = [
            f"# report_index: {idx}",
            f"# source: {report.source}",
            f"# rows: [{report.start_row}, {report.end_row})",
            f"# model: {report.model}",
            f"# prompt: {report.prompt_name}",
            f"# error_type: {report.error_type}"
            + (f" ({report.error_type_name})" if report.error_type_name else ""),
            f"# error_type_category: {report.error_type_category}",
            f"# endpoint: {report.endpoint}",
            f"# attempts: {report.attempts}",
            f"# error: {report.error}",
            f"# parse_error: {report.parse_error}",
            "",
            "=== Reasoning Summary ===",
            report.reasoning_summary or "(none)",
            "",
            "=== Raw LLM Response ===",
            report.raw_text or "(empty)",
        ]
        path.write_text("\n".join(blocks), encoding="utf-8")
        written += 1
    return written


# ---------------------------------------------------------------------------
# Summary / aggregates
# ---------------------------------------------------------------------------


def _top_clusters(clusters: Sequence[Cluster], limit: int = 20) -> List[Dict[str, Any]]:
    def _key(c: Cluster) -> tuple:
        return (
            -c.evidence_count,
            c.row if isinstance(c.row, int) else 1 << 30,
        )

    picked = sorted(clusters, key=_key)[:limit]
    out: List[Dict[str, Any]] = []
    for c in picked:
        syn = c.synthesis
        primary = (
            syn.primary_error_type
            if syn and syn.primary_error_type
            else (c.error_types[0] if c.error_types else "")
        )
        out.append(
            {
                "cluster_id": c.cluster_id,
                "row": c.row,
                "evidence_count": c.evidence_count,
                "primary_error_type": primary,
                "error_types": list(c.error_types),
                "columns": list(c.columns),
                "is_likely_false_positive": bool(
                    syn.is_likely_false_positive if syn else False
                ),
            }
        )
    return out


def build_summary(
    reports: Sequence[ChunkReport],
    flat_issues: Sequence[Mapping[str, Any]],
    clusters: Sequence[Cluster],
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    summary = summarise_reports(reports)

    column_counter: Dict[str, int] = {}
    row_counter: Dict[int, int] = {}
    error_type_counter: Dict[str, int] = {}
    issues_by_source: Dict[str, int] = {}
    for issue in flat_issues:
        for col in issue.get("columns", []) or []:
            column_counter[col] = column_counter.get(col, 0) + 1
        row = issue.get("row")
        if isinstance(row, int):
            row_counter[row] = row_counter.get(row, 0) + 1
        etype = str(issue.get("error_type", "unknown"))
        error_type_counter[etype] = error_type_counter.get(etype, 0) + 1
        src = str(issue.get("source", "") or "")
        if src:
            issues_by_source[src] = issues_by_source.get(src, 0) + 1

    top_columns = sorted(
        column_counter.items(), key=lambda kv: (-kv[1], kv[0])
    )[:15]
    top_rows = sorted(row_counter.items(), key=lambda kv: (-kv[1], kv[0]))[:15]
    top_error_types = sorted(
        error_type_counter.items(), key=lambda kv: (-kv[1], kv[0])
    )[:20]

    summary["top_columns"] = [{"column": c, "count": n} for c, n in top_columns]
    summary["top_rows"] = [{"row": r, "count": n} for r, n in top_rows]
    summary["top_error_types"] = [
        {"error_type": k, "count": n} for k, n in top_error_types
    ]

    # Cluster-level counters (Phase A + Phase B)
    summary.update(summarise_clusters(clusters))
    summary["top_clusters"] = _top_clusters(clusters)

    # Per-source breakdown for multi-CSV runs (also fine for single-CSV).
    clusters_by_source: Dict[str, int] = {}
    for c in clusters:
        src = c.source or ""
        if src:
            clusters_by_source[src] = clusters_by_source.get(src, 0) + 1
    summary["issues_by_source"] = dict(
        sorted(issues_by_source.items(), key=lambda kv: (-kv[1], kv[0]))
    )
    summary["clusters_by_source"] = dict(
        sorted(clusters_by_source.items(), key=lambda kv: (-kv[1], kv[0]))
    )
    summary["source_count"] = len(
        set(issues_by_source) | set(clusters_by_source)
    )

    summary["config"] = dict(config)
    return summary


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def _md_counter_table(title: str, counter: Mapping[str, int]) -> str:
    if not counter:
        return f"### {title}\n\n_(none)_\n"
    lines = [f"### {title}", "", "| Key | Count |", "| --- | ---: |"]
    for key, count in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"| {key} | {count} |")
    return "\n".join(lines) + "\n"


def _md_escape(value: Any) -> str:
    return str(value or "").replace("|", "\\|").replace("\n", " ")


def render_markdown(
    flat_issues: Sequence[Mapping[str, Any]],
    clusters: Sequence[Cluster],
    summary: Mapping[str, Any],
    run_metadata: Mapping[str, Any],
) -> str:
    lines: List[str] = []
    lines.append("# Data Cleaning Error Report")
    lines.append("")
    lines.append(f"- **Run:** `{run_metadata.get('run_id', '?')}`")
    lines.append(f"- **Model:** `{run_metadata.get('model', '?')}`")
    lines.append(
        f"- **Error types detected:** "
        f"{run_metadata.get('error_types_count', len(summary.get('issues_by_error_type', {})))}"
    )
    sources = run_metadata.get("sources") or []
    if sources and len(sources) > 1:
        lines.append(f"- **Sources:** {len(sources)} CSV file(s)")
        for s in sources[:20]:
            lines.append(f"  - `{s}`")
        if len(sources) > 20:
            lines.append(f"  - ... and {len(sources) - 20} more")
    else:
        lines.append(f"- **Source:** `{run_metadata.get('csv_path', '?')}`")
    lines.append(
        f"- **Chunks analysed:** {summary.get('chunks', 0)} "
        f"(LLM calls: {summary.get('llm_calls', 0)}, "
        f"errors: {summary.get('chunks_with_errors', 0)}, "
        f"parse errors: {summary.get('chunks_with_parse_errors', 0)})"
    )
    lines.append(f"- **Total detections:** {summary.get('total_issues', 0)}")
    lines.append(
        f"- **Clusters:** {summary.get('clusters_total', 0)} "
        f"(multi-type: {summary.get('clusters_multi_type', 0)}, "
        f"single-type: {summary.get('clusters_single_type', 0)}, "
        f"likely false positive: "
        f"{summary.get('clusters_likely_false_positives', 0)})"
    )
    lines.append("")

    clusters_by_source = summary.get("clusters_by_source") or {}
    if len(clusters_by_source) > 1:
        lines.append(_md_counter_table("Clusters by source", clusters_by_source))

    lines.append(
        _md_counter_table(
            "Detections by error type", summary.get("issues_by_error_type", {})
        )
    )
    lines.append(
        _md_counter_table(
            "Detections by error-type category",
            summary.get("issues_by_error_type_category", {}),
        )
    )
    lines.append(
        _md_counter_table(
            "Clusters by primary error type",
            summary.get("clusters_by_primary_error_type", {}),
        )
    )

    top_columns = summary.get("top_columns") or []
    if top_columns:
        lines.append("### Top columns by detection count")
        lines.append("")
        lines.append("| Column | Count |")
        lines.append("| --- | ---: |")
        for row in top_columns:
            lines.append(f"| {row['column']} | {row['count']} |")
        lines.append("")

    # Clusters section -------------------------------------------------
    lines.append("## Clusters (canonical issues)")
    lines.append("")
    if not clusters:
        lines.append("_(no clusters)_")
        lines.append("")
    else:
        ranked = sorted(
            clusters,
            key=lambda c: (
                -c.evidence_count,
                c.row if isinstance(c.row, int) else 1 << 30,
            ),
        )
        lines.append(
            "| Source | Row | Primary | #detections | All types | Explanation | Fix |"
        )
        lines.append("| --- | ---: | --- | ---: | --- | --- | --- |")
        for c in ranked:
            syn = c.synthesis
            primary = (
                syn.primary_error_type
                if syn and syn.primary_error_type
                else (c.error_types[0] if c.error_types else "")
            )
            explanation = (
                syn.canonical_explanation
                if syn and syn.canonical_explanation
                else (c.detections[0].explanation if c.detections else "")
            )
            fix = (
                syn.canonical_fix
                if syn and syn.canonical_fix
                else (c.detections[0].suggested_fix if c.detections else "")
            )
            fp_mark = " \u26a0 FP" if syn and syn.is_likely_false_positive else ""
            lines.append(
                f"| {_md_escape(c.source)} "
                f"| {c.row if c.row is not None else ''} "
                f"| {_md_escape(primary)}{fp_mark} "
                f"| {c.evidence_count} "
                f"| {_md_escape(', '.join(c.error_types))} "
                f"| {_md_escape(explanation)} "
                f"| {_md_escape(fix)} |"
            )
        lines.append("")

    # Raw detections section ------------------------------------------
    lines.append("## Raw detections")
    lines.append("")
    if not flat_issues:
        lines.append("_(no detections flagged)_")
        return "\n".join(lines)

    sorted_issues = sorted(
        flat_issues,
        key=lambda i: (
            i.get("row") if isinstance(i.get("row"), int) else 1 << 30,
        ),
    )

    lines.append(
        "| Source | Row | Error type | Columns | Observed | Explanation | Suggested fix |"
    )
    lines.append("| --- | ---: | --- | --- | --- | --- | --- |")
    for issue in sorted_issues:
        columns = ", ".join(issue.get("columns") or [])
        observed = _md_escape(issue.get("observed", ""))
        explanation = _md_escape(issue.get("explanation", ""))
        fix = issue.get("suggested_fix")
        fix_str = "" if fix is None else _md_escape(fix)
        etype_name = issue.get("error_type_name") or issue.get("error_type", "")
        lines.append(
            f"| {_md_escape(issue.get('source', ''))} "
            f"| {issue.get('row', '')} | "
            f"{_md_escape(etype_name)} | {columns} | {observed} | {explanation} | {fix_str} |"
        )

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# HTML dashboard
# ---------------------------------------------------------------------------


HTML_DATA_PLACEHOLDER = "/*__REPORT_DATA_JSON__*/"

# Source-data payload caps. A run with 20 CSVs x 100 rows x 10 cols is
# ~20k cells; we keep each cell short so the embedded JSON stays well
# under a megabyte even for large curated sets.
_SOURCE_DATA_MAX_CELL = 240
_SOURCE_DATA_MAX_ROWS_PER_SOURCE = 5000


def _build_source_data(
    source_frames: Optional[Mapping[str, pd.DataFrame]],
) -> Dict[str, Any]:
    """Serialise the analysed CSVs so the HTML dashboard can render the
    raw rows alongside the detections (used by the "Source data" view for
    false-negative review).

    The payload is:

    .. code-block:: python

        {
            "a.csv": {
                "columns": ["name", "age"],
                "rows":    [{"row": 0, "values": ["Alice", "30"]}, ...],
                "total_rows": 103,
                "truncated": false
            },
            ...
        }

    ``values`` is a positional list (same order as ``columns``) of short
    display strings - that keeps the JSON compact vs. a dict per row.
    """
    if not source_frames:
        return {}

    out: Dict[str, Any] = {}
    for name, frame in source_frames.items():
        if frame is None:
            continue
        columns = [str(c) for c in frame.columns]
        total = len(frame)
        limit = min(total, _SOURCE_DATA_MAX_ROWS_PER_SOURCE)
        rows: List[Dict[str, Any]] = []
        for i in range(limit):
            series = frame.iloc[i]
            values: List[str] = []
            for col in frame.columns:
                v = series[col]
                if pd.isna(v):
                    values.append("")
                else:
                    text = str(v)
                    if len(text) > _SOURCE_DATA_MAX_CELL:
                        text = text[: _SOURCE_DATA_MAX_CELL - 1] + "\u2026"
                    values.append(text)
            rows.append({"row": i, "values": values})
        out[str(name)] = {
            "columns": columns,
            "rows": rows,
            "total_rows": total,
            "truncated": total > limit,
        }
    return out


def render_html(
    template_path: Path,
    flat_issues: Sequence[Mapping[str, Any]],
    clusters: Sequence[Cluster],
    summary: Mapping[str, Any],
    run_metadata: Mapping[str, Any],
    source_frames: Optional[Mapping[str, pd.DataFrame]] = None,
    good_turing: Optional[Mapping[str, Any]] = None,
) -> str:
    """Render the self-contained HTML dashboard by inlining the data.

    Parameters
    ----------
    source_frames
        Optional mapping from CSV basename to its loaded DataFrame. When
        provided, the raw rows are embedded in the HTML payload so the
        dashboard can offer a "Source data" view for manually checking
        false negatives.
    good_turing
        Optional adaptive-mode payload (see the ``gt_dashboard`` dict
        assembled in ``experiments.run_cleaning_detection.main``). When
        provided, the dashboard shows a dedicated Good-Turing tab with
        the estimator progression, per-type / per-source false-negative
        estimates, and the seed-vs-variant prompt diff for every pass.
    """
    if not template_path.is_file():
        raise FileNotFoundError(f"HTML template not found: {template_path}")
    template = template_path.read_text(encoding="utf-8")
    if HTML_DATA_PLACEHOLDER not in template:
        raise ValueError(
            f"HTML template is missing the data placeholder "
            f"'{HTML_DATA_PLACEHOLDER}': {template_path}"
        )
    payload = {
        "run_metadata": dict(run_metadata),
        "summary": dict(summary),
        "issues": list(flat_issues),
        "clusters": [c.as_dict() for c in clusters],
        "source_data": _build_source_data(source_frames),
        "good_turing": dict(good_turing) if good_turing else None,
    }
    encoded = json.dumps(payload, ensure_ascii=False)
    # Guard against an accidental </script> inside the JSON payload.
    encoded = encoded.replace("</", "<\\/")
    return template.replace(HTML_DATA_PLACEHOLDER, encoded)


# ---------------------------------------------------------------------------
# Top-level convenience
# ---------------------------------------------------------------------------


def build_run_artifacts(
    reports: Sequence[ChunkReport],
    run_dir: Path,
    *,
    flat_issues: Sequence[Mapping[str, Any]],
    clusters: Sequence[Cluster],
    config: Mapping[str, Any],
    run_metadata: Mapping[str, Any],
    html_template_path: Optional[Path] = None,
    source_frames: Optional[Mapping[str, pd.DataFrame]] = None,
    good_turing: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Path]:
    """Write the full set of artifacts for a run and return their paths.

    The caller is responsible for producing ``flat_issues`` (via
    :func:`flatten_issues`) and ``clusters`` (via
    :func:`src.aggregator.cluster_issues` + optional Phase-B synthesis) so
    that this function stays purely a serialization layer - it does not
    itself depend on the LLM client.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    summary = build_summary(reports, flat_issues, clusters, config)

    paths: Dict[str, Path] = {}

    from .detector import write_reports_jsonl  # local import to avoid cycles

    paths["chunk_reports_jsonl"] = run_dir / "chunk_reports.jsonl"
    write_reports_jsonl(reports, paths["chunk_reports_jsonl"], include_raw=False)

    paths["chunk_reports_json"] = run_dir / "chunk_reports.json"
    write_reports_pretty(reports, paths["chunk_reports_json"])

    paths["issues_jsonl"] = run_dir / "issues.jsonl"
    write_jsonl(paths["issues_jsonl"], flat_issues)

    paths["issues_json"] = run_dir / "issues.json"
    write_json(paths["issues_json"], list(flat_issues))

    cluster_dicts = [c.as_dict() for c in clusters]
    paths["clusters_jsonl"] = run_dir / "clusters.jsonl"
    write_jsonl(paths["clusters_jsonl"], cluster_dicts)

    paths["clusters_json"] = run_dir / "clusters.json"
    write_json(paths["clusters_json"], cluster_dicts)

    paths["summary_json"] = run_dir / "summary.json"
    write_json(paths["summary_json"], summary)

    paths["report_md"] = run_dir / "report.md"
    paths["report_md"].write_text(
        render_markdown(flat_issues, clusters, summary, run_metadata),
        encoding="utf-8",
    )

    paths["raw_responses_dir"] = run_dir / "raw_responses"
    write_raw_response_sidecars(reports, paths["raw_responses_dir"])

    if html_template_path is not None:
        paths["report_html"] = run_dir / "report.html"
        html = render_html(
            html_template_path,
            flat_issues,
            clusters,
            summary,
            run_metadata,
            source_frames=source_frames,
            good_turing=good_turing,
        )
        paths["report_html"].write_text(html, encoding="utf-8")

    logger.info("Wrote %d artifacts under %s", len(paths), run_dir)
    return paths
