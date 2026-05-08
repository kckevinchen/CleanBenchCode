"""Column-wide web verification for under-explored columns.

The standard verifier (:mod:`experiments.run_web_verify`) only
inspects cells the LLM detector already flagged. When a column is
*budget-capped* (the adaptive run hit ``k_max`` while it was still
finding new defects), we suspect the detector missed cells. This
script audits **every distinct non-empty value** in a (source,
column) - including never-flagged cells - against the web, so the
operator can find missed errors.

Usage::

    # one column
    python -m experiments.run_web_verify_column \\
        --run-id 20260428T120000Z_gpt-5.2 \\
        --source contracts.csv --column "Org Name"

    # bulk: every (source, column) whose adaptive arms are
    # exhausted (``budget-capped`` per the labeling-effort artifact)
    python -m experiments.run_web_verify_column \\
        --run-id 20260428T120000Z_gpt-5.2 --all-budget-capped

The output lives at::

    outputs/<run_id>/column_verifications/<source_slug>__<column_slug>.jsonl

with one JSON record per distinct value. Each record carries the
agent's verdict + evidence URLs PLUS ``was_flagged_by_llm`` and the
covering ``cluster_ids`` so the labeling UI can highlight cells the
detector missed (false negatives).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.data_loader import load_csv  # noqa: E402, F401  (re-exported via run_web_verify)
from src.verifier import (  # noqa: E402
    AzureAgentBackend,
    GateConfig,
    Verifier,
    VerifierConfig,
)
from src.verifier.web_cache import WebCache  # noqa: E402

# Reuse the lookup builders from the standard verify script so we
# stay 1:1 on column resolution / NaN handling / dataframe access.
from experiments.run_web_verify import (  # noqa: E402
    _build_lookups,
    _frame_loader,
)

logger = logging.getLogger("run_web_verify_column")


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------


@dataclass
class ColumnAuditRecord:
    """One verification record for the column-audit artifact.

    JSONL one line per (source, column, distinct_value). When the same
    value appears in multiple rows we still issue ONE web call
    (cheaper, equivalent fidelity) and the ``rows`` field carries
    every row index it appears in.
    """

    source: str
    column: str
    value: str
    rows: List[int]
    n_rows: int = 0
    was_flagged_by_llm: bool = False
    cluster_ids: List[str] = field(default_factory=list)
    verdict: str = "ambiguous"
    canonical_value: Optional[str] = None
    confidence: float = 0.0
    evidence: List[Dict[str, str]] = field(default_factory=list)
    reasoning: str = ""
    queries: List[str] = field(default_factory=list)
    n_results: int = 0
    n_unique_urls: int = 0
    error: Optional[str] = None
    parse_error: Optional[str] = None
    llm_attempts: int = 0
    llm_endpoint: str = ""
    entity_kind: str = "other"
    ts: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_SLUG_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _slug(s: str, *, max_len: int = 60) -> str:
    """File-safe slug for use in artifact path components."""
    if not s:
        return "_"
    out = _SLUG_RE.sub("_", str(s))
    out = out.strip("_") or "_"
    if len(out) > max_len:
        # Keep both ends so distinct columns stay distinguishable.
        out = out[: max_len // 2] + "_" + out[-max_len // 2 :]
    return out


def column_artifact_path(run_dir: Path, source: str, column: str) -> Path:
    """Stable path for the column-audit JSONL of one (source, column)."""
    return (
        run_dir
        / "column_verifications"
        / f"{_slug(source)}__{_slug(column)}.jsonl"
    )


def index_artifact_path(run_dir: Path) -> Path:
    """Manifest listing every (source, column) that has been audited.

    One JSONL line per audited column with summary counts. Rebuilt on
    every run so it always reflects the on-disk state.
    """
    return run_dir / "column_verifications" / "_index.jsonl"


def progress_artifact_path(run_dir: Path) -> Path:
    """Live progress JSON for the in-flight column-audit run.

    The script rewrites this file atomically as it advances - once
    when the column list is known, every time it transitions to a
    new column, after every value-batch checkpoint, and once on
    completion. The labelling UI polls it to render a progress bar
    while a column-audit job is running.
    """
    return run_dir / "column_verifications" / "_progress.json"


class ProgressWriter:
    """Atomic JSON writer for the column-audit progress artifact.

    All fields are optional - missing keys mean "not yet known". The
    file is rewritten in full each tick (it stays small: <1 KiB).
    """

    def __init__(self, run_dir: Path) -> None:
        self.path = progress_artifact_path(run_dir)
        self._state: Dict[str, Any] = {
            "started_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": None,
            "done": False,
            "exit_code": None,
            "total_columns": 0,
            "current_column_index": 0,  # 1-based once set; 0 = not yet started
            "current_source": None,
            "current_column": None,
            "values_total": 0,
            "values_done": 0,
            "columns_done": 0,
            # Aggregate counts so the UI can show running totals
            # (FN candidates etc.) before the run finishes.
            "overall": {
                "confirmed_error": 0,
                "false_positive": 0,
                "ambiguous": 0,
                "errored": 0,
                "fn_candidates": 0,
                "total": 0,
            },
        }
        # Initial write so the file exists as soon as the script starts.
        self._flush()

    def update(self, **fields: Any) -> None:
        self._state.update(fields)
        self._flush()

    def update_overall(self, counts: Mapping[str, Any]) -> None:
        for k, v in counts.items():
            try:
                self._state["overall"][k] = self._state["overall"].get(k, 0) + int(v)
            except (TypeError, ValueError):
                continue
        self._flush()

    def finish(self, *, exit_code: int = 0) -> None:
        self._state["done"] = True
        self._state["exit_code"] = int(exit_code)
        self._state["values_total"] = self._state.get("values_total", 0)
        self._state["values_done"] = self._state.get("values_total", 0)
        self._flush()

    def _flush(self) -> None:
        self._state["updated_at"] = datetime.now(timezone.utc).isoformat()
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            with tmp.open("w", encoding="utf-8") as fh:
                json.dump(self._state, fh, ensure_ascii=False)
            os.replace(tmp, self.path)
        except OSError as exc:  # noqa: BLE001
            # Progress is best-effort: a missing-disk / permission
            # error must not crash the audit run.
            logger.debug("progress write failed: %s", exc)



def _flagged_clusters_for(
    clusters: Sequence[Mapping[str, Any]],
    source: str,
    column: str,
) -> Dict[int, List[str]]:
    """Return ``{row_idx: [cluster_id, ...]}`` for clusters whose
    primary column equals ``column`` in ``source``. Multi-column /
    multi-row clusters contribute every row in ``affected_rows`` if
    ``column`` is in their column set.
    """
    out: Dict[int, List[str]] = {}
    for c in clusters:
        if not isinstance(c, Mapping):
            continue
        if c.get("source") != source:
            continue
        cols = c.get("columns") or []
        if column not in (cols or []):
            continue
        cid = str(c.get("cluster_id") or "")
        # Primary row + every detected affected row.
        rows: List[int] = []
        primary = c.get("row")
        if isinstance(primary, int):
            rows.append(primary)
        for d in c.get("detections") or []:
            if not isinstance(d, Mapping):
                continue
            for r in d.get("affected_rows") or []:
                try:
                    rows.append(int(r))
                except (TypeError, ValueError):
                    continue
        for r in rows:
            out.setdefault(r, [])
            if cid and cid not in out[r]:
                out[r].append(cid)
    return out


def _distinct_values(
    frame: pd.DataFrame,
    column: str,
    *,
    max_value_chars: int,
    min_value_chars: int,
    max_distinct: Optional[int],
) -> List[Tuple[str, List[int]]]:
    """Return ``[(value, [row_idx, ...]), ...]`` ordered by first-seen row.

    Empty / NaN cells are skipped. Values longer than ``max_value_chars``
    or shorter than ``min_value_chars`` are skipped (they're either too
    big to web-verify cleanly or too short to disambiguate). When
    ``max_distinct`` is set, we cap to the most common N values.
    """
    if column not in frame.columns:
        return []
    series = frame[column]
    by_value: Dict[str, List[int]] = {}
    for idx, val in enumerate(series.tolist()):
        if val is None:
            continue
        try:
            if isinstance(val, float) and pd.isna(val):
                continue
        except Exception:  # noqa: BLE001
            pass
        s = str(val).strip()
        if not s:
            continue
        if len(s) < min_value_chars or len(s) > max_value_chars:
            continue
        by_value.setdefault(s, []).append(idx)
    items = list(by_value.items())
    if max_distinct is not None and len(items) > max_distinct:
        # Keep the most frequent values - they're the ones that
        # matter most for typo / coverage analysis.
        items.sort(key=lambda kv: (-len(kv[1]), kv[0]))
        items = items[:max_distinct]
    # Re-sort by first-seen row so output is reproducible per dataset.
    items.sort(key=lambda kv: kv[1][0])
    return [(v, rows) for v, rows in items]


def _budget_capped_columns(
    run_dir: Path,
) -> List[Tuple[str, str, List[str]]]:
    """Read the per-arm Good-Turing artifact and return every (source,
    column, [error_types]) where at least one arm reports
    ``status="exhausted"``.

    Returns an empty list when the artifact is missing or malformed.
    The returned ``error_types`` list contains every exhausted arm's
    ``error_type`` for that (source, column) - typically just one,
    but a column can be capped on multiple defect classes and we
    want the audit prompt to focus on all of them.
    """
    path = run_dir / "good_turing" / "per_arm_estimates.json"
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("could not read %s: %s", path, exc)
        return []
    arms = payload.get("arms") if isinstance(payload, Mapping) else None
    if not isinstance(arms, list):
        return []
    by_pair: Dict[Tuple[str, str], List[str]] = {}
    for arm in arms:
        if not isinstance(arm, Mapping):
            continue
        scope_id = arm.get("scope_id") or []
        if not isinstance(scope_id, list) or len(scope_id) < 2:
            continue
        if str(arm.get("status") or "") != "exhausted":
            continue
        et = str(arm.get("error_type") or "").strip()
        if not et:
            continue
        key = (str(scope_id[0]), str(scope_id[1]))
        bucket = by_pair.setdefault(key, [])
        if et not in bucket:
            bucket.append(et)
    out: List[Tuple[str, str, List[str]]] = [
        (s, c, ets) for (s, c), ets in by_pair.items()
    ]
    out.sort(key=lambda t: (t[0], t[1]))
    return out


def _capped_error_types_for(
    run_dir: Path, source: str, column: str,
) -> List[str]:
    """Look up the exhausted-arm error_types for ONE (source, column).

    Falls back to ``["typo"]`` when the per-arm artifact is missing
    or doesn't list an exhausted arm for this pair - mirrors the
    pre-error-type-aware default so the audit script still does
    something sensible on legacy runs / manual single-column runs.
    """
    for s, c, ets in _budget_capped_columns(run_dir):
        if s == source and c == column:
            return list(ets)
    return ["typo"]


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def write_records(path: Path, records: Sequence[ColumnAuditRecord]) -> None:
    """Atomically write one JSON record per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def load_records(path: Path) -> List[ColumnAuditRecord]:
    """Read previously-written records, or ``[]`` if missing."""
    if not path.exists():
        return []
    out: List[ColumnAuditRecord] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(d, dict):
                continue
            evidence_raw = d.get("evidence") or []
            evidence: List[Dict[str, str]] = []
            if isinstance(evidence_raw, list):
                for ev in evidence_raw:
                    if not isinstance(ev, dict):
                        continue
                    evidence.append({
                        "url": str(ev.get("url") or ""),
                        "title": str(ev.get("title") or ""),
                        "quote": str(ev.get("quote") or ""),
                    })
            rows_raw = d.get("rows") or []
            rows: List[int] = []
            if isinstance(rows_raw, list):
                for x in rows_raw:
                    try:
                        rows.append(int(x))
                    except (TypeError, ValueError):
                        continue
            out.append(ColumnAuditRecord(
                source=str(d.get("source") or ""),
                column=str(d.get("column") or ""),
                value=str(d.get("value") or ""),
                rows=rows,
                n_rows=int(d.get("n_rows") or len(rows)),
                was_flagged_by_llm=bool(d.get("was_flagged_by_llm")),
                cluster_ids=[str(x) for x in (d.get("cluster_ids") or [])],
                verdict=str(d.get("verdict") or "ambiguous"),
                canonical_value=(
                    str(d["canonical_value"])
                    if d.get("canonical_value") not in (None, "", "null")
                    else None
                ),
                confidence=float(d.get("confidence") or 0.0),
                evidence=evidence,
                reasoning=str(d.get("reasoning") or ""),
                queries=[str(q) for q in (d.get("queries") or [])],
                n_results=int(d.get("n_results") or 0),
                n_unique_urls=int(d.get("n_unique_urls") or 0),
                error=d.get("error") if d.get("error") else None,
                parse_error=d.get("parse_error") if d.get("parse_error") else None,
                llm_attempts=int(d.get("llm_attempts") or 0),
                llm_endpoint=str(d.get("llm_endpoint") or ""),
                entity_kind=str(d.get("entity_kind") or "other"),
                ts=str(d.get("ts") or ""),
            ))
    return out


def _summarise(records: Sequence[ColumnAuditRecord]) -> Dict[str, int]:
    counts = {"confirmed_error": 0, "false_positive": 0, "ambiguous": 0, "errored": 0}
    fn_candidates = 0
    for r in records:
        if r.error:
            counts["errored"] += 1
            continue
        counts[r.verdict] = counts.get(r.verdict, 0) + 1
        if r.verdict == "confirmed_error" and not r.was_flagged_by_llm:
            fn_candidates += 1
    counts["fn_candidates"] = fn_candidates
    counts["total"] = len(records)
    return counts


def _rebuild_index(run_dir: Path) -> None:
    """Walk ``column_verifications/*.jsonl`` and rewrite ``_index.jsonl``."""
    base = run_dir / "column_verifications"
    if not base.is_dir():
        return
    rows: List[Dict[str, Any]] = []
    for p in sorted(base.glob("*.jsonl")):
        if p.name.startswith("_"):
            continue
        recs = load_records(p)
        if not recs:
            continue
        # Every record in a file shares the (source, column) pair.
        first = recs[0]
        counts = _summarise(recs)
        rows.append({
            "source": first.source,
            "column": first.column,
            "path": str(p.relative_to(run_dir)),
            "counts": counts,
            "ts": max(
                (r.ts for r in recs if r.ts),
                default="",
            ),
        })
    out = base / "_index.jsonl"
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with tmp.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, out)


# ---------------------------------------------------------------------------
# Verification driver
# ---------------------------------------------------------------------------


def _synth_cluster(
    *,
    source: str,
    column: str,
    value: str,
    rows: List[int],
    cluster_ids: List[str],
    error_types: Sequence[str] = ("typo",),
) -> Dict[str, Any]:
    """Build a synthetic single-cell cluster suitable for ``Verifier.run``.

    The verifier reads ``verify_mode="column_audit"`` to swap the
    task-block framing to the neutral audit prompt. We anchor on the
    first row index so per-cell context (full row, sibling columns)
    still gets populated; we do NOT include sibling rows on the same
    value because column-audit prompts treat each distinct value as
    one decision regardless of row count.
    """
    primary_row = rows[0] if rows else None
    cid = (
        f"colaudit__{_slug(source)}__{_slug(column)}__"
        f"{_slug(value, max_len=80)}"
    )
    return {
        "cluster_id": cid,
        "source": source,
        "row": primary_row,
        "columns": [column],
        "error_types": list(error_types),
        "evidence_count": len(rows),
        "verify_mode": "column_audit",
        "_audit_value": value,  # debug only - not consumed by verifier
        "_audit_rows": list(rows),
        "_audit_cluster_ids": list(cluster_ids),
        "detections": [],
        "synthesis": {},
    }


def verify_column(
    run_dir: Path,
    source: str,
    column: str,
    *,
    agent_endpoint: Optional[str] = None,
    agent_id: Optional[str] = None,
    model: str = "gpt-5.2",
    reasoning_effort: str = "medium",
    concurrency: int = 4,
    max_distinct: Optional[int] = None,
    max_value_chars: int = 200,
    min_value_chars: int = 2,
    skip_flagged: bool = False,
    force: bool = False,
    progress: Optional[Any] = None,
    error_types: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Run the column-wide audit for one (source, column).

    Returns ``{counts, total, output_path}``. Skips values that
    already have a non-errored record on disk unless ``force`` is set.

    ``error_types`` are the detector error_type keys this column was
    budget-capped on (typically just one, e.g. ``["typo"]``). They
    are stamped onto every synthetic cluster so the verifier's
    column-audit task block can focus the agent's queries + verdict
    rubric on the SAME defect class the detector was hitting. When
    omitted, defaults to ``["typo"]`` for backwards compatibility
    with manual single-column runs.
    """
    et_focus = (
        [str(t) for t in error_types if str(t)]
        if error_types
        else _capped_error_types_for(run_dir, source, column)
    )
    summary_path = run_dir / "summary.json"
    clusters_path = run_dir / "clusters.json"
    if not clusters_path.exists():
        raise FileNotFoundError(f"clusters.json missing: {clusters_path}")
    summary: Mapping[str, Any] = {}
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8")) or {}
    csv_paths = (summary.get("config") or {}).get("csv_paths") or []
    frames = _frame_loader(csv_paths)
    if source not in frames:
        raise FileNotFoundError(
            f"source {source!r} not found in run's csv_paths "
            f"(loaded sources: {sorted(frames)})"
        )
    frame = frames[source]
    if column not in frame.columns:
        raise ValueError(
            f"column {column!r} not in source {source!r} "
            f"(columns: {list(frame.columns)})"
        )
    cell_lookup, column_samples, row_lookup = _build_lookups(frames)

    # Pre-flagged map so we can stamp ``was_flagged_by_llm`` and the
    # ``cluster_ids`` per-row.
    clusters = json.loads(clusters_path.read_text(encoding="utf-8"))
    if not isinstance(clusters, list):
        clusters = []
    flagged = _flagged_clusters_for(clusters, source, column)

    # Distinct values to verify.
    distinct = _distinct_values(
        frame,
        column,
        max_value_chars=max_value_chars,
        min_value_chars=min_value_chars,
        max_distinct=max_distinct,
    )
    if not distinct:
        logger.info(
            "no eligible distinct values in %s.%s (after value gates)",
            source, column,
        )
        return {
            "counts": _summarise([]),
            "total": 0,
            "output_path": column_artifact_path(run_dir, source, column),
        }

    # Skip cells already verified (unless --force).
    out_path = column_artifact_path(run_dir, source, column)
    prior = [] if force else load_records(out_path)
    prior_by_value: Dict[str, ColumnAuditRecord] = {r.value: r for r in prior}
    prior_keep: List[ColumnAuditRecord] = []
    todo: List[Tuple[str, List[int]]] = []
    for value, rows in distinct:
        # Optional skip: when ``skip_flagged`` is set the operator
        # only wants to find missed errors, so cells where ANY row
        # was flagged by the LLM are excluded (the standard
        # /web-verify path already handles those).
        if skip_flagged and any(r in flagged for r in rows):
            continue
        prev = prior_by_value.get(value)
        if prev is not None and not prev.error:
            prior_keep.append(prev)
            continue
        todo.append((value, rows))

    logger.info(
        "%s.%s: %d distinct values total; %d to verify, %d cached, "
        "%d skipped (flagged)",
        source, column, len(distinct), len(todo), len(prior_keep),
        sum(1 for v, rs in distinct if skip_flagged and any(r in flagged for r in rs)),
    )

    # Build synthetic clusters.
    synth: List[Dict[str, Any]] = []
    cluster_meta: Dict[str, Tuple[str, List[int], List[str]]] = {}
    for value, rows in todo:
        flagged_cids: List[str] = []
        for r in rows:
            for c in flagged.get(r, []):
                if c not in flagged_cids:
                    flagged_cids.append(c)
        c = _synth_cluster(
            source=source, column=column, value=value,
            rows=rows, cluster_ids=flagged_cids,
            error_types=et_focus,
        )
        synth.append(c)
        cluster_meta[c["cluster_id"]] = (value, rows, flagged_cids)

    # Cache + agent.
    cache_dir = run_dir / "web_cache"
    cache = WebCache(cache_dir)
    agent = AzureAgentBackend(
        endpoint=agent_endpoint or "",
        agent_id=agent_id or "",
        cache=cache,
    )
    vcfg = VerifierConfig(
        model=model,
        reasoning_effort=reasoning_effort,
        concurrency=max(1, concurrency),
        # Audit must allow person columns through; the column-audit
        # prompt itself is conservative on confirmed_error so this
        # does not regress the PII story.
        gate=GateConfig(verify_person=True),
        force=force,
    )
    verifier = Verifier(
        agent=agent,
        cell_lookup=cell_lookup,
        column_samples=column_samples,
        row_lookup=row_lookup,
        cfg=vcfg,
    )

    if progress is None:
        def progress(done: int, total: int) -> None:  # noqa: ARG001
            if done % 5 == 0 or done == total:
                logger.info("progress: %d / %d", done, total)

    fresh = verifier.run(synth, progress=progress)

    # Materialise into ColumnAuditRecord. The verifier returns
    # Verification objects keyed by ``cluster_id``; we map back to
    # the audit record via cluster_meta.
    fresh_records: List[ColumnAuditRecord] = []
    for v in fresh:
        meta = cluster_meta.get(v.cluster_id)
        if meta is None:
            continue
        value, rows, flagged_cids = meta
        rec = ColumnAuditRecord(
            source=source,
            column=column,
            value=value,
            rows=list(rows),
            n_rows=len(rows),
            was_flagged_by_llm=bool(flagged_cids),
            cluster_ids=list(flagged_cids),
            verdict=str(v.verdict or "ambiguous"),
            canonical_value=v.canonical_value,
            confidence=float(v.confidence or 0.0),
            evidence=[
                {"url": ev.url, "title": ev.title, "quote": ev.quote}
                for ev in (v.evidence or [])
            ],
            reasoning=v.reasoning or "",
            queries=list(v.queries or []),
            n_results=int(v.n_results or 0),
            n_unique_urls=int(v.n_unique_urls or 0),
            error=v.error,
            parse_error=v.parse_error,
            llm_attempts=int(v.llm_attempts or 0),
            llm_endpoint=v.llm_endpoint or "",
            entity_kind=v.entity_kind or "other",
            ts=v.ts or datetime.now(timezone.utc).isoformat(),
        )
        fresh_records.append(rec)

    # Persist: prior_keep + fresh_records, sorted by first row.
    merged = list(prior_keep) + fresh_records
    merged.sort(key=lambda r: (r.rows[0] if r.rows else 1 << 30, r.value))
    write_records(out_path, merged)
    _rebuild_index(run_dir)

    counts = _summarise(merged)
    return {
        "counts": counts,
        "total": len(merged),
        "output_path": out_path,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-id", required=True, help="Run directory under outputs/")
    p.add_argument(
        "--outputs-dir", type=Path,
        default=PROJECT_ROOT / "outputs",
        help="Root outputs directory (default: outputs/).",
    )
    p.add_argument(
        "--source", default=None,
        help="CSV name (e.g. contracts.csv). Required unless --all-budget-capped.",
    )
    p.add_argument(
        "--column", default=None,
        help="Column header. Required unless --all-budget-capped.",
    )
    p.add_argument(
        "--all-budget-capped", action="store_true",
        help="Run on every (source, column) whose adaptive arms include "
             "an exhausted state. Reads good_turing/per_arm_estimates.json.",
    )
    p.add_argument("--agent-endpoint", default=None)
    p.add_argument("--agent-id", default=None)
    p.add_argument("--model", default="gpt-5.2")
    p.add_argument("--reasoning-effort", default="medium")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument(
        "--max-distinct", type=int, default=None,
        help="Cap on distinct values per column (most frequent kept first).",
    )
    p.add_argument(
        "--max-value-chars", type=int, default=200,
        help="Skip values longer than this; the agent cannot disambiguate "
             "long opaque blobs anyway.",
    )
    p.add_argument(
        "--min-value-chars", type=int, default=2,
        help="Skip values shorter than this; too short to disambiguate.",
    )
    p.add_argument(
        "--skip-flagged", action="store_true",
        help="Skip values whose rows were already flagged by the LLM "
             "(only audit potentially-missed cells).",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Re-run cells that already have a non-errored verification.",
    )
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    _setup_logging(args.verbose)

    run_dir: Path = args.outputs_dir / args.run_id
    if not run_dir.is_dir():
        logger.error("run dir not found: %s", run_dir)
        return 2

    targets: List[Tuple[str, str, List[str]]] = []
    if args.all_budget_capped:
        targets = _budget_capped_columns(run_dir)
        if not targets:
            logger.error(
                "no budget-capped (status=exhausted) arms found in %s",
                run_dir / "good_turing" / "per_arm_estimates.json",
            )
            return 2
        logger.info("found %d budget-capped (source, column) pair(s)", len(targets))
    else:
        if not args.source or not args.column:
            logger.error(
                "--source and --column are required unless --all-budget-capped is set",
            )
            return 2
        # Single-column run: look up the capped error types for this
        # pair (falls back to ["typo"] when the artifact doesn't list
        # the pair) so the audit prompt's FOCUS preamble matches the
        # detector even outside --all-budget-capped mode.
        ets = _capped_error_types_for(run_dir, args.source, args.column)
        targets = [(args.source, args.column, ets)]

    overall = {"confirmed_error": 0, "false_positive": 0, "ambiguous": 0,
               "errored": 0, "fn_candidates": 0, "total": 0}

    # Live progress writer: persists the current column + value
    # counts to ``column_verifications/_progress.json`` so the
    # labelling UI can render a progress bar while we run. Best-
    # effort (any IO error is logged at DEBUG and ignored).
    pwriter = ProgressWriter(run_dir)
    pwriter.update(total_columns=len(targets))

    exit_code = 0
    try:
        for col_idx, (src, col, ets) in enumerate(targets, start=1):
            logger.info(
                "==== auditing %s.%s (focus error_types=%s) ====",
                src, col, ets or ["(none)"],
            )
            # Reset per-column counters; values_total gets set below as
            # soon as we know how many distinct values this column has.
            pwriter.update(
                current_column_index=col_idx,
                current_source=src,
                current_column=col,
                values_total=0,
                values_done=0,
            )

            def _progress(done: int, total: int, _src=src, _col=col) -> None:
                # Update the on-disk progress + a periodic INFO line.
                pwriter.update(values_total=int(total), values_done=int(done))
                if done % 5 == 0 or done == total:
                    logger.info("progress: %d / %d", done, total)

            try:
                result = verify_column(
                    run_dir, src, col,
                    agent_endpoint=args.agent_endpoint,
                    agent_id=args.agent_id,
                    model=args.model,
                    reasoning_effort=args.reasoning_effort,
                    concurrency=args.concurrency,
                    max_distinct=args.max_distinct,
                    max_value_chars=args.max_value_chars,
                    min_value_chars=args.min_value_chars,
                    skip_flagged=args.skip_flagged,
                    force=args.force,
                    error_types=ets,
                    progress=_progress,
                )
            except (FileNotFoundError, ValueError) as exc:
                logger.error("audit failed for %s.%s: %s", src, col, exc)
                pwriter.update(columns_done=col_idx)
                continue
            for k, v in result["counts"].items():
                overall[k] = overall.get(k, 0) + int(v)
            # Surface running totals to the UI.
            pwriter.update_overall(result["counts"])
            pwriter.update(columns_done=col_idx)
            rel = (
                result["output_path"].relative_to(PROJECT_ROOT)
                if result["output_path"].is_relative_to(PROJECT_ROOT)
                else result["output_path"]
            )
            logger.info("done %s.%s -> %s (%s)", src, col, rel, result["counts"])

        logger.info("OVERALL %s", overall)
    except KeyboardInterrupt:
        exit_code = 130
        raise
    except Exception:  # noqa: BLE001
        exit_code = 1
        raise
    finally:
        pwriter.finish(exit_code=exit_code)
    return exit_code



if __name__ == "__main__":
    sys.exit(main())
