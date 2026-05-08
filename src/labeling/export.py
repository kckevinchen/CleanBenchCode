"""Materialise a per-cell benchmark JSONL from cluster labels.

Detectors and clusters speak "one row per defect". Benchmarks need "one
row per (source, row, column) cell" so they can be consumed by cleaners
as a direct ground-truth signal. This module bridges the two.

Mapping rule:

* Each cluster touches one row and one-or-more columns. The annotator's
  TP/FP verdict applies to every affected cell.
* ``tp``      -> cell label = ``"error"``, corrected_value carried over.
* ``fp``      -> cell label = ``"clean"``.
* ``partial`` -> ``"error"`` (detector was right to flag, description off).
* ``unsure``  -> ``"ambiguous"``.
* ``skip``    -> the entire ``(source, row)`` is excluded from the
  benchmark. Any sibling cluster on the same row (regardless of its
  own verdict) and any cell-level FN label on that row are dropped
  too, so a single skip cleanly removes an unusable row from the
  ground truth.
* ``fn``      -> cell-level label (not cluster-level) marking a real
  error the LLM missed; emitted as ``"error"`` with empty cluster_ids.

If two clusters disagree on the same cell (rare but possible when the
same row/column is clustered twice), ``"ambiguous"`` wins. A user-placed
FN that overlaps a TP cluster is automatically superseded by the TP
record (TP outranks FN at this layer too).
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional

from .schema import BenchmarkCell, LabelStatus, LabelTarget
from .store import LabelStore


# A ``value_resolver`` is an optional callback ``(source, row, column)
# -> str | None`` that returns the canonical CSV value for a cell.
# When provided, the materializer prefers it over the cluster's
# in-memory ``row_values`` snapshot - because some clusters carry
# sparse/empty ``row_values`` (e.g. same-value propagation, cluster
# pruning), but the consumer of an exported benchmark expects every
# row to carry the original cell value verbatim. ``None`` means the
# resolver couldn't read that cell (missing source, out-of-range row,
# unknown column); the materializer falls back to ``row_values`` and
# finally to the empty string.
ValueResolver = Callable[[str, int, str], Optional[str]]


_VERDICT: Dict[str, str] = {
    LabelStatus.TP.value: "error",
    LabelStatus.PARTIAL.value: "error",
    LabelStatus.FP.value: "clean",
    LabelStatus.UNSURE.value: "ambiguous",
}


def _cell_key(source: str, row: int, column: str) -> str:
    return f"{source}\x00{row}\x00{column}"


def _llm_suggestion_for_cluster(
    cluster: Mapping[str, Any],
) -> Dict[str, Optional[str]]:
    """Map column -> LLM-proposed corrected value for this cluster.

    Resolution order, per column:
      1. ``synthesis.canonical_fix`` if non-empty - applied to every
         column the cluster touches. This is the reconciled,
         operator-facing suggestion (Phase B for multi-type clusters,
         auto-synthesised "richest detection" for single-type ones).
      2. The longest non-empty ``suggested_fix`` across all detections
         that mention this specific column. Detector-level fixes are
         already keyed implicitly by ``columns``, so we map each
         detection's suggestion onto its ``columns`` list.
      3. ``None`` (no usable suggestion).

    Returned dict uses ``""`` as the key for row-level findings so the
    sentinel ``column=""`` benchmark record can pick it up too. Values
    can be ``None`` when no detector contributed a fix for that column.
    """
    out: Dict[str, Optional[str]] = {}
    syn = cluster.get("synthesis") or {}
    canonical: Optional[str] = None
    if isinstance(syn, Mapping):
        cf = syn.get("canonical_fix")
        if cf not in (None, "", "null"):
            canonical = str(cf).strip() or None

    columns = list(cluster.get("columns") or [])
    if not columns:
        columns = [""]

    # Pre-seed every column (and the row-level sentinel) with the
    # canonical fix - it's the cluster-level "best answer".
    for col in columns:
        out[col] = canonical

    # Detector-level overrides: a detector's ``suggested_fix`` is
    # already specific to the columns it surfaced, so prefer it for
    # those columns when canonical is unset OR when the detector's
    # fix is longer (a heuristic: longer literal values like full
    # corrected strings tend to be more useful than truncated
    # synthesis paraphrases).
    for det in cluster.get("detections") or []:
        if not isinstance(det, Mapping):
            continue
        fix = det.get("suggested_fix")
        if fix in (None, "", "null"):
            continue
        fix_str = str(fix).strip()
        if not fix_str:
            continue
        det_columns = list(det.get("columns") or [])
        if not det_columns:
            det_columns = [""]
        for col in det_columns:
            current = out.get(col)
            if current is None or len(fix_str) > len(current):
                out[col] = fix_str

    return out


def materialize_label_benchmark(
    clusters: Iterable[Mapping[str, Any]],
    store: LabelStore,
    *,
    value_resolver: Optional[ValueResolver] = None,
) -> List[BenchmarkCell]:
    """Return one :class:`BenchmarkCell` per (source, row, column) cell.

    Renamed from ``build_benchmark`` to disambiguate from
    :func:`experiments.build_synth_benchmark.build_benchmark` (which
    INJECTS synthetic errors); this function MATERIALIZES a per-cell
    ground-truth benchmark from human cluster labels.

    ``value_resolver`` (optional) is a callable ``(source, row, column)
    -> str | None`` used to look up the canonical CSV value for each
    emitted cell. When provided, it takes precedence over the
    cluster's ``row_values`` snapshot so consumers can rely on
    ``original_value`` always being the actual CSV cell, regardless
    of how the cluster was created (detector pass vs same-value
    propagation vs cell-level FN). ``None`` is treated as "resolver
    has no value for this cell" and falls back to ``row_values`` /
    ``""``.
    """
    labels = store.latest_by_target()
    cells: Dict[str, Dict[str, Any]] = {}

    def _lookup_value(
        source: str, row: int, column: str, fallback: str = "",
    ) -> str:
        """Return the canonical original value for ``(source, row,
        column)``. Resolver wins; otherwise fall back to the caller-
        provided ``fallback`` (typically the cluster's ``row_values``
        entry, which itself defaults to ``""``)."""
        if value_resolver is not None:
            try:
                resolved = value_resolver(source, row, column)
            except Exception:  # noqa: BLE001 - resolver is operator code; we MUST NOT crash the export.
                resolved = None
            if resolved is not None:
                return str(resolved)
        return fallback

    # First pass: any (source, row) with at least one ``skip`` cluster
    # verdict is dropped wholesale from the benchmark - including
    # sibling clusters on the same row that may carry tp/fp/partial
    # verdicts, and any FN cell labels on that row. Skip = "this row
    # is unusable as ground truth".
    skipped_rows: set = set()
    cluster_list = list(clusters)
    for cluster in cluster_list:
        cid = cluster.get("cluster_id")
        if not cid:
            continue
        label = labels.get(cid)
        if label is None or label.status != LabelStatus.SKIP:
            continue
        source = cluster.get("source") or ""
        row = cluster.get("row")
        if source and row is not None:
            skipped_rows.add((source, int(row)))

    for cluster in cluster_list:
        cid = cluster.get("cluster_id")
        label = labels.get(cid) if cid else None
        if label is None or label.status == LabelStatus.SKIP:
            continue
        row = cluster.get("row")
        source = cluster.get("source", "")
        if row is None or not source:
            continue
        if (source, int(row)) in skipped_rows:
            continue

        verdict = _VERDICT.get(label.status.value)
        if verdict is None:
            continue

        row_values = cluster.get("row_values") or {}
        # Label-side columns win: re-clustering can shift cluster.columns
        # after the verdict was recorded.
        columns = list(label.columns) if label.columns else (
            cluster.get("columns") or []
        )
        # Row-level findings (e.g. ``row_duplication``) carry an empty
        # column list because the verdict applies to the whole row, not
        # any specific cell. Emit a single sentinel record with
        # ``column=""`` so the benchmark still captures the verdict.
        if not columns:
            columns = [""]
        error_types = cluster.get("error_types") or []
        detection_ids = [
            d.get("issue_id", "") for d in cluster.get("detections") or [] if d.get("issue_id")
        ]
        # LLM-side suggestion: prefer the cluster's synthesis
        # (``canonical_fix``) which is either the Phase-B reconciled
        # value for multi-type clusters or - via auto-synthesis - the
        # representative detection's ``suggested_fix`` for single-type
        # clusters. Falls back to scanning the per-detection
        # ``suggested_fix`` map when the synthesis didn't fill one in.
        llm_suggestion_per_col = _llm_suggestion_for_cluster(cluster)

        for column in columns:
            key = _cell_key(source, row, column)
            # Prefer the live CSV value (via value_resolver) so the
            # exported benchmark is self-contained: row_values is a
            # clustering-time snapshot that may be missing keys on
            # propagated/sparse clusters.
            row_values_fallback = ""
            raw = row_values.get(column)
            if raw is not None:
                row_values_fallback = str(raw)
            original_value = _lookup_value(
                source, row, column, fallback=row_values_fallback,
            )

            corrected = (
                label.corrected_values.get(column)
                if label.corrected_values
                else label.corrected_value
            )

            existing = cells.get(key)
            if existing is None:
                cells[key] = {
                    "source": source,
                    "row": row,
                    "column": column,
                    "original_value": original_value,
                    "label": verdict,
                    "corrected_value": corrected if verdict == "error" else None,
                    "llm_suggested_value": llm_suggestion_per_col.get(column),
                    "error_types": list(error_types),
                    "detection_ids": list(detection_ids),
                    "cluster_ids": [cid] if cid else [],
                    "annotators": [label.annotator],
                    "note": label.note,
                }
                continue

            # Conflict resolution.
            if existing["label"] != verdict:
                existing["label"] = "ambiguous"
                existing["corrected_value"] = None
            elif verdict == "error" and corrected:
                existing["corrected_value"] = (
                    existing["corrected_value"] or corrected
                )
            # LLM suggestion: keep the longest non-empty across the
            # contributing clusters - same heuristic the auto-synthesis
            # uses to pick the "richest" detection. ``None`` only sticks
            # if NO contributing cluster proposed a fix for this cell.
            new_suggestion = llm_suggestion_per_col.get(column)
            if new_suggestion:
                cur = existing.get("llm_suggested_value")
                if not cur or len(new_suggestion) > len(cur):
                    existing["llm_suggested_value"] = new_suggestion
            for field_, value in (
                ("error_types", error_types),
                ("detection_ids", detection_ids),
                ("cluster_ids", [cid] if cid else []),
                ("annotators", [label.annotator]),
            ):
                for v in value:
                    if v and v not in existing[field_]:
                        existing[field_].append(v)

    # Fold in cell-level FN labels. A FN claims "this cell is an error
    # the detector missed", so we emit it as label="error" with an
    # empty cluster_ids list (caller can tell FN cells from TP cells by
    # the absence of cluster_ids). FN labels that fall on a cell
    # already marked TP / PARTIAL are silently superseded - the
    # annotator effectively beat themselves to the call via the cluster
    # path, so we keep the cluster record.
    for label in labels.values():
        if label.target != LabelTarget.CELL or label.status != LabelStatus.FN:
            continue
        if label.row is None or not label.column or not label.source:
            continue
        if (label.source, int(label.row)) in skipped_rows:
            # The annotator dismissed this row entirely; even FN cell
            # marks on it are excluded.
            continue
        key = _cell_key(label.source, int(label.row), label.column)
        existing = cells.get(key)
        if existing is not None and existing["label"] == "error":
            # Already an error record; just merge annotator note/credit.
            if label.annotator not in existing["annotators"]:
                existing["annotators"].append(label.annotator)
            if label.corrected_value and not existing.get("corrected_value"):
                existing["corrected_value"] = label.corrected_value
            continue
        if existing is not None and existing["label"] != "error":
            # Cluster says clean/ambiguous, cell says error: ambiguous.
            existing["label"] = "ambiguous"
            existing["corrected_value"] = None
            if label.annotator not in existing["annotators"]:
                existing["annotators"].append(label.annotator)
            continue
        cells[key] = {
            "source": label.source,
            "row": int(label.row),
            "column": label.column,
            # FN cells were never enclosed in a cluster, so we have no
            # row_values snapshot. Fall back to the resolver (live CSV
            # value) so the exported record is still self-contained.
            "original_value": _lookup_value(
                label.source, int(label.row), label.column,
            ),
            "label": "error",
            "corrected_value": label.corrected_value,
            # Pure FN cells have no enclosing cluster, hence no LLM
            # suggestion to record. The benchmark consumer can tell FN
            # cells from TP/PARTIAL ones via the empty ``cluster_ids``
            # list.
            "llm_suggested_value": None,
            "error_types": [],
            "detection_ids": [],
            "cluster_ids": [],
            "annotators": [label.annotator],
            "note": label.note,
        }

    # Fill the corrected value for every TP cell that has no
    # operator-supplied correction by falling back to the LLM's
    # suggestion. The benchmark exists primarily so cleaners can
    # learn ``original -> corrected`` pairs, so leaving
    # ``corrected_value`` empty on confirmed errors makes those rows
    # useless. The LLM suggestion is the cluster's reconciled
    # ``synthesis.canonical_fix`` (or the longest detection
    # ``suggested_fix``) - exactly the value the operator saw in the
    # "Original / Fix" panel when they accepted the verdict, so
    # promoting it to ``corrected_value`` faithfully reflects "the
    # operator approved this fix". Cells with neither operator nor
    # LLM value retain ``None`` so consumers can distinguish "no fix
    # known" from "fix is empty string".
    for payload in cells.values():
        if payload.get("label") != "error":
            continue
        if payload.get("corrected_value"):
            continue
        suggestion = payload.get("llm_suggested_value")
        if suggestion is None:
            continue
        suggestion_str = str(suggestion)
        if not suggestion_str:
            continue
        payload["corrected_value"] = suggestion_str

    return [BenchmarkCell(**payload) for payload in cells.values()]


def write_benchmark(
    out_path: Path,
    clusters: Iterable[Mapping[str, Any]],
    store: LabelStore,
    *,
    value_resolver: Optional[ValueResolver] = None,
) -> Path:
    """Materialise the benchmark and write it to ``out_path`` as JSONL.

    Atomic: data is written to a sibling ``.tmp`` file and then
    ``os.replace``-d into place so concurrent readers never observe a
    truncated file.

    See :func:`materialize_label_benchmark` for ``value_resolver``.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = materialize_label_benchmark(
        clusters, store, value_resolver=value_resolver,
    )
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as fh:
        for cell in rows:
            fh.write(json.dumps(cell.model_dump(), ensure_ascii=False) + "\n")
    os.replace(tmp_path, out_path)
    return out_path


# ---------------------------------------------------------------------
# Per-error-type benchmark export.
#
# Splits the materialised benchmark into one bucket per error_type so
# downstream cleaners can load a single canonical file for each defect
# class (typo.jsonl, row_duplication.jsonl, ...). A cell whose
# ``error_types`` lists multiple types is replicated into EACH bucket
# - this matches the natural "give me every typo" semantics: if the
# annotator confirmed both a typo AND a pattern_inconsistency on the
# same cell, that cell is ground truth for both classes.
#
# Cells with no error_types (FP cluster cells, FN cell labels filed
# without an enclosing cluster, ambiguous row-level findings) land in
# a synthetic ``_uncategorized`` bucket so the export is lossless -
# no labelled cell gets silently dropped from the per-type bundle.
# ---------------------------------------------------------------------


# Bucket name for cells whose ``error_types`` is empty after
# materialisation. Underscored prefix sorts first in directory listings
# and signals "synthetic / catch-all" to humans browsing the bundle.
UNCATEGORIZED_BUCKET = "_uncategorized"


_FILENAME_SAFE_RE = re.compile(r"[^A-Za-z0-9_.-]")


def _slug_for_error_type(et: str) -> str:
    """Filename-safe slug for an error_type identifier.

    The canonical taxonomy uses snake_case keys (``typo``,
    ``row_duplication``, ``inter_table_inconsistency``, ...) so this
    is a no-op for healthy inputs. The slugger is defensive against
    legacy / freeform error_types that crept into a run via custom
    detectors: anything outside ``[A-Za-z0-9_.-]`` is replaced with
    ``_`` so the resulting filename is always portable.
    """
    s = (et or "").strip()
    if not s:
        return UNCATEGORIZED_BUCKET
    cleaned = _FILENAME_SAFE_RE.sub("_", s)
    # Collapse runs of underscores and strip leading dots so we never
    # produce hidden files on POSIX.
    cleaned = re.sub(r"_+", "_", cleaned).lstrip(".") or UNCATEGORIZED_BUCKET
    return cleaned


def materialize_label_benchmark_by_error_type(
    clusters: Iterable[Mapping[str, Any]],
    store: LabelStore,
    *,
    value_resolver: Optional[ValueResolver] = None,
) -> Dict[str, List[BenchmarkCell]]:
    """Materialise the benchmark grouped by ``error_type``.

    Returns a mapping ``{error_type: [BenchmarkCell, ...]}``. Each
    canonical cell is replicated once per error_type in its
    ``error_types`` list. Cells with empty ``error_types`` are
    bucketed under :data:`UNCATEGORIZED_BUCKET`.

    The replication is intentional - downstream callers want to
    ``cat typo.jsonl`` and see every cell where the annotator
    confirmed a typo, even if the same cell was also clustered as
    e.g. a pattern_inconsistency. Per-cell uniqueness is preserved
    *within* each bucket via the cell's ``(source, row, column)``
    triple, which the upstream materializer already deduplicates.

    See :func:`materialize_label_benchmark` for ``value_resolver``.
    """
    cells = materialize_label_benchmark(
        clusters, store, value_resolver=value_resolver,
    )
    buckets: Dict[str, List[BenchmarkCell]] = {}
    for cell in cells:
        types = list(cell.error_types) if cell.error_types else []
        if not types:
            buckets.setdefault(UNCATEGORIZED_BUCKET, []).append(cell)
            continue
        # Stable order within a bucket: rely on the materializer's
        # iteration order (insertion order of the underlying dict).
        seen: set = set()
        for et in types:
            slug = _slug_for_error_type(et)
            if slug in seen:
                # Two error_type strings collapsing to the same slug
                # (rare; only happens with legacy non-canonical types
                # that share a sanitised form). Avoid duplicate emit.
                continue
            seen.add(slug)
            buckets.setdefault(slug, []).append(cell)
    return buckets


def write_benchmark_by_error_type(
    out_dir: Path,
    clusters: Iterable[Mapping[str, Any]],
    store: LabelStore,
    *,
    value_resolver: Optional[ValueResolver] = None,
) -> Dict[str, Path]:
    """Write one JSONL per error_type into ``out_dir``.

    Returns ``{error_type_slug: path}``. The directory is created if
    missing. Existing per-type files in ``out_dir`` that no longer
    correspond to any labelled cell are left in place so the caller
    can detect stale buckets if needed - we never delete files we
    didn't write.

    Each file is written via the same ``.tmp + os.replace`` dance
    used by :func:`write_benchmark` so a concurrent reader never
    observes a truncated bucket.

    See :func:`materialize_label_benchmark` for ``value_resolver``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    buckets = materialize_label_benchmark_by_error_type(
        clusters, store, value_resolver=value_resolver,
    )
    written: Dict[str, Path] = {}
    for slug, rows in buckets.items():
        path = out_dir / f"{slug}.jsonl"
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as fh:
            for cell in rows:
                fh.write(
                    json.dumps(cell.model_dump(), ensure_ascii=False) + "\n"
                )
        os.replace(tmp_path, path)
        written[slug] = path
    return written

