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
* ``skip``    -> the SKIPPED CLUSTER alone is dropped from the
  benchmark. Sibling clusters on the same row (regardless of their
  own verdict) and any cell-level FN labels on that row pass through
  unchanged. Skip = "this verdict is undecided", not "this row is
  bad". Whole-row / whole-CSV removal is the exclusions API's job
  (row-scope / csv-scope exclusions); the benchmark exporter does
  not infer a row-level dismissal from a single cluster's skip.
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
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

from ._helpers import _has_whitespace_anomaly, _normalize_trivial
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
      1. ``synthesis.canonical_record[col]`` if present - the Phase-B
         literal canonical replacement.
      2. Any detection's ``corrected_value[col]`` literal - per-cell
         canonical replacement supplied by Phase-A.
      3. ``synthesis.canonical_fix`` (free-form text) if non-empty -
         applied to every column the cluster touches.
      4. The longest non-empty ``suggested_fix`` (free-form text)
         across all detections that mention this specific column.
      5. ``None`` (no usable suggestion).

    Returned dict uses ``""`` as the key for row-level findings so the
    sentinel ``column=""`` benchmark record can pick it up too. Values
    can be ``None`` when no detector contributed a fix for that column.
    """
    out: Dict[str, Optional[str]] = {}
    syn = cluster.get("synthesis") or {}
    canonical: Optional[str] = None
    canonical_record: Mapping[str, Any] = {}
    if isinstance(syn, Mapping):
        cf = syn.get("canonical_fix")
        if cf not in (None, "", "null"):
            canonical = str(cf).strip() or None
        cr = syn.get("canonical_record")
        if isinstance(cr, Mapping):
            canonical_record = cr

    columns = list(cluster.get("columns") or [])
    if not columns:
        columns = [""]

    # Pre-seed every column (and the row-level sentinel) with the
    # canonical fix - it's the cluster-level "best answer".
    for col in columns:
        out[col] = canonical

    # Tier 1: synthesis.canonical_record - literal per-column override
    # beats the free-form ``canonical_fix`` text.
    for col in columns:
        cv = canonical_record.get(col)
        if cv in (None, "", "null"):
            continue
        out[col] = str(cv).strip() or out.get(col)

    # Tier 2 / 4 (detector-level): a detector's ``corrected_value``
    # literal beats its ``suggested_fix`` prose for the same column,
    # and either beats the cluster-level ``canonical_fix`` text only
    # when no literal canonical_record was supplied for that column.
    for det in cluster.get("detections") or []:
        if not isinstance(det, Mapping):
            continue
        det_columns = list(det.get("columns") or [])
        if not det_columns:
            det_columns = [""]

        det_cv = det.get("corrected_value")
        if isinstance(det_cv, Mapping):
            for col in det_columns:
                # Only honour a per-detection literal if synthesis did
                # not already pin a literal for this column.
                if canonical_record.get(col) not in (None, "", "null"):
                    continue
                val = det_cv.get(col)
                if val in (None, "", "null"):
                    continue
                s = str(val).strip()
                if s:
                    out[col] = s

        fix = det.get("suggested_fix")
        if fix in (None, "", "null"):
            continue
        fix_str = str(fix).strip()
        if not fix_str:
            continue
        for col in det_columns:
            # Skip if we already have a literal (synthesis or detector).
            if canonical_record.get(col) not in (None, "", "null"):
                continue
            if isinstance(det_cv, Mapping) and det_cv.get(col) not in (None, "", "null"):
                continue
            current = out.get(col)
            if current is None or len(fix_str) > len(current):
                out[col] = fix_str

    return out


def materialize_label_benchmark(
    clusters: Iterable[Mapping[str, Any]],
    store: LabelStore,
    *,
    value_resolver: Optional[ValueResolver] = None,
    cell_note_overrides: Optional[Mapping[Tuple[str, int, str], str]] = None,
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

    # ``skip`` is per-cluster only: the skipped cluster's cells are
    # omitted, but sibling clusters and FN labels on the same row
    # pass through unchanged. Row- or CSV-wide dismissal is the
    # exclusions API's job (row-scope / csv-scope), which writes
    # cluster-level skip labels via a separate cascade. The exporter
    # never infers a row-level dismissal from a single cluster's
    # skip verdict.
    cluster_list = list(clusters)

    for cluster in cluster_list:
        cid = cluster.get("cluster_id")
        label = labels.get(cid) if cid else None
        if label is None or label.status == LabelStatus.SKIP:
            continue
        row = cluster.get("row")
        source = cluster.get("source", "")
        if row is None or not source:
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

        # Propagate the cluster verdict across every row the cluster
        # covers: the primary ``row`` plus every entry in any
        # detection's ``affected_rows``. Relation rules (and
        # row_duplication groups) fire on N rows but collapse to ONE
        # cluster — without this fan-out, the benchmark would only
        # carry a verdict for the representative row and silently
        # drop the rest. De-duped + sorted so the emit order is
        # deterministic; the primary row stays first.
        rows_to_emit: List[int] = [int(row)]
        _seen_rows = {int(row)}
        for det in cluster.get("detections") or []:
            if not isinstance(det, Mapping):
                continue
            for r in det.get("affected_rows") or ():
                try:
                    r_int = int(r)
                except (TypeError, ValueError):
                    continue
                if r_int in _seen_rows:
                    continue
                _seen_rows.add(r_int)
                rows_to_emit.append(r_int)

        for emit_row in rows_to_emit:
            for column in columns:
                key = _cell_key(source, emit_row, column)
                # Prefer the live CSV value (via value_resolver) so the
                # exported benchmark is self-contained: row_values is a
                # clustering-time snapshot that may be missing keys on
                # propagated/sparse clusters.
                row_values_fallback = ""
                # row_values is only populated for the primary row; for
                # peer / affected rows we always go through the live
                # resolver. If the resolver fails the fallback is "".
                if emit_row == row:
                    raw = row_values.get(column)
                    if raw is not None:
                        row_values_fallback = str(raw)
                original_value = _lookup_value(
                    source, emit_row, column, fallback=row_values_fallback,
                )

                corrected = (
                    label.corrected_values.get(column)
                    if label.corrected_values
                    else label.corrected_value
                )
                # Per-row correction guard: a cluster's
                # ``corrected_value`` is the operator's fix for THIS
                # cluster's primary row only. When the cluster also
                # fans out via ``detections[*].affected_rows`` to
                # peer rows (because the detection rule fired on N
                # rows of the same anomaly pattern), those peer
                # rows almost always have THEIR OWN per-row cluster
                # with THEIR OWN distinct correction. Applying the
                # primary row's correction to peer rows produced
                # cluster-contamination in the benchmark
                # (e.g. row20 'EAST VALLEY (YAKIMA)' inheriting
                # row19's correction 'EAST VALLEY (SPOKANE)').
                # Verdict (TP/FP/ambiguous) and cluster_ids /
                # detection_ids still fan out -- only the
                # row-specific ``corrected_value`` is gated.
                corrected_for_emit = corrected if emit_row == row else None

                # Is this contribution the cell's OWN anchor (the
                # cluster is literally anchored at this row) or a
                # verdict fanned out from another cluster's
                # ``affected_rows``? An own-anchor verdict is
                # authoritative for the cell; a fanned-out verdict is
                # only a default that any own-anchor verdict overrides.
                # Without this, a TP cluster whose detection carries an
                # over-broad ``affected_rows`` (e.g. an aggregate
                # "examples: ..." detection spanning unrelated rows)
                # would stamp "error" onto peer rows the operator
                # explicitly labelled "clean" on their own cluster,
                # manufacturing a phantom ``ambiguous`` conflict.
                is_anchor = emit_row == row

                existing = cells.get(key)
                if existing is None:
                    cells[key] = {
                        "source": source,
                        "row": emit_row,
                        "column": column,
                        "original_value": original_value,
                        "label": verdict,
                        "corrected_value": (
                            corrected_for_emit if verdict == "error" else None
                        ),
                        "llm_suggested_value": llm_suggestion_per_col.get(column),
                        "error_types": list(error_types),
                        "detection_ids": list(detection_ids),
                        "cluster_ids": [cid] if cid else [],
                        "annotators": [label.annotator],
                        "note": label.note,
                        "_anchor": is_anchor,
                    }
                    continue

                # Conflict resolution.
                if existing["label"] != verdict:
                    existing_anchor = existing.get("_anchor", False)
                    if is_anchor and not existing_anchor:
                        # Authoritative own-row verdict supersedes a
                        # verdict that was merely fanned out here.
                        existing["label"] = verdict
                        existing["corrected_value"] = (
                            corrected_for_emit if verdict == "error" else None
                        )
                        existing["_anchor"] = True
                        existing["note"] = label.note
                    elif existing_anchor and not is_anchor:
                        # Keep the authoritative own-row verdict; ignore
                        # the incoming fan-out entirely.
                        pass
                    else:
                        # Both own-anchor (two clusters on the same cell)
                        # or both fan-outs: a genuine disagreement.
                        existing["label"] = "ambiguous"
                        existing["corrected_value"] = None
                else:
                    # Verdicts agree. Still record whether THIS
                    # contribution is the cell's own anchor, so a later
                    # disagreeing fan-out can't downgrade an
                    # already-authoritative verdict to ``ambiguous``.
                    if is_anchor:
                        existing["_anchor"] = True
                    if verdict == "error" and corrected_for_emit:
                        existing["corrected_value"] = (
                            existing["corrected_value"] or corrected_for_emit
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
        if label.row is None or not label.source:
            continue
        # A FN with an EMPTY column is a ROW-LEVEL finding: the only
        # producer is the row-duplication trivial tagger, which writes
        # ``column=""`` at every byte-identical (exact-duplicate) row the
        # LLM missed. It exports as a ``row_duplication`` "error" record
        # using the same sentinel ``column=""`` the cluster path uses for
        # row_duplication verdicts, so the by-error-type bundle buckets
        # it correctly. Cell-level FNs (whitespace tagger etc.) keep their
        # real column and an empty ``error_types``.
        fn_column = label.column or ""
        is_row_level = fn_column == ""
        key = _cell_key(label.source, int(label.row), fn_column)
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
            "column": fn_column,
            # FN cells were never enclosed in a cluster, so we have no
            # row_values snapshot. Fall back to the resolver (live CSV
            # value) so the exported record is still self-contained.
            # Row-level findings have no single cell value, so they carry
            # an empty ``original_value`` (matching the cluster path).
            "original_value": (
                "" if is_row_level
                else _lookup_value(label.source, int(label.row), fn_column)
            ),
            "label": "error",
            "corrected_value": label.corrected_value,
            # Pure FN cells have no enclosing cluster, hence no LLM
            # suggestion to record. The benchmark consumer can tell FN
            # cells from TP/PARTIAL ones via the empty ``cluster_ids``
            # list.
            "llm_suggested_value": None,
            "error_types": ["row_duplication"] if is_row_level else [],
            "detection_ids": [],
            "cluster_ids": [],
            "annotators": [label.annotator],
            "note": label.note,
        }

    # Trivial whitespace cells have a DETERMINISTIC fix (collapse the
    # whitespace), so a peer row must NEVER inherit a sibling's correction
    # that fanned out via a multi-row cluster's ``affected_rows`` -- the
    # ``llm_suggested`` fallback below would otherwise stamp e.g. row 1's
    # "SIMPLY GUM - TPR" onto every peer ``DRY SODA  - TPR`` row. For a
    # TRIVIAL-noted error cell that still has no own-anchor fix, recompute
    # the correction from that cell's OWN value. Own-anchor fixes (operator
    # or LLM) are non-empty here and are left untouched.
    for payload in cells.values():
        if payload.get("label") != "error" or payload.get("corrected_value"):
            continue
        if "TRIVIAL" not in (payload.get("note") or "").upper():
            continue
        ov = payload.get("original_value")
        if isinstance(ov, str) and _has_whitespace_anomaly(ov):
            payload["corrected_value"] = _normalize_trivial(ov)

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

    # Apply per-cell note overrides: a single (source, row, column)
    # may carry a free-text note that supersedes the cluster's note.
    # Used when one cell in a cluster needs a different annotation
    # from its siblings (e.g. the operator wants to strip the
    # auto-applied TRIVIAL tag for one row only). Empty / missing
    # entries fall through so the cluster note still wins.
    if cell_note_overrides:
        for payload in cells.values():
            key = (
                payload.get("source", ""),
                int(payload.get("row", 0)),
                payload.get("column", ""),
            )
            override = cell_note_overrides.get(key)
            if override is None:
                continue
            payload["note"] = override

    for payload in cells.values():
        payload.pop("_anchor", None)
    return [BenchmarkCell(**payload) for payload in cells.values()]


def write_benchmark(
    out_path: Path,
    clusters: Iterable[Mapping[str, Any]],
    store: LabelStore,
    *,
    value_resolver: Optional[ValueResolver] = None,
    cell_note_overrides: Optional[Mapping[Tuple[str, int, str], str]] = None,
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
        cell_note_overrides=cell_note_overrides,
    )
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as fh:
        for cell in rows:
            fh.write(json.dumps(cell.model_dump(), ensure_ascii=False) + "\n")
    os.replace(tmp_path, out_path)
    return out_path


# ---------------------------------------------------------------------
# Pre-export consistency check.
#
# Same value, same column, same source -- but the reviewer landed on
# different verdicts row-to-row. Most of the time this is an
# accidental click ("error" on row 7 while every other row of the
# same value reads "clean") rather than an intentional ambiguity
# signal; the exporter surfaces the groups so the operator can fix
# them up before downloading the benchmark.
# ---------------------------------------------------------------------


def find_label_conflicts(
    cells: Iterable[BenchmarkCell],
) -> List[Dict[str, Any]]:
    """Group benchmark cells where the same (source, column,
    original_value) triple carries divergent ``label`` verdicts.

    Returns one group per conflicting triple::

        {
            "source": str,
            "column": str,
            "original_value": str,
            "labels": [<distinct label strings, sorted>],
            "cells": [
                {"row": int, "label": str,
                 "corrected_value": Any, "cluster_ids": [...],
                 "note": str},
                ...
            ],
        }

    Comparison is byte-exact on ``original_value`` (matching how the
    JSONL serialises it). Cells with an empty ``column`` are skipped
    because they describe row-level findings whose ``original_value``
    isn't a column value; cells with an empty ``original_value`` are
    skipped because blank cells legitimately reuse the same key and
    would generate high-noise warnings.

    Output is sorted (source, column, original_value) and each
    group's ``cells`` list is sorted by row for stable presentation.
    """
    groups: Dict[tuple, List[BenchmarkCell]] = {}
    for cell in cells:
        if not cell.column:
            continue
        if not cell.original_value:
            continue
        key = (cell.source, cell.column, cell.original_value)
        groups.setdefault(key, []).append(cell)

    out: List[Dict[str, Any]] = []
    for (source, column, original_value), members in groups.items():
        labels_set = {m.label for m in members}
        if len(labels_set) <= 1:
            continue
        ordered = sorted(members, key=lambda m: (m.row, m.label))
        out.append({
            "source": source,
            "column": column,
            "original_value": original_value,
            "labels": sorted(labels_set),
            "cells": [
                {
                    "row": m.row,
                    "label": m.label,
                    "corrected_value": m.corrected_value,
                    "cluster_ids": list(m.cluster_ids),
                    "note": m.note,
                }
                for m in ordered
            ],
        })
    out.sort(key=lambda g: (g["source"], g["column"], g["original_value"]))
    return out


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
    cell_note_overrides: Optional[Mapping[Tuple[str, int, str], str]] = None,
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
        cell_note_overrides=cell_note_overrides,
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
    cell_note_overrides: Optional[Mapping[Tuple[str, int, str], str]] = None,
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
        cell_note_overrides=cell_note_overrides,
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

