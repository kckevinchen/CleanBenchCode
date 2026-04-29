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
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

from .schema import BenchmarkCell, LabelStatus, LabelTarget
from .store import LabelStore


_VERDICT: Dict[str, str] = {
    LabelStatus.TP.value: "error",
    LabelStatus.PARTIAL.value: "error",
    LabelStatus.FP.value: "clean",
    LabelStatus.UNSURE.value: "ambiguous",
}


def _cell_key(source: str, row: int, column: str) -> str:
    return f"{source}\x00{row}\x00{column}"


def materialize_label_benchmark(
    clusters: Iterable[Mapping[str, Any]],
    store: LabelStore,
) -> List[BenchmarkCell]:
    """Return one :class:`BenchmarkCell` per (source, row, column) cell.

    Renamed from ``build_benchmark`` to disambiguate from
    :func:`experiments.build_synth_benchmark.build_benchmark` (which
    INJECTS synthetic errors); this function MATERIALIZES a per-cell
    ground-truth benchmark from human cluster labels.
    """
    labels = store.latest_by_target()
    cells: Dict[str, Dict[str, Any]] = {}

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
        columns = cluster.get("columns") or []
        error_types = cluster.get("error_types") or []
        detection_ids = [
            d.get("issue_id", "") for d in cluster.get("detections") or [] if d.get("issue_id")
        ]

        for column in columns:
            key = _cell_key(source, row, column)
            original_value = ""
            raw = row_values.get(column)
            if raw is not None:
                original_value = str(raw)

            existing = cells.get(key)
            if existing is None:
                cells[key] = {
                    "source": source,
                    "row": row,
                    "column": column,
                    "original_value": original_value,
                    "label": verdict,
                    "corrected_value": label.corrected_value if verdict == "error" else None,
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
            elif verdict == "error" and label.corrected_value:
                existing["corrected_value"] = (
                    existing["corrected_value"] or label.corrected_value
                )
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
            "original_value": "",
            "label": "error",
            "corrected_value": label.corrected_value,
            "error_types": [],
            "detection_ids": [],
            "cluster_ids": [],
            "annotators": [label.annotator],
            "note": label.note,
        }

    return [BenchmarkCell(**payload) for payload in cells.values()]


def write_benchmark(
    out_path: Path,
    clusters: Iterable[Mapping[str, Any]],
    store: LabelStore,
) -> Path:
    """Materialise the benchmark and write it to ``out_path`` as JSONL."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = materialize_label_benchmark(clusters, store)
    with out_path.open("w", encoding="utf-8") as fh:
        for cell in rows:
            fh.write(json.dumps(cell.model_dump(), ensure_ascii=False) + "\n")
    return out_path
