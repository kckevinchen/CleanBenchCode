"""Pydantic schema for label records and API payloads.

Labels are append-only: one JSON line per (annotator, target, ts) write.
The latest entry per (annotator, target_id) wins at read time - see
``store.LabelStore.latest_by_target``. This keeps history + undo trivial
while staying crash-safe (every POST fsyncs a single line).
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class LabelStatus(str, Enum):
    """Annotator verdict on a cluster (or single detection or cell)."""

    TP = "tp"            # true positive - real error
    FP = "fp"            # false positive - detector was wrong
    PARTIAL = "partial"  # real error but description/fix is off
    UNSURE = "unsure"    # needs a second opinion
    SKIP = "skip"        # defer; keep at bottom of queue
    FN = "fn"            # false negative - cell error the LLM missed


class LabelTarget(str, Enum):
    CLUSTER = "cluster"
    ISSUE = "issue"
    CELL = "cell"        # ad-hoc per-(source,row,column) cell label, used
                         # by the FN flow. target_id is built by
                         # :func:`cell_target_id` so it stays stable
                         # across reloads.


def cell_target_id(source: str, row: int, column: str) -> str:
    """Stable target_id for a cell-level label. Uses NUL separators so
    we never collide with cluster_id / issue_id formats."""
    return f"cell\x00{source}\x00{int(row)}\x00{column}"


class LabelIn(BaseModel):
    """POST /api/labels payload."""

    target: LabelTarget = LabelTarget.CLUSTER
    target_id: str                   # cluster_id or issue_id
    source: str                      # CSV basename (for quick filtering)
    row: Optional[int] = None
    column: Optional[str] = None     # nullable: some labels are row-level
    # Full column set the verdict applies to. Stored on the label so the
    # exporter doesn't have to re-derive it from clusters.json (which
    # can drift if clusters are regenerated). Empty => fall back to
    # ``[column]`` then to the cluster's ``columns``.
    columns: List[str] = Field(default_factory=list)
    status: LabelStatus
    corrected_value: Optional[str] = None
    # Per-column corrections; empty => use the scalar ``corrected_value``
    # for every column in ``columns`` (back-compat with single-column).
    corrected_values: Dict[str, str] = Field(default_factory=dict)
    note: str = ""
    # Annotator defaults to the server's --annotator flag if omitted.
    annotator: Optional[str] = None


class Label(LabelIn):
    """Persisted label record (LabelIn + server-assigned fields)."""

    annotator: str
    ts: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class LabelStats(BaseModel):
    """Per-status counts for a slice of clusters/issues."""

    total: int
    tp: int = 0
    fp: int = 0
    partial: int = 0
    unsure: int = 0
    skip: int = 0
    unlabeled: int = 0
    fn: int = 0


class BenchmarkCell(BaseModel):
    """One row of the exported benchmark (per-cell ground truth)."""

    source: str
    row: int
    column: str
    original_value: str
    label: str                       # error | clean | ambiguous
    corrected_value: Optional[str] = None
    # Concrete replacement value the detector / synthesiser proposed
    # for THIS cell, captured at label time. Independent of the
    # operator's verdict: an FP cell still records the LLM's
    # (rejected) suggestion so downstream callers can A/B compare
    # human vs LLM corrections without re-loading the run's clusters.
    # ``None`` when no detector for this cell produced a usable
    # suggestion (e.g. row-level findings, FN cells filed without an
    # enclosing cluster, or detectors that legitimately can't propose
    # a fix).
    llm_suggested_value: Optional[str] = None
    error_types: List[str] = Field(default_factory=list)
    detection_ids: List[str] = Field(default_factory=list)
    cluster_ids: List[str] = Field(default_factory=list)
    annotators: List[str] = Field(default_factory=list)
    note: str = ""


# ---------------------------------------------------------------------
# Multi-table inconsistency labels.
#
# Independent of the cluster/cell label vocabulary above. The flow is:
# the relation-discovery pipeline emits one ``violation_id`` per cross-
# sheet finding (1+ rows per finding, all sharing that id); the human
# reviewer compares the linked rows side-by-side and writes ONE verdict
# per ``violation_id``. Statuses match the LLM verifier verdicts
# verbatim so users can directly agree/disagree with the LLM's call.
# ---------------------------------------------------------------------


class MultiTableLabelStatus(str, Enum):
    """Annotator verdict on a multi-table inconsistency finding."""

    CONFIRMED_INCONSISTENCY = "confirmed_inconsistency"  # real cross-sheet error
    BENIGN_VARIATION = "benign_variation"                # acceptable variant
    NOT_A_VIOLATION = "not_a_violation"                  # false alarm
    UNSURE = "unsure"                                    # needs second opinion


class MultiTableLabelIn(BaseModel):
    """POST /api/runs/{run_id}/multi-table-labels payload."""

    violation_id: str                # shared across all linked rows
    workbook: str = ""               # for filtering / context only
    group_id: str = ""
    kind: str = ""                   # e.g. duplicate_row, attribute_conflict
    status: MultiTableLabelStatus
    note: str = ""
    annotator: Optional[str] = None


class MultiTableLabel(MultiTableLabelIn):
    """Persisted multi-table label record."""

    annotator: str
    ts: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# ---------------------------------------------------------------------
# Exclusions: per-source decisions to skip whole CSVs / rows / columns
# during labelling.
#
# Independent of the cluster/cell label vocabulary above. The flow:
# the operator marks a CSV file (or specific rows / columns within it)
# as too messy to label; the run keeps an append-only record of those
# decisions in ``<run_dir>/exclusions.jsonl`` so a later step can
# reconstruct a sanitised dataset by removing the excluded slices.
# Soft-delete (tombstoning) is supported via an explicit ``deleted``
# flag on a follow-up record with the same ``id`` so the undo path
# stays append-only - no rewriting prior history.
# ---------------------------------------------------------------------


class ExclusionScope(str, Enum):
    """What part of a source CSV is being excluded."""

    CSV = "csv"          # the entire file - rows/columns must be empty
    ROWS = "rows"        # specific row indices (uses .rows)
    COLUMNS = "columns"  # specific column names (uses .columns)


class ExclusionIn(BaseModel):
    """POST /api/runs/{run_id}/exclusions payload."""

    scope: ExclusionScope
    source: str                            # CSV basename
    rows: List[int] = Field(default_factory=list)
    columns: List[str] = Field(default_factory=list)
    note: str = ""
    annotator: Optional[str] = None


class Exclusion(ExclusionIn):
    """Persisted exclusion record."""

    id: str                                # uuid - stable across tombstoning
    annotator: str
    ts: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    # Tombstone marker - a follow-up record with the same ``id`` and
    # ``deleted=True`` retracts the prior exclusion. The store reduces
    # the JSONL to "active ids" by latest-wins so the API can present
    # an undo-able list without rewriting history.
    deleted: bool = False

