"""Pydantic schema for label records and API payloads.

Labels are append-only: one JSON line per (annotator, target, ts) write.
The latest entry per (annotator, target_id) wins at read time - see
``store.LabelStore.latest_by_target``. This keeps history + undo trivial
while staying crash-safe (every POST fsyncs a single line).
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional

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
    status: LabelStatus
    corrected_value: Optional[str] = None
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
    error_types: List[str] = Field(default_factory=list)
    detection_ids: List[str] = Field(default_factory=list)
    cluster_ids: List[str] = Field(default_factory=list)
    annotators: List[str] = Field(default_factory=list)
    note: str = ""
