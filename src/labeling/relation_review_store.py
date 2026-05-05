"""Append-only JSONL store for human reviews of discovered relations.

One file, ``<run_dir>/relation_reviews.jsonl``, one review record per
line. Modelled after :mod:`src.labeling.store` (cell/cluster labels):
all writes are O_APPEND + ``fsync``, the in-memory index is rebuilt on
each reload, and the *latest* entry per ``relation_id`` wins.

Three decisions are supported:

* ``approved``  - reviewer agrees with a discovery-emitted relation
* ``rejected``  - reviewer rejects a discovery-emitted relation
* ``added``     - reviewer authored a brand-new relation that was not
                  emitted by the LLM. ``relation`` carries the full
                  snapshot.

A ``cleared`` decision is also persisted to undo a prior approve/reject
without losing history.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from pydantic import BaseModel, Field


class ReviewDecision(str, Enum):
    APPROVED = "approved"
    REJECTED = "rejected"
    ADDED = "added"
    CLEARED = "cleared"


class RelationSnapshot(BaseModel):
    """Frozen relation payload, used both for ``added`` (authoritative)
    and for approve/reject (context preserved at review time so the
    review still makes sense if the underlying ``relations.jsonl`` is
    regenerated and the species_key shifts)."""

    kind: str
    columns: List[str] = Field(default_factory=list)
    lhs: List[str] = Field(default_factory=list)
    rhs: List[str] = Field(default_factory=list)
    predicate: str = ""
    executable: Optional[str] = None
    rationale: str = ""
    source: Optional[str] = None
    error_type: Optional[str] = None


class ReviewIn(BaseModel):
    """POST payload for a review action."""

    relation_id: str  # species_key of the relation being reviewed
    decision: ReviewDecision
    note: str = ""
    author: Optional[str] = None
    relation: Optional[RelationSnapshot] = None


class ReviewRecord(ReviewIn):
    """Persisted review record (ReviewIn + server-assigned fields)."""

    review_id: str
    created_at: datetime


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


class RelationReviewStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self._records: List[ReviewRecord] = []
        self._latest: Dict[str, ReviewRecord] = {}  # relation_id -> latest
        self.reload()

    # ------------------------------------------------------------------ IO

    def reload(self) -> None:
        with self._lock:
            self._records = []
            self._latest = {}
            if not self.path.exists():
                return
            with self.path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = ReviewRecord.model_validate_json(line)
                    except Exception:  # noqa: BLE001 - skip malformed lines
                        continue
                    self._records.append(record)
                    self._latest[record.relation_id] = record

    def append(self, payload: ReviewIn) -> ReviewRecord:
        record = ReviewRecord(
            review_id=uuid.uuid4().hex,
            created_at=_now_utc(),
            **payload.model_dump(),
        )
        line = record.model_dump_json() + "\n"
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(
                self.path,
                os.O_WRONLY | os.O_CREAT | os.O_APPEND,
                0o644,
            )
            try:
                os.write(fd, line.encode("utf-8"))
                os.fsync(fd)
            finally:
                os.close(fd)
            self._records.append(record)
            self._latest[record.relation_id] = record
        return record

    # -------------------------------------------------------------- queries

    def all(self) -> List[ReviewRecord]:
        with self._lock:
            return list(self._records)

    def latest_by_relation(self) -> Dict[str, ReviewRecord]:
        with self._lock:
            return dict(self._latest)

    def latest_for(self, relation_id: str) -> Optional[ReviewRecord]:
        with self._lock:
            return self._latest.get(relation_id)

    def added_relations(self) -> List[ReviewRecord]:
        """User-authored relations whose latest decision is ``added``."""
        with self._lock:
            return [
                r for r in self._latest.values()
                if r.decision == ReviewDecision.ADDED
            ]
