"""Append-only JSONL store for human reviews of discovered relations.

One file, ``<run_dir>/relation_reviews.jsonl``, keyed by ``relation_id``
with latest-wins. See :mod:`src.labeling._jsonl_store` for the shared
IO/index/self-healing.

Four decisions are supported:

* ``approved``  - reviewer agrees with a discovery-emitted relation
* ``rejected``  - reviewer rejects a discovery-emitted relation
* ``added``     - reviewer authored a brand-new relation that was not
                  emitted by the LLM. ``relation`` carries the full
                  snapshot.
* ``cleared``   - undo a prior approve/reject without losing history.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from ._jsonl_store import JsonlStore


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


class RelationReviewStore(JsonlStore[ReviewRecord, str]):
    model = ReviewRecord

    def _key(self, record: ReviewRecord) -> str:
        return record.relation_id

    # -------------------------------------------------------------- writes

    def append(self, payload: ReviewIn) -> ReviewRecord:
        record = ReviewRecord(
            review_id=uuid.uuid4().hex,
            created_at=_now_utc(),
            **payload.model_dump(),
        )
        return self._append(record)

    # -------------------------------------------------------------- queries

    def all(self) -> List[ReviewRecord]:
        return self._all()

    def latest_by_relation(self) -> Dict[str, ReviewRecord]:
        return self._latest_map()

    def latest_for(self, relation_id: str) -> Optional[ReviewRecord]:
        return self._latest_get(relation_id)

    def added_relations(self) -> List[ReviewRecord]:
        """User-authored relations whose latest decision is ``added``."""
        return [
            r for r in self._latest_map().values()
            if r.decision == ReviewDecision.ADDED
        ]
