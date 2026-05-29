"""Append-only JSONL store for exclusions.

One file, ``<run_dir>/exclusions.jsonl``, keyed by ``id`` with latest-wins.
Tracks per-source exclusion decisions made during labelling so a later
step can rebuild a sanitised dataset by removing whole CSVs, specific
rows, or specific columns the operator deemed too messy to label.

Soft-delete (undo) is a tombstone: a follow-up record with the same ``id``
and ``deleted=True`` retracts the original; :meth:`active` applies
latest-wins so callers see a consistent view without rewriting history.
See :mod:`src.labeling._jsonl_store` for the shared IO/index/self-healing.
"""

from __future__ import annotations

import uuid
from typing import Iterator, List, Optional

from ._jsonl_store import JsonlStore
from .schema import Exclusion, ExclusionIn


class ExclusionStore(JsonlStore[Exclusion, str]):
    model = Exclusion

    def _key(self, record: Exclusion) -> str:
        return record.id

    # -------------------------------------------------------------- writes

    def add(self, payload: ExclusionIn, *, annotator: str) -> Exclusion:
        """Persist a new exclusion. Returns the saved record (with id/ts)."""
        record = Exclusion(
            id=uuid.uuid4().hex,
            annotator=payload.annotator or annotator,
            scope=payload.scope,
            source=payload.source,
            rows=list(payload.rows or []),
            columns=list(payload.columns or []),
            note=payload.note,
            deleted=False,
        )
        return self._append(record)

    def remove(self, exclusion_id: str, *, annotator: str) -> Optional[Exclusion]:
        """Soft-delete an exclusion by id. Returns the tombstone record,
        or ``None`` if no active record matches."""
        existing = self._latest_get(exclusion_id)
        if existing is None or existing.deleted:
            return None
        tomb = existing.model_copy(update={
            "deleted": True,
            "annotator": annotator,
            "ts": Exclusion(
                id=exclusion_id,
                annotator=annotator,
                scope=existing.scope,
                source=existing.source,
            ).ts,
        })
        return self._append(tomb)

    # -------------------------------------------------------------- queries

    def all_records(self) -> List[Exclusion]:
        """Every record on disk, including tombstones, in write order."""
        return self._all()

    def active(self) -> List[Exclusion]:
        """Currently-active exclusions (latest record per id, tombstones
        filtered out). Stable order: by ts asc."""
        actives = [r for r in self._latest_map().values() if not r.deleted]
        return sorted(actives, key=lambda r: r.ts)

    def active_for_source(self, source: str) -> List[Exclusion]:
        return [r for r in self.active() if r.source == source]

    def __iter__(self) -> Iterator[Exclusion]:
        return iter(self.active())
