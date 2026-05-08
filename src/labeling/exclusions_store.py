"""Append-only JSONL store for exclusions.

Mirrors :class:`store.LabelStore` (one file, ``<run_dir>/exclusions.jsonl``,
one record per line, fsync per write). The store tracks per-source
exclusion decisions made during labelling so a later step can rebuild a
sanitised dataset by removing whole CSVs, specific rows, or specific
columns the operator deemed too messy to label.

Soft-delete (undo) is implemented via tombstone records: a follow-up
record with the same ``id`` and ``deleted=True`` retracts the original.
The reduce-to-active step here applies latest-wins so callers see a
consistent view without rewriting history.
"""

from __future__ import annotations

import os
import threading
import uuid
from pathlib import Path
from typing import Dict, Iterator, List, Optional

from .schema import Exclusion, ExclusionIn


class ExclusionStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self._records: List[Exclusion] = []
        # id -> latest record (active OR tombstoned). Active set is
        # derived by filtering out the tombstones at query time.
        self._latest: Dict[str, Exclusion] = {}
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
                        record = Exclusion.model_validate_json(line)
                    except Exception:  # noqa: BLE001 - skip malformed lines
                        continue
                    self._records.append(record)
                    self._latest[record.id] = record

    def _append_record(self, record: Exclusion) -> Exclusion:
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
            self._latest[record.id] = record
        return record

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
        return self._append_record(record)

    def remove(self, exclusion_id: str, *, annotator: str) -> Optional[Exclusion]:
        """Soft-delete an exclusion by id. Returns the tombstone record,
        or ``None`` if no active record matches."""
        with self._lock:
            existing = self._latest.get(exclusion_id)
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
        return self._append_record(tomb)

    # -------------------------------------------------------------- queries

    def all_records(self) -> List[Exclusion]:
        """Every record on disk, including tombstones, in write order."""
        with self._lock:
            return list(self._records)

    def active(self) -> List[Exclusion]:
        """Currently-active exclusions (latest record per id, tombstones
        filtered out). Stable order: by ts asc."""
        with self._lock:
            actives = [r for r in self._latest.values() if not r.deleted]
        return sorted(actives, key=lambda r: r.ts)

    def active_for_source(self, source: str) -> List[Exclusion]:
        return [r for r in self.active() if r.source == source]

    def __iter__(self) -> Iterator[Exclusion]:
        return iter(self.active())
