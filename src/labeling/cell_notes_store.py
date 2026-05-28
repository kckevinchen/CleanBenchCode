"""Append-only JSONL store for per-cell note overrides.

Mirrors :class:`store.LabelStore` (one file, ``<run_dir>/cell_notes.jsonl``,
one record per line, fsync per write). Each record overlays the cluster
label's ``note`` for a single (source, row, column) cell at export time.

Latest record per (source, row, column) wins. A record whose ``note`` is
empty (after stripping) is a tombstone -- callers should treat it as
"no override", which makes the exporter fall back to the cluster note.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Dict, List, Tuple

from .schema import CellNote, CellNoteIn

# Key for the active map: (source, row, column).
CellKey = Tuple[str, int, str]


def _key(source: str, row: int, column: str) -> CellKey:
    return (str(source), int(row), str(column))


class CellNotesStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self._records: List[CellNote] = []
        # Latest record per (source, row, column).
        self._latest: Dict[CellKey, CellNote] = {}
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
                        rec = CellNote.model_validate_json(line)
                    except Exception:  # noqa: BLE001 - skip malformed
                        continue
                    self._records.append(rec)
                    self._latest[_key(rec.source, rec.row, rec.column)] = rec

    def _append_record(self, record: CellNote) -> CellNote:
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
            self._latest[_key(record.source, record.row, record.column)] = record
        return record

    # -------------------------------------------------------------- writes

    def upsert(self, payload: CellNoteIn, *, annotator: str) -> CellNote:
        """Persist a new per-cell note override.

        An empty ``note`` is recorded verbatim and acts as a tombstone:
        :meth:`active_overrides` drops it so the exporter falls back to
        the cluster's note.
        """
        record = CellNote(
            source=payload.source,
            row=int(payload.row),
            column=payload.column,
            note=payload.note or "",
            annotator=payload.annotator or annotator,
        )
        return self._append_record(record)

    # -------------------------------------------------------------- queries

    def latest_for(self, source: str, row: int, column: str) -> CellNote | None:
        """Return the latest record for a cell, or ``None`` if unset.

        Tombstones (empty note) are returned verbatim; callers that
        want the resolved-active note should use
        :meth:`active_overrides` instead.
        """
        with self._lock:
            return self._latest.get(_key(source, row, column))

    def active_overrides(self) -> Dict[CellKey, str]:
        """Return ``{(source, row, column): note}`` for every cell with
        a non-empty override. Tombstones are filtered out."""
        with self._lock:
            return {
                k: rec.note
                for k, rec in self._latest.items()
                if rec.note and rec.note.strip()
            }

    def all_records(self) -> List[CellNote]:
        """Every record on disk, in write order. Includes tombstones."""
        with self._lock:
            return list(self._records)
