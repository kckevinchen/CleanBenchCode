"""Append-only JSONL store for per-cell corrected-value overrides.

One file, ``<run_dir>/cell_notes.jsonl``, keyed by (source, row, column)
with latest-wins. The ``note`` field carries the operator's literal
per-cell FIX: at export time it supersedes the cluster's
``corrected_value`` for that single ERROR cell, so each affected row of a
fan-out cluster can get its own correction (see
:func:`src.labeling.export.materialize_label_benchmark`). A record whose
``note`` is empty (after stripping) is a tombstone -- :meth:`active_overrides`
drops it so the exporter falls back to the cluster fix. The ``note`` name
is retained for storage/endpoint back-compat; per-cell *notes* are no
longer a feature. See :mod:`src.labeling._jsonl_store` for the shared
IO/index/self-healing.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from ._jsonl_store import JsonlStore
from .schema import CellNote, CellNoteIn

# Key for the active map: (source, row, column).
CellKey = Tuple[str, int, str]


def _cell_key(source: str, row: int, column: str) -> CellKey:
    return (str(source), int(row), str(column))


class CellNotesStore(JsonlStore[CellNote, CellKey]):
    model = CellNote

    def _key(self, record: CellNote) -> CellKey:
        return _cell_key(record.source, record.row, record.column)

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
        return self._append(record)

    # -------------------------------------------------------------- queries

    def latest_for(self, source: str, row: int, column: str) -> CellNote | None:
        """Return the latest record for a cell, or ``None`` if unset.

        Tombstones (empty note) are returned verbatim; callers that want
        the resolved-active note should use :meth:`active_overrides`.
        """
        return self._latest_get(_cell_key(source, row, column))

    def active_overrides(self) -> Dict[CellKey, str]:
        """Return ``{(source, row, column): note}`` for every cell with a
        non-empty override. Tombstones are filtered out."""
        return {
            k: rec.note
            for k, rec in self._latest_map().items()
            if rec.note and rec.note.strip()
        }

    def all_records(self) -> List[CellNote]:
        """Every record on disk, in write order. Includes tombstones."""
        return self._all()
