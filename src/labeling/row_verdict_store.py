"""Append-only JSONL store for per-(source, row, column) verdict overrides.

One file, ``<run_dir>/row_verdicts.jsonl``, keyed by (source, row, column)
with latest-wins. Each record overrides the enclosing cluster's verdict
for a single cell -- this is how an operator labels ONE row of an
``fd_violation`` conflict group independently of the cluster's verdict
(which otherwise fans out identically to every conflicting row). The
override wins for its keyed cell at export time and in the viewer.

A record whose ``status`` is ``unlabeled`` is a tombstone (= "revert this
cell to the cluster's fanned-out default"); :meth:`active_overrides`
drops it so the exporter / viewer fall back to the cluster verdict. See
:mod:`src.labeling._jsonl_store` for the shared IO/index/self-healing.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from ._jsonl_store import JsonlStore
from .schema import LabelStatus, RowVerdict, RowVerdictIn

# Key for the active map: (source, row, column).
RowKey = Tuple[str, int, str]


def _row_key(source: str, row: int, column: str) -> RowKey:
    return (str(source), int(row), str(column))


class RowVerdictStore(JsonlStore[RowVerdict, RowKey]):
    model = RowVerdict

    def _key(self, record: RowVerdict) -> RowKey:
        return _row_key(record.source, record.row, record.column)

    # -------------------------------------------------------------- writes

    def upsert(self, payload: RowVerdictIn, *, annotator: str) -> RowVerdict:
        """Persist a new per-row verdict override.

        A ``status`` of ``unlabeled`` is recorded verbatim and acts as a
        tombstone: :meth:`active_overrides` drops it so the exporter and
        viewer fall back to the cluster's fanned-out verdict.
        """
        record = RowVerdict(
            source=payload.source,
            row=int(payload.row),
            column=payload.column,
            status=payload.status,
            corrected_value=payload.corrected_value,
            note=payload.note or "",
            annotator=payload.annotator or annotator,
        )
        return self._append(record)

    # -------------------------------------------------------------- queries

    def latest_for(self, source: str, row: int, column: str) -> RowVerdict | None:
        """Return the latest record for a cell, or ``None`` if unset.

        Tombstones (``unlabeled``) are returned verbatim; callers that
        want the resolved-active overrides should use
        :meth:`active_overrides`.
        """
        return self._latest_get(_row_key(source, row, column))

    def active_overrides(self) -> Dict[RowKey, RowVerdict]:
        """Return ``{(source, row, column): RowVerdict}`` for every cell
        with a non-tombstone override. ``unlabeled`` records are filtered
        out so the cluster default wins again."""
        return {
            k: rec
            for k, rec in self._latest_map().items()
            if rec.status != LabelStatus.UNLABELED
        }

    def all_records(self) -> List[RowVerdict]:
        """Every record on disk, in write order. Includes tombstones."""
        return self._all()
