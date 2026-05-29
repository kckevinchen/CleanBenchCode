"""Append-only JSONL store for column-clusters labels.

Mirrors :class:`src.labeling.multi_table_label_store.MultiTableLabelStore`
but keyed by ``(source, column)``. The labelling artifact for the
"inconsistency" error type lives at ``<run_dir>/column_clusters.jsonl``;
one JSON line per write, latest record per (source, column) wins at read
time. See :mod:`src.labeling._jsonl_store` for the shared crash-safe IO,
index, and out-of-band-write self-healing.
"""

from __future__ import annotations

from typing import Dict, Iterator, List, Optional, Tuple

from ._jsonl_store import JsonlStore
from .schema import ColumnClusters

# Key type alias for ergonomic typing at call sites.
ColumnKey = Tuple[str, str]  # (source, column)


class ColumnClusterStore(JsonlStore[ColumnClusters, ColumnKey]):
    model = ColumnClusters

    def _key(self, record: ColumnClusters) -> ColumnKey:
        return (record.source, record.column)

    # -------------------------------------------------------------- writes

    def append(self, record: ColumnClusters) -> ColumnClusters:
        return self._append(record)

    # -------------------------------------------------------------- queries

    def all(self) -> List[ColumnClusters]:
        return self._all()

    def latest_by_column(self) -> Dict[ColumnKey, ColumnClusters]:
        """Reduce the JSONL to the latest record per (source, column)."""
        return self._latest_map()

    def latest_for(self, source: str, column: str) -> Optional[ColumnClusters]:
        return self._latest_get((source, column))

    def history(self, source: str, column: str) -> List[ColumnClusters]:
        return [
            r for r in self._all()
            if r.source == source and r.column == column
        ]

    def __iter__(self) -> Iterator[ColumnClusters]:
        return iter(self.all())
