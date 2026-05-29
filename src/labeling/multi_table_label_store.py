"""Append-only JSONL store for multi-table inconsistency labels.

Mirrors :class:`src.labeling.store.LabelStore` for the cluster/cell
vocabulary, but keyed by ``violation_id`` so all linked rows of a single
finding (duplicate_row, attribute_conflict, surface_drift, ...) collapse
to one verdict. ``<run_dir>/multi_table_labels.jsonl`` holds one record
per write; the latest record per ``violation_id`` wins at read time. See
:mod:`src.labeling._jsonl_store` for the shared IO/index/self-healing.
"""

from __future__ import annotations

from typing import Dict, Iterator, List, Optional

from ._jsonl_store import JsonlStore
from .schema import MultiTableLabel


class MultiTableLabelStore(JsonlStore[MultiTableLabel, str]):
    model = MultiTableLabel

    def _key(self, record: MultiTableLabel) -> str:
        return record.violation_id

    # -------------------------------------------------------------- writes

    def append(self, label: MultiTableLabel) -> MultiTableLabel:
        return self._append(label)

    # -------------------------------------------------------------- queries

    def all(self) -> List[MultiTableLabel]:
        return self._all()

    def latest_by_violation(self) -> Dict[str, MultiTableLabel]:
        return self._latest_map()

    def latest_for(self, violation_id: str) -> Optional[MultiTableLabel]:
        return self._latest_get(violation_id)

    def history(self, violation_id: str) -> List[MultiTableLabel]:
        return [r for r in self._all() if r.violation_id == violation_id]

    def __iter__(self) -> Iterator[MultiTableLabel]:
        return iter(self.all())
