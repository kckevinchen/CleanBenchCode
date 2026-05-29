"""Append-only JSONL store for labels.

One file, ``<run_dir>/labels.jsonl``, one label record per line, keyed by
``target_id`` (latest wins). See :mod:`src.labeling._jsonl_store` for the
shared IO/index/self-healing machinery; this module only adds the
label-specific vocabulary: ``UNLABELED`` tombstones (filtered from the
"current" views but retained in history) and the batched
:meth:`append_many`.
"""

from __future__ import annotations

from typing import Dict, Iterator, List, Optional

from ._jsonl_store import JsonlStore
from .schema import Label, LabelStatus


class LabelStore(JsonlStore[Label, str]):
    model = Label

    def _key(self, record: Label) -> str:
        return record.target_id

    # -------------------------------------------------------------- writes

    def append(self, label: Label) -> Label:
        return self._append(label)

    def append_many(self, labels: List[Label]) -> List[Label]:
        return self._append_many(labels)

    # -------------------------------------------------------------- queries

    def all(self) -> List[Label]:
        return self._all()

    def latest_by_target(self) -> Dict[str, Label]:
        """Map ``target_id`` to its most recent label (None if unlabelled).

        ``LabelStatus.UNLABELED`` records are tombstones written when an
        auto-skip-from-exclusion is reverted; they are filtered out here
        so downstream consumers see "no label" for the target without
        needing to know about the unlabel mechanism. ``history`` / ``all``
        still expose them for the audit trail.
        """
        return {
            tid: rec for tid, rec in self._latest_map().items()
            if rec.status != LabelStatus.UNLABELED
        }

    def history(self, target_id: str) -> List[Label]:
        return [r for r in self._all() if r.target_id == target_id]

    def latest_for(self, target_id: str) -> Optional[Label]:
        rec = self._latest_get(target_id)
        if rec is not None and rec.status == LabelStatus.UNLABELED:
            return None
        return rec

    def __iter__(self) -> Iterator[Label]:
        return iter(self.all())
