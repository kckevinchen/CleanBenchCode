"""Append-only JSONL store for multi-table inconsistency labels.

Mirrors :class:`src.labeling.store.LabelStore` for the cluster/cell
vocabulary, but keyed by ``violation_id`` so all linked rows of a
single finding (duplicate_row, attribute_conflict, surface_drift, ...)
collapse to one verdict. ``<run_dir>/multi_table_labels.jsonl`` holds
one record per write; the latest record per ``violation_id`` wins at
read time.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Dict, Iterator, List, Optional

from .schema import MultiTableLabel


class MultiTableLabelStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self._records: List[MultiTableLabel] = []
        self._latest: Dict[str, MultiTableLabel] = {}
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
                        record = MultiTableLabel.model_validate_json(line)
                    except Exception:  # noqa: BLE001 - skip malformed lines
                        continue
                    self._records.append(record)
                    self._latest[record.violation_id] = record

    def append(self, label: MultiTableLabel) -> MultiTableLabel:
        line = label.model_dump_json() + "\n"
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
            self._records.append(label)
            self._latest[label.violation_id] = label
        return label

    # -------------------------------------------------------------- queries

    def all(self) -> List[MultiTableLabel]:
        with self._lock:
            return list(self._records)

    def latest_by_violation(self) -> Dict[str, MultiTableLabel]:
        with self._lock:
            return dict(self._latest)

    def latest_for(self, violation_id: str) -> Optional[MultiTableLabel]:
        with self._lock:
            return self._latest.get(violation_id)

    def history(self, violation_id: str) -> List[MultiTableLabel]:
        with self._lock:
            return [r for r in self._records if r.violation_id == violation_id]

    def __iter__(self) -> Iterator[MultiTableLabel]:
        return iter(self.all())
