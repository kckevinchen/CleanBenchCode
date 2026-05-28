"""Append-only JSONL store for column-clusters labels.

Mirrors :class:`src.labeling.multi_table_label_store.MultiTableLabelStore`
but keyed by ``(source, column)``. The labelling artifact for the
"inconsistency" error type lives at ``<run_dir>/column_clusters.jsonl``;
one JSON line per write, latest record per (source, column) wins at
read time. Crash-safe (every append fsyncs a single line), undo is
trivial (write a new record), and the JSONL retains the full audit
trail.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

from .schema import ColumnClusters

# Key type alias for ergonomic typing at call sites.
ColumnKey = Tuple[str, str]  # (source, column)

logger = logging.getLogger(__name__)


class ColumnClusterStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self._records: List[ColumnClusters] = []
        self._latest: Dict[ColumnKey, ColumnClusters] = {}
        self.reload()

    # ------------------------------------------------------------------ IO

    def reload(self) -> None:
        with self._lock:
            self._records = []
            self._latest = {}
            if not self.path.exists():
                return
            with self.path.open("r", encoding="utf-8") as fh:
                for line_no, line in enumerate(fh, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = ColumnClusters.model_validate_json(line)
                    except Exception as exc:  # noqa: BLE001 - skip malformed lines
                        # Crash-safe by design: a partial write or
                        # manual edit leaves a malformed line, and
                        # later writes still succeed. Log at DEBUG so
                        # operators can diagnose if needed without
                        # spamming production logs.
                        logger.debug(
                            "ColumnClusterStore: skipping malformed line "
                            "%d in %s (%s): %r",
                            line_no, self.path, exc, line[:200],
                        )
                        continue
                    self._records.append(record)
                    self._latest[(record.source, record.column)] = record

    def append(self, record: ColumnClusters) -> ColumnClusters:
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
            self._latest[(record.source, record.column)] = record
        return record

    # -------------------------------------------------------------- queries

    def all(self) -> List[ColumnClusters]:
        with self._lock:
            return list(self._records)

    def latest_by_column(self) -> Dict[ColumnKey, ColumnClusters]:
        """Reduce the JSONL to the latest record per (source, column)."""
        with self._lock:
            return dict(self._latest)

    def latest_for(self, source: str, column: str) -> Optional[ColumnClusters]:
        with self._lock:
            return self._latest.get((source, column))

    def history(self, source: str, column: str) -> List[ColumnClusters]:
        with self._lock:
            return [
                r for r in self._records
                if r.source == source and r.column == column
            ]

    def __iter__(self) -> Iterator[ColumnClusters]:
        return iter(self.all())
