"""Append-only JSONL store for labels.

One file, ``<run_dir>/labels.jsonl``, one label record per line. All
writes are O_APPEND + ``fsync`` for durability; the store keeps an
in-memory index (``latest_by_target``) rebuilt on each reload.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Dict, Iterator, List, Optional

from .schema import Label


class LabelStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self._records: List[Label] = []
        self._latest: Dict[str, Label] = {}  # target_id -> latest Label
        self.reload()

    # ------------------------------------------------------------------ IO

    def reload(self) -> None:
        """Re-read the JSONL from disk. Called on startup and on-demand."""
        with self._lock:
            self._records = []
            self._latest = {}
            if not self.path.exists():
                return
            with self.path.open("r", encoding="utf-8") as fh:
                for line_no, line in enumerate(fh, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = Label.model_validate_json(line)
                    except Exception:  # noqa: BLE001 - skip malformed lines
                        continue
                    self._records.append(record)
                    self._latest[record.target_id] = record

    def append(self, label: Label) -> Label:
        """Persist ``label`` and update the in-memory index atomically."""
        line = label.model_dump_json() + "\n"
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # O_APPEND guarantees atomic append on POSIX for writes <= PIPE_BUF.
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
            self._latest[label.target_id] = label
        return label

    # -------------------------------------------------------------- queries

    def all(self) -> List[Label]:
        with self._lock:
            return list(self._records)

    def latest_by_target(self) -> Dict[str, Label]:
        """Map ``target_id`` to its most recent label (None if unlabelled)."""
        with self._lock:
            return dict(self._latest)

    def history(self, target_id: str) -> List[Label]:
        with self._lock:
            return [r for r in self._records if r.target_id == target_id]

    def latest_for(self, target_id: str) -> Optional[Label]:
        with self._lock:
            return self._latest.get(target_id)

    def __iter__(self) -> Iterator[Label]:
        return iter(self.all())
