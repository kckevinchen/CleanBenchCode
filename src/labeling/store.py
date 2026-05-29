"""Append-only JSONL store for labels.

One file, ``<run_dir>/labels.jsonl``, one label record per line. All
writes are O_APPEND + ``fsync`` for durability; the store keeps an
in-memory index (``latest_by_target``) rebuilt on each reload.

The store is self-healing against *out-of-band* writes: offline CLIs
(``apply_benchmark_labels``, ``auto_label_*``, ...) or a second worker
can append to ``labels.jsonl`` while a server holds this store cached.
Each read checks the file's ``(mtime, size)`` and reloads if it changed
underneath us. Our own appends update that signature, so in-process
writes never trigger a redundant reload.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

from .schema import Label, LabelStatus


class LabelStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self._records: List[Label] = []
        self._latest: Dict[str, Label] = {}  # target_id -> latest Label
        # (mtime, size) of the file as of our last read/write. Lets us
        # detect an out-of-band append and reload, without thrashing on
        # our own in-process writes.
        self._file_sig: Optional[Tuple[float, int]] = None
        self.reload()

    # ------------------------------------------------------------------ IO

    def _current_sig(self) -> Optional[Tuple[float, int]]:
        try:
            st = self.path.stat()
            return (st.st_mtime, st.st_size)
        except OSError:
            return None

    def _reload_locked(self) -> None:
        """Re-read the JSONL. Caller must hold ``self._lock``."""
        self._records = []
        self._latest = {}
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = Label.model_validate_json(line)
                    except Exception:  # noqa: BLE001 - skip malformed lines
                        continue
                    self._records.append(record)
                    self._latest[record.target_id] = record
        self._file_sig = self._current_sig()

    def reload(self) -> None:
        """Re-read the JSONL from disk. Called on startup and on-demand."""
        with self._lock:
            self._reload_locked()

    def _reload_if_changed(self) -> None:
        """Reload iff the file changed underneath us since our last
        read/write (i.e. an out-of-band writer appended). Our own
        appends refresh ``_file_sig`` so they never trigger a reload."""
        with self._lock:
            if self._current_sig() != self._file_sig:
                self._reload_locked()

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
            self._file_sig = self._current_sig()
        return label

    def append_many(self, labels: List[Label]) -> List[Label]:
        """Persist ``labels`` in a single fsync.

        Equivalent to calling :meth:`append` per record but folds the
        write + fsync into one syscall pair, which matters on slow
        disks (~10 ms fsync each on the data volume): a 30-cell
        propagation drops from ~300 ms to ~10 ms.
        """
        if not labels:
            return []
        payload = "".join(lab.model_dump_json() + "\n" for lab in labels)
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(
                self.path,
                os.O_WRONLY | os.O_CREAT | os.O_APPEND,
                0o644,
            )
            try:
                os.write(fd, payload.encode("utf-8"))
                os.fsync(fd)
            finally:
                os.close(fd)
            for lab in labels:
                self._records.append(lab)
                self._latest[lab.target_id] = lab
            self._file_sig = self._current_sig()
        return list(labels)

    # -------------------------------------------------------------- queries

    def all(self) -> List[Label]:
        self._reload_if_changed()
        with self._lock:
            return list(self._records)

    def latest_by_target(self) -> Dict[str, Label]:
        """Map ``target_id`` to its most recent label (None if unlabelled).

        ``LabelStatus.UNLABELED`` records are tombstones written when an
        auto-skip-from-exclusion is reverted; they are filtered out
        here so downstream consumers see "no label" for the target
        without needing to know about the unlabel mechanism.
        ``history`` / ``all`` still expose them for the audit trail.
        """
        self._reload_if_changed()
        with self._lock:
            return {
                tid: rec for tid, rec in self._latest.items()
                if rec.status != LabelStatus.UNLABELED
            }

    def history(self, target_id: str) -> List[Label]:
        self._reload_if_changed()
        with self._lock:
            return [r for r in self._records if r.target_id == target_id]

    def latest_for(self, target_id: str) -> Optional[Label]:
        self._reload_if_changed()
        with self._lock:
            rec = self._latest.get(target_id)
            if rec is not None and rec.status == LabelStatus.UNLABELED:
                return None
            return rec

    def __iter__(self) -> Iterator[Label]:
        return iter(self.all())
