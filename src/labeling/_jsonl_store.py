"""Shared base for the append-only JSONL label/decision stores.

Every store under ``src/labeling`` follows the same recipe: one file, one
JSON record per line, ``O_APPEND`` + ``fsync`` per write, an in-memory
``latest``-per-key index rebuilt on reload, and latest-wins semantics.
``JsonlStore`` factors that recipe out so the six concrete stores
(``LabelStore``, ``ColumnClusterStore``, ``CellNotesStore``,
``ExclusionStore``, ``MultiTableLabelStore``, ``RelationReviewStore``)
each shrink to: declare the record ``model``, implement ``_key`` (the
dedup key), and expose their domain-specific query/write *names* as thin
wrappers over the primitives here.

The base is also **self-healing against out-of-band writes**: offline CLIs
(``apply_benchmark_labels``, ``auto_label_*``, ...) or a second worker can
append to the same file while a server holds a store cached. Each read
checks the file's ``(mtime, size)`` and reloads if it changed underneath
us; our own appends refresh that signature so in-process writes never
trigger a redundant reload. (Previously only ``LabelStore`` and
``ColumnClusterStore`` had this; centralising it fixes the other four.)

Design notes for subclasses:

* Public method *names* deliberately live on the subclass (``append`` vs
  ``upsert`` vs ``add``; ``latest_by_target`` vs ``active`` vs ...), so
  call sites and tests are unchanged. They delegate to the protected
  primitives ``_append`` / ``_append_many`` / ``_all`` / ``_latest_map`` /
  ``_latest_get``.
* The read primitives self-heal (``_reload_if_changed``) before reading,
  so every subclass query sees out-of-band writes for free.
* Tombstone semantics (UNLABELED status, empty note, ``deleted=True``)
  stay in the subclass: the base only keeps the latest record per key;
  what "active" means is the subclass's call.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Dict, Generic, List, Optional, Tuple, TypeVar

from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)
K = TypeVar("K")


class JsonlStore(Generic[T, K]):
    #: pydantic record model; subclass sets this (e.g. ``model = Label``).
    model: type[T]

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self._records: List[T] = []
        self._latest: Dict[K, T] = {}
        # (mtime, size) of the file as of our last read/write. Lets us
        # detect an out-of-band append and reload without thrashing on
        # our own in-process writes.
        self._file_sig: Optional[Tuple[float, int]] = None
        self.reload()

    # ----------------------------------------------------- subclass hook

    def _key(self, record: T) -> K:
        """Extract the latest-wins dedup key from a record."""
        raise NotImplementedError

    # --------------------------------------------------------------- IO

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
                for line_no, raw in enumerate(fh, start=1):
                    line = raw.strip()
                    if not line:
                        continue
                    try:
                        record = self.model.model_validate_json(line)
                    except Exception as exc:  # noqa: BLE001 - skip malformed
                        # Crash-safe by design: a partial write or manual
                        # edit leaves a malformed line; later writes still
                        # succeed and we simply skip the bad line.
                        logger.debug(
                            "%s: skipping malformed line %d in %s (%s): %r",
                            type(self).__name__, line_no, self.path, exc,
                            line[:200],
                        )
                        continue
                    self._records.append(record)
                    self._latest[self._key(record)] = record
        self._file_sig = self._current_sig()

    def reload(self) -> None:
        """Re-read the JSONL from disk. Called on startup and on-demand."""
        with self._lock:
            self._reload_locked()

    def _reload_if_changed(self) -> None:
        """Reload iff the file changed underneath us since our last
        read/write (an out-of-band writer appended). Our own appends
        refresh ``_file_sig`` so they never trigger a reload."""
        with self._lock:
            if self._current_sig() != self._file_sig:
                self._reload_locked()

    def _write_lines_locked(self, payload: str) -> None:
        """Append ``payload`` (already newline-terminated) durably. Caller
        must hold ``self._lock``."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # O_APPEND guarantees atomic append on POSIX for writes <= PIPE_BUF.
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            os.write(fd, payload.encode("utf-8"))
            os.fsync(fd)
        finally:
            os.close(fd)

    def _append(self, record: T) -> T:
        """Persist ``record`` and update the in-memory index atomically."""
        line = record.model_dump_json() + "\n"
        with self._lock:
            self._write_lines_locked(line)
            self._records.append(record)
            self._latest[self._key(record)] = record
            self._file_sig = self._current_sig()
        return record

    def _append_many(self, records: List[T]) -> List[T]:
        """Persist ``records`` in a single fsync.

        Equivalent to calling :meth:`_append` per record but folds the
        write + fsync into one syscall pair, which matters on slow disks
        (~10 ms fsync each): a 30-record batch drops from ~300 ms to ~10 ms.
        """
        if not records:
            return []
        payload = "".join(r.model_dump_json() + "\n" for r in records)
        with self._lock:
            self._write_lines_locked(payload)
            for r in records:
                self._records.append(r)
                self._latest[self._key(r)] = r
            self._file_sig = self._current_sig()
        return list(records)

    # ----------------------------------------- read primitives (self-heal)

    def _all(self) -> List[T]:
        self._reload_if_changed()
        with self._lock:
            return list(self._records)

    def _latest_map(self) -> Dict[K, T]:
        self._reload_if_changed()
        with self._lock:
            return dict(self._latest)

    def _latest_get(self, key: K) -> Optional[T]:
        self._reload_if_changed()
        with self._lock:
            return self._latest.get(key)
