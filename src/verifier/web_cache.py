"""Disk-backed cache for web-search responses keyed by query string.

We make the same query many times during a labelling session (e.g. the
operator clicks Refresh, or the same typo appears in two clusters).
Caching is the difference between "free" and "expensive" verification.

Storage layout: a single directory of ``<sha256(query)>.json`` files
plus a sidecar ``index.json`` for human-readable inspection. Each file
records the provider name, ts, and the raw provider payload. Pruning
is left to the caller (delete the directory to wipe).

Thread-safe within a process via a single lock; safe across processes
via atomic ``os.replace`` writes.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

logger = logging.getLogger(__name__)


def _hash_key(provider: str, query: str, **extra: Any) -> str:
    """Deterministic SHA-256 of (provider, query, extra) for cache filename."""
    payload = json.dumps(
        {"provider": provider, "query": query, "extra": extra},
        sort_keys=True, ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class WebCache:
    """Filesystem cache for web-search responses.

    Use ``ttl_seconds=0`` (default) to keep entries forever; set a
    positive value to expire stale entries on read.
    """

    root: Path
    ttl_seconds: int = 0
    _lock: threading.Lock = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    # --- internal --------------------------------------------------

    def _path(self, key: str) -> Path:
        return self.root / f"{key}.json"

    # --- public ----------------------------------------------------

    def get(
        self, provider: str, query: str, **extra: Any,
    ) -> Optional[Mapping[str, Any]]:
        """Return cached payload or ``None``. Honours ``ttl_seconds``."""
        key = _hash_key(provider, query, **extra)
        path = self._path(key)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("web cache read failed for %s: %s", path.name, exc)
            return None
        if self.ttl_seconds > 0:
            ts = float(data.get("ts") or 0)
            if ts and (time.time() - ts) > self.ttl_seconds:
                return None
        return data.get("payload")

    def put(
        self, provider: str, query: str, payload: Mapping[str, Any],
        **extra: Any,
    ) -> None:
        """Atomically store ``payload`` under ``(provider, query, extra)``."""
        key = _hash_key(provider, query, **extra)
        path = self._path(key)
        record = {
            "provider": provider,
            "query": query,
            "extra": extra,
            "ts": time.time(),
            "payload": payload,
        }
        tmp = path.with_suffix(".json.tmp")
        with self._lock:
            tmp.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, path)


__all__ = ["WebCache"]
