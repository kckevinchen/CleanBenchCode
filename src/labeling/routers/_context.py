"""Shared state container passed to every router module's ``register()``.

The labelling workbench is a single-tenant FastAPI app: every endpoint
needs the same handful of project-scoped objects (caches, registries,
the job manager, the resolved outputs root). Wrapping them in this
dataclass lets each router module take one argument instead of seven.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Tuple

from fastapi import HTTPException


@dataclass
class RouterContext:
    project_root: Path
    outputs_dir: Path
    static_dir: Path
    annotator: str
    registry: Any  # RunRegistry - avoids circular import
    jobs: Any  # JobManager - avoids circular import
    cache: Any  # RunCache - avoids circular import

    _diff_lock: threading.Lock = field(default_factory=threading.Lock)
    _diff_cache: Dict[Tuple[Any, ...], Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Shared helpers (formerly closures in ``create_app``).
    # ------------------------------------------------------------------

    def safe_run_dir(self, run_id: str) -> Path:
        """Resolve ``outputs_dir / run_id``, rejecting traversal attempts.

        Symlinked run dirs whose target escapes the outputs root are
        rejected so ``shutil.rmtree`` and friends cannot follow them
        outside the workbench.
        """
        if (
            not run_id
            or run_id in (".", "..")
            or "/" in run_id
            or "\\" in run_id
            or "\x00" in run_id
        ):
            raise HTTPException(400, f"invalid run_id: {run_id!r}")
        outputs_root = self.outputs_dir.resolve()
        rd = (self.outputs_dir / run_id).resolve()
        try:
            rd.relative_to(outputs_root)
        except ValueError as exc:
            raise HTTPException(400, f"invalid run_id: {run_id!r}") from exc
        if rd == outputs_root or rd.parent != outputs_root:
            raise HTTPException(400, f"invalid run_id: {run_id!r}")
        return rd

    def run_dir_or_404(self, run_id: str) -> Path:
        rd = self.safe_run_dir(run_id)
        if not rd.is_dir():
            raise HTTPException(404, f"run not found: {run_id}")
        return rd

    def get_diff(self, run_a: str, run_b: str) -> Dict[str, Any]:
        """Memoised ``diff_runs(a, b)``; key includes both issues.jsonl mtimes.

        Bounded to 8 cached pair-results so we don't accumulate stale
        diffs as the operator clicks through run-comparison pairs.
        """
        from experiments.compare_runs import diff_runs

        a_dir = self.run_dir_or_404(run_a)
        b_dir = self.run_dir_or_404(run_b)
        try:
            mtime_a = (a_dir / "issues.jsonl").stat().st_mtime
            mtime_b = (b_dir / "issues.jsonl").stat().st_mtime
        except FileNotFoundError as exc:
            raise HTTPException(404, f"missing issues.jsonl: {exc}") from exc
        key = (run_a, run_b, mtime_a, mtime_b)
        with self._diff_lock:
            cached = self._diff_cache.get(key)
            if cached is not None:
                return cached
            if len(self._diff_cache) >= 8:
                self._diff_cache.clear()
        result = diff_runs(a_dir, b_dir)
        with self._diff_lock:
            self._diff_cache[key] = result
        return result
