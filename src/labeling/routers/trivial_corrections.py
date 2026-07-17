"""Async LLM correction jobs for trivially-tagged cells marked PARTIAL.

When the operator converts trivially-tagged cells to ``partial`` (via
``convert-trivial-to-skip`` with ``status=partial``), they can ask an LLM to
generate a better ``corrected_value`` for each cell. Because a batch can be
hundreds of cells and each is one (slow) LLM call, the work runs as a
background job with progress polling, mirroring
``routers/correction_jobs.py``. The actual logic lives in
:mod:`src.labeling.trivial_corrector`.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from fastapi import Body, FastAPI, HTTPException

from ...core.llm_client import LLMClient
from .. import trivial_corrector
from ._context import RouterContext

logger = logging.getLogger(__name__)


# Factory for the LLM client, indirected through a module attribute so tests
# can inject a stub without a live Azure service / ``az login``.
def _default_llm_factory() -> LLMClient:
    return LLMClient()


_llm_factory: Callable[[], Any] = _default_llm_factory


@dataclass
class _CorrectionJob:
    job_id: str
    run_id: str
    total: int = 0
    done: int = 0
    status: str = "queued"            # queued | running | done | error
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    error: Optional[str] = None
    summary: Dict[str, int] = field(default_factory=dict)
    results: List[Dict[str, Any]] = field(default_factory=list)


class _JobRegistry:
    def __init__(self, max_workers: int = 2) -> None:
        self._lock = threading.Lock()
        self._jobs: Dict[str, _CorrectionJob] = {}
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="trivial-fix-job",
        )

    def create(self, run_id: str, total: int) -> _CorrectionJob:
        job = _CorrectionJob(job_id=uuid.uuid4().hex[:16], run_id=run_id, total=total)
        with self._lock:
            self._jobs[job.job_id] = job
        return job

    def get(self, job_id: str) -> Optional[_CorrectionJob]:
        with self._lock:
            return self._jobs.get(job_id)

    def submit(self, job: _CorrectionJob, fn) -> None:
        self._executor.submit(fn, job)


_jobs = _JobRegistry()


def register(app: FastAPI, router_ctx: RouterContext) -> None:
    cache = router_ctx.cache
    annotator = router_ctx.annotator

    @app.post("/api/runs/{run_id}/labels/trivial-partial-corrections")
    def start_trivial_partial_corrections(
        run_id: str, payload: Dict[str, Any] = Body(default_factory=dict),
    ) -> Dict[str, Any]:
        """Start a background job that uses an LLM to (re)generate the
        ``corrected_value`` of every TRIVIAL-noted ``partial`` cell.

        Targets are labels at ``status=partial`` whose note carries ``TRIVIAL``
        and does NOT yet carry the ``[LLM-fix]`` marker (so re-runs are
        idempotent). Each is re-appended as a ``partial`` label with the LLM's
        fix and ``[LLM-fix]`` added to its note.

        Payload (all optional):
          * ``source`` - restrict to one CSV.
          * ``dry_run`` - return the target list/count WITHOUT calling the LLM.
          * ``model`` / ``reasoning_effort`` - LLM knobs.

        Returns ``{job_id, total_targets, status}``; poll
        ``GET .../trivial-partial-corrections/{job_id}``. When there are no
        targets, returns ``job_id=None`` and ``status="done"``. A ``dry_run``
        returns ``{dry_run, total_targets, targets}`` and starts no job.
        """
        ctx = cache.get(run_id)
        source_filter = payload.get("source")
        if source_filter is not None:
            if not isinstance(source_filter, str):
                raise HTTPException(400, "source filter must be a string")
            if source_filter not in ctx.sources:
                raise HTTPException(404, f"source not found: {source_filter}")
        model = str(payload.get("model", trivial_corrector.DEFAULT_MODEL))
        reasoning_effort = str(
            payload.get("reasoning_effort", trivial_corrector.DEFAULT_REASONING_EFFORT)
        )

        targets = trivial_corrector.collect_partial_trivial_targets(
            ctx, source_filter=source_filter,
        )

        if bool(payload.get("dry_run")):
            return {
                "dry_run": True,
                "total_targets": len(targets),
                "targets": [
                    {
                        "target_id": t.target_id, "target": t.target,
                        "source": t.source, "row": t.row, "column": t.column,
                        "original_value": t.original_value,
                    }
                    for t in targets[:200]
                ],
            }

        if not targets:
            return {"job_id": None, "total_targets": 0, "status": "done"}

        job = _jobs.create(run_id, len(targets))

        def _run(job: _CorrectionJob) -> None:
            job.status = "running"
            job.started_at = time.time()

            def _progress(done: int, total: int) -> None:
                job.done = done
                job.total = total

            try:
                llm_client = _llm_factory()
            except Exception as exc:  # noqa: BLE001
                job.status = "error"
                job.error = f"LLM client unavailable: {exc}"
                job.finished_at = time.time()
                return
            try:
                results = trivial_corrector.generate_corrections(
                    ctx, targets, llm_client=llm_client, model=model,
                    reasoning_effort=reasoning_effort, progress_cb=_progress,
                )
                job.summary = trivial_corrector.write_corrections(
                    ctx, results, annotator=annotator,
                )
                job.results = [
                    {
                        "source": r.source, "row": r.row, "column": r.column,
                        "original_value": r.original_value,
                        "corrected_value": r.corrected_value,
                        "changed": r.changed, "error": r.error,
                    }
                    for r in results
                ]
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "trivial-partial correction job failed run=%s job=%s",
                    job.run_id, job.job_id,
                )
                job.status = "error"
                job.error = str(exc)
                job.finished_at = time.time()
                return
            job.status = "done"
            job.finished_at = time.time()

        _jobs.submit(job, _run)
        return {
            "job_id": job.job_id,
            "total_targets": len(targets),
            "status": job.status,
        }

    @app.get("/api/runs/{run_id}/trivial-partial-corrections/{job_id}")
    def get_trivial_partial_correction_job(
        run_id: str, job_id: str,
    ) -> Dict[str, Any]:
        """Poll a trivial-partial correction job. ``summary`` / ``results`` are
        populated once ``status == "done"``."""
        job = _jobs.get(job_id)
        if job is None or job.run_id != run_id:
            raise HTTPException(404, f"correction job not found: {job_id}")
        return {
            "job_id": job.job_id,
            "run_id": job.run_id,
            "status": job.status,
            "done": job.done,
            "total": job.total,
            "started_at": job.started_at,
            "finished_at": job.finished_at,
            "error": job.error,
            "summary": job.summary if job.status == "done" else {},
            "results": job.results if job.status == "done" else [],
        }
