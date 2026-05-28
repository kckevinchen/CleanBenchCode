"""Async batch correction-suggestion jobs.

The operator can kick off LLM correction-suggestion runs over every
TP/unsure column-cluster verdict in a run; this owns the job registry
(a process-global, bounded-concurrency executor) and the start/poll
endpoints. The actual suggestion work lives in
:mod:`src.labeling.correction_suggester`.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from fastapi import Body, FastAPI, HTTPException

from .. import correction_suggester
from ...core.llm_client import LLMClient
from ..schema import LabelStatus, LabelTarget
from ._context import RouterContext

logger = logging.getLogger(__name__)


@dataclass
class _CorrectionJob:
    """State for one async batch correction job.

    ``status`` transitions: ``queued`` -> ``running`` -> ``done`` |
    ``error``. ``progress`` is updated by the dispatch progress callback
    so the UI can render a "12 / 87 LLM calls done" hint.
    """

    job_id: str
    run_id: str
    targets: List[Tuple[str, str, str]] = field(default_factory=list)
    status: str = "queued"            # queued | running | done | error
    phase: str = ""                    # "phase A" / "phase B" / ""
    done: int = 0
    total: int = 0
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    error: Optional[str] = None
    # Per-cluster suggestion payload, ready for the UI. Keyed by
    # ``f"{source}\x00{column}\x00{cluster_local_id}"`` for JSON-safe
    # transport (tuple keys can't be serialised).
    results: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)


class _CorrectionJobRegistry:
    """Thread-safe registry + bounded executor for correction jobs.

    Concurrency is intentionally low (max_workers=2): the underlying
    ``dispatch_groups`` already fans out per-LLM-call concurrency
    internally, so two simultaneous batch jobs would otherwise hammer
    the LLM endpoint.
    """

    def __init__(self, max_workers: int = 2) -> None:
        self._lock = threading.Lock()
        self._jobs: Dict[str, _CorrectionJob] = {}
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="correction-job",
        )

    def create(
        self, run_id: str, targets: List[Tuple[str, str, str]],
    ) -> _CorrectionJob:
        job = _CorrectionJob(
            job_id=uuid.uuid4().hex[:16],
            run_id=run_id,
            targets=list(targets),
        )
        with self._lock:
            self._jobs[job.job_id] = job
        return job

    def get(self, job_id: str) -> Optional[_CorrectionJob]:
        with self._lock:
            return self._jobs.get(job_id)

    def submit(
        self, job: _CorrectionJob, fn,
    ) -> None:
        """Enqueue ``fn(job)`` on the background executor."""
        self._executor.submit(fn, job)


# Single registry per process. ``create_app`` doesn't know about
# multi-tenancy, so a module-global is fine here (mirrors how
# ``RunCache`` and the launcher's ``JobManager`` are scoped).
_correction_jobs = _CorrectionJobRegistry()



def register(app: FastAPI, router_ctx: RouterContext) -> None:
    cache = router_ctx.cache

    @app.post("/api/runs/{run_id}/correction-jobs")
    def start_correction_job(
        run_id: str,
        payload: Dict[str, Any] = Body(default_factory=dict),
    ) -> Dict[str, Any]:
        """Start a background correction-suggestion job.

        Body fields (all optional):
          * ``source`` (str) - restrict the job to this source CSV.
            Default: every source in the run.
          * ``samples``, ``samples_threshold``, ``retry_samples``,
            ``model``, ``reasoning_effort``, ``row_context_rows`` -
            same defaults as ``suggest-corrections``.

        Returns ``{"job_id", "total_targets", "status": "queued"}``.
        Poll ``GET /correction-jobs/{job_id}`` for progress + final
        results.
        """
        ctx = cache.get(run_id)
        source_filter = payload.get("source")
        if source_filter is not None:
            if not isinstance(source_filter, str):
                raise HTTPException(400, "source filter must be a string")
            if source_filter not in ctx.sources:
                raise HTTPException(404, f"source not found: {source_filter}")

        # Walk every saved column-cluster Label record and pick out
        # those with a TP/unsure verdict. We iterate the LabelStore
        # rather than re-parsing drafts so we naturally get only the
        # clusters the operator has actually reviewed.
        seen: set = set()
        targets: List[Tuple[str, str, str]] = []
        for lab in ctx.store.all():
            if lab.target != LabelTarget.COLUMN_CLUSTER:
                continue
            if lab.status not in (LabelStatus.TP, LabelStatus.UNSURE):
                continue
            tid = lab.target_id
            if tid in seen:
                continue  # latest-wins; first hit per id is the latest
            seen.add(tid)
            parsed = correction_suggester.parse_colclust_target_id(tid)
            if parsed is None:
                continue
            s, c, lid = parsed
            if source_filter is not None and s != source_filter:
                continue
            # latest_for is order-independent; use it as the source of truth
            latest = ctx.store.latest_for(tid)
            if latest is None or latest.status not in (
                LabelStatus.TP, LabelStatus.UNSURE,
            ):
                continue
            targets.append((s, c, lid))

        if not targets:
            raise HTTPException(
                404,
                "no TP/unsure column-cluster verdicts in this run "
                + (f"for source {source_filter!r}" if source_filter else ""),
            )

        samples = int(payload.get("samples", 3))
        samples_threshold = float(payload.get("samples_threshold", 0.5))
        retry_samples = int(payload.get("retry_samples", 4))
        model = str(payload.get("model", "gpt-5.2"))
        reasoning_effort = str(payload.get("reasoning_effort", "medium"))
        row_context_rows = int(payload.get("row_context_rows", 3))

        job = _correction_jobs.create(run_id, targets)

        def _run(job: _CorrectionJob) -> None:
            job.status = "running"
            job.started_at = time.time()

            def _progress(phase: str, done: int, total: int) -> None:
                # Dispatch fires this once per finished LLM call; the
                # registry is module-global so writes are visible to
                # the GET endpoint without further locking (single
                # writer per job).
                job.phase = phase
                job.done = done
                job.total = total

            try:
                llm_client = LLMClient()
            except Exception as exc:  # noqa: BLE001
                job.status = "error"
                job.error = f"LLM client unavailable: {exc}"
                job.finished_at = time.time()
                return

            try:
                results = correction_suggester.suggest_corrections_for_clusters(
                    ctx.run_dir,
                    targets=job.targets,
                    llm_client=llm_client,
                    model=model,
                    reasoning_effort=reasoning_effort,
                    samples=samples,
                    samples_threshold=samples_threshold,
                    retry_samples=retry_samples,
                    row_context_rows=row_context_rows,
                    concurrency=8,
                    progress_cb=_progress,
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "correction batch job failed run=%s job=%s",
                    job.run_id, job.job_id,
                )
                job.status = "error"
                job.error = str(exc)
                job.finished_at = time.time()
                return

            for (s, c, lid), suggs in results.items():
                key = f"{s}\x00{c}\x00{lid}"
                job.results[key] = correction_suggester.suggestions_to_payload(suggs)
            job.status = "done"
            job.finished_at = time.time()

        _correction_jobs.submit(job, _run)
        return {
            "job_id": job.job_id,
            "total_targets": len(targets),
            "status": job.status,
        }

    @app.get("/api/runs/{run_id}/correction-jobs/{job_id}")
    def get_correction_job(run_id: str, job_id: str) -> Dict[str, Any]:
        """Return job state for polling. Result payload is included in
        the same response when ``status == "done"``."""
        job = _correction_jobs.get(job_id)
        if job is None or job.run_id != run_id:
            raise HTTPException(404, f"correction job not found: {job_id}")
        return {
            "job_id": job.job_id,
            "run_id": job.run_id,
            "status": job.status,
            "phase": job.phase,
            "done": job.done,
            "total": job.total,
            "total_targets": len(job.targets),
            "started_at": job.started_at,
            "finished_at": job.finished_at,
            "error": job.error,
            "results": job.results if job.status == "done" else {},
        }

