"""Pipeline that asks an LLM to detect data-cleaning issues in CSV chunks.

The detector is exhaustive-by-default: for each CSV chunk, it issues one LLM
call *per error type* in the taxonomy (see :mod:`src.error_taxonomy`). Each
such call produces a :class:`ChunkReport` that is tagged with the originating
``error_type``. Downstream code merges all reports into a single flat list of
issues for reporting.

Calls are dispatched through a :class:`concurrent.futures.ThreadPoolExecutor`
so that the many (chunk x type) jobs can run concurrently across the pool of
Azure OpenAI endpoints managed by :class:`~src.llm_client.LLMClient`.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from .data_loader import CsvChunk
from .error_taxonomy import ErrorTaxonomy, ErrorType
from .llm_client import LLMClient, LLMResponse
from .table_digest import TableDigest

logger = logging.getLogger(__name__)


from .jsonl_io import JSON_FENCE_RE as _JSON_BLOCK_RE, extract_json_payloads


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ChunkReport:
    """One LLM call's result for a single (chunk, error_type) pair."""

    source: str
    start_row: int
    end_row: int
    model: str
    prompt_name: str
    issues: List[Dict[str, Any]] = field(default_factory=list)
    raw_text: str = ""
    reasoning_summary: str = ""
    endpoint: str = ""
    attempts: int = 0
    error: Optional[str] = None
    parse_error: Optional[str] = None
    # Per-type metadata (populated by PerTypeCleaningDetector). When the
    # detector runs in "holistic" mode (one call per chunk for ALL types),
    # these remain empty strings.
    error_type: str = ""
    error_type_name: str = ""
    error_type_category: str = ""
    # Free-form per-report metadata. Populated by detectors that need
    # to flag operational signals (e.g. ``cluster_budget_exhausted``
    # from the row_duplication clusterer pipeline). Consumers should
    # treat unknown keys as advisory.
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self, *, include_raw: bool = True) -> Dict[str, Any]:
        payload = asdict(self)
        if not include_raw:
            payload.pop("raw_text", None)
            payload.pop("reasoning_summary", None)
        return payload


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------


def _coerce_issues(payload: Any) -> List[Dict[str, Any]]:
    """Normalise the model's JSON response into a list of issue dicts."""
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("issues", "errors", "findings", "results"):
            if key in payload and isinstance(payload[key], list):
                return [item for item in payload[key] if isinstance(item, dict)]
        return [payload]
    return []


def parse_llm_json(text: str) -> tuple[List[Dict[str, Any]], Optional[str]]:
    """Best-effort extraction of JSON from a model response.

    Tries fenced blocks first (``​```json ...​```​``), then the whole
    response stripped, then a bracket-balanced scan that pulls every
    ``{...}`` or ``[...]`` payload out of surrounding prose. The
    bracket scan handles the common case where the LLM forgot the
    fence and emitted "Here are my findings: {\"issues\": [...]}".
    """
    if not text:
        return [], "empty response"

    candidates: List[str] = []
    for match in _JSON_BLOCK_RE.finditer(text):
        candidates.append(match.group(1))
    candidates.append(text.strip())
    # Bracket-balanced fallback: handles non-fenced prose+JSON.
    candidates.extend(extract_json_payloads(text))

    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate:
            continue
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        issues = _coerce_issues(payload)
        return issues, None

    return [], "failed to parse JSON from response"


# ---------------------------------------------------------------------------
# Per-error-type detector
# ---------------------------------------------------------------------------


class PerTypeCleaningDetector:
    """Exhaustive detector: one LLM call per (chunk, error_type) pair.

    The detector iterates over the cartesian product of chunks and selected
    error types. Jobs are submitted to a thread pool so that up to
    ``concurrency`` calls are in flight simultaneously; the underlying
    :class:`LLMClient` load-balances them across its endpoint pool.

    Parameters
    ----------
    llm
        Shared, thread-safe LLM client.
    taxonomy
        The error-type catalogue (shared system prompt + per-type entries).
    model
        Deployment name (e.g. ``"gpt-5.2"``).
    error_types
        Optional list of ``ErrorType`` objects to run. Defaults to the full
        taxonomy.
    reasoning_effort, reasoning_summary
        Passed through to the Responses API when applicable.
    concurrency
        Maximum number of concurrent LLM calls. Defaults to the number of
        endpoints in ``llm`` (cap 8). Use 1 for deterministic, sequential
        execution (useful when debugging).
    """

    def __init__(
        self,
        llm: LLMClient,
        taxonomy: ErrorTaxonomy,
        model: str,
        *,
        error_types: Optional[Sequence[ErrorType]] = None,
        reasoning_effort: str = "medium",
        reasoning_summary: str = "detailed",
        concurrency: Optional[int] = None,
        digests: Optional[Dict[str, TableDigest]] = None,
    ) -> None:
        self.llm = llm
        self.taxonomy = taxonomy
        self.model = model
        self.error_types = list(error_types) if error_types is not None else list(
            taxonomy.types
        )
        if not self.error_types:
            raise ValueError("PerTypeCleaningDetector needs at least one error type")
        self.reasoning_effort = reasoning_effort
        self.reasoning_summary = reasoning_summary
        # Optional whole-table digests, keyed by stringified source
        # path. Used to render a TABLE OVERVIEW block at the top of
        # every per-chunk prompt so the LLM can reason about
        # multi-column / multi-row error types without us having to
        # ship the full CSV in each call. ``None`` (or a missing
        # entry for a given source) leaves the placeholder empty.
        self.digests: Dict[str, TableDigest] = dict(digests or {})
        default_concurrency = min(8, max(1, len(self.llm.endpoints)))
        self.concurrency = concurrency if concurrency and concurrency > 0 else default_concurrency

    # ------------------------------------------------------------------
    # Single-job execution
    # ------------------------------------------------------------------

    def _analyse(self, chunk: CsvChunk, error_type: ErrorType) -> ChunkReport:
        digest = self.digests.get(str(chunk.source))
        rendered = self.taxonomy.render_user_prompt(
            error_type, chunk, digest=digest
        )
        prompt_name = f"error_types.{error_type.key}"
        report = ChunkReport(
            source=str(chunk.source),
            start_row=chunk.start_row,
            end_row=chunk.end_row,
            model=self.model,
            prompt_name=prompt_name,
            error_type=error_type.key,
            error_type_name=error_type.name,
            error_type_category=error_type.category,
        )

        response: LLMResponse = self.llm.query(
            self.model,
            rendered,
            reasoning_effort=self.reasoning_effort,
            reasoning_summary=self.reasoning_summary,
            system_prompt=self.taxonomy.system_prompt,
        )

        report.endpoint = response.endpoint
        report.attempts = response.attempts
        report.raw_text = response.text
        report.reasoning_summary = response.reasoning_summary
        report.error = response.error
        # When ``self.model`` is a comma-list rotation pool, record the
        # specific family member that actually served this call so
        # downstream attribution shows ``gpt-5.4`` rather than the
        # full pool string. Empty (e.g. on hard failure) stays as the
        # pool spec so the run record is still labelled.
        if response.model:
            report.model = response.model

        if response.ok:
            issues, parse_err = parse_llm_json(response.text)
            # Stamp each issue with its error-type identity (category is also
            # useful for downstream aggregation).
            for issue in issues:
                issue.setdefault("error_type", error_type.key)
                issue.setdefault("error_type_name", error_type.name)
                issue.setdefault("error_type_category", error_type.category)
                # Backfill the generic `category` field used by older
                # consumers with the taxonomy category (kept lowercase for
                # compatibility with existing aggregations).
                issue.setdefault("category", error_type.category.lower())
            report.issues = issues
            report.parse_error = parse_err
        return report

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def run(
        self,
        chunks: Iterable[CsvChunk],
        *,
        skip_predicate: Optional[Callable[[CsvChunk, ErrorType], bool]] = None,
    ) -> List[ChunkReport]:
        """Run detection across all (chunk, error_type) pairs.

        Parameters
        ----------
        chunks
            Iterable of CSV chunks to probe.
        skip_predicate
            Optional callable. When provided, every (chunk, error_type)
            pair for which it returns ``True`` is dropped from the job
            list before scheduling. The progress bar's ``total`` reflects
            the *remaining* jobs only - so an adaptive runner that
            prunes saturated per-arm work each pass will see the bar
            shrink from one pass to the next.
        """
        chunks = list(chunks)
        jobs: List[tuple[int, int, CsvChunk, ErrorType]] = []
        skipped = 0
        for c_idx, chunk in enumerate(chunks):
            for t_idx, et in enumerate(self.error_types):
                if skip_predicate is not None and skip_predicate(chunk, et):
                    skipped += 1
                    continue
                jobs.append((c_idx, t_idx, chunk, et))

        total = len(jobs)
        if skipped:
            logger.info(
                "Per-type detection: %d chunks x %d error types -> "
                "%d LLM calls after skipping %d saturated-arm pair(s) "
                "(concurrency=%d)",
                len(chunks),
                len(self.error_types),
                total,
                skipped,
                self.concurrency,
            )
        else:
            logger.info(
                "Per-type detection: %d chunks x %d error types = %d LLM calls "
                "(concurrency=%d)",
                len(chunks),
                len(self.error_types),
                total,
                self.concurrency,
            )

        if total == 0:
            # Every (chunk, type) pair was pruned. Return empty so the
            # caller's bookkeeping path stays uniform.
            return []

        results: List[Optional[ChunkReport]] = [None] * total
        progress = tqdm(
            total=total,
            desc="Detect",
            unit="call",
            dynamic_ncols=True,
            leave=True,
        )
        progress_lock = threading.Lock()
        error_count = {"n": 0}

        def _on_complete(report: ChunkReport, chunk: CsvChunk, et: ErrorType) -> None:
            """Update the progress bar and log only on LLM / parse errors."""
            with progress_lock:
                if report.error:
                    error_count["n"] += 1
                    # Keep the bar tidy: route error details through the
                    # logger and just show a running error count as a
                    # postfix on the bar itself.
                    logger.warning(
                        "LLM call failed: source=%s rows=[%d,%d) type=%s attempts=%d error=%s",
                        Path(str(chunk.source)).name,
                        chunk.start_row,
                        chunk.end_row,
                        et.key,
                        report.attempts,
                        report.error,
                    )
                elif report.parse_error:
                    # Parse errors are less noisy but still worth surfacing.
                    logger.warning(
                        "Response parse failed: source=%s rows=[%d,%d) type=%s parse_error=%s",
                        Path(str(chunk.source)).name,
                        chunk.start_row,
                        chunk.end_row,
                        et.key,
                        report.parse_error,
                    )
                progress.set_postfix(errors=error_count["n"], refresh=False)
                progress.update(1)

        def _run_job(position: int, chunk: CsvChunk, et: ErrorType) -> tuple[int, ChunkReport]:
            report = self._analyse(chunk, et)
            _on_complete(report, chunk, et)
            return position, report

        # `logging_redirect_tqdm` routes any log records emitted while
        # the bar is alive through ``tqdm.write`` so they do not corrupt
        # the progress line (they land cleanly above it instead of
        # interleaving with bar redraws).
        try:
            with logging_redirect_tqdm():
                if self.concurrency == 1:
                    for position, (_c, _t, chunk, et) in enumerate(jobs):
                        _, report = _run_job(position, chunk, et)
                        results[position] = report
                else:
                    with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
                        futures = [
                            pool.submit(_run_job, position, chunk, et)
                            for position, (_c, _t, chunk, et) in enumerate(jobs)
                        ]
                        for fut in as_completed(futures):
                            position, report = fut.result()
                            results[position] = report
        finally:
            progress.close()

        if error_count["n"]:
            logger.warning(
                "Per-type detection finished with %d/%d failed call(s). "
                "See warnings above and chunk_reports.jsonl for details.",
                error_count["n"],
                total,
            )

        # Stable ordering: first by chunk index, then by taxonomy order.
        # `jobs` was already built in that order, so `results` preserves it.
        final: List[ChunkReport] = [r for r in results if r is not None]
        return final


# ---------------------------------------------------------------------------
# Report writers / aggregators (shared with report_builder)
# ---------------------------------------------------------------------------


def write_reports_jsonl(
    reports: Iterable[ChunkReport],
    path: Path,
    *,
    include_raw: bool = False,
) -> int:
    """Write one chunk report per line. Slim by default (no raw_text / reasoning).

    The raw LLM text and reasoning summaries are written as sidecar files by
    :func:`src.report_builder.write_raw_response_sidecars`, which keeps the
    JSONL compact and easy to grep/diff while preserving full fidelity on disk.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as fh:
        for report in reports:
            fh.write(
                json.dumps(report.as_dict(include_raw=include_raw), ensure_ascii=False)
                + "\n"
            )
            count += 1
    logger.info("Wrote %d chunk reports to %s", count, path)
    return count


def summarise_reports(reports: Iterable[ChunkReport]) -> Dict[str, Any]:
    reports = list(reports)
    total_issues = sum(len(r.issues) for r in reports)
    failed = [r for r in reports if r.error]
    parse_failed = [r for r in reports if r.parse_error and not r.error]

    # Distinct CSV chunks (by (source, start_row, end_row)) - reports may
    # duplicate per error-type, so counting them directly over-states chunks.
    chunk_keys = {(r.source, r.start_row, r.end_row) for r in reports}

    category_counter: Dict[str, int] = {}
    error_type_counter: Dict[str, int] = {}
    error_type_category_counter: Dict[str, int] = {}
    for report in reports:
        for issue in report.issues:
            cat = str(issue.get("category", "unknown")).lower()
            category_counter[cat] = category_counter.get(cat, 0) + 1
            etype = str(issue.get("error_type") or report.error_type or "unknown")
            error_type_counter[etype] = error_type_counter.get(etype, 0) + 1
            etcat = str(
                issue.get("error_type_category")
                or report.error_type_category
                or "unknown"
            )
            error_type_category_counter[etcat] = (
                error_type_category_counter.get(etcat, 0) + 1
            )

    return {
        "chunks": len(chunk_keys),
        "llm_calls": len(reports),
        "chunks_with_errors": len(failed),
        "chunks_with_parse_errors": len(parse_failed),
        "total_issues": total_issues,
        "issues_by_category": category_counter,
        "issues_by_error_type": error_type_counter,
        "issues_by_error_type_category": error_type_category_counter,
    }
