"""Multi-table cross-sheet pipeline orchestration.

The four-step cross-sheet pipeline used to live inline inside
``experiments.run_relation_discovery._run_multi_table_pass``. We split
it here so two runners can compose its phases:

* ``run_relation_discovery`` runs **only steps 1+2** (priors → LLM
  group confirmation) and emits ``multi_table_groups.jsonl``. The
  output is the "relation" the user asked the LLM to discover.
* ``run_cleaning_detection`` runs **steps 3+4** (semantic detector
  *or* deterministic compiler + LLM verifier) and emits
  ``multi_table_violations.jsonl`` plus
  ``multi_table_verifications.jsonl``. The cleaning runner can either
  drive both halves end-to-end inline or consume groups from a
  previous discovery run via ``--multi-table-from``.

Public helpers (used by both runners):

* :func:`group_workbooks`        - parent-dir grouping convention
  shared by every multi-table-aware runner. CSVs whose parent
  directory matches share a workbook id; flat directories collapse
  to a single ``"_root"`` workbook (and skip cross-sheet work).
* :func:`resolve_groups_pass`    - steps 1+2.
* :func:`detect_violations_pass` - step 3 (+ optional step 4 for the
  compiler strategy).

Both passes reuse the existing module-level building blocks
(:mod:`multi_table_priors`, :mod:`multi_table_resolver`,
:mod:`multi_table_violations`, :mod:`multi_table_verifier`,
:mod:`multi_table_semantic_detector`) - this file is purely glue +
progress reporting.
"""

from __future__ import annotations

import concurrent.futures as _cf
import logging
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from src.core.multi_table_resolver import (
    ResolvedWorkbook,
    load_confirm_prompt,
    resolve_workbook,
    write_resolved_workbooks_jsonl,
)
from src.core.multi_table_semantic_detector import (
    SemanticPassResult,
    detect_workbook_inconsistencies,
    expected_semantic_call_count,
    load_detect_prompt,
    summarize as summarize_semantic,
)
from src.core.multi_table_verifier import (
    load_verify_prompt,
    summarize as summarize_verifications,
    verify_workbook_violations,
    write_verifications_jsonl,
)
from src.core.multi_table_violations import (
    compile_violations,
    write_violations_jsonl,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Workbook grouping
# ---------------------------------------------------------------------------


def group_workbooks(
    csv_paths: Sequence[Path],
    *,
    csv_dir: Optional[Path] = None,
) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    """Group CSV paths by parent directory into workbook → sheet lists.

    Returns ``(workbooks, source_keys)`` where:

    * ``workbooks`` maps a workbook id (parent directory name, or
      ``"_root"`` when the file sits directly under ``csv_dir``) to
      its list of source keys (basenames, with relative path
      disambiguation when basenames collide).
    * ``source_keys`` maps each absolute CSV path back to the source
      key the caller used in its ``source_frames`` dict, so callers
      that already populated ``source_frames`` keyed by basename can
      align the two views.

    Both this function and the convention it encodes are shared
    between the relation-discovery and cleaning-detection runners so
    artifacts cross-reference cleanly across runs.
    """
    workbooks: Dict[str, List[str]] = {}
    source_keys: Dict[str, str] = {}
    seen_keys: set[str] = set()
    for path in csv_paths:
        name = path.name
        if name in seen_keys:
            try:
                rel = (
                    path.relative_to(csv_dir)
                    if csv_dir is not None
                    else path
                )
            except ValueError:
                rel = path
            name = str(rel).replace("\\", "/")
        seen_keys.add(name)
        source_keys[str(path)] = name
        wb_id = path.parent.name or "_root"
        workbooks.setdefault(wb_id, []).append(name)
    return workbooks, source_keys


# ---------------------------------------------------------------------------
# Step 1+2: priors + LLM group confirmation
# ---------------------------------------------------------------------------


_EMPTY_RESOLVE_STATS: Dict[str, Any] = {
    "n_workbooks": 0,
    "n_groups": 0,
    "n_confirmed_groups": 0,
    "n_workbooks_errored": 0,
}


def resolve_groups_pass(
    *,
    run_dir: Path,
    workbooks: Mapping[str, Sequence[str]],
    source_frames: Mapping[str, Any],
    digests: Mapping[str, Any],
    llm: Any,
    model: str,
    prompt_path: Path,
    concurrency: int,
) -> Tuple[List[ResolvedWorkbook], Dict[str, Any]]:
    """Run the priors → LLM-confirm phase (steps 1+2).

    Writes ``multi_table_groups.jsonl`` under ``run_dir`` and returns
    ``(resolved, stats)``. Workbooks with fewer than 2 sheets are
    skipped silently. When no eligible workbooks remain the function
    returns ``([], _EMPTY_RESOLVE_STATS)`` and writes nothing.
    """
    eligible: Dict[str, List[str]] = {
        wb: list(sheets)
        for wb, sheets in workbooks.items()
        if len(sheets) >= 2
    }
    if not eligible:
        return [], dict(_EMPTY_RESOLVE_STATS)

    prompt_cfg = load_confirm_prompt(prompt_path)
    logger.info(
        "Multi-table cross-sheet pass: %d workbook(s) eligible "
        "(prompt=%s, model=%s)",
        len(eligible), Path(prompt_path).name, model,
    )

    resolved_list: List[Optional[ResolvedWorkbook]] = [None] * len(eligible)
    items = list(eligible.items())

    def _work(idx: int, wb: str, sheets: List[str]) -> None:
        try:
            resolved_list[idx] = resolve_workbook(
                workbook=wb,
                sheets=sheets,
                frames=source_frames,
                digests=digests,
                llm=llm,
                model=model,
                prompt_cfg=prompt_cfg,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "multi-table resolve failed for workbook=%s: %s", wb, exc,
            )

    with logging_redirect_tqdm():
        progress = tqdm(
            total=len(items),
            desc="Multi-table confirm (step 2)",
            unit="wb",
            dynamic_ncols=True,
            leave=True,
        )
        if concurrency <= 1 or len(items) == 1:
            for idx, (wb, sheets) in enumerate(items):
                _work(idx, wb, sheets)
                progress.update(1)
        else:
            with _cf.ThreadPoolExecutor(max_workers=concurrency) as ex:
                futures = [
                    ex.submit(_work, idx, wb, sheets)
                    for idx, (wb, sheets) in enumerate(items)
                ]
                for fut in _cf.as_completed(futures):
                    fut.result()
                    progress.update(1)
        progress.close()

    resolved: List[ResolvedWorkbook] = [r for r in resolved_list if r is not None]

    write_resolved_workbooks_jsonl(
        resolved, run_dir / "multi_table_groups.jsonl",
    )

    n_groups = sum(len(r.groups) for r in resolved)
    n_confirmed = sum(1 for r in resolved for g in r.groups if g.confirmed)
    n_errored = sum(1 for r in resolved if r.error)

    stats: Dict[str, Any] = {
        "n_workbooks": len(resolved),
        "n_groups": n_groups,
        "n_confirmed_groups": n_confirmed,
        "n_workbooks_errored": n_errored,
    }
    logger.info(
        "Multi-table groups: %d workbook(s), %d group(s) "
        "(%d confirmed); %d workbook(s) errored.",
        len(resolved), n_groups, n_confirmed, n_errored,
    )
    return resolved, stats


# ---------------------------------------------------------------------------
# Step 3 (+ optional step 4): violations + verifier
# ---------------------------------------------------------------------------


def detect_violations_pass(
    *,
    run_dir: Path,
    resolved: Sequence[ResolvedWorkbook],
    source_frames: Mapping[str, Any],
    llm: Any,
    model: str,
    strategy: str = "compiler",
    detect_prompt_path: Optional[Path] = None,
    verify_prompt_path: Optional[Path] = None,
    verify_enabled: bool = True,
    semantic_n_sample_rows: int = 50,
    semantic_max_pages: Optional[int] = None,
    concurrency: int = 1,
) -> Dict[str, Any]:
    """Run step 3 (and optional step 4) of the multi-table pipeline.

    ``strategy='compiler'`` (legacy):
        Deterministic Python compiler emits violations from the row
        data using the column mapping. Then, when ``verify_enabled``
        is true and ``verify_prompt_path`` exists, the LLM verifier
        runs once per workbook and writes
        ``multi_table_verifications.jsonl``.

    ``strategy='semantic'``:
        One LLM call per (group, page) finds inconsistencies
        directly. The detector emits its own per-finding verdicts so
        step 4 is skipped to avoid double-charging.

    Writes ``multi_table_violations.jsonl`` and (when applicable)
    ``multi_table_verifications.jsonl`` under ``run_dir``. Returns a
    stats dict ready for inclusion in ``summary.json``.
    """
    if not resolved:
        return {
            "n_violations": 0,
            "violations_by_kind": {},
            "verification": None,
            "semantic": None,
            "strategy": strategy,
        }

    all_violations: List[Any] = []
    semantic_stats: Optional[Dict[str, Any]] = None
    semantic_verifications: List[Any] = []

    if strategy == "semantic":
        if detect_prompt_path is None:
            raise ValueError(
                "strategy='semantic' requires detect_prompt_path"
            )
        try:
            detect_cfg = load_detect_prompt(detect_prompt_path)
        except Exception:  # noqa: BLE001
            logger.exception(
                "Could not load detect prompt at %s; aborting semantic "
                "pass.",
                detect_prompt_path,
            )
            detect_cfg = None

        if detect_cfg is not None:
            logger.info(
                "Multi-table semantic detector (step 3, LLM): "
                "prompt=%s, model=%s, n_sample_rows=%d, max_pages=%s",
                Path(detect_prompt_path).name, model,
                semantic_n_sample_rows,
                semantic_max_pages if semantic_max_pages is not None else "all",
            )

            results_per_wb: List[Optional[Any]] = [None] * len(resolved)

            expected_calls_per_wb = [
                expected_semantic_call_count(
                    resolved=r,
                    frames=source_frames,
                    n_sample_rows=semantic_n_sample_rows,
                    max_pages=semantic_max_pages,
                )
                for r in resolved
            ]
            n_confirmed_groups = sum(
                1 for r in resolved for g in r.groups if g.confirmed
            )
            expected_calls_total = sum(expected_calls_per_wb)
            logger.info(
                "Multi-table semantic detector: %d expected LLM call(s) "
                "across %d confirmed group(s) in %d workbook(s)",
                expected_calls_total, n_confirmed_groups, len(resolved),
            )

            def _detect_one(
                idx: int,
                r: Any,
                progress_cb: Optional[Callable[[int], None]] = None,
            ) -> None:
                try:
                    results_per_wb[idx] = detect_workbook_inconsistencies(
                        resolved=r,
                        frames=source_frames,
                        llm=llm,
                        model=model,
                        prompt_cfg=detect_cfg,
                        n_sample_rows=semantic_n_sample_rows,
                        max_pages=semantic_max_pages,
                        progress_cb=progress_cb,
                    )
                except Exception:  # noqa: BLE001
                    logger.exception(
                        "semantic detect failed for workbook=%s",
                        r.workbook,
                    )
                    if (
                        progress_cb is not None
                        and 0 <= idx < len(expected_calls_per_wb)
                    ):
                        try:
                            remaining = expected_calls_per_wb[idx]
                            if remaining > 0:
                                progress_cb(remaining)
                        except Exception:  # noqa: BLE001
                            pass

            with logging_redirect_tqdm():
                progress = tqdm(
                    total=expected_calls_total,
                    desc=(
                        f"Multi-table semantic (step 3, "
                        f"{n_confirmed_groups} confirmed groups)"
                    ),
                    unit="call",
                    dynamic_ncols=True,
                    leave=True,
                )
                if concurrency <= 1 or len(resolved) == 1:
                    for idx, r in enumerate(resolved):
                        _detect_one(idx, r, progress_cb=progress.update)
                else:
                    with _cf.ThreadPoolExecutor(max_workers=concurrency) as ex:
                        futures = [
                            ex.submit(_detect_one, idx, r, progress.update)
                            for idx, r in enumerate(resolved)
                        ]
                        for fut in _cf.as_completed(futures):
                            fut.result()
                progress.close()

            n_calls_total = 0
            n_errored_total = 0
            n_parse_errored_total = 0
            for res in results_per_wb:
                if res is None:
                    continue
                all_violations.extend(res.violations)
                semantic_verifications.extend(res.verifications)
                n_calls_total += res.n_calls
                n_errored_total += res.n_errored
                n_parse_errored_total += res.n_parse_errored

            write_verifications_jsonl(
                semantic_verifications,
                run_dir / "multi_table_verifications.jsonl",
            )
            agg = SemanticPassResult(
                violations=all_violations,
                verifications=semantic_verifications,
                n_calls=n_calls_total,
                n_errored=n_errored_total,
                n_parse_errored=n_parse_errored_total,
            )
            semantic_stats = summarize_semantic(agg)
            logger.info(
                "Multi-table semantic detector: %d call(s), "
                "%d finding(s) (%s); %d errored, %d parse-errored.",
                n_calls_total,
                semantic_stats.get("n_findings", 0),
                ", ".join(
                    f"{k}={v}"
                    for k, v in sorted(
                        (semantic_stats.get("by_kind") or {}).items()
                    )
                ) or "none",
                n_errored_total, n_parse_errored_total,
            )
    else:
        for r in resolved:
            try:
                all_violations.extend(
                    compile_violations(resolved=r, frames=source_frames),
                )
            except Exception:  # noqa: BLE001
                logger.exception(
                    "compile_violations failed for workbook=%s", r.workbook,
                )

    write_violations_jsonl(
        all_violations, run_dir / "multi_table_violations.jsonl",
    )

    by_kind: Dict[str, int] = {}
    for v in all_violations:
        by_kind[v.kind] = by_kind.get(v.kind, 0) + 1
    logger.info(
        "Multi-table detect: %d workbook(s), %d violation(s) (%s).",
        len(resolved),
        len(all_violations),
        ", ".join(f"{k}={v}" for k, v in sorted(by_kind.items())) or "none",
    )

    # Step 4: per-violation LLM verification (compiler strategy only).
    verification_stats: Optional[Dict[str, Any]] = None
    if (
        strategy != "semantic"
        and verify_enabled
        and all_violations
        and verify_prompt_path is not None
    ):
        try:
            verify_cfg = load_verify_prompt(verify_prompt_path)
        except Exception:  # noqa: BLE001
            logger.exception(
                "Could not load verify prompt at %s; skipping step 4.",
                verify_prompt_path,
            )
        else:
            logger.info(
                "Multi-table verifier (step 4): prompt=%s, model=%s",
                Path(verify_prompt_path).name, model,
            )
            verifications_per_wb: List[List[Any]] = [[] for _ in resolved]

            def _verify_one(idx: int, r: Any) -> None:
                try:
                    verifications_per_wb[idx] = verify_workbook_violations(
                        resolved=r,
                        violations=all_violations,
                        frames=source_frames,
                        llm=llm,
                        model=model,
                        prompt_cfg=verify_cfg,
                    )
                except Exception:  # noqa: BLE001
                    logger.exception(
                        "verify failed for workbook=%s", r.workbook,
                    )

            with logging_redirect_tqdm():
                progress = tqdm(
                    total=len(resolved),
                    desc="Multi-table verify (step 4)",
                    unit="wb",
                    dynamic_ncols=True,
                    leave=True,
                )
                if concurrency <= 1 or len(resolved) == 1:
                    for idx, r in enumerate(resolved):
                        _verify_one(idx, r)
                        progress.update(1)
                else:
                    with _cf.ThreadPoolExecutor(max_workers=concurrency) as ex:
                        futures = [
                            ex.submit(_verify_one, idx, r)
                            for idx, r in enumerate(resolved)
                        ]
                        for fut in _cf.as_completed(futures):
                            fut.result()
                            progress.update(1)
                progress.close()

            all_verifications: List[Any] = []
            for chunk in verifications_per_wb:
                all_verifications.extend(chunk)
            write_verifications_jsonl(
                all_verifications,
                run_dir / "multi_table_verifications.jsonl",
            )
            verification_stats = summarize_verifications(all_verifications)
            logger.info(
                "Multi-table verifier: %d verification(s) (%s)",
                verification_stats.get("n_verifications", 0),
                ", ".join(
                    f"{k}={v}"
                    for k, v in sorted(
                        (verification_stats.get("by_verdict") or {}).items()
                    )
                ) or "none",
            )

    return {
        "n_violations": len(all_violations),
        "violations_by_kind": by_kind,
        "verification": verification_stats,
        "semantic": semantic_stats,
        "strategy": strategy,
    }
