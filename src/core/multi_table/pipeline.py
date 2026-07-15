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
import json
import logging
import re
from dataclasses import replace
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

from .resolver import (
    ResolvedWorkbook,
    _infer_union_key,
    load_confirm_prompt,
    resolve_workbook,
    write_resolved_workbooks_jsonl,
)
from .discovery_adaptive import (
    AdaptiveDiscoveryConfig,
    resolve_workbook_adaptive,
    summarize_discovery,
)
from .semantic_detector import (
    SemanticPassResult,
    detect_workbook_inconsistencies,
    expected_semantic_call_count,
    load_detect_prompt,
    structural_complement,
    summarize as summarize_semantic,
)
from .verifier import (
    load_verify_prompt,
    summarize as summarize_verifications,
    verify_workbook_violations,
    write_verifications_jsonl,
)
from .violations import (
    compile_violations,
    write_violations_jsonl,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Workbook grouping
# ---------------------------------------------------------------------------


GROUP_BY_CHOICES = ("parent-dir", "prefix", "regex")


def _workbook_id_for(
    path: Path,
    *,
    group_by: str,
    group_pattern: Optional["re.Pattern[str]"],
) -> str:
    """Compute the workbook id for one CSV path under a grouping mode.

    * ``parent-dir`` (default): the parent directory name, or
      ``"_root"`` when the file sits directly under the scan root.
      This is the historical behaviour and works for layouts like
      ``data/bing21_grouped/<sha>.xlsx/*.csv``.
    * ``prefix``: the basename up to the first ``"_"``. Targets flat
      corpora such as PBI where every table of one report is dumped
      into a single directory as ``<reportID>_<TableName>.csv`` - the
      report id before the underscore is the true workbook key.
    * ``regex``: the first capture group (or the whole match when the
      pattern has no groups) of ``group_pattern`` applied to the
      basename. Files that do not match are given a unique singleton
      id so they are never merged into an unrelated group.
    """
    if group_by == "parent-dir":
        return path.parent.name or "_root"
    if group_by == "prefix":
        stem = path.name
        head = stem.split("_", 1)[0]
        # An empty head (leading underscore / empty name) is useless
        # as a key; fall back to the full basename so the file stays
        # in its own singleton group instead of a junk bucket.
        return head or path.name
    if group_by == "regex":
        if group_pattern is None:
            raise ValueError("group_by='regex' requires group_regex")
        m = group_pattern.search(path.name)
        if m is None:
            # No match -> singleton id keyed on the basename so the
            # file is never grouped with anything else.
            return f"_ungrouped::{path.name}"
        if m.groups():
            captured = m.group(1)
        else:
            captured = m.group(0)
        return captured or f"_ungrouped::{path.name}"
    raise ValueError(
        f"unknown group_by={group_by!r}; expected one of {GROUP_BY_CHOICES}"
    )


def group_workbooks(
    csv_paths: Sequence[Path],
    *,
    csv_dir: Optional[Path] = None,
    group_by: str = "parent-dir",
    group_regex: Optional[str] = None,
) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    """Group CSV paths into workbook → sheet lists.

    Returns ``(workbooks, source_keys)`` where:

    * ``workbooks`` maps a workbook id to its list of source keys
      (basenames, with relative path disambiguation when basenames
      collide). The id is derived per ``group_by`` (see
      :func:`_workbook_id_for`).
    * ``source_keys`` maps each absolute CSV path back to the source
      key the caller used in its ``source_frames`` dict, so callers
      that already populated ``source_frames`` keyed by basename can
      align the two views.

    ``group_by`` selects the grouping rule:

    * ``"parent-dir"`` (default) - parent directory name; flat
      directories collapse to a single ``"_root"`` workbook.
    * ``"prefix"`` - basename up to the first ``"_"`` (flat PBI-style
      ``<reportID>_<TableName>.csv`` corpora).
    * ``"regex"`` - first capture group of ``group_regex`` applied to
      the basename.

    Both this function and the convention it encodes are shared
    between the relation-discovery and cleaning-detection runners so
    artifacts cross-reference cleanly across runs.
    """
    if group_by not in GROUP_BY_CHOICES:
        raise ValueError(
            f"unknown group_by={group_by!r}; expected one of "
            f"{GROUP_BY_CHOICES}"
        )
    group_pattern: Optional["re.Pattern[str]"] = None
    if group_by == "regex":
        if not group_regex:
            raise ValueError("group_by='regex' requires a non-empty group_regex")
        try:
            group_pattern = re.compile(group_regex)
        except re.error as exc:
            raise ValueError(
                f"invalid group_regex {group_regex!r}: {exc}"
            ) from exc

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
        wb_id = _workbook_id_for(
            path, group_by=group_by, group_pattern=group_pattern,
        )
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
    adaptive: Optional[AdaptiveDiscoveryConfig] = None,
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
    disc_stats_list: List[Optional[Dict[str, Any]]] = [None] * len(eligible)
    items = list(eligible.items())
    adaptive_on = bool(adaptive is not None and adaptive.enabled)
    if adaptive_on:
        logger.info(
            "Multi-table discovery is ADAPTIVE (k_min=%d, k_max=%d, "
            "stop_when=%s, shuffle_rows=%s, prompt_ensemble=%s).",
            adaptive.k_min, adaptive.k_max, adaptive.stop_when,
            adaptive.shuffle_rows, adaptive.prompt_ensemble,
        )

    def _work(idx: int, wb: str, sheets: List[str]) -> None:
        try:
            if adaptive_on:
                rw, dstats = resolve_workbook_adaptive(
                    workbook=wb,
                    sheets=sheets,
                    frames=source_frames,
                    digests=digests,
                    llm=llm,
                    model=model,
                    prompt_cfg=prompt_cfg,
                    cfg=adaptive,
                )
                resolved_list[idx] = rw
                disc_stats_list[idx] = dstats
            else:
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
    if adaptive_on:
        disc = [d for d in disc_stats_list if d is not None]
        adaptive_summary = summarize_discovery(disc)
        adaptive_summary["config"] = {
            "k_min": adaptive.k_min, "k_max": adaptive.k_max,
            "stop_when": adaptive.stop_when,
            "shuffle_rows": adaptive.shuffle_rows,
            "prompt_ensemble": adaptive.prompt_ensemble,
        }
        stats["discovery_adaptive"] = adaptive_summary
        try:
            gt_dir = run_dir / "good_turing"
            gt_dir.mkdir(parents=True, exist_ok=True)
            (gt_dir / "multi_table_discovery.json").write_text(
                json.dumps(
                    {"summary": adaptive_summary, "per_workbook": disc},
                    indent=2, ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        except Exception:  # noqa: BLE001
            logger.exception("failed to write multi_table_discovery.json")
        logger.info(
            "Multi-table adaptive discovery: %d workbook(s), %d total pass(es) "
            "(mean %.1f/wb), mean missing-mass %.3f; stop reasons %s.",
            adaptive_summary["n_workbooks"],
            adaptive_summary["total_passes"],
            adaptive_summary.get("mean_passes_per_workbook", 0.0),
            adaptive_summary["mean_missing_mass"],
            adaptive_summary["stopped_reasons"],
        )
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
    semantic_key_align: bool = True,
    semantic_key_targeted: bool = False,
    semantic_structural: bool = True,
    guard_enabled: bool = True,
    guard_density_threshold: float = 0.4,
    guard_min_findings: int = 10,
    drop_same_sheet_pairs: bool = True,
    drop_fk_orphan: bool = True,
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
        directly. Pages are key-aligned by default
        (``semantic_key_align``) so rows sharing a primary/join key
        co-locate; ``semantic_key_targeted`` further restricts each
        call to cross-sheet keys (budget mode). When
        ``semantic_structural`` is set, the deterministic compiler
        additionally contributes any exact fk_orphan / pk_collision /
        dtype_drift / header_synonym findings the LLM pass missed. The
        detector emits its own per-finding verdicts so step 4 is
        skipped to avoid double-charging.

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

    # Recall safety net: a confirmed union with no primary_key can't be
    # row-paired, so cross-sheet conflicts are missed. resolve_workbook
    # already infers a key for fresh runs, but groups loaded from disk
    # (``--multi-table-from``) may predate that fix - re-infer here so
    # detection is robust regardless of how the groups were produced.
    for rw in resolved:
        if not getattr(rw, "groups", None):
            continue
        patched = False
        new_groups = []
        for g in rw.groups:
            if g.confirmed and g.kind == "union" and not g.primary_key:
                sub = {s: source_frames[s] for s in g.sheets if s in source_frames}
                inferred = _infer_union_key(g, sub)
                if inferred:
                    g = replace(g, primary_key=inferred)
                    patched = True
            new_groups.append(g)
        if patched:
            rw.groups[:] = new_groups

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
                    key_align=semantic_key_align,
                    key_targeted=semantic_key_targeted,
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
                        key_align=semantic_key_align,
                        key_targeted=semantic_key_targeted,
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

            # Structural complement: merge any exact deterministic
            # findings (fk_orphan / pk_collision / dtype_drift /
            # header_synonym) the LLM pass did not already surface.
            n_structural = 0
            if semantic_structural:
                extra_v, extra_ver = structural_complement(
                    resolved_list=resolved,
                    frames=source_frames,
                    semantic_violations=all_violations,
                )
                n_structural = len(extra_v)
                if extra_v:
                    logger.info(
                        "Multi-table structural complement: +%d "
                        "deterministic violation row(s) (%d finding(s)) "
                        "the semantic pass did not surface.",
                        len(extra_v), len(extra_ver),
                    )
                all_violations.extend(extra_v)
                semantic_verifications.extend(extra_ver)

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
            semantic_stats["n_structural_complement"] = n_structural
            semantic_stats["key_align"] = bool(semantic_key_align)
            semantic_stats["key_targeted"] = bool(semantic_key_targeted)
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

    # Scope filter: inter_table_inconsistency is a CROSS-sheet concern.
    # Drop same-sheet row pairs (a row_duplication concern) and orphan
    # foreign keys (join inference across same-file CSVs is unreliable)
    # before anything else looks at the artifacts. Failures never break.
    scope_stats: Optional[Dict[str, Any]] = None
    if drop_same_sheet_pairs or drop_fk_orphan:
        try:
            from .group_guard import _read_jsonl, filter_scope_leak_dicts

            vpath = run_dir / "multi_table_violations.jsonl"
            verpath = run_dir / "multi_table_verifications.jsonl"
            vdicts = _read_jsonl(vpath)
            verdicts = _read_jsonl(verpath)
            new_v, new_ver, s_stats = filter_scope_leak_dicts(
                violation_dicts=vdicts, verification_dicts=verdicts,
                drop_same_sheet_pairs=drop_same_sheet_pairs,
                drop_fk_orphan=drop_fk_orphan,
            )
            n_dropped = s_stats["dropped_same_sheet"] + s_stats["dropped_fk_orphan"]
            if n_dropped:
                vpath.write_text(
                    "".join(
                        json.dumps(r, ensure_ascii=False) + "\n" for r in new_v
                    ),
                    encoding="utf-8",
                )
                verpath.write_text(
                    "".join(
                        json.dumps(r, ensure_ascii=False) + "\n"
                        for r in new_ver
                    ),
                    encoding="utf-8",
                )
                logger.info(
                    "Multi-table scope filter: dropped %d same-sheet pair(s) "
                    "and %d orphan-FK finding(s) (%d -> %d).",
                    s_stats["dropped_same_sheet"], s_stats["dropped_fk_orphan"],
                    s_stats["findings_before"], s_stats["findings_after"],
                )
            scope_stats = s_stats
        except Exception:  # noqa: BLE001
            logger.exception("Multi-table scope filter failed; skipping.")

    # Group-level plausibility guard: an implausible row-pairing flood
    # (positional/ordinal key, or a union that disagrees on a high
    # fraction of its keys = periods/snapshots) means NO error AND a
    # wrong discovered relation, so the whole flood is DROPPED (no
    # finding emitted; only a count in stats). Runs on the just-written
    # artifacts so it covers BOTH strategies uniformly. Failures never
    # break the pass.
    guard_stats: Optional[Dict[str, Any]] = None
    if guard_enabled:
        try:
            from .group_guard import _read_jsonl, guard_finding_dicts

            vpath = run_dir / "multi_table_violations.jsonl"
            verpath = run_dir / "multi_table_verifications.jsonl"
            vdicts = _read_jsonl(vpath)
            verdicts = _read_jsonl(verpath)
            group_lookup = {
                (r.workbook, g.group_id): {**g.to_dict(), "workbook": r.workbook}
                for r in resolved for g in r.groups
            }

            def _load(workbook: str, sheets: Sequence[str]) -> Dict[str, Any]:
                return {
                    s: source_frames[s] for s in sheets if s in source_frames
                }

            new_v, new_ver, g_stats = guard_finding_dicts(
                violation_dicts=vdicts, verification_dicts=verdicts,
                group_lookup=group_lookup, load_frames=_load,
                density_threshold=guard_density_threshold,
                min_findings=guard_min_findings,
            )
            if g_stats.get("findings_suppressed"):
                vpath.write_text(
                    "".join(
                        json.dumps(r, ensure_ascii=False) + "\n" for r in new_v
                    ),
                    encoding="utf-8",
                )
                verpath.write_text(
                    "".join(
                        json.dumps(r, ensure_ascii=False) + "\n"
                        for r in new_ver
                    ),
                    encoding="utf-8",
                )
                by_kind = {}
                seen_g: set = set()
                for r in new_v:
                    vid = r.get("violation_id")
                    if vid in seen_g:
                        continue
                    seen_g.add(vid)
                    by_kind[r["kind"]] = by_kind.get(r["kind"], 0) + 1
                logger.info(
                    "Multi-table group guard: dropped %d implausible "
                    "group(s), suppressed %d finding(s) (%d -> %d).",
                    g_stats["groups_collapsed"], g_stats["findings_suppressed"],
                    g_stats.get("findings_before", 0),
                    g_stats.get("findings_after", 0),
                )
            guard_stats = {
                k: g_stats[k]
                for k in (
                    "groups_collapsed", "findings_suppressed", "notes_added",
                    "density_threshold", "min_findings",
                )
            }
        except Exception:  # noqa: BLE001
            logger.exception("Multi-table group guard failed; skipping.")

    # Recompute final counts from the on-disk artifact so the scope filter
    # and guard rewrites are both reflected in the returned stats.
    final_n = len(all_violations)
    if scope_stats is not None or guard_stats is not None:
        try:
            from .group_guard import _read_jsonl

            final_dicts = _read_jsonl(run_dir / "multi_table_violations.jsonl")
            by_kind = {}
            seen_f: set = set()
            for r in final_dicts:
                vid = r.get("violation_id")
                if vid in seen_f:
                    continue
                seen_f.add(vid)
                by_kind[r.get("kind")] = by_kind.get(r.get("kind"), 0) + 1
            final_n = len(seen_f)
        except Exception:  # noqa: BLE001
            logger.exception("Multi-table final count recompute failed.")

    return {
        "n_violations": final_n,
        "violations_by_kind": by_kind,
        "verification": verification_stats,
        "semantic": semantic_stats,
        "strategy": strategy,
        "scope_filter": scope_stats,
        "guard": guard_stats,
    }
