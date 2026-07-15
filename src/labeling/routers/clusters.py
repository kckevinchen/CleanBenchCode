"""Cluster queue, detail, web-verification, and correction-suggestion endpoints.

Serves the cluster label-queue (one row per underlying defect) and its
drill-down, the web-grounded verification verdict for a cluster, and the
LLM correction-suggestion flow. ``_cluster_summary`` shapes a raw cluster
record into the slim queue item the UI renders.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from fastapi import Body, FastAPI, HTTPException, Query

from .. import correction_suggester
from ...core.llm_client import LLMClient
from .._helpers import _scalar
from ..schema import Label, LabelStatus, LabelTarget
from ._context import RouterContext

logger = logging.getLogger(__name__)

# Error types whose detections legitimately aggregate many rows into ONE
# cluster, so surfacing the union of their ``affected_rows`` truthfully
# tells the labeller "this single TP/FP click covers N rows": the relation
# pipeline (FD / calculation / cross-table) plus row_duplication. The
# single-column types (typo, inconsistency, semantic_incompatible,
# numeric_outlier) are the opposite case - ``cluster_issues`` *fans* a
# multi-row issue OUT into one cluster per affected row so each cell stays
# individually labellable, yet every fanned cluster still carries the FULL
# ``affected_rows`` list. Surfacing it there makes all N clusters re-claim
# the same N rows, so one cell shows up N times in the queue. Gate the
# surface to genuine aggregates; fanned clusters then each show only their
# own row.
_MULTIROW_AGGREGATE_TYPES = frozenset({
    "row_duplication",
    "fd_violation",
    "row_calculation_inconsistency",
    "string_calculation_inconsistency",
    "inter_table_inconsistency",
})


def _cluster_summary(c: Mapping[str, Any]) -> Dict[str, Any]:
    syn = c.get("synthesis") or {}
    primary = None
    if isinstance(syn, Mapping):
        primary = syn.get("primary_error_type")
    error_types = list(c.get("error_types") or [])
    # Surface the union of every detection's ``affected_rows`` plus
    # the primary row so the queue panel can show "rows: 5, 12, 27"
    # at a glance instead of forcing the operator to open each
    # cluster and scan the context table. Originally this was only
    # populated for ``row_duplication`` clusters; relation rules
    # (string_calculation_inconsistency, FD, etc.) also fire on many
    # rows but collapse to ONE cluster after the per-incident
    # aggregation in ``compile_code_violations``, so the same
    # affected-rows surface is what tells the labeller "this single
    # TP click covers N rows".
    affected_rows: List[int] = []
    if set(error_types) & _MULTIROW_AGGREGATE_TYPES:
        seen: set = set()
        primary_row = c.get("row")
        if isinstance(primary_row, (int, float)):
            seen.add(int(primary_row))
        for d in c.get("detections") or []:
            if not isinstance(d, Mapping):
                continue
            for r in d.get("affected_rows") or ():
                try:
                    seen.add(int(r))
                except (TypeError, ValueError):
                    continue
        # Only surface when there are MORE rows than just the primary;
        # singleton clusters get an empty list so the existing UI guards
        # (`affected_rows.length > 1`) keep their meaning.
        if len(seen) > 1:
            affected_rows = sorted(seen)
    # Verdict / strategy: row_duplication clusters carry these on
    # each detection. We surface the strongest verdict (duplicate
    # beats near_duplicate) and the union of strategies so the queue
    # panel can show a "near-dup" badge alongside the row list.
    verdict = ""
    discovery_strategies: List[str] = []
    cluster_budget_exhausted = False
    max_groups_per_source = 0
    if "row_duplication" in error_types:
        verdict_rank = {"duplicate": 2, "near_duplicate": 1}
        seen_strats: set = set()
        best_rank = -1
        for d in c.get("detections") or []:
            if not isinstance(d, Mapping):
                continue
            v = str(d.get("verdict", "") or "").strip().lower()
            if v in verdict_rank and verdict_rank[v] > best_rank:
                best_rank = verdict_rank[v]
                verdict = v
            s = str(d.get("discovery_strategy", "") or "").strip()
            if s and s not in seen_strats:
                seen_strats.add(s)
                discovery_strategies.append(s)
            if d.get("cluster_budget_exhausted"):
                cluster_budget_exhausted = True
                try:
                    cap = int(d.get("max_groups_per_source") or 0)
                except (TypeError, ValueError):
                    cap = 0
                if cap > max_groups_per_source:
                    max_groups_per_source = cap
    return {
        "cluster_id": c.get("cluster_id"),
        "source": c.get("source"),
        "row": c.get("row"),
        "columns": list(c.get("columns") or []),
        "error_types": error_types,
        "evidence_count": c.get("evidence_count"),
        "primary_error_type": primary,
        "is_multi_type": len(error_types) >= 2,
        "affected_rows": affected_rows,
        "verdict": verdict,
        "discovery_strategies": discovery_strategies,
        "cluster_budget_exhausted": cluster_budget_exhausted,
        "max_groups_per_source": max_groups_per_source,
    }


def make_anchor_value_fn(ctx: Any) -> Callable[..., Optional[str]]:
    """Build a function that returns the ``_scalar`` anchor cell value of
    a single-column, concrete-row, non-aggregate cluster.

    The anchor value is the grouping key used to collapse the fanned
    same-value detections (``cluster_issues`` fans one single-column
    issue into one cluster per affected row) into ONE labellable defect
    that is counted ONCE. Returns ``None`` for clusters that must NOT be
    grouped: multi-column, row-less, multi-row aggregate types
    (row_duplication / FD / calculation / cross-table), or when the
    frame / cell can't be read. The per-source frame cache is closed
    over so a caller iterating every cluster reads each CSV at most once.
    """
    frame_cache: Dict[str, Any] = {}

    def anchor(
        source: Optional[str],
        row: Any,
        columns: Any,
        error_types: Any,
    ) -> Optional[str]:
        cols = list(columns or [])
        if len(cols) != 1 or not isinstance(row, int):
            return None
        if set(error_types or []) & _MULTIROW_AGGREGATE_TYPES:
            return None
        src = source or ""
        if src not in frame_cache:
            try:
                frame_cache[src] = ctx.load_frame(src) if src else None
            except FileNotFoundError:
                frame_cache[src] = None
        frame = frame_cache[src]
        if frame is None:
            return None
        col = cols[0]
        if col not in frame.columns or not (0 <= row < len(frame)):
            return None
        return _scalar(frame.iat[row, frame.columns.get_loc(col)])

    return anchor


def group_cluster_ids_by_value(ctx: Any) -> List[List[str]]:
    """Collapse fanned same-value clusters into per-issue groups.

    Returns a list of groups, each a list of ``cluster_id`` strings.
    Single-column clusters sharing a ``(source, column, anchor value)``
    form one group (the user labels / counts the defect ONCE); every
    other cluster (multi-column, row-less, aggregate type, empty value)
    is its own singleton group. Mirrors the grouping ``list_clusters``
    applies to the queue so the stats bar counts per-issue, not per
    fanned row.
    """
    anchor = make_anchor_value_fn(ctx)
    groups: Dict[Tuple[str, str, str], List[str]] = {}
    # Clusters that don't value-group but DO resolve to the same canonical
    # (source, row, columns) cell are collapsed too, mirroring the
    # (source, row, frozenset(columns)) dedup ``list_clusters`` applies to
    # the queue. Without this, multi-column / empty-value / aggregate-type
    # duplicate-cell clusters count once in the queue but N times here, so
    # the stats total drifts above the queue count.
    cell_groups: Dict[Tuple[str, int, Tuple[str, ...]], List[str]] = {}
    singletons: List[List[str]] = []
    for c in ctx.clusters:
        cid = c.get("cluster_id")
        if not cid:
            continue
        cols = list(c.get("columns") or [])
        val = anchor(
            c.get("source"), c.get("row"), cols, c.get("error_types"),
        )
        if val is None or val == "":
            row = c.get("row")
            cols_key = tuple(sorted(x for x in cols if isinstance(x, str)))
            # Same dedup gate as the queue: only when the column set is
            # non-empty AND the row is concrete (row-less / column-less
            # defects are dataset-wide flags that stay separate).
            if cols_key and isinstance(row, int):
                cell_groups.setdefault(
                    (c.get("source") or "", row, cols_key), []
                ).append(cid)
            else:
                singletons.append([cid])
            continue
        groups.setdefault((c.get("source") or "", cols[0], val), []).append(cid)
    return list(groups.values()) + list(cell_groups.values()) + singletons


# Status precedence for collapsing a group's many member verdicts into
# the single verdict the group is COUNTED as: the most-advanced label
# wins so a partly-labelled group (e.g. after same-value propagation)
# reads as labelled. Shared by the queue grouping and the stats bar.
GROUP_STATUS_RANK = {
    "tp": 5, "partial": 4, "fp": 3, "fn": 2, "unsure": 1, "skip": 0,
}


def group_status(statuses: List[str]) -> str:
    """Return the most-advanced status among a group's members, or
    ``"unlabeled"`` when none carry a concrete verdict."""
    best = "unlabeled"
    best_rank = -1
    for st in statuses:
        r = GROUP_STATUS_RANK.get(st, -1)
        if r > best_rank:
            best_rank = r
            best = st
    return best


def register(app: FastAPI, router_ctx: RouterContext) -> None:
    cache = router_ctx.cache
    annotator = router_ctx.annotator

    @app.get("/api/runs/{run_id}/clusters")
    def list_clusters(
        run_id: str,
        source: Optional[str] = None,
        workbook: Optional[str] = None,
        status: Optional[str] = None,
        error_type: Optional[str] = None,
        limit: int = Query(500, ge=1, le=5000),
        offset: int = Query(0, ge=0),
    ) -> Dict[str, Any]:
        ctx = cache.get(run_id)
        latest = ctx.store.latest_by_target()
        filtered: List[Dict[str, Any]] = []

        # Anchor value (= grouping key) for collapsing the fanned
        # same-value detections into one labellable item. Shared with the
        # stats bar via ``make_anchor_value_fn`` so the queue and the
        # per-status counts agree on what "one issue" is.
        _anchor = make_anchor_value_fn(ctx)

        def _anchor_value(item_: Dict[str, Any]) -> Optional[str]:
            return _anchor(
                item_.get("source"), item_.get("row"),
                item_.get("columns"), item_.get("error_types"),
            )

        for c in ctx.clusters:
            if source and c.get("source") != source:
                continue
            if workbook:
                src_name = c.get("source", "") or ""
                if ctx.source_workbook.get(src_name, "") != workbook:
                    continue
            if error_type and error_type not in (c.get("error_types") or []):
                continue
            label = latest.get(c.get("cluster_id", ""))
            label_status = label.status.value if label else "unlabeled"
            if status and status != label_status:
                continue

            item = _cluster_summary(ctx.canonical_cluster(c.get("cluster_id", "")) or c)
            item["label_status"] = label_status
            if label is not None:
                item["corrected_value"] = label.corrected_value
                item["note"] = label.note
                item["annotator"] = label.annotator
                item["ts"] = label.ts
            item["_anchor_value"] = _anchor_value(item)
            filtered.append(item)

        # Dedup: collapse clusters that resolve to the SAME canonical
        # (source, row, frozenset(columns)) key. This is the historical
        # version of the same fix that ``flatten_issues`` now applies
        # pre-clustering: the LLM occasionally returns differently-
        # mangled forms of the same multi-line header, the aggregator
        # clusters on raw strings (so it doesn't merge them), and the
        # serve-time canonicalizer maps both back to the actual frame
        # column - producing apparent "two clusters on the same cell"
        # in the UI. Picking a primary keeps existing labels visible
        # while the duplicates remain reachable by direct cluster_id
        # lookup (so prior labels on absorbed clusters never disappear).
        primary_by_key: Dict[Any, Dict[str, Any]] = {}
        absorbed_count = 0
        for item in filtered:
            cols = tuple(sorted(item.get("columns") or []))
            row = item.get("row")
            key = (item.get("source") or "", row, cols)
            # Only dedup when the column set is non-empty AND the row
            # is concrete; row-less / column-less defects are dataset-
            # wide flags that legitimately exist as separate clusters.
            if not cols or not isinstance(row, int):
                primary_by_key[id(item)] = item
                continue
            existing = primary_by_key.get(key)
            if existing is None:
                primary_by_key[key] = item
                continue
            absorbed_count += 1
            # Pick the better cluster as the primary. Preference order:
            #   1. labelled > unlabeled (don't hide an annotator's work),
            #   2. higher evidence_count (more corroboration),
            #   3. lower cluster_id (stable, deterministic).
            def _rank(it: Dict[str, Any]) -> tuple:
                labelled = (it.get("label_status") or "unlabeled") != "unlabeled"
                return (
                    not labelled,
                    -int(it.get("evidence_count") or 0),
                    str(it.get("cluster_id") or ""),
                )
            if _rank(item) < _rank(existing):
                primary_by_key[key] = item
                # Track absorbed predecessor so the UI can show "+N more".
                merged = list(existing.get("merged_cluster_ids") or [])
                merged.append(existing.get("cluster_id"))
                merged.extend(item.get("merged_cluster_ids") or [])
                item["merged_cluster_ids"] = [m for m in merged if m]
            else:
                merged = list(existing.get("merged_cluster_ids") or [])
                merged.append(item.get("cluster_id"))
                merged.extend(item.get("merged_cluster_ids") or [])
                existing["merged_cluster_ids"] = [m for m in merged if m]
        # Rebuild the filtered list from the dedup'd values, preserving
        # whatever ordering hash-iteration gives us; the explicit sort
        # below is the authoritative final order.
        filtered = list(primary_by_key.values())
        if absorbed_count:
            logger.debug(
                "queue-dedup: collapsed %d duplicate cluster(s) on run=%s",
                absorbed_count, run_id,
            )

        # ---- Group fanned single-column same-value detections --------
        # Single-column types (typo / inconsistency / semantic /
        # numeric_outlier) are fanned by ``cluster_issues`` into one
        # cluster per affected row, so the SAME bad value in one column
        # shows up as N queue rows (one primary + N-1 inferred). Collapse
        # every cluster sharing a (source, column, cell value) into ONE
        # representative queue item so the operator labels the defect
        # ONCE and it counts ONCE. Auto-apply still fans the verdict back
        # out to every underlying row on save (``/clusters/propagate-
        # status`` for the flagged siblings + ``/labels/propagate-same-
        # value`` for the LLM-missed rows), so no row is lost.
        #
        # Multi-row aggregate types (row_duplication / FD / calculation /
        # cross-table) are excluded - they already aggregate many rows
        # into ONE cluster and have their own affected-rows surface.
        _status_rank = GROUP_STATUS_RANK
        groups: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
        passthrough: List[Dict[str, Any]] = []
        for item in filtered:
            gval = item.get("_anchor_value")
            cols = item.get("columns") or []
            # Groupable only when we resolved a non-empty anchor value
            # (which already implies single-column, concrete row, and a
            # non-aggregate type via ``_anchor_value``). Empty values are
            # never grouped - they would collapse every blank cell in the
            # column together.
            if gval is None or gval == "":
                item.pop("_anchor_value", None)
                passthrough.append(item)
                continue
            groups.setdefault(
                (item.get("source") or "", cols[0], gval), []
            ).append(item)

        grouped_reps: List[Dict[str, Any]] = []
        for members in groups.values():
            if len(members) == 1:
                rep = members[0]
                rep.pop("_anchor_value", None)
                grouped_reps.append(rep)
                continue
            # Representative = the member best suited to anchor the group:
            # prefer a primary (non-inferred) detection so the detail pane
            # shows a real detection, then the most-advanced label, then
            # the lowest row for a stable cursor order.
            def _rep_rank(it: Dict[str, Any]) -> tuple:
                row_ = it.get("row")
                return (
                    1 if it.get("inferred") else 0,
                    -_status_rank.get(it.get("label_status") or "unlabeled", -1),
                    row_ if isinstance(row_, int) else (1 << 30),
                    str(it.get("cluster_id") or ""),
                )

            rep = min(members, key=_rep_rank)
            rows_sorted = sorted({
                m.get("row") for m in members if isinstance(m.get("row"), int)
            })
            rep["group_member_ids"] = [
                m.get("cluster_id") for m in members if m.get("cluster_id")
            ]
            rep["group_rows"] = rows_sorted
            rep["group_size"] = len(rep["group_member_ids"])
            # Group verdict = the most-advanced label across members so a
            # partly-labelled group (e.g. post-propagation) reads as
            # labelled and the dominant verdict colours the queue dot.
            best_member = max(
                members,
                key=lambda m: _status_rank.get(
                    m.get("label_status") or "unlabeled", -1
                ),
            )
            best_status = best_member.get("label_status") or "unlabeled"
            rep["label_status"] = best_status
            if best_status != "unlabeled" and best_member is not rep:
                for k in ("corrected_value", "note", "annotator", "ts"):
                    if k in best_member:
                        rep[k] = best_member[k]
            # Surface the full row span so the queue shows a "xN" badge
            # ("one TP/FP click labels them all") and the detail pane can
            # list the covered rows.
            if len(rows_sorted) > 1:
                rep["affected_rows"] = rows_sorted
            rep.pop("_anchor_value", None)
            grouped_reps.append(rep)

        grouped_count = len(filtered) - (len(passthrough) + len(grouped_reps))
        filtered = passthrough + grouped_reps
        if grouped_count:
            logger.debug(
                "queue-group: collapsed %d fanned same-value cluster(s) on "
                "run=%s", grouped_count, run_id,
            )

        # Sort by location so the operator sees clusters in the same
        # order they'd read the data: workbook (parent dir / table_group)
        # first so siblings of one logical workbook are adjacent, then
        # CSV name, then row index, then the first flagged column's
        # position in the CSV (so the cursor walks left-to-right within
        # a row instead of jumping around).
        col_pos_cache: Dict[str, Dict[str, int]] = {}

        def _first_col_pos(item_: Dict[str, Any]) -> int:
            cols = item_.get("columns") or []
            if not cols:
                return -1
            src = item_.get("source") or ""
            pos_map = col_pos_cache.get(src)
            if pos_map is None:
                try:
                    headers = list(ctx.load_frame(src).columns) if src else []
                except FileNotFoundError:
                    headers = []
                pos_map = {h: i for i, h in enumerate(headers)}
                col_pos_cache[src] = pos_map
            return min(
                (pos_map.get(c, len(pos_map)) for c in cols),
                default=len(pos_map),
            )

        def _sort_key(item_: Dict[str, Any]):
            src = item_.get("source") or ""
            wb = ctx.source_workbook.get(src, "") or ""
            row = item_.get("row")
            return (
                wb,
                src,
                row if isinstance(row, int) else 1 << 30,
                _first_col_pos(item_),
                item_.get("cluster_id", ""),
            )

        filtered.sort(key=_sort_key)

        return {
            "total": len(filtered),
            "offset": offset,
            "limit": limit,
            "items": filtered[offset : offset + limit],
        }

    @app.get("/api/runs/{run_id}/clusters/{cluster_id}")
    def get_cluster(run_id: str, cluster_id: str) -> Dict[str, Any]:
        ctx = cache.get(run_id)
        c = ctx.canonical_cluster(cluster_id)
        if c is None:
            raise HTTPException(404, f"cluster not found: {cluster_id}")
        label = ctx.store.latest_for(cluster_id)
        return {
            "cluster": c,
            "label": label.model_dump() if label else None,
            "history": [h.model_dump() for h in ctx.store.history(cluster_id)],
            "web_verification": ctx.verification(cluster_id),
        }

    @app.get("/api/runs/{run_id}/clusters/{cluster_id}/web-verification")
    def get_web_verification(
        run_id: str, cluster_id: str,
    ) -> Dict[str, Any]:
        """Return the persisted web-verification record for one cluster.

        ``404`` when the verifier has never run on this cluster (the
        operator can still label manually). Run ``python -m
        experiments.run_web_verify --run-id <run>`` to populate the
        store.
        """
        ctx = cache.get(run_id)
        if ctx.cluster(cluster_id) is None:
            raise HTTPException(404, f"cluster not found: {cluster_id}")
        ver = ctx.verification(cluster_id)
        if ver is None:
            raise HTTPException(
                404,
                "no web verification on file for this cluster "
                "(run experiments.run_web_verify to populate)",
            )
        return ver

    @app.post("/api/runs/{run_id}/clusters/{cluster_id}/web-verification")
    def create_web_verification(
        run_id: str, cluster_id: str,
        retry: bool = Query(
            False,
            description=(
                "When ``true``, replace an existing **errored** "
                "verification record (e.g. an Azure agent timeout) "
                "instead of returning 409. Has no effect when the "
                "prior record is non-errored - those still 409 to "
                "prevent accidental overwrite."
            ),
        ),
    ) -> Dict[str, Any]:
        """Trigger a single-cluster web-verification on demand.

        Used by the "Verify on web" button in the labeling UI for
        clusters the batch verifier didn't pick up. Synchronous - the
        UI shows a spinner while we make the Foundry agent call
        (typically 5-30 s per cluster, depending on how many web
        searches the agent decides to run). Returns the new
        verification record on success.

        Status codes:
        * ``200`` - verification produced (verdict may be any of
          confirmed_error / false_positive / ambiguous / errored).
        * ``404`` - cluster not found in this run.
        * ``409`` - cluster already has a non-errored verification
          (UI bug or double-click; refuse to clobber). For an
          errored prior record, the UI may pass ``?retry=true`` to
          force a re-run.
        * ``422`` - cluster is gate-rejected (PII column / non-entity
          column with no whitelisted error_type / unsuitable value).
        * ``500`` - missing agent config, prompt file unreadable, etc.
        """
        ctx = cache.get(run_id)
        if ctx.cluster(cluster_id) is None:
            raise HTTPException(404, f"cluster not found: {cluster_id}")
        # Importing inside the handler keeps the labeling-app
        # importable even when the verifier package has unmet
        # dependencies (the only network-side dep is azure-ai-agents,
        # which is required by requirements.txt; be defensive anyway).
        try:
            from experiments.run_web_verify import (  # noqa: WPS433
                VerifyError,
                verify_single_cluster,
            )
        except ImportError as exc:
            raise HTTPException(500, f"verifier unavailable: {exc}")
        try:
            # ``verify_single_cluster`` defaults to
            # ``include_person=True`` so person-name columns flow
            # through with the lenient "valid if anyone has the
            # name" criterion. See ``GateConfig.verify_person`` for
            # the underlying gate.
            record = verify_single_cluster(
                ctx.run_dir, cluster_id,
                retry_errored=retry,
            )
        except VerifyError as exc:
            code_map = {
                "not_found": 404,
                "already_verified": 409,
                "gate_rejected": 422,
                "internal": 500,
            }
            raise HTTPException(
                code_map.get(exc.code, 500),
                exc.message,
            )
        except FileNotFoundError as exc:
            # Most likely a missing prompt file; surface as 500.
            raise HTTPException(500, str(exc))
        # ``verify_single_cluster`` already wrote the record to disk
        # atomically. ctx.verification() re-reads via mtime check and
        # returns the canonical dict shape (matching what the GET
        # endpoint serves).
        result = ctx.verification(cluster_id)
        if result is None:
            # Should never happen - the file was just written - but
            # fall back to converting the in-memory record so the
            # client gets *something*.
            result = {
                "cluster_id": record.cluster_id,
                "verdict": record.verdict,
                "canonical_value": record.canonical_value,
                "confidence": record.confidence,
                "error": record.error,
            }
        return result

    # -------- column-cluster value-corrections (LLM suggest + persist) --
    #
    # Live counterpart to the offline
    # ``experiments.export_inconsistency_benchmark_with_llm`` script.
    # The shared logic lives in :mod:`src.labeling.correction_suggester`
    # so the on-disk LLM cache is binary-compatible between the two
    # entry points - re-running the export after the operator has
    # used these endpoints will pick up cached samples for free.
    #
    # The persisted output is a per-cluster ``value_corrections``
    # mapping on the latest column-cluster Label record (see
    # :class:`LabelIn.value_corrections`). Operators may save partial
    # mappings; an absent key means "drop this value from the
    # benchmark export" rather than "use LLM suggestion".

    def _parse_cluster_target(cluster_id: str) -> Tuple[str, str, str]:
        """Decode the ``colclust\\x00src\\x00col\\x00cN`` target.

        Raises ``HTTPException(400)`` for malformed payloads so the
        endpoint can short-circuit with a friendly error.
        """
        parts = correction_suggester.parse_colclust_target_id(cluster_id)
        if parts is None:
            raise HTTPException(
                400,
                f"invalid column-cluster id: {cluster_id!r} "
                f"(expected 'colclust\\x00src\\x00col\\x00cN')",
            )
        return parts

    def _saved_value_corrections_for(
        ctx: "RunContext", target_id: str,
    ) -> Dict[str, Union[str, List[str]]]:
        """Fetch the latest persisted ``value_corrections`` for a target.

        Empty dict when the cluster has no label record yet, or its
        latest record carries an empty mapping.
        """
        latest = ctx.store.latest_for(target_id)
        if latest is None:
            return {}
        return dict(latest.value_corrections or {})

    @app.post(
        "/api/runs/{run_id}/column-cluster-suggest-corrections"
    )
    def suggest_cluster_corrections(
        run_id: str,
        cluster_id: str = Query(
            ...,
            description=(
                "Column-cluster target id "
                "(colclust\\x00src\\x00col\\x00cN). Passed as a query "
                "param, not a path segment, so column names containing "
                "'/' (e.g. 'Shipping State/Province') don't break routing."
            ),
        ),
        payload: Dict[str, Any] = Body(default_factory=dict),
    ) -> Dict[str, Any]:
        """Run the LLM correction suggester for ONE column-cluster.

        Synchronous; typically completes in a few seconds when the
        cache is warm and 30-90 s on a cold cluster (K=3 + retry-4).
        The UI uses this for the per-cluster "Generate corrections"
        button.

        Body fields (all optional):
          * ``samples`` (int, default 3) - K self-consistency samples.
          * ``samples_threshold`` (float, default 0.5) - vote frac.
          * ``retry_samples`` (int, default 4) - extra samples when
            consensus = [original_value].
          * ``model`` (str, default ``"gpt-5.2"``).
          * ``reasoning_effort`` (str, default ``"medium"``).
          * ``row_context_rows`` (int, default 3).

        Returns ``{"suggestions": [...], "saved_value_corrections": {...}}``
        where each suggestion describes one (value, rows) tuple in the
        cluster.
        """
        ctx = cache.get(run_id)
        s, c, lid = _parse_cluster_target(cluster_id)
        if s not in ctx.sources:
            raise HTTPException(404, f"source not found: {s}")

        samples = int(payload.get("samples", 3))
        samples_threshold = float(payload.get("samples_threshold", 0.5))
        retry_samples = int(payload.get("retry_samples", 4))
        model = str(payload.get("model", "gpt-5.2"))
        reasoning_effort = str(payload.get("reasoning_effort", "medium"))
        row_context_rows = int(payload.get("row_context_rows", 3))

        try:
            llm_client = LLMClient()
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(500, f"LLM client unavailable: {exc}")

        try:
            results = correction_suggester.suggest_corrections_for_clusters(
                ctx.run_dir,
                targets=[(s, c, lid)],
                llm_client=llm_client,
                model=model,
                reasoning_effort=reasoning_effort,
                samples=samples,
                samples_threshold=samples_threshold,
                retry_samples=retry_samples,
                row_context_rows=row_context_rows,
                concurrency=8,
            )
        except FileNotFoundError as exc:
            raise HTTPException(
                404,
                f"missing run artifact for correction suggestion: {exc}",
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "suggest_corrections failed for run=%s cluster=%s",
                run_id, cluster_id,
            )
            raise HTTPException(500, f"correction suggestion failed: {exc}")

        suggestions = results.get((s, c, lid), [])
        if not suggestions:
            raise HTTPException(
                404,
                f"no draft cluster matched target {cluster_id!r} "
                f"(check column_cluster_drafts.jsonl in this run)",
            )
        return {
            "cluster_id": cluster_id,
            "source": s,
            "column": c,
            "cluster_local_id": lid,
            "suggestions": correction_suggester.suggestions_to_payload(suggestions),
            "saved_value_corrections": _saved_value_corrections_for(
                ctx, cluster_id,
            ),
        }

    @app.get(
        "/api/runs/{run_id}/column-cluster-corrections"
    )
    def get_cluster_corrections(
        run_id: str,
        cluster_id: str = Query(
            ...,
            description=(
                "Column-cluster target id "
                "(colclust\\x00src\\x00col\\x00cN). Query param (not a "
                "path segment) so '/'-containing column names route."
            ),
        ),
    ) -> Dict[str, Any]:
        """Return the latest persisted ``value_corrections`` for a cluster
        plus any cached LLM suggestions (no LLM calls).

        The UI calls this when opening a cluster: if
        ``saved_value_corrections`` is non-empty, the UI shows the
        operator's curated mapping; if it is empty AND
        ``cached_suggestions`` are available, the UI offers a one-click
        "Use cached suggestions" affordance. Otherwise the operator
        must invoke ``POST /suggest-corrections`` to spend LLM budget.
        """
        ctx = cache.get(run_id)
        s, c, lid = _parse_cluster_target(cluster_id)
        if s not in ctx.sources:
            raise HTTPException(404, f"source not found: {s}")

        saved = _saved_value_corrections_for(ctx, cluster_id)

        # Cache-only path: pass dry_run=True so dispatch_groups returns
        # whatever lives in inconsistency_llm_cache.jsonl without ever
        # constructing an LLM client.
        cached_suggestions: List[Dict[str, Any]] = []
        try:
            results = correction_suggester.suggest_corrections_for_clusters(
                ctx.run_dir,
                targets=[(s, c, lid)],
                llm_client=None,
                samples=3,
                samples_threshold=0.5,
                retry_samples=4,
                row_context_rows=3,
                concurrency=1,
                dry_run=True,
            )
            cached_suggestions = correction_suggester.suggestions_to_payload(
                results.get((s, c, lid), []),
            )
        except FileNotFoundError:
            # Run still missing draft artifact - that's fine; saved
            # corrections (if any) are still returned below.
            cached_suggestions = []
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "get_cluster_corrections cache-replay failed for "
                "run=%s cluster=%s: %s",
                run_id, cluster_id, exc,
            )
            cached_suggestions = []

        return {
            "cluster_id": cluster_id,
            "source": s,
            "column": c,
            "cluster_local_id": lid,
            "saved_value_corrections": saved,
            "cached_suggestions": cached_suggestions,
        }

    @app.post(
        "/api/runs/{run_id}/column-cluster-corrections"
    )
    def save_cluster_corrections(
        run_id: str,
        cluster_id: str = Query(
            ...,
            description=(
                "Column-cluster target id "
                "(colclust\\x00src\\x00col\\x00cN). Query param (not a "
                "path segment) so '/'-containing column names route."
            ),
        ),
        payload: Dict[str, Any] = Body(...),
    ) -> Dict[str, Any]:
        """Persist a ``value_corrections`` mapping for one cluster.

        Body fields:
          * ``value_corrections`` (dict, required) - mapping from each
            original surface form in the cluster to its correction
            (scalar = one-direction, list = multi-direction). An ABSENT
            key means "drop this value from the benchmark export".
          * ``note_tag`` (str, optional, default ``"[manual]"``) -
            short prefix appended to the existing note for audit trail.
            Common values: ``"[llm-suggested]"`` (operator accepted the
            LLM consensus verbatim), ``"[llm-edited]"`` (operator edited
            the LLM consensus), ``"[manual]"`` (operator typed the
            corrections from scratch).

        Builds a fresh Label record from the latest cluster verdict's
        fields (status / source / column / columns / confirmed_canonical /
        note) plus the new ``value_corrections`` and the appended
        ``note_tag``. Requires a prior verdict in {tp, unsure}; saving
        corrections on an FP/skip cluster is rejected.
        """
        ctx = cache.get(run_id)
        s, c, lid = _parse_cluster_target(cluster_id)
        if s not in ctx.sources:
            raise HTTPException(404, f"source not found: {s}")

        raw = payload.get("value_corrections")
        if not isinstance(raw, dict):
            raise HTTPException(
                400, "value_corrections must be a JSON object",
            )
        # Coerce + validate: keys must be strings, values must be
        # str | list[str]; reject any other shape so we don't silently
        # store junk that the export pipeline later mishandles.
        normalized: Dict[str, Union[str, List[str]]] = {}
        for k, v in raw.items():
            if not isinstance(k, str):
                raise HTTPException(
                    400, f"value_corrections keys must be strings, got {k!r}",
                )
            if isinstance(v, str):
                normalized[k] = v
            elif isinstance(v, list):
                if not all(isinstance(x, str) for x in v):
                    raise HTTPException(
                        400,
                        f"value_corrections[{k!r}] list elements must be strings",
                    )
                normalized[k] = list(v)
            else:
                raise HTTPException(
                    400,
                    f"value_corrections[{k!r}] must be string or list of strings, "
                    f"got {type(v).__name__}",
                )

        latest = ctx.store.latest_for(cluster_id)
        if latest is None:
            raise HTTPException(
                404,
                f"cluster {cluster_id!r} has no prior verdict; "
                f"label it tp/unsure before saving corrections",
            )
        if latest.status not in (LabelStatus.TP, LabelStatus.UNSURE):
            raise HTTPException(
                400,
                f"cluster verdict is {latest.status!r}; "
                f"corrections are only valid for status in {{tp, unsure}}",
            )

        note_tag = str(payload.get("note_tag") or "[manual]").strip() or "[manual]"
        prior_note = (latest.note or "").strip()
        if note_tag in prior_note:
            new_note = prior_note
        elif prior_note:
            new_note = f"{note_tag}; {prior_note}"
        else:
            new_note = note_tag

        new_label = Label(
            target=LabelTarget.COLUMN_CLUSTER,
            target_id=cluster_id,
            source=latest.source,
            row=latest.row,
            column=latest.column,
            columns=list(latest.columns),
            status=latest.status,
            corrected_value=latest.corrected_value,
            corrected_values=dict(latest.corrected_values),
            confirmed_canonical=latest.confirmed_canonical,
            value_corrections=normalized,
            note=new_note,
            annotator=(payload.get("annotator") or annotator).strip()
                or annotator,
        )
        ctx.store.append(new_label)
        return {
            "cluster_id": cluster_id,
            "saved_value_corrections": normalized,
            "note": new_note,
        }

    # -------- batch correction-suggestion job (async) ------------------
    #
    # Enumerates every TP/unsure column-cluster in the run (optionally
    # filtered to one source), kicks off the same shared dispatcher in
    # a background thread, and returns a job-id the UI can poll. Cache
    # hits make most batches finish in seconds; cold runs may take
    # several minutes.

