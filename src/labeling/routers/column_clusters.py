"""Column-cluster (inconsistency) labelling endpoints.

Backed by ``<run_dir>/column_clusters.jsonl`` (operator-saved value
groups) and ``column_cluster_drafts.jsonl`` (LLM drafts). Serves the
inconsistency-column queue, per-column value/cluster/draft views (with
both ``.../values/{column:path}`` and ``.../{column}/values`` URL
shapes), the saved-record CRUD, and per-column stats.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException, Query

from ..schema import (
    ColumnClusters,
    ColumnClustersIn,
    ColumnLabelStatus,
    column_cluster_target_id,
)
from ._context import RouterContext

logger = logging.getLogger(__name__)


def register(app: FastAPI, router_ctx: RouterContext) -> None:
    cache = router_ctx.cache
    annotator = router_ctx.annotator

    def _column_value_counts(
        ctx: "RunContext", source: str, column: str,
    ) -> Tuple[Dict[str, int], int]:
        """Return ``(value -> count, n_rows)`` for one CSV column.

        Empty strings (``""``) are excluded from the count map -- a
        standardization cluster over a blank value is meaningless. The
        empty-cell tally is still reflected in the returned ``n_rows``
        for context.
        """
        frame = ctx.load_frame(source)
        if column not in frame.columns:
            raise HTTPException(
                404, f"column not found in {source}: {column}",
            )
        series = frame[column].astype(str)
        n_rows = int(len(series))
        counts: Dict[str, int] = {}
        for val in series:
            if val == "":
                continue
            counts[val] = counts.get(val, 0) + 1
        return counts, n_rows

    @app.get("/api/runs/{run_id}/inconsistency-columns")
    def list_inconsistency_columns(
        run_id: str,
        source: Optional[str] = None,
        min_distinct: int = Query(2, ge=1),
        # A column whose distinct-value ratio (n_distinct / n_rows)
        # exceeds this is hidden. Defaults to ``1.0`` so every
        # column with >= ``min_distinct`` distinct values surfaces
        # for human review (matches the column-clustering driver,
        # which sends them all to the LLM proposer). Lower the
        # value to re-enable the historical PK / free-text guard.
        max_distinct_ratio: float = Query(1.0, gt=0.0, le=1.0),
        # Maximum total candidates returned in one response. Defaults
        # generously; the UI is expected to scope by ``source``.
        limit: int = Query(2000, ge=1, le=20000),
    ) -> Dict[str, Any]:
        """Enumerate candidate columns for standardization labelling.

        For each (source, column) the response carries:
          * ``n_rows`` - total rows in the source.
          * ``n_distinct`` - distinct non-empty values.
          * ``top_values`` - up to 3 most-frequent values + counts
            (so the UI can render a queue hint).
          * ``status`` / ``n_clusters`` / ``annotator`` / ``ts`` from
            the latest ``column_clusters.jsonl`` record, or
            ``"unlabeled"`` when nothing has been written yet.

        Cardinality filter: ``min_distinct`` excludes columns that
        mathematically can't form a cluster (< 2 distinct values).
        ``max_distinct_ratio`` defaults to ``1.0`` -- every other
        column is surfaced for human review. Callers that want the
        old PK / free-text guard can pass a lower value.
        """
        ctx = cache.get(run_id)
        store = ctx.column_cluster_store
        latest = store.latest_by_column()

        # Build a (source, column) -> n_draft_clusters index from
        # column_cluster_drafts.jsonl. Only a fraction of candidate
        # columns actually produce non-empty drafts (~7% in practice),
        # so the frontend's "Has draft" filter and the queue sort
        # need this signal up front - otherwise the operator clicks
        # through dozens of empty-draft columns before finding one
        # the LLM had anything to say about. Returns an empty map if
        # the file is missing (single-pass / non-adaptive runs).
        n_draft_index: Dict[Tuple[str, str], int] = {}
        try:
            rd = ctx.run_dir
            drafts_path = rd / "column_cluster_drafts.jsonl"
            if drafts_path.exists():
                with drafts_path.open("r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            d = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        s = d.get("source") or ""
                        c = d.get("column") or ""
                        if not s or not c:
                            continue
                        dc = d.get("draft_clusters") or []
                        n_draft_index[(s, c)] = len(dc)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "inconsistency-columns: failed to index drafts: %s", exc,
            )
            n_draft_index = {}

        items: List[Dict[str, Any]] = []
        sources = [source] if source else list(ctx.sources.keys())
        for src in sources:
            if src not in ctx.sources:
                continue
            try:
                frame = ctx.load_frame(src)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "inconsistency-columns: failed to load %s: %s",
                    src, exc,
                )
                continue
            n_rows = int(len(frame))
            if n_rows == 0:
                continue
            for col in frame.columns:
                series = frame[col].astype(str)
                counts: Dict[str, int] = {}
                for val in series:
                    if val == "":
                        continue
                    counts[val] = counts.get(val, 0) + 1
                n_distinct = len(counts)
                if n_distinct < min_distinct:
                    continue
                if n_rows > 0 and n_distinct / n_rows > max_distinct_ratio:
                    continue
                top = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:3]
                rec = latest.get((src, col))
                # ``n_draft_clusters`` is the count of LLM-proposed
                # candidate clusters for this column (0 means the LLM
                # ran but didn't find anything, distinguishable from
                # "no LLM pass ran at all" only via the run's overall
                # drafts file existence -- which the queue payload
                # doesn't surface).
                n_drafts = n_draft_index.get((src, col), 0)
                items.append({
                    "source": src,
                    "column": col,
                    "n_rows": n_rows,
                    "n_distinct": n_distinct,
                    "top_values": [
                        {"value": v, "count": c} for v, c in top
                    ],
                    "status": rec.status.value if rec else
                        ColumnLabelStatus.UNLABELED.value,
                    "n_clusters": len(rec.clusters) if rec else 0,
                    "n_draft_clusters": n_drafts,
                    "annotator": rec.annotator if rec else None,
                    "ts": rec.ts if rec else None,
                })

        # Stable order: unlabeled first (most actionable), then
        # columns with at least one LLM-proposed cluster (so the
        # operator lands on something the LLM actually flagged
        # instead of scrolling through dozens of empty drafts),
        # then by source / column.
        order = {
            ColumnLabelStatus.UNLABELED.value: 0,
            ColumnLabelStatus.LABELED.value: 1,
            ColumnLabelStatus.NO_INCONSISTENCY.value: 2,
            ColumnLabelStatus.DROPPED_OVERLAPPING.value: 3,
            ColumnLabelStatus.DROPPED_MESSY.value: 4,
        }
        items.sort(key=lambda x: (
            order.get(x["status"], 99),
            0 if x["n_draft_clusters"] > 0 else 1,
            x["source"],
            x["column"],
        ))
        total = len(items)
        # Headline count of how many entries in the full queue
        # carry an LLM-proposed cluster -- the frontend uses this
        # to decide whether to surface a "with LLM draft" filter
        # button.
        total_with_draft = sum(
            1 for it in items if it["n_draft_clusters"] > 0
        )
        return {
            "total": total,
            "total_with_draft": total_with_draft,
            "items": items[:limit],
        }

    @app.get(
        "/api/runs/{run_id}/columns/{source}/{column}/values"
    )
    def get_column_values(
        run_id: str, source: str, column: str,
        # When the column has thousands of distinct values, return
        # only the top-N by frequency. The labelling UI shows them
        # ordered; rare-tail singletons are surfaced via a separate
        # tray in Slice 6.
        limit: int = Query(500, ge=1, le=10000),
    ) -> Dict[str, Any]:
        """Return distinct values + per-value counts for one column.

        Used by the column-board frontend (Slice 6) to render the
        clusterable value list. Empty cells are excluded from the
        value list but counted in the ``n_empty`` summary so the
        operator can tell whether the column has missing data.
        """
        ctx = cache.get(run_id)
        if source not in ctx.sources:
            raise HTTPException(404, f"source not found: {source}")
        counts, n_rows = _column_value_counts(ctx, source, column)
        items = [
            {"value": v, "count": c}
            for v, c in sorted(
                counts.items(), key=lambda kv: (-kv[1], kv[0]),
            )
        ]
        n_empty = n_rows - sum(counts.values())
        return {
            "source": source,
            "column": column,
            "n_rows": n_rows,
            "n_empty": n_empty,
            "n_distinct": len(counts),
            "values": items[:limit],
            "truncated": len(items) > limit,
        }

    @app.get(
        "/api/runs/{run_id}/columns/{source}/{column}/clusters"
    )
    def get_column_clusters(
        run_id: str, source: str, column: str,
    ) -> Dict[str, Any]:
        """Return the latest column-clusters record for (source, column).

        ``record`` is ``null`` when nothing has been written yet.
        ``history`` carries every append-only record for the (source,
        column) pair in write order (oldest first).
        """
        store = cache.get(run_id).column_cluster_store
        latest = store.latest_for(source, column)
        history = store.history(source, column)
        return {
            "record": latest.model_dump(mode="json") if latest else None,
            "history": [h.model_dump(mode="json") for h in history],
        }

    @app.post(
        "/api/runs/{run_id}/columns/{source}/{column}/clusters"
    )
    def post_column_clusters(
        run_id: str, source: str, column: str,
        payload: ColumnClustersIn,
    ) -> Dict[str, Any]:
        """Write a column-clusters record.

        The path's ``source`` / ``column`` win over the payload's --
        the frontend sends them in the body for convenience but the
        URL is the source of truth. Cluster invariants are enforced
        by :class:`ColumnClustersIn`'s validators; this handler
        additionally validates that every clustered value actually
        appears in the column (so a stale draft can't poison the
        record).
        """
        ctx = cache.get(run_id)
        if source not in ctx.sources:
            raise HTTPException(404, f"source not found: {source}")
        # Validate that values referenced by clusters exist in the
        # column. We don't validate the count fields against the
        # actual count -- the human/LLM annotator may quote a stale
        # count from a draft; the scorer joins against the live CSV
        # anyway. But if a value isn't in the column at all the
        # record is unusable downstream.
        counts, _ = _column_value_counts(ctx, source, column)
        unknown: List[str] = []
        for cl in payload.clusters:
            for cv in cl.values:
                if cv.value not in counts:
                    unknown.append(cv.value)
        if unknown:
            raise HTTPException(
                400,
                f"values not present in {source}/{column}: "
                f"{unknown[:5]}{'...' if len(unknown) > 5 else ''}",
            )

        store = cache.get(run_id).column_cluster_store
        record = ColumnClusters(
            source=source,
            column=column,
            status=payload.status,
            drop_reason=payload.drop_reason,
            clusters=payload.clusters,
            note=payload.note,
            annotator=(
                (payload.annotator or annotator).strip() or annotator
            ),
        )
        saved = store.append(record)
        return {
            "record": saved.model_dump(mode="json"),
            "history": [
                h.model_dump(mode="json")
                for h in store.history(source, column)
            ],
        }

    @app.get(
        "/api/runs/{run_id}/columns/{source}/{column}/draft"
    )
    def get_column_draft(
        run_id: str, source: str, column: str,
    ) -> Dict[str, Any]:
        """Return the LLM-proposed draft clusters for (source, column).

        Reads ``<run_dir>/column_cluster_drafts.jsonl`` (written by
        ``experiments.run_column_clustering``). When the run has no
        draft file, or no draft for this column, returns
        ``{"draft": null}``. When multiple drafts exist for the same
        (source, column), latest-wins (the file is append-only).
        """
        rd = cache._safe_run_dir(run_id)
        if not rd.is_dir():
            raise HTTPException(404, f"run not found: {run_id}")
        drafts_path = rd / "column_cluster_drafts.jsonl"
        if not drafts_path.exists():
            return {"draft": None}
        found: Optional[Dict[str, Any]] = None
        try:
            with drafts_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:  # noqa: BLE001 - skip malformed
                        continue
                    if (rec.get("source") == source
                            and rec.get("column") == column):
                        found = rec
        except OSError as exc:
            logger.warning("get_column_draft: read failed: %s", exc)
            return {"draft": None}
        return {"draft": found}

    # ------------------------------------------------------------------
    # Column endpoints with the action *before* the column name + a
    # ``{column:path}`` converter so the column can contain literal
    # slashes (e.g. "Project Name/Description"). Starlette decodes
    # ``%2F`` to ``/`` before path matching, so the original routes
    # of the form ``.../{column}/<action>`` 404 on columns with a
    # slash. These aliases mount the *same* handlers under a path
    # shape that places ``column`` as the trailing path segment.
    # The original routes are kept for backward compatibility.
    # ------------------------------------------------------------------

    @app.get("/api/runs/{run_id}/columns/{source}/values/{column:path}")
    def get_column_values_alias(
        run_id: str, source: str, column: str,
        limit: int = Query(500, ge=1, le=10000),
    ) -> Dict[str, Any]:
        return get_column_values(run_id, source, column, limit)

    @app.get("/api/runs/{run_id}/columns/{source}/clusters/{column:path}")
    def get_column_clusters_alias(
        run_id: str, source: str, column: str,
    ) -> Dict[str, Any]:
        return get_column_clusters(run_id, source, column)

    @app.post("/api/runs/{run_id}/columns/{source}/clusters/{column:path}")
    def post_column_clusters_alias(
        run_id: str, source: str, column: str,
        payload: ColumnClustersIn,
    ) -> Dict[str, Any]:
        return post_column_clusters(run_id, source, column, payload)

    @app.get("/api/runs/{run_id}/columns/{source}/draft/{column:path}")
    def get_column_draft_alias(
        run_id: str, source: str, column: str,
    ) -> Dict[str, Any]:
        return get_column_draft(run_id, source, column)

    @app.get("/api/runs/{run_id}/column-clusters")
    def list_column_clusters(
        run_id: str,
        source: Optional[str] = None,
        status: Optional[str] = None,
        include_drafts: bool = True,
        limit: int = Query(5000, ge=1, le=20000),
    ) -> Dict[str, Any]:
        """Flatten every saved column-cluster into queue-ready items.

        One JSON object per ``StdCluster`` across every (source,
        column) record in ``column_clusters.jsonl``. Each item carries
        the data the cell-level label page needs to:
          * render a queue row (source, column, kind, n_values),
          * paint the CSV (full ``values`` list + canonical),
          * render the right pane (kind, canonical, agreement, llm
            suggestion if drafted), and
          * route a TP/FP/Partial/Unsure/Skip verdict back through
            ``POST /labels`` (synthetic ``target_id``).

        The verdict (``label_status``) is joined from the labels
        store so the queue dot color is consistent with cell-level
        clusters on first paint - no extra fetch needed.

        When ``include_drafts`` is true (the default), LLM-proposed
        draft clusters are *also* surfaced for columns whose
        ``column_clusters.jsonl`` record is missing or not in the
        ``labeled`` state. Synthetic items use ``c{idx+1}`` local
        ids matching the draft order and carry
        ``draft_pending: true``. The first verdict on such an item
        lazy-promotes the column's full draft list to a real
        record (see ``post_label``'s COLUMN_CLUSTER branch) so the
        cluster_id assignment stays deterministic.

        Filters:
          * ``source`` - exact CSV basename match.
          * ``status`` - one of ``unlabeled``/``tp``/``fp``/``partial``
            /``unsure``/``skip`` (matches the joined verdict).
          * ``include_drafts`` - set to ``false`` to hide unsaved
            drafts (matches the legacy behavior).
        """
        ctx = cache.get(run_id)
        store = ctx.column_cluster_store
        latest_records = store.latest_by_column()
        label_index = ctx.store.latest_by_target()

        # Pre-load draft clusters so we can attach the agreement /
        # canonical-suggestion to each saved cluster whose value-set
        # matches a draft cluster's value-set. The match key is the
        # ``frozenset`` of cluster values (canonical-agnostic) - the
        # operator may have chosen a different canonical than the
        # LLM did but the cluster's *identity* is the value-set.
        # We deliberately don't surface drafts here that were never
        # saved into a record (see endpoint docstring).
        draft_lookup: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
        try:
            drafts_path = ctx.run_dir / "column_cluster_drafts.jsonl"
            if drafts_path.exists():
                with drafts_path.open("r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            d = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        key = (d.get("source") or "", d.get("column") or "")
                        if not key[0] or not key[1]:
                            continue
                        draft_lookup[key] = d.get("draft_clusters") or []
        except Exception as exc:  # noqa: BLE001
            logger.warning("list_column_clusters: draft index failed: %s", exc)
            draft_lookup = {}

        items: List[Dict[str, Any]] = []
        for (src, col), rec in latest_records.items():
            if source and src != source:
                continue
            # Only surface clusters from records the operator actively
            # *labelled* (status=labeled). Records flagged
            # no_inconsistency / dropped_* by the operator carry no
            # clusters anyway, so this is mostly defensive.
            if rec.status != ColumnLabelStatus.LABELED:
                continue
            drafts = draft_lookup.get((src, col), [])
            for cl in rec.clusters:
                target_id = column_cluster_target_id(src, col, cl.cluster_id)
                lbl = label_index.get(target_id)
                label_status = (
                    lbl.status.value if lbl is not None else "unlabeled"
                )
                if status and label_status != status:
                    continue
                # Match this cluster's value-set against the LLM draft
                # to surface the LLM's proposed canonical + agreement.
                # Use frozenset so the order doesn't matter.
                value_set = frozenset(cv.value for cv in cl.values)
                llm_canonical: Optional[str] = None
                llm_agreement: Optional[float] = None
                llm_kind: Optional[str] = None
                llm_target_style: Optional[str] = None
                for dc in drafts:
                    dvs = frozenset(dc.get("values") or [])
                    if dvs == value_set:
                        llm_canonical = dc.get("canonical")
                        llm_agreement = dc.get("agreement")
                        llm_kind = dc.get("kind")
                        ts = dc.get("target_style")
                        llm_target_style = ts if isinstance(ts, str) and ts else None
                        break
                items.append({
                    "target_id": target_id,
                    "source": src,
                    "column": col,
                    "cluster_local_id": cl.cluster_id,
                    "kind": cl.kind.value,
                    "canonical_value": cl.canonical_value,
                    "values": [
                        {"value": cv.value, "count": cv.count}
                        for cv in cl.values
                    ],
                    "n_values": len(cl.values),
                    "annotator": rec.annotator,
                    "ts": rec.ts,
                    "label_status": label_status,
                    "label_note": (lbl.note if lbl else ""),
                    "label_confirmed_canonical": (
                        lbl.confirmed_canonical if lbl else None
                    ),
                    "label_value_corrections": (
                        dict(lbl.value_corrections) if lbl and lbl.value_corrections else {}
                    ),
                    "llm_canonical": llm_canonical,
                    "llm_agreement": llm_agreement,
                    "llm_kind": llm_kind,
                    "llm_target_style": llm_target_style,
                    "from_draft": llm_canonical is not None,
                    "draft_pending": False,
                    "is_col_cluster": True,
                    "error_types": ["inconsistency"],
                })
                if len(items) >= limit:
                    break
            if len(items) >= limit:
                break

        # Surface LLM drafts as queue items for columns whose record
        # is missing or not in the ``labeled`` state. The synthetic
        # ``target_id`` uses ``c{idx+1}`` matching the draft order so
        # the lazy-promotion in ``post_label`` lines up. Drafts in
        # columns that ARE labelled are skipped here -- they only
        # participate as a cross-reference for the LLM hint (handled
        # above) since their value-sets are already represented by
        # the saved clusters.
        if include_drafts and len(items) < limit:
            labeled_columns: set = {
                (src, col)
                for (src, col), rec in latest_records.items()
                if rec.status == ColumnLabelStatus.LABELED
            }
            for (src, col), drafts in draft_lookup.items():
                if source and src != source:
                    continue
                if (src, col) in labeled_columns:
                    continue
                for idx, dc in enumerate(drafts, start=1):
                    raw_values = dc.get("values") or []
                    # Normalise to {value, count} dicts. Drafts written
                    # by ``run_column_clustering`` may store plain
                    # strings; backfill count=1 so the frontend's
                    # ``data-raw`` paint path still finds the cell.
                    norm_values: List[Dict[str, Any]] = []
                    for v in raw_values:
                        if isinstance(v, dict):
                            val = v.get("value")
                            cnt = v.get("count") or 1
                        else:
                            val = v
                            cnt = 1
                        if not isinstance(val, str) or not val:
                            continue
                        norm_values.append({"value": val, "count": int(cnt)})
                    if len(norm_values) < 2:
                        continue
                    local_id = f"c{idx}"
                    target_id = column_cluster_target_id(src, col, local_id)
                    lbl = label_index.get(target_id)
                    label_status = (
                        lbl.status.value if lbl is not None else "unlabeled"
                    )
                    if status and label_status != status:
                        continue
                    canonical = dc.get("canonical") or dc.get("canonical_value")
                    seen = {nv["value"] for nv in norm_values}
                    if canonical not in seen:
                        canonical = None
                    ts = dc.get("target_style")
                    target_style = ts if isinstance(ts, str) and ts else None
                    items.append({
                        "target_id": target_id,
                        "source": src,
                        "column": col,
                        "cluster_local_id": local_id,
                        "kind": dc.get("kind") or "untagged",
                        "canonical_value": canonical,
                        "values": norm_values,
                        "n_values": len(norm_values),
                        "annotator": None,
                        "ts": None,
                        "label_status": label_status,
                        "label_note": (lbl.note if lbl else ""),
                        "label_confirmed_canonical": (
                            lbl.confirmed_canonical if lbl else None
                        ),
                        "label_value_corrections": (
                            dict(lbl.value_corrections) if lbl and lbl.value_corrections else {}
                        ),
                        "llm_canonical": canonical,
                        "llm_agreement": dc.get("agreement"),
                        "llm_kind": dc.get("kind") or "untagged",
                        "llm_target_style": target_style,
                        "from_draft": True,
                        "draft_pending": True,
                        "is_col_cluster": True,
                        "error_types": ["inconsistency"],
                    })
                    if len(items) >= limit:
                        break
                if len(items) >= limit:
                    break
        # Stable order: source, column, then cluster_local_id. The
        # frontend re-bins by source for the sidebar groups, but a
        # deterministic order keeps the queue snapshot stable
        # across reloads.
        items.sort(key=lambda it: (it["source"], it["column"], it["cluster_local_id"]))
        return {"items": items, "total": len(items)}

    @app.get("/api/runs/{run_id}/column-cluster-stats")
    def column_cluster_stats(run_id: str) -> Dict[str, Any]:
        """Dashboard summary: per-status counts of (source, column) pairs.

        Only counts pairs that have an explicit record. The frontend
        derives ``inconsistency_columns_total`` (the queue size) from
        the ``/inconsistency-columns`` listing, not this endpoint.
        """
        store = cache.get(run_id).column_cluster_store
        latest = store.latest_by_column()
        by_status: Dict[str, int] = {}
        n_clusters_total = 0
        for rec in latest.values():
            by_status[rec.status.value] = by_status.get(rec.status.value, 0) + 1
            n_clusters_total += len(rec.clusters)
        return {
            "labelled_columns": len(latest),
            "by_status": by_status,
            "n_clusters_total": n_clusters_total,
        }

