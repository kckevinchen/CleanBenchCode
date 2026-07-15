"""Label read/write endpoints under ``/api/runs/{run_id}/labels``.

The heart of the workbench: ``GET`` returns the run's full label history
+ latest-per-target map; ``POST`` records a verdict for any target kind
(cluster, issue, cell, or column-cluster), canonicalising the target_id
and lazily promoting an LLM draft to a real column-cluster record on the
operator's first verdict.
"""

from __future__ import annotations

from typing import Any, Dict

from fastapi import FastAPI, HTTPException

from .._cells import _load_drafts_for_column
from ..schema import (
    ColumnClusters,
    ColumnLabelStatus,
    Label,
    LabelIn,
    LabelStatus,
    LabelTarget,
    cell_target_id,
    column_cluster_target_id,
)
from ._context import RouterContext


def register(app: FastAPI, router_ctx: RouterContext) -> None:
    cache = router_ctx.cache
    annotator = router_ctx.annotator

    @app.get("/api/runs/{run_id}/labels")
    def list_labels(run_id: str) -> Dict[str, Any]:
        ctx = cache.get(run_id)
        latest = ctx.store.latest_by_target()
        return {
            "latest": {k: v.model_dump() for k, v in latest.items()},
            "history_count": len(ctx.store.all()),
        }

    @app.post("/api/runs/{run_id}/labels")
    def post_label(run_id: str, payload: LabelIn) -> Label:
        ctx = cache.get(run_id)
        if payload.target == LabelTarget.CLUSTER:
            if ctx.cluster(payload.target_id) is None:
                raise HTTPException(404, f"cluster not found: {payload.target_id}")
        elif payload.target == LabelTarget.ISSUE:
            if ctx.issue(payload.target_id) is None:
                raise HTTPException(404, f"issue not found: {payload.target_id}")
        elif payload.target == LabelTarget.COLUMN_CLUSTER:
            # Column-cluster verdicts are independent per cluster:
            # labelling c1 does not finalize the column's clustering.
            # We accept the verdict if the cluster_id exists either
            # in a saved ``column_clusters.jsonl`` record (column
            # board has run) OR in the column's draft JSONL (LLM
            # proposal not yet finalized). Drafts use ``c{idx+1}``
            # in proposal order, matching the synthetic target_ids
            # surfaced by ``list_column_clusters``.
            #
            # We deliberately do NOT auto-promote drafts to a
            # ``ColumnClusters`` record: that record represents a
            # disjoint partition (only the column-board UI builds
            # one), whereas LLM drafts may overlap. Persisting a
            # subset of drafts here would mark the column as
            # ``labeled`` and hide the remaining drafts from the
            # queue, blocking the operator from voting on them.
            if not payload.source or not payload.column:
                raise HTTPException(
                    400, "column_cluster labels require source and column",
                )
            if payload.source not in ctx.sources:
                raise HTTPException(404, f"source not found: {payload.source}")
            # ``target_id`` of the form ``colclust\x00src\x00col\x00cN``.
            parts = payload.target_id.split("\x00")
            if len(parts) != 4 or parts[0] != "colclust":
                raise HTTPException(
                    400,
                    f"invalid column_cluster target_id: {payload.target_id!r}",
                )
            local_id = parts[3]
            store = ctx.column_cluster_store
            rec = store.latest_for(payload.source, payload.column)
            in_record = (
                rec is not None
                and rec.status == ColumnLabelStatus.LABELED
                and any(cl.cluster_id == local_id for cl in rec.clusters)
            )
            in_drafts = False
            if not in_record:
                drafts = _load_drafts_for_column(
                    ctx.run_dir, payload.source, payload.column,
                )
                # Mirror the c{idx+1}-in-draft-order convention used
                # by ``list_column_clusters`` when surfacing drafts.
                # A draft is a usable target if it has >= 2 string
                # values (same minimum the listing enforces).
                draft_ids: set = set()
                for idx, dr in enumerate(drafts, start=1):
                    raw = dr.get("values") or []
                    str_count = sum(
                        1 for v in raw
                        if isinstance(v, str) and v
                        or (isinstance(v, dict)
                            and isinstance(v.get("value"), str)
                            and v.get("value"))
                    )
                    if str_count >= 2:
                        draft_ids.add(f"c{idx}")
                in_drafts = local_id in draft_ids
            if not (in_record or in_drafts):
                raise HTTPException(
                    404,
                    f"cluster {local_id!r} not found in "
                    f"{payload.source!r}/{payload.column!r}; "
                    f"no saved record or matching draft",
                )
            # Validate ``confirmed_canonical`` (if set) against the
            # cluster's actual value-set. The operator may only
            # endorse one of the cluster's surface forms; rejecting
            # arbitrary strings here keeps the synthesised
            # ``ColumnClusters`` records valid (their schema enforces
            # the same constraint).
            if payload.confirmed_canonical is not None:
                cluster_values: set = set()
                if in_record:
                    for cl in rec.clusters:
                        if cl.cluster_id == local_id:
                            cluster_values = {cv.value for cv in cl.values}
                            break
                else:
                    drafts = _load_drafts_for_column(
                        ctx.run_dir, payload.source, payload.column,
                    )
                    idx = int(local_id[1:]) - 1
                    if 0 <= idx < len(drafts):
                        for v in drafts[idx].get("values") or []:
                            if isinstance(v, str) and v:
                                cluster_values.add(v)
                            elif (isinstance(v, dict)
                                  and isinstance(v.get("value"), str)
                                  and v.get("value")):
                                cluster_values.add(v["value"])
                if payload.confirmed_canonical not in cluster_values:
                    raise HTTPException(
                        400,
                        f"confirmed_canonical "
                        f"{payload.confirmed_canonical!r} is not a "
                        f"value of cluster {local_id!r}",
                    )
            payload = payload.model_copy(update={
                "target_id": column_cluster_target_id(
                    payload.source, payload.column, local_id,
                ),
            })
            # Preserve operator-saved per-value corrections. A verdict
            # POST carries only the verdict fields, so without this an
            # operator who re-affirms or changes a column-cluster verdict
            # AFTER saving corrections would clobber them: the new record
            # becomes the latest and ``LabelIn.value_corrections``
            # defaults to ``{}``. Inherit the latest saved mapping when
            # the incoming payload didn't set one (mirrors
            # ``save_cluster_corrections``, which preserves the verdict).
            # Operators clear corrections explicitly via the corrections
            # endpoint, so an empty incoming mapping never *means* clear.
            if not payload.value_corrections:
                prev = ctx.store.latest_for(payload.target_id)
                if prev is not None and prev.value_corrections:
                    payload = payload.model_copy(update={
                        "value_corrections": dict(prev.value_corrections),
                    })
        else:  # CELL
            # Cell labels carry FN (false negative). Require the same
            # (source, row, column) triple in the payload so the
            # benchmark exporter can find them later. We rebuild the
            # canonical target_id here regardless of what the client
            # sent so two clients can't drift out of sync on key format.
            if payload.row is None or not payload.column or not payload.source:
                raise HTTPException(
                    400, "cell labels require source, row, and column",
                )
            if payload.source not in ctx.sources:
                raise HTTPException(404, f"source not found: {payload.source}")
            # Cell labels carry an operator-only claim that this cell
            # is (or was) an error the LLM missed. ``TP`` is deliberately
            # excluded: a true-positive is a verdict on an LLM detection,
            # which only the cluster path owns -- there is no detection
            # at an unflagged cell to affirm. The supported verdicts
            # carry the operator's confidence on the FN claim; FP and
            # UNLABELED are tombstones that clear the FN.
            _allowed_cell_statuses = {
                LabelStatus.FN, LabelStatus.PARTIAL,
                LabelStatus.UNSURE, LabelStatus.SKIP,
                LabelStatus.FP, LabelStatus.UNLABELED,
            }
            if payload.status not in _allowed_cell_statuses:
                raise HTTPException(
                    400,
                    "cell labels support fn/partial/unsure/skip "
                    "(verdict on the operator's claim) or fp / "
                    "unlabeled (clear the cell); tp belongs to the "
                    "cluster path",
                )
            payload = payload.model_copy(update={
                "target_id": cell_target_id(
                    payload.source, int(payload.row), payload.column,
                ),
            })

        record = Label(
            **payload.model_dump(exclude={"annotator"}),
            annotator=(payload.annotator or annotator).strip() or annotator,
        )
        return ctx.store.append(record)

