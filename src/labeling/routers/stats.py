"""Run-level label stats endpoint under ``/api/runs/{run_id}/stats``.

Aggregates per-status counts across cell-clusters, column-cluster queue
items, and cell-level FN labels (with a ``shadowed`` tally for FN cells
also covered by a TP/PARTIAL cluster) for the label-page stats bar.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from fastapi import FastAPI

from ..column_cluster_queue import iter_column_cluster_targets
from ..schema import LabelStats, LabelStatus, LabelTarget
from ._context import RouterContext

logger = logging.getLogger(__name__)


def register(app: FastAPI, router_ctx: RouterContext) -> None:
    cache = router_ctx.cache

    @app.get("/api/runs/{run_id}/stats")
    def stats(run_id: str) -> Dict[str, Any]:
        ctx = cache.get(run_id)
        latest = ctx.store.latest_by_target()

        def _bucket(cluster_ids: List[str]) -> LabelStats:
            counts = {"tp": 0, "fp": 0, "partial": 0, "unsure": 0, "skip": 0, "unlabeled": 0}
            for cid in cluster_ids:
                label = latest.get(cid)
                key = label.status.value if label else "unlabeled"
                counts[key] = counts.get(key, 0) + 1
            return LabelStats(total=len(cluster_ids), **counts)

        by_source: Dict[str, List[str]] = {}
        all_ids: List[str] = []
        for c in ctx.clusters:
            cid = c.get("cluster_id")
            if not cid:
                continue
            all_ids.append(cid)
            by_source.setdefault(c.get("source", ""), []).append(cid)

        # Column-cluster (inconsistency) queue items also live on the
        # label page, so the stats bar must include them. Enumerate
        # them the same way ``list_column_clusters`` does: saved
        # labelled records contribute one target per cluster_id; LLM
        # drafts in unlabelled columns contribute one target per
        # ``c{idx}`` (gated by ≥2 live values).
        try:
            cc_targets = iter_column_cluster_targets(ctx.run_dir)
        except Exception as exc:  # noqa: BLE001 - never break the stats endpoint on a read error
            logger.warning("stats: column-cluster enumeration failed: %s", exc)
            cc_targets = []
        for src, _col, target_id in cc_targets:
            all_ids.append(target_id)
            by_source.setdefault(src, []).append(target_id)

        # Cell-level FN counts. We tally only the *latest* state per
        # cell key, so re-labelling a cell as FP doesn't double-count.
        # Cells that are simultaneously TP via cluster are reported as
        # ``shadowed`` so the operator sees the double-labelling.
        fn_total = 0
        fn_shadowed = 0
        fn_by_source: Dict[str, int] = {}
        # Build a quick (source,row,col) -> cluster-state lookup so we
        # can detect shadowing without re-scanning every cluster.
        covered: Dict[tuple, str] = {}
        for c in ctx.clusters:
            cid = c.get("cluster_id", "")
            lab = latest.get(cid)
            if lab is None or lab.target != LabelTarget.CLUSTER:
                continue
            st = lab.status.value
            if st not in ("tp", "partial"):
                continue
            row = c.get("row")
            if not isinstance(row, int):
                continue
            for col in (c.get("columns") or []):
                covered[(c.get("source", ""), row, col)] = st
        for lab in latest.values():
            if lab.target != LabelTarget.CELL or lab.status != LabelStatus.FN:
                continue
            if lab.row is None or not lab.column:
                continue
            fn_total += 1
            fn_by_source[lab.source] = fn_by_source.get(lab.source, 0) + 1
            if (lab.source, int(lab.row), lab.column) in covered:
                fn_shadowed += 1

        return {
            "overall": _bucket(all_ids).model_dump(),
            "by_source": {src: _bucket(ids).model_dump() for src, ids in by_source.items()},
            "fn": {
                "total": fn_total,
                "shadowed": fn_shadowed,
                "by_source": fn_by_source,
            },
        }

