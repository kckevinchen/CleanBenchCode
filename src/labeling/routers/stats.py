"""Run-level label stats endpoint under ``/api/runs/{run_id}/stats``.

Aggregates per-status counts across cell-clusters, column-cluster queue
items, and cell-level FN labels (with a ``shadowed`` tally for FN cells
also covered by a TP/PARTIAL cluster) for the label-page stats bar.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Tuple

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

    @app.get("/api/runs/{run_id}/labeling-effort")
    def labeling_effort(run_id: str) -> Dict[str, Any]:
        """Per-(source, column) labeling effort projection.

        Driven by the per-arm Good-Turing artifact written by adaptive
        runs (``good_turing/per_arm_estimates.json``). Each arm =
        ``(scope_kind=column, (source, column), error_type)`` exposes:

        * ``n_observed``         - distinct defect species seen so far
        * ``chao1.estimated_total`` - Chao1 point estimate of total
          species (observed + projected unseen).
        * ``good_turing_missing_mass`` - probability the next perturbed
          run surfaces a never-before-seen species.
        * ``status``             - saturated / exhausted / active /
          insufficient / not_applicable.

        We aggregate arms over error_types within each (source, column)
        cell to project the labelling-effort the operator should plan
        for. Returns ``{"available": false}`` when the run has no
        per-arm artifact (single-pass / fixed-K runs).
        """
        ctx = cache.get(run_id)
        payload = ctx.per_arm_payload()
        if not payload or not isinstance(payload.get("arms"), list):
            return {
                "available": False,
                "reason": (
                    "Per-arm Good-Turing estimates not available for this "
                    "run (only adaptive multi-pass runs at "
                    "stop_scope=column produce per-(source, column) arms)."
                ),
            }
        scope_kind = payload.get("scope_kind", "column")
        if scope_kind != "column":
            return {
                "available": False,
                "reason": (
                    "Per-arm artifact uses scope_kind="
                    f"{scope_kind!r}; per-column effort projection "
                    "requires stop_scope=column."
                ),
            }

        # Already-labelled counts per (source, column). Used to show
        # progress vs. the projected total. We only count cluster-level
        # verdicts (tp/partial) plus FN cell-level labels - skipped /
        # unsure clusters do not contribute to "labelled".
        latest = ctx.store.latest_by_target()
        labelled_by_cell: Dict[Tuple[str, str], int] = {}
        for c in ctx.clusters:
            cid = c.get("cluster_id")
            lab = latest.get(cid) if cid else None
            if lab is None or lab.target != LabelTarget.CLUSTER:
                continue
            if lab.status.value not in ("tp", "fp", "partial"):
                continue
            src = c.get("source") or ""
            for col in (c.get("columns") or []):
                k = (src, str(col))
                labelled_by_cell[k] = labelled_by_cell.get(k, 0) + 1
        for lab in latest.values():
            if lab.target != LabelTarget.CELL or lab.status != LabelStatus.FN:
                continue
            if lab.column is None:
                continue
            k = (lab.source, str(lab.column))
            labelled_by_cell[k] = labelled_by_cell.get(k, 0) + 1

        # Group arms by (source, column).
        cells: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for arm in payload["arms"]:
            scope_id = arm.get("scope_id") or []
            if len(scope_id) < 2:
                continue
            source = str(scope_id[0])
            column = str(scope_id[1])
            etype = str(arm.get("error_type", ""))
            n_obs = float(arm.get("n_observed", 0))
            chao1 = arm.get("chao1") or {}
            est_total = float(chao1.get("estimated_total", n_obs))
            est_missed = max(est_total - n_obs, 0.0)
            ci = chao1.get("ci_95") or [est_total, est_total]
            missing_mass = float(arm.get("good_turing_missing_mass", 0.0))
            status = str(arm.get("status", "active"))

            cell = cells.setdefault((source, column), {
                "source": source,
                "column": column,
                "n_observed": 0.0,
                "chao1_estimated_total": 0.0,
                "chao1_estimated_missed": 0.0,
                "chao1_ci_low": 0.0,
                "chao1_ci_high": 0.0,
                # Mass aggregation: weight each arm's missing mass by
                # its hit count so a tiny arm with mass=1.0 doesn't
                # dominate. Falls back to a simple max if no hits yet.
                "_mass_weight_sum": 0.0,
                "_mass_weighted": 0.0,
                "_mass_max": 0.0,
                "arms_total": 0,
                "arms_saturated": 0,
                "arms_active": 0,
                "arms_insufficient": 0,
                "arms_not_applicable": 0,
                "by_error_type": [],
            })
            cell["n_observed"] += n_obs
            cell["chao1_estimated_total"] += est_total
            cell["chao1_estimated_missed"] += est_missed
            cell["chao1_ci_low"] += float(ci[0]) if ci else est_total
            cell["chao1_ci_high"] += float(ci[1]) if ci else est_total
            # Hit-count weighted missing mass (n_observed proxy for
            # active probe count). Falls back to max() when n_obs=0.
            cell["_mass_weight_sum"] += n_obs
            cell["_mass_weighted"] += missing_mass * n_obs
            cell["_mass_max"] = max(cell["_mass_max"], missing_mass)
            cell["arms_total"] += 1
            if status in ("saturated", "exhausted"):
                cell["arms_saturated"] += 1
            elif status == "not_applicable":
                cell["arms_not_applicable"] += 1
            elif status == "insufficient":
                cell["arms_insufficient"] += 1
            else:
                cell["arms_active"] += 1
            cell["by_error_type"].append({
                "error_type": etype,
                "n_observed": int(n_obs),
                "estimated_total": est_total,
                "estimated_missed": est_missed,
                "missing_mass": missing_mass,
                "status": status,
            })

        # Finalise + attach labelled-so-far counts.
        out: List[Dict[str, Any]] = []
        for (src, col), c in cells.items():
            ws = c.pop("_mass_weight_sum")
            mw = c.pop("_mass_weighted")
            mmax = c.pop("_mass_max")
            c["good_turing_missing_mass"] = (mw / ws) if ws > 0 else mmax
            c["labelled_so_far"] = labelled_by_cell.get((src, col), 0)
            # "Effort remaining" projects how many MORE distinct
            # defect species the operator should expect to see if
            # they fully cover this cell. Based on Chao1's missed
            # estimate net of what they've already labelled (clamped
            # to zero - already-labelled can exceed estimated_missed
            # for very-saturated cells).
            still_to_label = (
                c["chao1_estimated_total"] - c["labelled_so_far"]
            )
            c["effort_remaining"] = max(still_to_label, 0.0)
            # Sort the per-error-type list by est_missed desc so the
            # frontend can show the top contributors.
            c["by_error_type"].sort(
                key=lambda e: e["estimated_missed"], reverse=True
            )
            out.append(c)
        # Default sort: largest projected missed first - the cells the
        # operator should focus on if they want to maximise expected
        # FN coverage per minute spent.
        out.sort(
            key=lambda c: c["chao1_estimated_missed"], reverse=True,
        )

        # Run-level totals so the frontend can show a headline pill.
        total_observed = sum(c["n_observed"] for c in out)
        total_estimated = sum(c["chao1_estimated_total"] for c in out)
        total_missed = sum(c["chao1_estimated_missed"] for c in out)
        total_labelled = sum(c["labelled_so_far"] for c in out)

        return {
            "available": True,
            "scope_kind": scope_kind,
            "num_runs_completed": payload.get("num_runs_completed"),
            "totals": {
                "n_observed": total_observed,
                "chao1_estimated_total": total_estimated,
                "chao1_estimated_missed": total_missed,
                "labelled_so_far": total_labelled,
                "effort_remaining": max(total_estimated - total_labelled, 0.0),
                "cells": len(out),
            },
            "cells": out,
        }

