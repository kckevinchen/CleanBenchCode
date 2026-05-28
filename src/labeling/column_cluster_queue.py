"""Enumerator for the label-page's column-cluster queue.

The label page surfaces an ``inconsistency`` queue item for every
column-cluster the operator can verdict. The set is the union of:

1. **Saved records** in ``column_clusters.jsonl`` whose ``status`` is
   :class:`~src.labeling.schema.ColumnLabelStatus.LABELED` — one queue
   item per saved cluster_id (``c1..cN``).

2. **LLM drafts** in ``column_cluster_drafts.jsonl`` for any
   ``(source, column)`` *without* a labelled saved record — one queue
   item per draft cluster (``c{idx+1}`` by file order, gated by the
   ``len(values) >= 2`` filter the queue endpoint applies).

This module wraps that enumeration in a single function so the
``/api/runs/{id}/stats`` endpoint and the exclusion-cascade helper
agree on the target_id set without duplicating the discovery logic.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from .column_cluster_store import ColumnClusterStore
from .schema import ColumnLabelStatus, column_cluster_target_id


logger = logging.getLogger(__name__)


def iter_column_cluster_targets(
    run_dir: Path,
) -> List[Tuple[str, str, str]]:
    """Return ``[(source, column, target_id)]`` for every column-cluster
    queue item the label page would surface, in a deterministic order
    (saved records first by ``(source, column, cluster_id)``, then
    drafts by ``(source, column, idx)``).

    Saved labelled records take precedence: if a ``(source, column)``
    has a labelled saved record, its drafts are intentionally skipped
    (the label page does the same, since the saved record's clusters
    *are* the canonical interpretation of the LLM proposals for that
    column).

    Drafts with fewer than 2 live values are skipped to mirror the
    ``list_column_clusters`` endpoint, which only renders ≥2-value
    drafts in the queue.
    """
    out: List[Tuple[str, str, str]] = []
    cc_path = run_dir / "column_clusters.jsonl"
    drafts_path = run_dir / "column_cluster_drafts.jsonl"

    labeled_columns: Set[Tuple[str, str]] = set()
    if cc_path.exists():
        try:
            store = ColumnClusterStore(cc_path)
        except Exception as exc:  # noqa: BLE001 - never break callers on a read error
            logger.warning(
                "iter_column_cluster_targets: ColumnClusterStore load "
                "failed for %s: %s", cc_path, exc,
            )
            store = None  # type: ignore[assignment]
        if store is not None:
            for (src, col), rec in store.latest_by_column().items():
                if rec.status == ColumnLabelStatus.LABELED:
                    labeled_columns.add((src, col))
                    for cl in rec.clusters:
                        out.append((
                            src, col,
                            column_cluster_target_id(src, col, cl.cluster_id),
                        ))

    if drafts_path.exists():
        latest_drafts: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
        try:
            with drafts_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    src = d.get("source") or ""
                    col = d.get("column") or ""
                    if not src or not col:
                        continue
                    latest_drafts[(src, col)] = d.get("draft_clusters") or []
        except OSError as exc:
            logger.warning(
                "iter_column_cluster_targets: draft index read failed: %s",
                exc,
            )
            latest_drafts = {}

        for (src, col), drafts in latest_drafts.items():
            if (src, col) in labeled_columns:
                continue
            for idx, dc in enumerate(drafts, start=1):
                raw_values = dc.get("values") or []
                live = 0
                for v in raw_values:
                    if isinstance(v, dict):
                        val = v.get("value")
                    else:
                        val = v
                    if isinstance(val, str) and val:
                        live += 1
                if live < 2:
                    continue
                local_id = f"c{idx}"
                out.append((
                    src, col,
                    column_cluster_target_id(src, col, local_id),
                ))

    return out
