"""Materialise per-cell inconsistency benchmark records for a run.

Shared between two callers:

  * ``experiments.export_inconsistency_benchmark_with_llm`` -- the
    offline CLI used to bake an inconsistency benchmark with fresh
    LLM calls (writes JSONL to ``label_result/``).
  * ``src.labeling.app::export_benchmark`` -- the live FastAPI
    endpoint behind the "Benchmark" download button. It calls this
    module in **cache-only** mode (``dry_run=True``,
    ``llm_client=None``) so the UI export instantly returns whatever
    has already been baked into the on-disk LLM cache plus any
    operator-saved ``value_corrections``. No LLM dollars are spent
    on a button click.

The two callers diverge only in I/O: the CLI writes JSONL and prints
stats; the live endpoint merges this list with the existing
``materialize_label_benchmark`` cells. The materialization logic
itself -- including operator-override short-circuiting and
cluster-rescue safety net -- lives here so both paths emit
byte-identical cells for a given (run-dir, cache) snapshot.

See ``experiments.export_inconsistency_benchmark_with_llm`` module
docstring for the full label / corrected_value semantics.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from src.core.data_loader import load_csv
from src.core.llm_client import LLMClient
from src.labeling.correction_suggester import (
    INTERESTING_STATUSES,
    CellGroup,
    ClusterCtx,
    aggregate_samples,
    build_cluster_index,
    dispatch_groups,
    load_cache,
    load_column_cluster_records,
    load_drafts,
    load_labels,
    resolve_csv_map,
)
from src.labeling.schema import BenchmarkCell

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Cell-group enumeration (same as the offline CLI's _build_cell_groups
# but lives here so both callers share the implementation).
# ---------------------------------------------------------------------


def build_cell_groups(
    cluster_index: Dict[Tuple[str, str], List[ClusterCtx]],
    csv_map: Dict[str, Path],
    *,
    csv_cache: Dict[str, pd.DataFrame],
    stats: Counter,
) -> List[CellGroup]:
    """Produce one ``CellGroup`` per (source, column, value) tuple
    that has at least one row in the CSV AND at least one involved
    cluster with verdict in :data:`INTERESTING_STATUSES`.
    """
    groups: List[CellGroup] = []
    interesting_values_per_sc: Dict[Tuple[str, str], set[str]] = defaultdict(set)
    for (s, c), ctxs in cluster_index.items():
        for ctx in ctxs:
            if ctx.status in INTERESTING_STATUSES:
                interesting_values_per_sc[(s, c)].update(ctx.values)

    for (s, c), values in sorted(interesting_values_per_sc.items()):
        path = csv_map.get(s)
        if path is None:
            logger.warning(
                "No CSV path for source %s; skipping %d candidate value(s).",
                s, len(values),
            )
            stats["skip_missing_csv"] += 1
            continue
        if s not in csv_cache:
            try:
                df = load_csv(path)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Failed to load CSV %s: %s; skipping.", path, exc,
                )
                stats["skip_csv_load_error"] += 1
                continue
            csv_cache[s] = df
        df = csv_cache[s]
        if c not in df.columns:
            logger.warning("Column %r not in %s; skipping.", c, s)
            stats["skip_missing_column"] += 1
            continue
        col_series = df[c].astype(str)

        ctxs = cluster_index.get((s, c), [])
        for v in sorted(values):
            mask = col_series == v
            row_idx = [int(i) for i in col_series.index[mask]]
            if not row_idx:
                logger.warning(
                    "Value %r has no rows in %s/%s; stale label?",
                    v, s, c,
                )
                stats["empty_value_match"] += 1
                continue
            involved = [ctx for ctx in ctxs if v in ctx.value_set]
            if not involved:
                stats["empty_involved"] += 1
                continue
            groups.append(CellGroup(
                source=s,
                column=c,
                value=v,
                rows=row_idx,
                involved=involved,
            ))
            stats["cell_groups"] += 1
            stats["matched_rows"] += len(row_idx)
    return groups


# ---------------------------------------------------------------------
# BenchmarkCell construction.
# ---------------------------------------------------------------------


def _label_from_involved(involved: Iterable[ClusterCtx]) -> str:
    """TP-wins union rule. Caller must ensure ``involved`` is non-empty
    and contains at least one TP or unsure entry."""
    statuses = {ctx.status for ctx in involved}
    if "tp" in statuses:
        return "error"
    if "unsure" in statuses:
        return "ambiguous"
    return "ambiguous"


def _coerce_corrected_value(
    values: Optional[List[str]],
) -> Optional[Any]:
    """Polymorphic coercion for ``BenchmarkCell.corrected_value``.

    None / [] -> None (no correction). Length-1 -> scalar
    (one-directional). Length-2+ -> sorted list (bi-directional).
    """
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return sorted(set(values))


def build_cells_for_group(
    group: CellGroup,
    *,
    possible_values: Optional[List[str]],
    agreement: Optional[float] = None,
    n_samples: Optional[int] = None,
) -> List[BenchmarkCell]:
    """Emit one ``BenchmarkCell`` per CSV row covered by ``group``."""
    label = _label_from_involved(group.involved)
    cluster_ids = [ctx.global_id for ctx in group.involved]
    cluster_kinds = [ctx.kind for ctx in group.involved]
    llm_hint: Optional[str] = None
    for ctx in group.involved:
        if ctx.confirmed_canonical:
            llm_hint = ctx.confirmed_canonical
            break
    if llm_hint is None:
        for ctx in group.involved:
            if ctx.llm_canonical:
                llm_hint = ctx.llm_canonical
                break
    annotators = sorted({
        ctx.annotator for ctx in group.involved if ctx.annotator
    })
    notes = sorted({
        ctx.note for ctx in group.involved if ctx.note
    })
    note_parts: List[str] = list(notes)
    if agreement is not None and n_samples is not None and n_samples > 0:
        note_parts.append(
            f"llm_agreement={agreement:.2f} (samples={n_samples})"
        )
    note_text = "; ".join(p for p in note_parts if p)
    corrected = _coerce_corrected_value(possible_values)

    cells: List[BenchmarkCell] = []
    for row in group.rows:
        cells.append(BenchmarkCell(
            source=group.source,
            row=row,
            column=group.column,
            original_value=group.value,
            label=label,
            corrected_value=corrected,
            llm_suggested_value=llm_hint,
            error_types=["inconsistency"],
            detection_ids=[],
            cluster_ids=cluster_ids,
            cluster_kinds=cluster_kinds,
            annotators=annotators,
            note=note_text,
        ))
    return cells


# ---------------------------------------------------------------------
# Top-level materialiser.
# ---------------------------------------------------------------------


def materialize_inconsistency_benchmark(
    run_dir: Path,
    *,
    llm_client: Optional[LLMClient] = None,
    model: str = "gpt-5.2",
    reasoning_effort: str = "medium",
    samples: int = 3,
    samples_threshold: float = 0.5,
    retry_samples: int = 4,
    row_context_rows: int = 3,
    cache_path: Optional[Path] = None,
    csv_cache: Optional[Dict[str, pd.DataFrame]] = None,
    concurrency: int = 8,
    progress_cb: Optional[Callable[[str, int, int], None]] = None,
    dry_run: bool = True,
    limit: Optional[int] = None,
    stats: Optional[Counter] = None,
) -> List[BenchmarkCell]:
    """Materialise per-cell inconsistency benchmark cells for a run.

    Cache-only behaviour (``dry_run=True``, ``llm_client=None``) is
    safe to call from request handlers: every prompt is looked up in
    ``inconsistency_llm_cache.jsonl`` and groups without a cache hit
    are aggregated as if the LLM returned an empty list (which then
    falls back to either the operator's saved ``value_corrections``
    or the cluster-rescue null-emit path).

    Pass ``llm_client=LLMClient()`` and ``dry_run=False`` for the
    offline CLI flow that actually queries the model and appends new
    cache entries.

    ``run_dir`` must contain ``labels.jsonl``,
    ``column_cluster_drafts.jsonl``, and ``summary.json``. Returns
    ``[]`` (not an error) if the run has no TP/unsure column-cluster
    verdicts -- that's the natural result for runs whose labels are
    all FP / clean / pending.
    """
    run_dir = Path(run_dir)
    if stats is None:
        stats = Counter()
    if csv_cache is None:
        csv_cache = {}
    if cache_path is None:
        cache_path = run_dir / "inconsistency_llm_cache.jsonl"

    labels_path = run_dir / "labels.jsonl"
    drafts_path = run_dir / "column_cluster_drafts.jsonl"
    records_path = run_dir / "column_clusters.jsonl"
    # Need labels plus at least one cluster source: the LLM drafts file
    # or the operator-curated column_clusters.jsonl (operator-ADDED
    # "missed inconsistency" clusters live only in the latter).
    if not labels_path.exists() or not (
        drafts_path.exists() or records_path.exists()
    ):
        return []

    labels = load_labels(labels_path)
    drafts = load_drafts(drafts_path)
    records = load_column_cluster_records(records_path)
    cluster_index = build_cluster_index(drafts, labels, records=records)

    n_interesting = sum(
        1 for ctxs in cluster_index.values()
        for ctx in ctxs if ctx.status in INTERESTING_STATUSES
    )
    if n_interesting == 0:
        return []

    csv_map = resolve_csv_map(run_dir)
    groups = build_cell_groups(
        cluster_index, csv_map,
        csv_cache=csv_cache, stats=stats,
    )
    if limit is not None:
        groups = groups[: limit]
    if not groups:
        return []

    cache = load_cache(cache_path)
    n_samples = max(1, int(samples))
    n_retry = max(0, int(retry_samples))

    results_by_idx = dispatch_groups(
        groups=groups,
        csv_cache=csv_cache,
        llm_client=llm_client,
        cache_path=cache_path,
        cache=cache,
        model=model,
        reasoning_effort=reasoning_effort,
        samples=n_samples,
        samples_threshold=samples_threshold,
        retry_samples=n_retry,
        row_context_rows=row_context_rows,
        concurrency=concurrency,
        progress_cb=progress_cb,
        stats=stats,
        dry_run=dry_run,
    )

    # Pass 1.5: operator-saved value_corrections short-circuit the LLM
    # consensus path. Their decision is final.
    overrides_by_idx: Dict[int, dict] = {}
    n_override_accept = 0
    n_override_drop = 0
    for i, grp in enumerate(groups):
        chosen_ctx = None
        for ctx in grp.involved:
            if ctx.status not in ("tp", "unsure"):
                continue
            if not ctx.value_corrections:
                continue
            chosen_ctx = ctx
            break
        if chosen_ctx is None:
            continue
        if grp.value in chosen_ctx.value_corrections:
            raw = chosen_ctx.value_corrections[grp.value]
            if isinstance(raw, list):
                consensus = [str(x) for x in raw if isinstance(x, str)]
            elif isinstance(raw, str):
                consensus = [raw]
            else:
                consensus = None
            if consensus:
                overrides_by_idx[i] = {
                    "consensus": consensus,
                    "agreement": 1.0,
                    "n_valid": 1,
                    "would_drop": (
                        len(consensus) == 1 and consensus[0] == grp.value
                    ),
                    "from_saved": True,
                }
                n_override_accept += 1
        else:
            overrides_by_idx[i] = {
                "consensus": None,
                "agreement": 0.0,
                "n_valid": 0,
                "would_drop": True,
                "from_saved": True,
                "operator_drop": True,
            }
            n_override_drop += 1
    if overrides_by_idx:
        stats["groups_with_saved_corrections_accept"] = n_override_accept
        stats["groups_with_saved_corrections_drop"] = n_override_drop

    # Pass 1: per-group consensus.
    per_group: Dict[int, dict] = {}
    agreement_buckets: Counter = Counter()
    for i, grp in enumerate(groups):
        if i in overrides_by_idx:
            ov = overrides_by_idx[i]
            per_group[i] = {
                "consensus": ov["consensus"],
                "agreement": ov["agreement"],
                "n_valid": ov["n_valid"],
                "would_drop": ov["would_drop"],
            }
            if ov["consensus"]:
                agreement_buckets["all_agree"] += 1
            continue
        sample_lists = results_by_idx[i] if i < len(results_by_idx) else []
        consensus, agreement, n_valid = aggregate_samples(
            sample_lists, threshold_frac=samples_threshold,
        )
        n_slots = len(sample_lists)
        if n_valid == 0:
            stats["groups_no_valid_sample"] += 1
        elif n_valid < n_slots:
            stats["groups_partial_samples"] += 1
        if n_valid > 0:
            if agreement >= 1.0:
                agreement_buckets["all_agree"] += 1
            elif agreement >= 0.66:
                agreement_buckets["majority"] += 1
            else:
                agreement_buckets["split"] += 1
        would_drop = bool(
            consensus is not None
            and len(consensus) == 1
            and consensus[0] == grp.value
        )
        per_group[i] = {
            "consensus": consensus,
            "agreement": agreement,
            "n_valid": n_valid,
            "would_drop": would_drop,
        }

    # Pass 2: cluster rescue. Operator-saved overrides are exempt.
    cluster_to_groups: Dict[str, List[int]] = defaultdict(list)
    cluster_status: Dict[str, str] = {}
    cluster_has_saved_override: Dict[str, bool] = defaultdict(bool)
    for i, grp in enumerate(groups):
        has_override = i in overrides_by_idx
        for ctx in grp.involved:
            if ctx.status in ("tp", "unsure"):
                cluster_to_groups[ctx.global_id].append(i)
                cluster_status[ctx.global_id] = ctx.status
                if has_override:
                    cluster_has_saved_override[ctx.global_id] = True
    rescued_indices: set = set()
    for gid, idxs in cluster_to_groups.items():
        if not idxs:
            continue
        if cluster_has_saved_override.get(gid):
            continue
        if all(per_group[i]["would_drop"] for i in idxs):
            for i in idxs:
                rescued_indices.add(i)
    stats["groups_rescued_from_full_drop"] = len(rescued_indices)

    # Pass 3: emit cells.
    out_cells: List[BenchmarkCell] = []
    for i, grp in enumerate(groups):
        info = per_group[i]
        consensus = info["consensus"]
        agreement = info["agreement"]
        n_valid = info["n_valid"]
        if info["would_drop"] and i not in rescued_indices:
            # A "would_drop" value is one whose canonical form equals the
            # value itself -- either the LLM consensus said so (Pass 1) or
            # the operator dropped it via ``value_corrections`` (an absent
            # key in Pass 1.5). For an ERROR cell (a TP cluster covers it)
            # that genuinely means "the canonical form of a confirmed error
            # is not itself an error", so we drop it from the benchmark
            # (it exports as clean by being absent).
            #
            # But an UNSURE-only cell exports as ``ambiguous`` (FP immunity):
            # the reviewer could not confidently call the cluster an error,
            # so a model that flags ANY of its values must not be penalised.
            # Dropping such a value would silently strip that immunity (the
            # cell becomes absent -> scored as clean -> a flag is an FP),
            # which is exactly the accidental data-loss the per-value
            # "Drop" default causes. Ambiguous cells carry no
            # ``corrected_value`` anyway, so we KEEP the cell (emitted with
            # no correction) instead of dropping it.
            if _label_from_involved(grp.involved) != "ambiguous":
                stats["cell_groups_already_canonical"] += 1
                stats["rows_already_canonical"] += len(grp.rows)
                continue
            stats["ambiguous_groups_kept_despite_drop"] += 1
            stats["ambiguous_rows_kept_despite_drop"] += len(grp.rows)
            possible_for_cell: Optional[List[str]] = None
        elif i in rescued_indices:
            possible_for_cell = None
            stats["cells_emitted_null_cluster_rescue"] += len(grp.rows)
        elif n_valid == 0 or consensus is None:
            possible_for_cell = None
            stats["cells_emitted_null_no_llm_fix"] += len(grp.rows)
        else:
            possible_for_cell = consensus
        cells = build_cells_for_group(
            grp,
            possible_values=possible_for_cell,
            agreement=agreement if n_valid > 0 else None,
            n_samples=n_valid,
        )
        out_cells.extend(cells)
    stats["agreement_all_agree"] = agreement_buckets.get("all_agree", 0)
    stats["agreement_majority"] = agreement_buckets.get("majority", 0)
    stats["agreement_split"] = agreement_buckets.get("split", 0)
    return out_cells
