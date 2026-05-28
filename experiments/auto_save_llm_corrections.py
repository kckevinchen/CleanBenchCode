"""Persist LLM-suggested per-value corrections for every TP/unsure
column-cluster verdict in a run.

For each column-cluster label whose status is ``tp`` or ``unsure``, the
script asks :func:`correction_suggester.suggest_corrections_for_clusters`
for the per-(source, column, value) consensus list, mirrors the
labelling UI's accept/drop heuristic, and appends a new
:class:`Label` record with the resulting ``value_corrections`` mapping
plus a ``[llm-suggested]`` note tag.

The default mode is **cache-only**: ``dry_run=True`` + ``llm_client=None``
so the script consults
``<run-dir>/inconsistency_llm_cache.jsonl`` and never spends new LLM
budget. Pass ``--llm`` to fall back to the model for cache misses.

Accept/drop rule (mirrors ``frontend/labeling/label.js`` line ~4180):

    * consensus is empty / no LLM samples -> drop the value
    * consensus is exactly ``[value]`` -> drop ("LLM said no fix needed")
    * otherwise -> accept; scalar consensus becomes a string,
      multi-element consensus becomes a list.

Clusters whose mapping ends up empty after applying the rule are
skipped entirely (no append) so labels.jsonl stays clean.

Usage::

    python -m experiments.auto_save_llm_corrections \\
        --run-id 20260513T234733Z_gpt5-family \\
        --annotator kaiwen \\
        [--scope tp_unsure|tp|missing] \\
        [--dry-run] \\
        [--llm] [--model gpt-5.2] [--reasoning-effort medium]
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Make ``src`` importable when run as a script as well as a module.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.llm_client import LLMClient  # noqa: E402
from src.labeling import correction_suggester  # noqa: E402
from src.labeling.schema import (  # noqa: E402
    Label,
    LabelStatus,
    LabelTarget,
)
from src.labeling.store import LabelStore  # noqa: E402

logger = logging.getLogger("auto_save_llm_corrections")


_INTERESTING = (LabelStatus.TP, LabelStatus.UNSURE)


def _consensus_to_correction(
    value: str,
    suggestion: Optional[List[str]],
) -> Optional[Any]:
    """Return the saved correction for ``value`` or ``None`` to drop.

    Mirrors the per-row heuristic baked into ``saveCorrectionsFromUI``
    + the row defaulter in ``renderCorrectionsBlock``.
    """
    if not suggestion:
        return None
    if len(suggestion) == 1:
        only = suggestion[0]
        if only == value:
            # LLM confidently said "no fix needed" -> mirror UI default
            # (action=drop).
            return None
        return only
    # multi-canonical / bi-directional: accept as a list (sorted +
    # de-duplicated to match _aggregate_samples' canonical shape).
    return sorted(set(suggestion))


def _select_targets(
    store: LabelStore,
    *,
    scope: str,
) -> List[Tuple[str, str, str, Label]]:
    """Pick TP/unsure column-cluster verdicts to act on.

    ``scope``:
      * ``tp_unsure`` (default) - every TP or unsure cluster.
      * ``tp`` - TP only.
      * ``missing`` - TP/unsure clusters with empty
        ``value_corrections`` (won't overwrite manual / prior LLM saves).
    """
    out: List[Tuple[str, str, str, Label]] = []
    seen: set = set()
    # ``store.all()`` is in append order; iterate latest_by_target
    # to honour latest-wins.
    for tid, lab in store.latest_by_target().items():
        if lab.target != LabelTarget.COLUMN_CLUSTER:
            continue
        if scope == "tp" and lab.status != LabelStatus.TP:
            continue
        if scope in ("tp_unsure", "missing") and lab.status not in _INTERESTING:
            continue
        if scope == "missing" and lab.value_corrections:
            continue
        parsed = correction_suggester.parse_colclust_target_id(tid)
        if parsed is None:
            continue
        if tid in seen:
            continue
        seen.add(tid)
        s, c, lid = parsed
        out.append((s, c, lid, lab))
    return out


def _run(args: argparse.Namespace) -> int:
    project_root = Path(args.project_root).resolve()
    if args.run_dir is not None:
        run_dir = Path(args.run_dir).resolve()
    else:
        run_dir = project_root / "outputs" / args.run_id
    if not run_dir.is_dir():
        logger.error("run directory does not exist: %s", run_dir)
        return 2

    labels_path = run_dir / "labels.jsonl"
    store = LabelStore(labels_path)

    targets_with_label = _select_targets(store, scope=args.scope)
    logger.info(
        "selected %d column-cluster target(s) under scope=%s "
        "(labels.jsonl=%s)",
        len(targets_with_label), args.scope, labels_path,
    )
    if not targets_with_label:
        logger.warning("nothing to do; exiting")
        return 0

    targets: List[Tuple[str, str, str]] = [
        (s, c, lid) for (s, c, lid, _) in targets_with_label
    ]
    label_by_target: Dict[Tuple[str, str, str], Label] = {
        (s, c, lid): lab for (s, c, lid, lab) in targets_with_label
    }

    llm_client: Optional[LLMClient] = None
    if args.llm:
        try:
            llm_client = LLMClient()
        except Exception as exc:  # noqa: BLE001
            logger.error("failed to initialise LLM client: %s", exc)
            return 3

    def _progress(phase: str, done: int, total: int) -> None:
        if total <= 0:
            return
        if done % 50 == 0 or done == total:
            logger.info("[%s] %d / %d", phase, done, total)

    logger.info(
        "running suggest_corrections_for_clusters "
        "(dry_run=%s, llm=%s, samples=%d, retry=%d, model=%s, "
        "reasoning_effort=%s, concurrency=%d)",
        not args.llm, args.llm, args.samples, args.retry_samples,
        args.model, args.reasoning_effort, args.concurrency,
    )
    results = correction_suggester.suggest_corrections_for_clusters(
        run_dir,
        targets=targets,
        llm_client=llm_client,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        samples=args.samples,
        samples_threshold=args.samples_threshold,
        retry_samples=args.retry_samples,
        row_context_rows=args.row_context_rows,
        concurrency=args.concurrency,
        progress_cb=_progress,
        dry_run=not args.llm,
    )

    stats: Counter = Counter()
    new_labels: List[Label] = []
    annotator = (args.annotator or "").strip()
    if not annotator:
        logger.error("--annotator is required")
        return 2

    for key, suggs in results.items():
        s, c, lid = key
        prior = label_by_target.get(key)
        if prior is None:
            stats["skip_no_prior_label"] += 1
            continue

        if not suggs:
            stats["skip_no_suggestions"] += 1
            continue

        mapping: Dict[str, Any] = {}
        n_drop = 0
        n_no_consensus = 0
        for sg in suggs:
            corr = _consensus_to_correction(sg.value, sg.suggestion)
            if corr is None:
                if sg.suggestion:
                    n_drop += 1
                else:
                    n_no_consensus += 1
                continue
            mapping[sg.value] = corr

        if not mapping:
            stats["skip_all_dropped"] += 1
            stats["values_drop_per_lone"] += n_drop
            stats["values_no_consensus"] += n_no_consensus
            continue

        # Detect overwrite vs net-new.
        prior_map = dict(prior.value_corrections or {})
        if prior_map == mapping:
            stats["unchanged"] += 1
            continue
        if prior_map:
            stats["overwriting_existing"] += 1
        else:
            stats["new_corrections"] += 1
        stats["values_accepted"] += len(mapping)
        stats["values_drop_per_lone"] += n_drop
        stats["values_no_consensus"] += n_no_consensus

        prior_note = (prior.note or "").strip()
        # Always tag as `[llm-suggested]` -- repeated runs of this
        # script should not invent a synthetic "edited" tag just
        # because a prior auto-save already saved a (potentially
        # different) suggestion.
        base_tag = "[llm-suggested]"
        if base_tag in prior_note:
            new_note = prior_note
        elif prior_note:
            new_note = f"{base_tag}; {prior_note}"
        else:
            new_note = base_tag

        new_label = Label(
            target=LabelTarget.COLUMN_CLUSTER,
            target_id=prior.target_id,
            source=prior.source,
            row=prior.row,
            column=prior.column,
            columns=list(prior.columns),
            status=prior.status,
            corrected_value=prior.corrected_value,
            corrected_values=dict(prior.corrected_values),
            confirmed_canonical=prior.confirmed_canonical,
            value_corrections=mapping,
            note=new_note,
            annotator=annotator,
        )
        new_labels.append(new_label)

    logger.info("would write %d new label record(s); stats=%s",
                len(new_labels), dict(stats))

    if args.dry_run:
        # Emit a peek so the user can eyeball a few before committing.
        for lab in new_labels[: args.preview]:
            logger.info(
                "  [%s] %s/%s/%s -> %d correction(s): %s",
                lab.status, lab.source, lab.column,
                correction_suggester.parse_colclust_target_id(
                    lab.target_id,
                )[2],
                len(lab.value_corrections),
                {k: v for k, v in list(lab.value_corrections.items())[:3]},
            )
        logger.info("--dry-run set; no writes performed")
        return 0

    if not new_labels:
        logger.info("no new labels to write; exiting")
        return 0

    written = store.append_many(new_labels)
    logger.info("appended %d label record(s) to %s",
                len(written), labels_path)
    return 0


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--run-id", help="Run id (under <project-root>/outputs/).")
    g.add_argument("--run-dir", help="Absolute path to the run directory.")
    p.add_argument("--annotator", default="auto-llm",
                   help="Annotator string for the appended labels "
                        "(default: auto-llm).")
    p.add_argument("--scope", default="tp_unsure",
                   choices=["tp_unsure", "tp", "missing"],
                   help="Which clusters to act on (default: tp_unsure).")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would be written but do not append.")
    p.add_argument("--preview", type=int, default=10,
                   help="Number of preview rows to print in dry-run.")
    p.add_argument("--llm", action="store_true",
                   help="Allow LLM calls for cache misses (default off: "
                        "cache-only).")
    p.add_argument("--model", default="gpt-5.2")
    p.add_argument("--reasoning-effort", default="medium",
                   choices=["minimal", "low", "medium", "high"])
    p.add_argument("--samples", type=int, default=3)
    p.add_argument("--samples-threshold", type=float, default=0.5)
    p.add_argument("--retry-samples", type=int, default=4)
    p.add_argument("--row-context-rows", type=int, default=3)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--project-root",
                   default=str(REPO_ROOT),
                   help="Workbench project root (used to resolve --run-id).")
    p.add_argument("--log-level", default="INFO")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
