"""Empirical validation of discovered relations.

Loads ``relations*.jsonl`` from a completed relation-discovery run, asks
the LLM to write code (or refuse) for each relation, runs the code in
a restricted sandbox against the source DataFrames, and writes
``relation_verifications.jsonl`` next to the relations files.

The dashboard (``experiments.build_relation_report``) picks up the
file automatically and renders pass-rates + sample violations on
each relation card.

Example::

    python -m experiments.run_relation_validation \\
        outputs/20260429T215852Z_gpt-5.2/ \\
        --layer auto \\
        --concurrency 4

``--layer auto`` (default) prefers the highest-level rule layer
present in the run dir: ``generalized`` > ``aggregated`` > originals.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.data_loader import load_csv  # noqa: E402
from src.core.jsonl_io import read_jsonl as _read_jsonl  # noqa: E402
from src.core.llm_client import LLMClient  # noqa: E402
from src.core.relation_validator import (  # noqa: E402
    load_validation_prompt,
    validate_relations,
    write_verifications_jsonl,
)


logger = logging.getLogger("run_relation_validation")


_LAYER_FILES = {
    "generalized": "relations_generalized.jsonl",
    "aggregated": "relations_aggregated.jsonl",
    "originals": "relations.jsonl",
}


def _resolve_layer(
    run_dir: Path, layer: str,
) -> List[tuple[str, Dict[str, Any]]]:
    """Return ``[(layer_name, relation), ...]`` for the requested layer.

    With ``layer == "union"`` (default), validate the highest-level
    rule available for each rule lineage:

      * Start with all ``generalized`` rules.
      * Add ``aggregated`` rules whose ``species_key`` was NOT absorbed
        by some generalized rule's ``generalized_from``.
      * Add ``originals`` whose ``species_key`` was NOT absorbed by a
        kept aggregated rule's ``aggregated_from`` and not absorbed by
        a generalized rule's ``generalized_from``.

    This guarantees every original rule is represented in the
    validation set exactly once, at the highest abstraction layer it
    was lifted to.

    With ``layer == "auto"`` (legacy), walk generalized -> aggregated
    -> originals and return the first non-empty file. Other values
    pick exactly one file.
    """
    if layer == "union":
        gen = _read_jsonl(run_dir / _LAYER_FILES["generalized"])
        agg = _read_jsonl(run_dir / _LAYER_FILES["aggregated"])
        orig = _read_jsonl(run_dir / _LAYER_FILES["originals"])

        def _src_keys(rels, field):
            absorbed: set[str] = set()
            for r in rels:
                for k in (r.get(field) or []):
                    if k:
                        absorbed.add(str(k))
            return absorbed

        absorbed_by_gen = _src_keys(gen, "generalized_from")
        kept_agg = [
            r for r in agg
            if str(r.get("species_key") or "") not in absorbed_by_gen
        ]
        absorbed_by_agg = _src_keys(kept_agg, "aggregated_from")
        kept_orig = [
            r for r in orig
            if str(r.get("species_key") or "") not in absorbed_by_gen
            and str(r.get("species_key") or "") not in absorbed_by_agg
        ]
        out: List[tuple[str, Dict[str, Any]]] = []
        out.extend(("generalized", r) for r in gen)
        out.extend(("aggregated", r) for r in kept_agg)
        out.extend(("originals", r) for r in kept_orig)
        return out

    if layer == "auto":
        for name in ("generalized", "aggregated", "originals"):
            rels = _read_jsonl(run_dir / _LAYER_FILES[name])
            if rels:
                return [(name, r) for r in rels]
        return []
    if layer not in _LAYER_FILES:
        raise ValueError(
            f"unknown layer {layer!r}; expected one of "
            f"{list(_LAYER_FILES) + ['union', 'auto']}"
        )
    rels = _read_jsonl(run_dir / _LAYER_FILES[layer])
    return [(layer, r) for r in rels]


def _load_source_frames(
    run_dir: Path,
    needed_sources: Sequence[str],
    *,
    csv_dirs: Sequence[Path] = (),
) -> Dict[str, pd.DataFrame]:
    """Load source CSVs referenced by the relations.

    Resolution order for each ``<source>.csv`` basename:
      1. ``summary.json`` -> ``config.csv_paths`` (recorded by
         ``run_relation_discovery``).
      2. Recursive search under each ``--csv-dir`` provided on the
         command line. First match wins.
    """
    summary_path = run_dir / "summary.json"
    csv_paths: List[Path] = []
    if summary_path.is_file():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if isinstance(summary, dict):
                # Top-level (current) and nested under 'config' (legacy).
                for p in summary.get("csv_paths") or []:
                    csv_paths.append(Path(p))
                cfg = summary.get("config") or {}
                if isinstance(cfg, dict):
                    for p in cfg.get("csv_paths") or []:
                        csv_paths.append(Path(p))
        except json.JSONDecodeError:
            logger.warning("could not parse %s", summary_path)
    needed = set(needed_sources)
    by_name: Dict[str, Path] = {}
    for p in csv_paths:
        if p.name in needed and p.is_file():
            by_name.setdefault(p.name, p)
    # Fallback: recursive scan of the user-provided csv-dirs for any
    # still-unresolved basenames.
    unresolved = needed - set(by_name)
    if unresolved and csv_dirs:
        for root in csv_dirs:
            root = Path(root)
            if not root.is_dir():
                logger.warning("--csv-dir not a directory: %s", root)
                continue
            for path in root.rglob("*.csv"):
                if path.name in unresolved:
                    by_name.setdefault(path.name, path)
            unresolved = needed - set(by_name)
            if not unresolved:
                break
    frames: Dict[str, pd.DataFrame] = {}
    missing: List[str] = []
    for name in needed:
        path = by_name.get(name)
        if path is None:
            missing.append(name)
            continue
        try:
            frames[name] = load_csv(path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("failed to load %s: %s", path, exc)
    if missing:
        logger.warning(
            "%d source(s) referenced by relations could not be located "
            "(pass --csv-dir to point at the source tree). Examples: %s",
            len(missing), sorted(missing)[:5],
        )
    return frames


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Empirically validate a relation-discovery run by running "
            "LLM-written check() code against each relation."
        ),
    )
    p.add_argument(
        "run_dir", type=Path,
        help="Run directory produced by experiments.run_relation_discovery.",
    )
    p.add_argument(
        "--layer", default="union",
        choices=["union", "auto", "generalized", "aggregated", "originals"],
        help="Which rule layer to validate. 'union' (default) takes "
        "every rule at the highest abstraction layer it was lifted "
        "to: all generalized rules + aggregated rules not absorbed "
        "by a generalization + originals not absorbed by either. "
        "'auto' (legacy) picks the first non-empty layer in order "
        "generalized > aggregated > originals.",
    )
    p.add_argument(
        "--prompt", type=Path,
        default=PROJECT_ROOT / "prompts" / "validate_relation.yaml",
        help="Validation prompt YAML.",
    )
    p.add_argument(
        "--model", default=None,
        help="Override the model from the prompt YAML.",
    )
    p.add_argument(
        "--reasoning-effort", default=None,
        help="Override reasoning_effort from the prompt YAML.",
    )
    p.add_argument(
        "--concurrency", type=int, default=4,
        help="Parallel LLM calls.",
    )
    p.add_argument(
        "--sample-rows", type=int, default=30,
        help="How many head rows to include in the prompt.",
    )
    p.add_argument(
        "--timeout-seconds", type=float, default=8.0,
        help="Wall-clock cap per check() execution.",
    )
    p.add_argument(
        "--max-violation-samples", type=int, default=5,
        help="How many violating rows to record per relation.",
    )
    p.add_argument(
        "--limit", type=int, default=None,
        help="Validate at most N relations (useful for smoke tests).",
    )
    p.add_argument(
        "--output", type=Path, default=None,
        help="Where to write verifications JSONL. Defaults to "
        "<run_dir>/relation_verifications.jsonl.",
    )
    p.add_argument(
        "--csv-dir", type=Path, action="append", default=[],
        help="Directory to search recursively for source CSVs when "
        "summary.json doesn't record csv_paths (e.g. older runs). "
        "May be passed multiple times.",
    )
    p.add_argument(
        "--k-max", type=int, default=1,
        help="Max LLM samples per relation (recall-first union of "
        "violators across runs). 1 = single-shot (default, legacy "
        "behaviour). >=2 enables the K-sample ensemble path.",
    )
    p.add_argument(
        "--k-min", type=int, default=2,
        help="Minimum samples before the Good-Turing stopping rule "
        "may fire. Only meaningful with --k-max >= 2.",
    )
    p.add_argument(
        "--epsilon", type=float, default=0.05,
        help="Good-Turing stop threshold for the K-sample ensemble.",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    args = parse_args(argv)

    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        logger.error("run directory not found: %s", run_dir)
        return 2

    tagged = _resolve_layer(run_dir, args.layer)
    if not tagged:
        logger.error(
            "no relations found for layer %s in %s", args.layer, run_dir,
        )
        return 2
    if args.limit is not None:
        tagged = tagged[: args.limit]
    layer_counts: Dict[str, int] = {}
    for ln, _ in tagged:
        layer_counts[ln] = layer_counts.get(ln, 0) + 1
    logger.info(
        "validating %d relations (layer=%s, breakdown=%s)",
        len(tagged), args.layer, layer_counts,
    )

    # Source frames. Multi-table relations carry the workbook id in
    # ``source`` and the actual sheet filenames in ``tables``. We need
    # frames for BOTH so the validator can resolve every per-sheet
    # reference - using only ``source`` skipped sheet frames and made
    # every multi-table record fail with "missing source frame".
    needed_set: set = set()
    for _, r in tagged:
        src = str(r.get("source") or "").strip()
        if src:
            needed_set.add(src)
        tables = r.get("tables") or []
        if isinstance(tables, (list, tuple)):
            for t in tables:
                t_str = str(t).strip()
                if t_str:
                    needed_set.add(t_str)
    needed_sources = sorted(needed_set)
    source_frames = _load_source_frames(
        run_dir, needed_sources, csv_dirs=args.csv_dir or [],
    )

    # Prompt + model.
    prompt_cfg = load_validation_prompt(args.prompt)
    model = args.model or prompt_cfg.get("model") or "gpt-5.2"
    if args.reasoning_effort:
        prompt_cfg = dict(prompt_cfg)
        prompt_cfg["reasoning_effort"] = args.reasoning_effort

    llm = LLMClient()

    # Build the optional arm-config resolver for the K-sample
    # ensemble path. When k-max == 1 (legacy default), pass None so
    # validate_relations preserves single-shot behaviour byte-for-byte.
    arm_config_resolver = None
    if int(args.k_max) > 1:
        from src.adaptive.scoped_gt import ArmConfig
        _shared_arm_cfg = ArmConfig(
            epsilon=float(args.epsilon),
            epsilon_chao1=float(args.epsilon),
            stop_when="gt",
            k_min=max(1, int(args.k_min)),
            k_max=int(args.k_max),
            require_two_consecutive=False,
        )
        def arm_config_resolver(_rel: Dict[str, Any]) -> ArmConfig:  # type: ignore[no-redef]
            return _shared_arm_cfg
        logger.info(
            "K-sample ensemble enabled: k_min=%d k_max=%d epsilon=%.3f",
            _shared_arm_cfg.k_min, _shared_arm_cfg.k_max, _shared_arm_cfg.epsilon,
        )

    total_all = len(tagged)
    progress_done = {"n": 0}

    # Group relations by their source layer so each verification
    # record carries the correct ``layer`` field. Within a group the
    # batch validator parallelizes LLM calls.
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for ln, rel in tagged:
        groups.setdefault(ln, []).append(rel)

    records: List[Any] = []
    for ln, rels in groups.items():
        def _progress(done: int, total: int, _ln: str = ln) -> None:
            # Aggregate progress across groups for a smooth global view.
            if done == total:
                progress_done["n"] += done
                logger.info(
                    "validated %d/%d relations (layer=%s done)",
                    progress_done["n"], total_all, _ln,
                )

        recs = validate_relations(
            rels,
            source_frames=source_frames,
            llm=llm,
            model=model,
            prompt_cfg=prompt_cfg,
            concurrency=max(1, int(args.concurrency)),
            sample_rows=int(args.sample_rows),
            timeout_seconds=float(args.timeout_seconds),
            max_violation_samples=int(args.max_violation_samples),
            layer=ln,
            progress_cb=_progress,
            arm_config_resolver=arm_config_resolver,
        )
        records.extend(recs)

    out_path = args.output or (run_dir / "relation_verifications.jsonl")
    n = write_verifications_jsonl(records, out_path)
    logger.info("wrote %d verifications to %s", n, out_path)

    # Summary stats.
    n_verifiable = sum(1 for r in records if r.verifiable)
    n_with_passrate = sum(1 for r in records if r.pass_rate is not None)
    n_errors = sum(1 for r in records if r.error)
    n_perfect = sum(
        1 for r in records
        if r.pass_rate is not None and r.pass_rate >= 0.999
    )
    logger.info(
        "summary: verifiable=%d/%d  scored=%d  perfect=%d  errors=%d",
        n_verifiable, len(records), n_with_passrate, n_perfect, n_errors,
    )

    # Optionally re-render the dashboard so the new pass-rates surface
    # without a manual rebuild step.
    try:
        from experiments.build_relation_report import build_relation_report  # noqa: WPS433
        template_path = PROJECT_ROOT / "frontend" / "relations_template.html"
        if template_path.is_file():
            build_relation_report(run_dir, template_path=template_path)
            logger.info("dashboard refreshed: %s", run_dir / "report.html")
    except Exception as exc:  # noqa: BLE001 - non-fatal
        logger.warning("dashboard refresh failed: %s", exc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
