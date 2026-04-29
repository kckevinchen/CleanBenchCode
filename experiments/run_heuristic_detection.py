"""Run heuristic detectors over a CSV directory.

Produces a run dir whose layout mirrors ``experiments.run_cleaning_detection``
(``issues.jsonl``, ``clusters.json``, ``summary.json``,
optional ``report.html``) so the same dashboard, labeling UI, and
comparison harness consume both kinds of runs.

Usage::

    python -m experiments.run_heuristic_detection \\
        --csv-dir data/curated_pbi_full/271841717 \\
        --output-dir outputs \\
        --detectors whitespace,unicode_norm,casing_fingerprint,\\
                    pattern_minority,key_collision_typo,fd_violation,\\
                    numeric_outlier

The ``--detectors`` flag defaults to "all registered". The run is
fully offline and deterministic.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.aggregator import cluster_issues  # noqa: E402
from src.core.data_loader import load_csv  # noqa: E402
from src.core.report_builder import build_run_artifacts  # noqa: E402
from src.heuristics import REGISTRY, HeuristicConfig  # noqa: E402

logger = logging.getLogger("run_heuristic_detection")


def _discover_csvs(csv_dir: Path) -> List[Path]:
    """Find all CSVs under ``csv_dir``.

    Supports three layouts transparently:

    * Single CSV file.
    * Flat directory: ``csv_dir/*.csv``.
    * Pre-grouped tree (e.g. ``data/curated_bing21_grouped/<wb>.xlsx/<sheet>.csv``):
      one level of subdirectories, each containing CSVs.

    Hidden / underscore-prefixed dirs (``.lang_cache``, ``_unmatched``)
    are skipped. Detection is automatic: if the top level contains any
    visible directory, we walk one level deep.
    """
    if csv_dir.is_file() and csv_dir.suffix.lower() == ".csv":
        return [csv_dir]
    entries = list(csv_dir.iterdir())
    has_subdirs = any(
        p.is_dir() and not p.name.startswith(".") and not p.name.startswith("_")
        for p in entries
    )
    paths: List[Path] = []
    if has_subdirs:
        for d in entries:
            if not d.is_dir():
                continue
            if d.name.startswith(".") or d.name.startswith("_"):
                continue
            paths.extend(p for p in d.iterdir()
                         if p.is_file() and p.suffix.lower() == ".csv")
    else:
        paths.extend(p for p in entries
                     if p.is_file() and p.suffix.lower() == ".csv")
    return sorted(paths)


def _resolve_detectors(spec: Optional[str]) -> List[str]:
    if not spec or spec.strip() in ("", "all"):
        return sorted(REGISTRY)
    requested = [s.strip() for s in spec.split(",") if s.strip()]
    unknown = [s for s in requested if s not in REGISTRY]
    if unknown:
        raise SystemExit(
            f"unknown detector(s): {unknown}; available: {sorted(REGISTRY)}"
        )
    return requested


def run_heuristics(
    csv_paths: List[Path],
    detector_names: List[str],
    *,
    config: HeuristicConfig,
    base_dir: Path,
) -> tuple[List[Mapping[str, Any]], Dict[str, pd.DataFrame], List[Dict[str, Any]]]:
    """Run all selected detectors over every CSV.

    Returns a tuple of (flat_issues, source_frames, per_detector_stats).
    The flat-issue shape matches what ``cluster_issues`` consumes.
    """
    detectors = [REGISTRY[name](config=config) for name in detector_names]
    flat_issues: List[Mapping[str, Any]] = []
    source_frames: Dict[str, pd.DataFrame] = {}
    per_detector: Dict[str, Dict[str, Any]] = {
        d.name: {"matches": 0, "elapsed_s": 0.0, "by_source": {}}
        for d in detectors
    }

    for path in csv_paths:
        source_name = path.name
        frame = load_csv(path)
        # Reset index so heuristic detectors get 0..N-1 regardless of
        # how the CSV was loaded.
        frame = frame.reset_index(drop=True)
        source_frames[source_name] = frame
        for det in detectors:
            t0 = time.perf_counter()
            try:
                matches = det.detect(frame, source=source_name)
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "detector %s failed on %s: %s", det.name, source_name, exc
                )
                continue
            elapsed = time.perf_counter() - t0
            per_detector[det.name]["matches"] += len(matches)
            per_detector[det.name]["elapsed_s"] += elapsed
            per_detector[det.name]["by_source"].setdefault(source_name, 0)
            per_detector[det.name]["by_source"][source_name] += len(matches)
            for i, m in enumerate(matches):
                issue = m.as_issue(
                    issue_id=f"h::{det.name}::{source_name}::{m.row}::{m.column}::{i}"
                )
                # Add absolute path so downstream tools that key by
                # full path keep working; ``source`` (basename) is the
                # primary key used by the labeling UI.
                issue["source_path"] = str(path)
                flat_issues.append(issue)

    stats: List[Dict[str, Any]] = [
        {"name": name, **payload} for name, payload in per_detector.items()
    ]
    return flat_issues, source_frames, stats


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv-dir", type=Path, required=True,
                   help="Directory of CSVs (or single CSV file).")
    p.add_argument("--output-dir", type=Path, default=Path("outputs"),
                   help="Parent directory for the new run dir.")
    p.add_argument("--detectors", default="all",
                   help="Comma list of detector names, or 'all' (default).")
    p.add_argument("--list-detectors", action="store_true",
                   help="Print the registered detector names and exit.")
    p.add_argument("--observed-max-chars", type=int, default=80)
    p.add_argument("--min-column-size", type=int, default=5)
    p.add_argument("--write-html", action="store_true", default=True,
                   help=("Render report.html (default: on). The dashboard "
                         "uses the same ``frontend/report_template.html`` "
                         "as the LLM runs."))
    p.add_argument("--no-html", dest="write_html", action="store_false")
    p.add_argument("--run-id", default=None,
                   help="Override the auto-generated run id.")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def _run_id(prefix: str = "heuristic") -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{ts}_{prefix}"


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    if args.list_detectors:
        for name in sorted(REGISTRY):
            print(name)
        return 0

    csv_paths = _discover_csvs(args.csv_dir)
    if not csv_paths:
        raise SystemExit(f"no CSVs found under {args.csv_dir}")
    logger.info("found %d CSV file(s) under %s", len(csv_paths), args.csv_dir)

    detectors = _resolve_detectors(args.detectors)
    logger.info("running detectors: %s", detectors)

    config = HeuristicConfig(
        observed_max_chars=args.observed_max_chars,
        min_column_size=args.min_column_size,
    )

    flat_issues, source_frames, det_stats = run_heuristics(
        csv_paths, detectors, config=config, base_dir=args.csv_dir,
    )
    logger.info("emitted %d flat issues", len(flat_issues))

    clusters = cluster_issues(flat_issues)

    run_id = args.run_id or _run_id()
    run_dir = args.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info("writing run dir %s", run_dir)

    run_metadata = {
        "kind": "heuristic",
        "run_id": run_id,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "detectors": detectors,
        "detector_stats": det_stats,
        "csv_dir": str(args.csv_dir),
        "n_sources": len(csv_paths),
    }
    config_payload = {
        "csv_paths": [str(p) for p in csv_paths],
        "sources": [p.name for p in csv_paths],
        "detectors": detectors,
        "observed_max_chars": args.observed_max_chars,
        "min_column_size": args.min_column_size,
        # Marker so build_report.py / the dashboard can distinguish a
        # heuristic run from an LLM run at a glance.
        "run_kind": "heuristic",
    }

    template = (PROJECT_ROOT / "frontend" / "report_template.html") if args.write_html else None
    paths = build_run_artifacts(
        reports=[],  # heuristic runs have no chunk reports
        run_dir=run_dir,
        flat_issues=flat_issues,
        clusters=clusters,
        config=config_payload,
        run_metadata=run_metadata,
        html_template_path=template,
        source_frames=source_frames,
    )
    # Persist the per-detector stats next to summary.json for the
    # comparison harness.
    (run_dir / "heuristic_stats.json").write_text(
        json.dumps(det_stats, indent=2, sort_keys=True), encoding="utf-8",
    )
    logger.info("wrote %d artifact(s)", len(paths))
    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
