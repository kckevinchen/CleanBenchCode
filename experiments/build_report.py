"""Regenerate ``report.html`` from a persisted run directory.

``experiments.run_cleaning_detection`` emits all machine-readable
artifacts (clusters.jsonl, issues.jsonl, summary.json, chunk_reports.*,
and the optional ``good_turing/`` subdir) into a timestamped run
directory but deliberately does **not** produce the HTML dashboard.
This script is the second stage: it loads those artifacts back and
renders ``report.html`` via the same
:func:`src.report_builder.render_html` used to ship with the run.

Running it is free (no LLM calls) and fully offline, so dashboards can
be rebuilt against a newer ``frontend/report_template.html`` without
re-running detection.

Usage
-----

::

    python -m experiments.build_report outputs/20260424T010203Z_gpt-5.2

    # Pick a different template:
    python -m experiments.build_report <run_dir> \\
        --template frontend/report_template.html

    # Write to a different path:
    python -m experiments.build_report <run_dir> --output /tmp/report.html

The run directory is expected to contain ``summary.json``,
``clusters.jsonl`` and ``issues.jsonl``. If ``good_turing/
gt_dashboard.json`` is present it will be embedded as the adaptive-mode
tab payload.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.data_loader import load_csv  # noqa: E402
from src.core.report_builder import render_html  # noqa: E402

logger = logging.getLogger("build_report")


@dataclass
class _ClusterShim:
    """Tiny stand-in for :class:`src.aggregator.Cluster`.

    ``render_html`` only needs ``.as_dict()``; we short-circuit that by
    returning the already-serialised cluster record straight from
    ``clusters.jsonl``.
    """
    data: Dict[str, Any]

    def as_dict(self) -> Dict[str, Any]:
        return self.data


from src.core.jsonl_io import read_jsonl as _read_jsonl  # noqa: E402


def _load_source_frames(
    csv_paths: List[str],
) -> Dict[str, pd.DataFrame]:
    """Re-load the source CSVs recorded in ``summary.config.csv_paths``.

    Matches the collision-handling rule used at detection time: when two
    CSVs share a basename, later ones fall back to a path-based key so
    the dashboard's per-source breakdown stays consistent.
    """
    frames: Dict[str, pd.DataFrame] = {}
    for raw in csv_paths:
        path = Path(raw)
        if not path.is_file():
            logger.warning(
                "source CSV %s no longer exists; skipping (dashboard "
                "will omit the 'Source data' view for it)", path,
            )
            continue
        df = load_csv(path)
        name = path.name
        if name in frames:
            name = str(path).replace("\\", "/")
        frames[name] = df
    return frames


def _build_run_metadata(
    summary: Dict[str, Any], run_dir: Path,
) -> Dict[str, Any]:
    """Reconstruct the ``run_metadata`` dict the detector would have
    passed to ``render_html`` at run time.

    Most fields live under ``summary["config"]``; the run-id and
    generated-at timestamp come from the run directory name
    (``<YYYYMMDDTHHMMSSZ>_<model>``).
    """
    config = dict(summary.get("config", {}))
    sources = config.get("sources") or []
    source_count = len(sources)
    run_id = run_dir.name
    timestamp = run_id.split("_", 1)[0] if "_" in run_id else ""
    metadata = dict(config)
    metadata.update({
        "run_id": run_id,
        "generated_at": timestamp,
        "source": sources[0] if source_count == 1 else f"{source_count} files",
        "sources": sources,
        "error_types_count": len(config.get("error_types") or []),
    })
    return metadata


def build_report(
    run_dir: Path,
    *,
    template_path: Path,
    output_path: Optional[Path] = None,
) -> Path:
    summary_path = run_dir / "summary.json"
    clusters_path = run_dir / "clusters.jsonl"
    issues_path = run_dir / "issues.jsonl"
    for req in (summary_path, clusters_path, issues_path):
        if not req.is_file():
            raise FileNotFoundError(f"missing artifact: {req}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    clusters = [_ClusterShim(d) for d in _read_jsonl(clusters_path)]
    issues = _read_jsonl(issues_path)

    gt_dashboard_path = run_dir / "good_turing" / "gt_dashboard.json"
    good_turing: Optional[Dict[str, Any]] = None
    if gt_dashboard_path.is_file():
        good_turing = json.loads(gt_dashboard_path.read_text(encoding="utf-8"))

    csv_paths = summary.get("config", {}).get("csv_paths") or []
    source_frames = _load_source_frames(csv_paths)

    run_metadata = _build_run_metadata(summary, run_dir)
    html = render_html(
        template_path,
        issues,
        clusters,
        summary,
        run_metadata,
        source_frames=source_frames,
        good_turing=good_turing,
    )
    out = output_path or (run_dir / "report.html")
    out.write_text(html, encoding="utf-8")
    return out


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Render report.html from a saved run directory.",
    )
    p.add_argument(
        "run_dir", type=Path,
        help="Run directory produced by experiments.run_cleaning_detection.",
    )
    p.add_argument(
        "--template", type=Path,
        default=PROJECT_ROOT / "frontend" / "report_template.html",
        help="Path to the HTML dashboard template.",
    )
    p.add_argument(
        "--output", type=Path, default=None,
        help="Where to write the rendered HTML. Defaults to <run_dir>/report.html.",
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
    if not args.template.is_file():
        logger.error("HTML template not found: %s", args.template)
        return 2
    out = build_report(
        run_dir,
        template_path=args.template,
        output_path=args.output,
    )
    logger.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
