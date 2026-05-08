"""Run :mod:`find_errors_by_normalization` on a CSV dataset and emit a
run dir whose layout matches ``experiments.run_cleaning_detection``.

The output run dir contains ``issues.jsonl`` / ``clusters.json`` /
``summary.json`` / ``report.html`` with the same schema the LLM
pipeline produces, so the existing dashboard, labeling UI, and the
``experiments.compare_runs`` harness all consume it without
modification. The original column-level findings emitted by the
external script are also preserved verbatim under
``external_normalization.jsonl`` for traceability.

Usage (from the project root)::

    python -m external.run_normalization_experiment \\
        --csv-dir data/curated_bing21_min_filtered \\
        --output-dir outputs

The script reuses the heuristic from
``external/find_errors_by_normalization.py`` untouched: it imports
``find_errors_by_normalization`` and only adapts I/O + output shape.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Reuse the heuristic untouched. ``external/`` has no ``__init__.py``,
# so add its directory to sys.path and import the module by name.
_EXT_DIR = Path(__file__).resolve().parent
if str(_EXT_DIR) not in sys.path:
    sys.path.insert(0, str(_EXT_DIR))
from find_errors_by_normalization import find_errors_by_normalization  # noqa: E402

from src.core.aggregator import cluster_issues  # noqa: E402
from src.core.data_loader import load_csv  # noqa: E402
from src.core.report_builder import build_run_artifacts  # noqa: E402

logger = logging.getLogger("run_normalization_experiment")


# The external heuristic flags two-or-more variants of the same
# entity in one column - the "text" sub-pattern of the merged
# ``inconsistency`` umbrella in the LLM taxonomy. Tagging every
# emitted issue with this single bucket keeps
# ``experiments.compare_runs`` (whose ERROR_TYPE_BUCKETS table maps
# types onto buckets) able to diff the heuristic against the LLM
# directly. Refer to ``error_type.csv`` / ``prompts/error_types.yaml``
# for the canonical names.
EXT_ERROR_TYPE = "inconsistency"
EXT_ERROR_TYPE_NAME = "Inconsistency"
EXT_ERROR_TYPE_CATEGORY = "Single-Column / Inconsistency"
EXT_CATEGORY = EXT_ERROR_TYPE_CATEGORY.lower()


def _discover_csvs(csv_dir: Path) -> List[Path]:
    """Mirror ``experiments.run_heuristic_detection._discover_csvs``.

    Handles three layouts:
      * single CSV file
      * flat directory of CSVs
      * one-level-nested (``<csv_dir>/<group>/<file>.csv``)
    The grouped layout is what ``data/curated_bing21_min_filtered``
    uses (each ``<hash>.xlsx/`` holds one or more CSVs).
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


def _run_id(prefix: str = "external_norm") -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{ts}_{prefix}"


def _to_columns_dict(frame: pd.DataFrame) -> Dict[str, List[str]]:
    """Build the ``{column -> list[str_values]}`` dict the heuristic
    expects. Row order is preserved so per-row indices line up with
    the source frame.
    """
    out: Dict[str, List[str]] = {}
    for col in frame.columns:
        out[str(col)] = [
            "" if pd.isna(v) else str(v).strip()
            for v in frame[col].tolist()
        ]
    return out


def _full_rows(frame: pd.DataFrame) -> List[Dict[str, str]]:
    """Materialise per-row column dicts for ``full_row_values``."""
    cols = [str(c) for c in frame.columns]
    rows: List[Dict[str, str]] = []
    for record in frame.itertuples(index=False, name=None):
        rows.append({
            c: ("" if pd.isna(v) else str(v))
            for c, v in zip(cols, record)
        })
    return rows


def expand_findings(
    findings: List[Mapping[str, Any]],
    *,
    source: str,
    columns_dict: Dict[str, List[str]],
    full_rows: List[Dict[str, str]],
    next_idx_start: int = 0,
) -> List[Dict[str, Any]]:
    """Lift column-level findings to per-cell flat issues.

    Each entry in ``findings`` describes one normalisation-key cluster
    of variants in one column. We emit one flat issue per row whose
    cell value is one of the variants, so downstream clustering /
    labeling / comparison all operate at the same cell granularity as
    the LLM output.
    """
    issues: List[Dict[str, Any]] = []
    next_idx = next_idx_start
    for finding in findings:
        col = finding["column"]
        variants = list(finding.get("variants") or [])
        if col not in columns_dict or not variants:
            continue
        variant_set = set(variants)
        col_values = columns_dict[col]
        for row, value in enumerate(col_values):
            if value not in variant_set:
                continue
            full_row = full_rows[row] if row < len(full_rows) else {}
            issues.append({
                "id": f"ext-{source}-{next_idx:06d}",
                "report_index": 0,
                "source": source,
                "model": "external/find_errors_by_normalization",
                "prompt": "external_normalization",
                "error_type": EXT_ERROR_TYPE,
                "error_type_name": EXT_ERROR_TYPE_NAME,
                "error_type_category": EXT_ERROR_TYPE_CATEGORY,
                "category": EXT_CATEGORY,
                "row": row,
                "columns": [col],
                "affected_rows": [],
                "observed": value,
                "explanation": (
                    f"Value {value!r} normalises to key {finding['key']!r}; "
                    f"column {col!r} has {len(variants)} variants sharing "
                    f"this key: {variants}"
                ),
                "suggested_fix": None,
                "row_values": {col: value},
                "full_row_values": full_row,
                "meta": {
                    "detector": "external/find_errors_by_normalization",
                    "norm_key": finding["key"],
                    "variants": variants,
                    "cell_count": finding.get("cell_count"),
                },
            })
            next_idx += 1
    return issues


def run(csv_paths: List[Path]) -> tuple[
    List[Mapping[str, Any]],
    Dict[str, pd.DataFrame],
    List[Dict[str, Any]],
]:
    flat_issues: List[Mapping[str, Any]] = []
    raw_findings: List[Dict[str, Any]] = []
    source_frames: Dict[str, pd.DataFrame] = {}
    next_idx = 0
    for path in csv_paths:
        source = path.name
        try:
            frame = load_csv(path).reset_index(drop=True)
        except Exception as exc:  # noqa: BLE001
            logger.warning("skipping %s: %s", path, exc)
            continue
        source_frames[source] = frame
        cols = _to_columns_dict(frame)
        full_rows = _full_rows(frame)
        findings = find_errors_by_normalization(cols)
        for f in findings:
            entry = dict(f)
            entry["file"] = source
            raw_findings.append(entry)
        per_file_issues = expand_findings(
            findings,
            source=source,
            columns_dict=cols,
            full_rows=full_rows,
            next_idx_start=next_idx,
        )
        next_idx += len(per_file_issues)
        flat_issues.extend(per_file_issues)
    return flat_issues, source_frames, raw_findings


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--csv-dir", type=Path, required=True,
                   help="Directory of CSVs (or single CSV file).")
    p.add_argument("--output-dir", type=Path, default=Path("outputs"),
                   help="Parent directory for the new run dir.")
    p.add_argument("--write-html", action="store_true", default=True,
                   help="Render report.html (default: on).")
    p.add_argument("--no-html", dest="write_html", action="store_false")
    p.add_argument("--run-id", default=None,
                   help="Override the auto-generated run id.")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    csv_paths = _discover_csvs(args.csv_dir)
    if not csv_paths:
        raise SystemExit(f"no CSVs found under {args.csv_dir}")
    logger.info("found %d CSV file(s) under %s", len(csv_paths), args.csv_dir)

    flat_issues, source_frames, raw_findings = run(csv_paths)
    logger.info(
        "emitted %d flat issues from %d raw findings across %d source(s)",
        len(flat_issues), len(raw_findings), len(source_frames),
    )

    clusters = cluster_issues(flat_issues)

    run_id = args.run_id or _run_id()
    run_dir = args.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info("writing run dir %s", run_dir)

    run_metadata = {
        "kind": "external_normalization",
        "run_id": run_id,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "detectors": ["external/find_errors_by_normalization"],
        "csv_dir": str(args.csv_dir),
        "n_sources": len(source_frames),
    }
    config_payload = {
        "csv_paths": [str(p) for p in csv_paths],
        "sources": [p.name for p in csv_paths],
        "detectors": ["external/find_errors_by_normalization"],
        "run_kind": "external_normalization",
        "error_type_bucket": EXT_ERROR_TYPE,
    }

    template = (
        PROJECT_ROOT / "frontend" / "report_template.html"
        if args.write_html else None
    )
    paths = build_run_artifacts(
        reports=[],
        run_dir=run_dir,
        flat_issues=flat_issues,
        clusters=clusters,
        config=config_payload,
        run_metadata=run_metadata,
        html_template_path=template,
        source_frames=source_frames,
    )

    raw_path = run_dir / "external_normalization.jsonl"
    with raw_path.open("w", encoding="utf-8") as fp:
        for f in raw_findings:
            fp.write(json.dumps(f, ensure_ascii=False) + "\n")
    paths["external_normalization_jsonl"] = raw_path

    logger.info("wrote %d artifact(s)", len(paths))
    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
