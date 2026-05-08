"""Sweep ``llm.reasoning_effort`` across the smoke-test datasets.

Goal: hold every other knob fixed and ask "if we change the detection
LLM's reasoning effort, will it stop catching the planted errors?".
For each ``(dataset, effort)`` cell we run one
``experiments.run_cleaning_detection`` invocation in ``mode=single``
(no Good-Turing perturbation, no Phase-B synthesis), then aggregate
the per-cell ``summary.json`` / ``issues.jsonl`` into a side-by-side
comparison report::

    outputs/reasoning_effort_sweep_<UTC ts>/
      runs/
        smoke_singlecol__low/  ...   # one full run dir per cell
      cells.jsonl                    # per-cell issue keys (canonical)
      comparison.tsv                 # detected counts per (dataset, effort)
      comparison.md                  # markdown table + delta-vs-medium
      summary.json                   # machine-readable rollup

Usage::

    python -m experiments.sweep_reasoning_effort                # full sweep
    python -m experiments.sweep_reasoning_effort --dry-run      # validate only
    python -m experiments.sweep_reasoning_effort \\
        --efforts low,high \\
        --datasets smoke_singlecol,smoke_olympic                # subset

The multi-table pipeline (smoke_multitable) pins
``reasoning_effort`` inside its prompt YAMLs (default ``medium``),
so the CLI ``--reasoning-effort`` flag alone would not propagate
there. The sweep generates per-effort copies of those YAMLs into a
temp dir under the sweep root and passes them via the matching
``--multi-table-*-prompt`` flags so multi-table runs honour the
sweep effort end-to-end.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("sweep_reasoning_effort")


# ---------------------------------------------------------------------------
# Cell definitions: each entry maps a smoke directory to the scope /
# error-type / column-mode combo that exercises the planted errors.
# ---------------------------------------------------------------------------

# Each dataset config is a flat dict so the orchestrator can build a
# CLI invocation with no per-dataset branching.
DATASETS: List[Dict[str, Any]] = [
    {
        "name": "smoke_singlecol",
        # Per-column scan; the file plants 1 error per single-column type.
        "csv_arg": "data/smoke_singlecol/employees.csv",
        "scopes": "single-column",
        "error_types": "typo,inconsistency,semantic_incompatible,numeric_outlier",
        "column_mode": "single",
        "uses_multi_table": False,
        # Hand-coded ground-truth keys (source, row, column, error_type).
        # Row indices are 0-based against DATA ROWS (header excluded),
        # matching the convention used by the per-type detector when
        # it emits ``row`` / ``affected_rows``. So data row 0 == file
        # line 2 == E001; data row 3 == file line 5 == E004.
        #
        # NOTE on the ``typo`` type: ``prompts/error_types.yaml``
        # deliberately FAST-SKIPS person-name and email columns (the
        # SEMANTIC VIEW carve-out). The fixture therefore plants no
        # typo errors -- recall on this file is reported against the
        # remaining three single-column types only. Add a planted
        # ``typo`` against an identifier-like vocabulary column
        # (status / unit / state / org-name) if you want to exercise
        # that detector here in the future.
        "expected": [
            ("employees.csv", 3, "status", "inconsistency"),         # E004 "active"
            ("employees.csv", 7, "phone", "inconsistency"),           # E008 missing dashes
            ("employees.csv", 9, "join_date", "inconsistency"),       # E010 en-dash
            ("employees.csv", 10, "discount_rate", "inconsistency"),  # E011 "10%"
            ("employees.csv", 11, "department", "inconsistency"),     # E012 "Eng."
            ("employees.csv", 12, "status", "inconsistency"),         # E013 "ACTIVE"
            ("employees.csv", 13, "annual_salary", "numeric_outlier"),# E014 5M
            ("employees.csv", 14, "phone", "inconsistency"),          # E015 "(555) ..."
            ("employees.csv", 15, "email", "semantic_incompatible"),  # E016 phone in email
            ("employees.csv", 16, "department", "inconsistency"),     # E017 "engineering dept"
            ("employees.csv", 17, "join_date", "inconsistency"),      # E018 en-dash
        ],
    },
    {
        "name": "smoke_multicol",
        "csv_arg": "data/smoke_multicol/orders.csv",
        "scopes": "multi-column",
        "error_types": (
            "fd_violation,row_calculation_inconsistency,"
            "string_calculation_inconsistency"
        ),
        "column_mode": "all",
        "uses_multi_table": False,
        "expected": [
            # O014 (data row 13): zip 30301 -> {NY, GA} elsewhere, here state=CA
            ("orders.csv", 13, "state", "fd_violation"),
            # O017 (data row 16): unit_price*quantity=50 but total=75
            ("orders.csv", 16, "total", "row_calculation_inconsistency"),
            # O019 (data row 18): full_name="Sam X Smith" != first+" "+last
            ("orders.csv", 18, "full_name", "string_calculation_inconsistency"),
        ],
    },
    {
        "name": "smoke_multirow",
        "csv_arg": "data/smoke_multirow/customers.csv",
        "scopes": "multi-row",
        "error_types": "row_duplication",
        "column_mode": "all",
        "uses_multi_table": False,
        # row_duplication emits ``columns: []``; we project to
        # column="" and credit a hit when EITHER member of the dup
        # pair appears in ``affected_rows``. Data rows 15/16/17 are
        # the second-occurrence rows of the three planted dups
        # (exact C002, near-Acme C016, near-Wayne C017).
        "expected": [
            ("customers.csv", 15, "", "row_duplication"),  # exact dup of C002 (row 1)
            ("customers.csv", 16, "", "row_duplication"),  # near-dup of C001 (row 0)
            ("customers.csv", 17, "", "row_duplication"),  # near-dup of C009 (row 8)
        ],
    },
    {
        "name": "smoke_olympic",
        "csv_arg": "data/smoke_olympic/medals.csv",
        # Mixed: planted error is single-column casing inconsistency
        # ("Great britain" vs "Great Britain"). Row indices are
        # 0-based against data rows: 2012 GB at line 13 -> data row 11;
        # 2016 GB at line 21 -> data row 19.
        "scopes": "single-column,multi-column",
        "error_types": (
            "typo,inconsistency,row_calculation_inconsistency"
        ),
        "column_mode": "auto",
        "uses_multi_table": False,
        "expected": [
            ("medals.csv", 11, "country", "inconsistency"),  # 2012 "Great britain"
            ("medals.csv", 19, "country", "inconsistency"),  # 2016 "Great britain"
        ],
    },
    {
        "name": "smoke_multitable",
        "csv_arg": "data/smoke_multitable/sales_book",
        "scopes": "multi-table",
        "error_types": "inter_table_inconsistency",
        "column_mode": "all",
        "uses_multi_table": True,
        # Multi-table findings live in ``multi_table_violations.jsonl``
        # (and are loaded separately below). Row indices follow the
        # SAME 0-based-against-data-rows convention as the per-chunk
        # detector. O005 -> data row 4; O013 -> data row 12.
        "expected": [
            # O005: orders says Cambridge but customers says Boston for C002
            ("orders.csv", 4, "customer_city", "inter_table_inconsistency"),
            # O013: customer_id C099 not in customers (FK orphan)
            ("orders.csv", 12, "customer_id", "inter_table_inconsistency"),
        ],
    },
]


# Multi-table prompt YAMLs that pin ``reasoning_effort:`` internally;
# the sweep rewrites these per-effort and passes the rewritten paths
# via ``--multi-table-prompt`` / ``--multi-table-detect-prompt``.
MULTI_TABLE_PROMPTS = {
    "confirm": "prompts/confirm_union_join.yaml",
    "detect": "prompts/detect_multi_table_inconsistency.yaml",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _materialise_multitable_prompts(
    effort: str, sweep_root: Path
) -> Dict[str, Path]:
    """Copy each multi-table prompt YAML into ``sweep_root/prompts_<effort>/``
    with ``reasoning_effort:`` rewritten to ``effort``.

    Returns a mapping from ``MULTI_TABLE_PROMPTS`` key to the rewritten
    file path.
    """
    out_dir = sweep_root / "prompts" / effort
    out_dir.mkdir(parents=True, exist_ok=True)
    rewritten: Dict[str, Path] = {}
    for key, rel_src in MULTI_TABLE_PROMPTS.items():
        src = PROJECT_ROOT / rel_src
        dst = out_dir / src.name
        text = src.read_text(encoding="utf-8")
        new_lines = []
        for line in text.splitlines():
            if line.startswith("reasoning_effort:"):
                new_lines.append(f"reasoning_effort: {effort}")
            else:
                new_lines.append(line)
        # Preserve trailing newline if the original had one.
        if text.endswith("\n"):
            new_lines.append("")
        dst.write_text("\n".join(new_lines), encoding="utf-8")
        rewritten[key] = dst
    return rewritten


def _build_invocation(
    dataset: Dict[str, Any],
    effort: str,
    *,
    output_dir: Path,
    run_name: str,
    multi_table_prompts: Optional[Dict[str, Path]],
    extra_args: Iterable[str] = (),
) -> List[str]:
    cmd: List[str] = [
        sys.executable,
        "-m",
        "experiments.run_cleaning_detection",
        "--config",
        "config/experiment.yaml",
        "--mode",
        "single",
        "--csv",
        dataset["csv_arg"],
        "--reasoning-effort",
        effort,
        "--scopes",
        dataset["scopes"],
        "--error-types",
        dataset["error_types"],
        "--column-mode",
        dataset["column_mode"],
        "--no-synthesis",
        "--output-dir",
        str(output_dir),
        "--run-name",
        run_name,
        # Force chunk_size large enough to fit any smoke source in 1
        # chunk, so chunk count isn't a confound.
        "--chunk-size",
        "200",
        # Cap row_duplication's collision-adjudication cost; smoke
        # data has at most a handful of groups.
        "--row-dup-max-groups-per-source",
        "10",
    ]
    if dataset.get("uses_multi_table") and multi_table_prompts is not None:
        cmd.extend([
            "--multi-table-prompt",
            str(multi_table_prompts["confirm"]),
            "--multi-table-detect-prompt",
            str(multi_table_prompts["detect"]),
        ])
    cmd.extend(extra_args)
    return cmd


def _run_cell(
    cmd: List[str], log_path: Path, *, dry_run: bool
) -> Tuple[int, float]:
    """Spawn one detection run and tee stdout+stderr to ``log_path``.

    Returns ``(exit_code, wall_seconds)``. In ``dry_run`` mode we
    append ``--validate-config`` so no LLM call is made.
    """
    if dry_run and "--validate-config" not in cmd:
        cmd = list(cmd) + ["--validate-config"]
    logger.info("    cmd: %s", " ".join(cmd))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)
    with log_path.open("w", encoding="utf-8") as fh:
        fh.write(f"# {started.isoformat()}\n# {' '.join(cmd)}\n\n")
        fh.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=fh,
            stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        rc = proc.wait()
    wall = (datetime.now(timezone.utc) - started).total_seconds()
    return rc, wall


# ---------------------------------------------------------------------------
# Result loaders + comparison builders
# ---------------------------------------------------------------------------


# Cell key shape: (source_filename, row_1based, column_or_empty, error_type).
CellKey = Tuple[str, int, str, str]


def _coerce_int(v: Any) -> Optional[int]:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _issue_to_keys(issue: Dict[str, Any]) -> List[CellKey]:
    """Project a flat issue record onto one or more canonical cell keys.

    Most detectors emit one row per (row, column) hit. Multi-cell
    findings (e.g. ``string_calculation_inconsistency`` covering
    multiple cells in a row) are exploded into one key per cell.
    Row-level findings without a column (e.g. ``row_duplication``)
    fall back to ``column=''``.

    Row coverage rules:
    * ``affected_rows`` (when present and non-empty) is the canonical
      list of rows the finding applies to; we emit one key per
      affected row. This is what catches multi-row findings such as
      a ``row_duplication`` adjudication that names rows ``[0, 16]``
      or an ``inconsistency`` finding whose primary ``row`` is the
      first hit but whose ``affected_rows`` lists every minority
      cell in the column.
    * Otherwise we fall back to the primary ``row`` field (most
      multi-column / calculation findings emit ``affected_rows: []``
      and only set ``row`` to the offending data row).
    """
    src = str(issue.get("source") or issue.get("source_path") or "")
    src_name = Path(src).name if src else ""
    et = str(issue.get("error_type") or "")
    rows: List[int] = []
    for r in issue.get("affected_rows", []) or []:
        ri = _coerce_int(r)
        if ri is not None:
            rows.append(ri)
    if not rows and "row" in issue and issue["row"] is not None:
        ri = _coerce_int(issue["row"])
        if ri is not None:
            rows.append(ri)
    if not rows:
        for r in issue.get("rows", []) or []:
            ri = _coerce_int(r)
            if ri is not None:
                rows.append(ri)
    if not rows:
        rows = [-1]
    cols: List[str] = []
    if issue.get("column"):
        cols.append(str(issue["column"]))
    for c in issue.get("columns", []) or []:
        if c:
            cols.append(str(c))
    if not cols:
        cols = [""]
    keys: List[CellKey] = []
    for r in rows:
        for c in cols:
            keys.append((src_name, r, c, et))
    return keys


def _multitable_violations_to_keys(
    run_dir: Path,
) -> List[Tuple[CellKey, Dict[str, Any]]]:
    """Project ``multi_table_violations.jsonl`` rows onto cell keys.

    The multi-table pipeline writes its findings into
    ``multi_table_violations.jsonl`` (and adjudicated verdicts into
    ``multi_table_verifications.jsonl``) instead of the per-chunk
    ``issues.jsonl`` consumed by the rest of the report. We map each
    violation to ``(sheet, row, column, "inter_table_inconsistency")``
    and pair it with the verifier verdict (when available) so the
    sweep can distinguish raw detections from confirmed ones.
    """
    violations_p = run_dir / "multi_table_violations.jsonl"
    verifications_p = run_dir / "multi_table_verifications.jsonl"
    groups_p = run_dir / "multi_table_groups.jsonl"
    if not violations_p.exists():
        return []
    # Build per-(group_id, sheet) canonical->raw lookup so we can
    # also emit keys under the raw column name when the LLM
    # canonicalises a header (e.g. orders.csv:customer_city is
    # canonicalised to "city" by confirm_union_join.yaml; planted
    # ground-truth keys reference the raw column).
    canonical_to_raw: Dict[Tuple[str, str], Dict[str, str]] = {}
    if groups_p.exists():
        with groups_p.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # The file is one workbook per line, each with a
                # ``groups`` list.
                groups_list = payload.get("groups")
                if not isinstance(groups_list, list):
                    # Fall back to flat per-line group records.
                    groups_list = [payload]
                for g in groups_list:
                    if not isinstance(g, dict):
                        continue
                    gid = str(g.get("group_id") or "")
                    cm = g.get("column_mapping") or {}
                    if not gid or not isinstance(cm, dict):
                        continue
                    for canonical, entries in cm.items():
                        if not isinstance(entries, list):
                            continue
                        for e in entries:
                            if not isinstance(e, dict):
                                continue
                            sh = Path(str(e.get("sheet") or "")).name
                            raw = str(e.get("column") or "")
                            if sh and raw:
                                canonical_to_raw.setdefault(
                                    (gid, sh), {}
                                )[str(canonical)] = raw
    verdicts: Dict[str, Dict[str, Any]] = {}
    if verifications_p.exists():
        with verifications_p.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    v = json.loads(line)
                except json.JSONDecodeError:
                    continue
                vid = v.get("violation_id")
                if isinstance(vid, str):
                    verdicts[vid] = v
    out: List[Tuple[CellKey, Dict[str, Any]]] = []
    with violations_p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                viol = json.loads(line)
            except json.JSONDecodeError:
                continue
            sheet = str(viol.get("sheet") or "")
            sheet_name = Path(sheet).name
            gid = str(viol.get("group_id") or "")
            row_v = _coerce_int(viol.get("row"))
            cols = [str(c) for c in (viol.get("columns") or []) if c]
            if not cols:
                cols = [""]

            def _expand_cols(sh: str, col: str) -> List[str]:
                """Return the canonical-as-emitted name plus the
                raw mapped column on this sheet (when known)."""
                if not col:
                    return [col]
                names: List[str] = [col]
                raw = canonical_to_raw.get((gid, sh), {}).get(col)
                if raw and raw not in names:
                    names.append(raw)
                return names

            for col in cols:
                for proj_col in _expand_cols(sheet_name, col):
                    key: CellKey = (
                        sheet_name,
                        row_v if row_v is not None else -1,
                        proj_col,
                        "inter_table_inconsistency",
                    )
                    out.append((key, {
                        "violation_id": viol.get("violation_id"),
                        "kind": viol.get("kind"),
                        "verdict": (
                            verdicts.get(viol.get("violation_id") or "", {})
                            .get("verdict")
                        ),
                        "confidence": (
                            verdicts.get(viol.get("violation_id") or "", {})
                            .get("confidence")
                        ),
                    }))
            # Also project ``linked`` rows in the same sheet space so
            # a detection that names the partner row credits both.
            for link in viol.get("linked", []) or []:
                if not isinstance(link, dict):
                    continue
                lsheet = str(link.get("sheet") or "")
                lsheet_name = Path(lsheet).name
                lrow = _coerce_int(link.get("row"))
                if lrow is None:
                    continue
                for col in cols:
                    for proj_col in _expand_cols(lsheet_name, col):
                        key = (
                            lsheet_name,
                            lrow,
                            proj_col,
                            "inter_table_inconsistency",
                        )
                        out.append((key, {
                            "violation_id": viol.get("violation_id"),
                            "kind": viol.get("kind"),
                            "linked": True,
                        }))
    return out


def _load_issues(run_dir: Path) -> List[Dict[str, Any]]:
    fp = run_dir / "issues.jsonl"
    if not fp.exists():
        return []
    out: List[Dict[str, Any]] = []
    with fp.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _load_summary(run_dir: Path) -> Dict[str, Any]:
    fp = run_dir / "summary.json"
    if not fp.exists():
        return {}
    try:
        return json.loads(fp.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _write_cells_jsonl(
    cells: List[Dict[str, Any]], path: Path
) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for c in cells:
            fh.write(json.dumps(c, sort_keys=True) + "\n")


def _build_comparison(
    cells: List[Dict[str, Any]],
    datasets: List[Dict[str, Any]],
    efforts: List[str],
) -> Dict[str, Any]:
    """Roll cells up into ``{dataset: {effort: stats}}`` for the report.

    For each cell we record:

    * ``count`` - number of issue keys emitted
    * ``unique_keys`` - count of distinct ``(row, column, error_type)``
    * ``recall`` - fraction of expected keys also detected (when
      ``expected`` is provided)
    * ``hits``, ``misses`` - per-key breakdown for the markdown table
    """
    by_cell: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for c in cells:
        by_cell[(c["dataset"], c["effort"])] = c
    rollup: Dict[str, Any] = {"datasets": {}, "efforts": efforts}
    for ds in datasets:
        ds_name = ds["name"]
        ds_block: Dict[str, Any] = {"by_effort": {}}
        expected = [
            (s, int(r), col, et) for (s, r, col, et) in (ds.get("expected") or [])
        ]
        ds_block["expected"] = [
            {"source": s, "row": r, "column": col, "error_type": et}
            for (s, r, col, et) in expected
        ]
        for ef in efforts:
            cell = by_cell.get((ds_name, ef))
            if cell is None:
                ds_block["by_effort"][ef] = {
                    "missing": True,
                    "count": 0,
                    "unique_keys": 0,
                    "hits": [],
                    "misses": [],
                    "recall": None,
                }
                continue
            keyset = {
                tuple(k)
                for k in cell.get("keys", [])
            }
            unique = len(keyset)
            hits: List[Dict[str, Any]] = []
            misses: List[Dict[str, Any]] = []
            for (s, r, col, et) in expected:
                if (s, r, col, et) in keyset:
                    hits.append(
                        {"source": s, "row": r, "column": col, "error_type": et}
                    )
                else:
                    misses.append(
                        {"source": s, "row": r, "column": col, "error_type": et}
                    )
            recall: Optional[float] = (
                len(hits) / len(expected) if expected else None
            )
            # Per-error-type counter (raw issues, not unique keys).
            per_type: Counter = Counter()
            for k in cell.get("keys", []):
                per_type[k[3]] += 1
            ds_block["by_effort"][ef] = {
                "count": cell.get("count", 0),
                "unique_keys": unique,
                "hits": hits,
                "misses": misses,
                "recall": recall,
                "per_error_type": dict(per_type),
                "rc": cell.get("rc"),
                "wall_s": cell.get("wall_s"),
                "run_dir": cell.get("run_dir"),
            }
        rollup["datasets"][ds_name] = ds_block
    return rollup


def _write_tsv(rollup: Dict[str, Any], path: Path) -> None:
    efforts: List[str] = list(rollup.get("efforts", []))
    cols = ["dataset", "effort", "expected", "unique_keys", "hits", "misses",
            "recall", "wall_s", "rc"]
    with path.open("w", encoding="utf-8") as fh:
        fh.write("\t".join(cols) + "\n")
        for ds_name, ds in rollup["datasets"].items():
            n_expected = len(ds.get("expected", []))
            for ef in efforts:
                cell = ds["by_effort"].get(ef, {})
                row = [
                    ds_name,
                    ef,
                    str(n_expected),
                    str(cell.get("unique_keys", 0)),
                    str(len(cell.get("hits", []))),
                    str(len(cell.get("misses", []))),
                    "" if cell.get("recall") is None
                    else f"{cell['recall']:.3f}",
                    "" if cell.get("wall_s") is None
                    else f"{cell['wall_s']:.1f}",
                    "" if cell.get("rc") is None else str(cell["rc"]),
                ]
                fh.write("\t".join(row) + "\n")


def _md_recall(cell: Dict[str, Any]) -> str:
    if cell.get("missing"):
        return "n/a"
    r = cell.get("recall")
    if r is None:
        return "n/a"
    return f"{r:.0%} ({len(cell.get('hits', []))}/" \
           f"{len(cell.get('hits', [])) + len(cell.get('misses', []))})"


def _write_markdown(
    rollup: Dict[str, Any], path: Path, *, sweep_meta: Dict[str, Any]
) -> None:
    efforts: List[str] = list(rollup.get("efforts", []))
    lines: List[str] = []
    lines.append("# Reasoning-effort sweep on smoke datasets\n")
    lines.append(f"- model: `{sweep_meta.get('model', 'n/a')}`")
    lines.append("- mode: `single` (no Good-Turing perturbation)")
    lines.append("- synthesis: disabled (`--no-synthesis`)")
    lines.append(f"- efforts: `{', '.join(efforts)}`")
    lines.append(f"- timestamp: `{sweep_meta.get('timestamp', '')}`\n")
    lines.append(
        "Recall is measured against the planted-error list documented "
        "in `experiments/sweep_reasoning_effort.py` (per-cell "
        "`expected` block). Misses at lower effort indicate the LLM "
        "stopped catching that planted error.\n"
    )

    # ---- Headline table ----
    lines.append("## Recall by (dataset x effort)\n")
    header = ["dataset", "expected"]
    for ef in efforts:
        header.append(f"recall@{ef}")
    for ef in efforts:
        header.append(f"raw@{ef}")
    for ef in efforts:
        header.append(f"wall_s@{ef}")
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for ds_name, ds in rollup["datasets"].items():
        n_expected = len(ds.get("expected", []))
        row = [ds_name, str(n_expected)]
        for ef in efforts:
            row.append(_md_recall(ds["by_effort"].get(ef, {})))
        for ef in efforts:
            cell = ds["by_effort"].get(ef, {})
            row.append(str(cell.get("count", 0)))
        for ef in efforts:
            cell = ds["by_effort"].get(ef, {})
            row.append(
                "" if cell.get("wall_s") is None
                else f"{cell['wall_s']:.1f}"
            )
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append(
        "`recall` = fraction of planted errors detected. "
        "`raw` = total emitted detections (issues + multi-table "
        "violations); large swings here at fixed recall are signal "
        "about effort-driven false-positive volume.\n"
    )

    # ---- Per-dataset miss tables ----
    for ds_name, ds in rollup["datasets"].items():
        if not ds.get("expected"):
            continue
        lines.append(f"## {ds_name}\n")
        lines.append("Planted errors (one row per expected hit). A "
                     "checkmark means the LLM emitted at least one "
                     "issue with that exact `(source, row, column, "
                     "error_type)` key.\n")
        header = ["row", "column", "error_type"] + [f"@{ef}" for ef in efforts]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join(["---"] * len(header)) + "|")
        for exp in ds["expected"]:
            row_md = [str(exp["row"]), exp.get("column") or "*", exp["error_type"]]
            for ef in efforts:
                cell = ds["by_effort"].get(ef, {})
                hits_set = {
                    (h["source"], h["row"], h["column"], h["error_type"])
                    for h in cell.get("hits", [])
                }
                key = (exp["source"], exp["row"],
                       exp["column"], exp["error_type"])
                row_md.append("YES" if key in hits_set else "miss")
            lines.append("| " + " | ".join(row_md) + " |")
        # ---- Per-error-type emitted counts ----
        all_types = sorted({
            t
            for ef in efforts
            for t in (ds["by_effort"].get(ef, {}).get("per_error_type") or {})
        })
        if all_types:
            lines.append("\nEmitted issue counts (raw, not unique):\n")
            tbl_header = ["error_type"] + efforts
            lines.append("| " + " | ".join(tbl_header) + " |")
            lines.append("|" + "|".join(["---"] * len(tbl_header)) + "|")
            for et in all_types:
                row = [et]
                for ef in efforts:
                    cnt = ds["by_effort"].get(ef, {}).get(
                        "per_error_type", {}
                    ).get(et, 0)
                    row.append(str(cnt))
                lines.append("| " + " | ".join(row) + " |")
        # ---- Wall-clock per cell ----
        lines.append("\nWall-clock seconds per cell:\n")
        wc_header = efforts
        lines.append("| " + " | ".join(wc_header) + " |")
        lines.append("|" + "|".join(["---"] * len(wc_header)) + "|")
        wc_row: List[str] = []
        for ef in efforts:
            cell = ds["by_effort"].get(ef, {})
            wc_row.append(
                "" if cell.get("wall_s") is None else f"{cell['wall_s']:.1f}"
            )
        lines.append("| " + " | ".join(wc_row) + " |")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--efforts",
        default="low,medium,high",
        help="Comma-separated list of reasoning_effort values to sweep.",
    )
    p.add_argument(
        "--datasets",
        default=None,
        help=(
            "Comma-separated subset of dataset names "
            f"({','.join(d['name'] for d in DATASETS)}). "
            "Default: all five."
        ),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs",
        help="Parent dir for the sweep root (default: outputs/).",
    )
    p.add_argument(
        "--sweep-name",
        default=None,
        help=(
            "Override the sweep root name. Default: "
            "reasoning_effort_sweep_<UTC ts>."
        ),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Run with --validate-config (no LLM calls). Useful to "
            "smoke-test the orchestrator + dataset definitions."
        ),
    )
    p.add_argument(
        "--rebuild-only",
        action="store_true",
        help=(
            "Skip detection runs and only re-aggregate existing "
            "per-cell run dirs into comparison.{tsv,md,json}. "
            "Useful when iterating on the report layout."
        ),
    )
    p.add_argument(
        "--keep-going",
        action="store_true",
        default=True,
        help=(
            "Continue the sweep on a per-cell failure (default). "
            "Failures are recorded in summary.json with rc != 0."
        ),
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    args = parse_args(argv)
    efforts = [e.strip() for e in args.efforts.split(",") if e.strip()]
    if args.datasets:
        wanted = {d.strip() for d in args.datasets.split(",") if d.strip()}
        datasets = [d for d in DATASETS if d["name"] in wanted]
        if not datasets:
            raise SystemExit(
                f"--datasets={args.datasets!r} matched none of: "
                f"{[d['name'] for d in DATASETS]}"
            )
    else:
        datasets = list(DATASETS)
    sweep_root_name = (
        args.sweep_name or f"reasoning_effort_sweep_{_utc_stamp()}"
    )
    sweep_root: Path = (
        args.output_dir if args.output_dir.is_absolute()
        else PROJECT_ROOT / args.output_dir
    ) / sweep_root_name
    runs_root = sweep_root / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    logs_root = sweep_root / "logs"
    logs_root.mkdir(parents=True, exist_ok=True)
    logger.info("sweep root: %s", sweep_root)

    # Pre-materialise per-effort multi-table prompt copies.
    mt_prompt_index: Dict[str, Dict[str, Path]] = {}
    if any(d.get("uses_multi_table") for d in datasets):
        for ef in efforts:
            mt_prompt_index[ef] = _materialise_multitable_prompts(ef, sweep_root)

    cells: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    for ds in datasets:
        for ef in efforts:
            run_name = f"{ds['name']}__{ef}"
            run_dir = runs_root / run_name
            log_path = logs_root / f"{run_name}.log"
            cmd = _build_invocation(
                ds, ef,
                output_dir=runs_root,
                run_name=run_name,
                multi_table_prompts=mt_prompt_index.get(ef),
            )
            if not args.rebuild_only:
                logger.info(
                    "[run] dataset=%s effort=%s -> %s",
                    ds["name"], ef, run_dir,
                )
                # Wipe any previous incomplete run dir to keep
                # summary.json / issues.jsonl in lockstep with the
                # new invocation.
                if run_dir.exists():
                    shutil.rmtree(run_dir)
                rc, wall = _run_cell(cmd, log_path, dry_run=args.dry_run)
                # Persist sweep-cell metadata next to the run dir so
                # ``--rebuild-only`` can recover wall_s without
                # re-running anything.
                if not args.dry_run and run_dir.exists():
                    try:
                        (run_dir / "_sweep_cell.json").write_text(
                            json.dumps(
                                {
                                    "dataset": ds["name"],
                                    "effort": ef,
                                    "rc": rc,
                                    "wall_s": wall,
                                    "cmd": cmd,
                                },
                                indent=2,
                            ),
                            encoding="utf-8",
                        )
                    except OSError:
                        logger.debug("could not write _sweep_cell.json",
                                     exc_info=True)
                if rc != 0:
                    failures.append(
                        {
                            "dataset": ds["name"],
                            "effort": ef,
                            "rc": rc,
                            "log": str(log_path),
                            "cmd": cmd,
                        }
                    )
                    if not args.keep_going:
                        logger.error(
                            "stopping sweep on failure (rc=%d) "
                            "for %s/%s. log=%s",
                            rc, ds["name"], ef, log_path,
                        )
                        break
            else:
                # Recover persisted timing / rc when rebuilding so
                # the markdown still shows wall-clock per cell.
                meta_path = run_dir / "_sweep_cell.json"
                rc = 0
                wall = None
                if meta_path.exists():
                    try:
                        meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError):
                        meta = {}
                    rc = int(meta.get("rc", 0) or 0)
                    wall = meta.get("wall_s")
            issues = (
                [] if args.dry_run else _load_issues(run_dir)
            )
            keys: List[CellKey] = []
            for issue in issues:
                keys.extend(_issue_to_keys(issue))
            mt_violations: List[Dict[str, Any]] = []
            if not args.dry_run and ds.get("uses_multi_table"):
                mt_pairs = _multitable_violations_to_keys(run_dir)
                for k, meta in mt_pairs:
                    keys.append(k)
                    mt_violations.append({"key": list(k), **meta})
            summary = _load_summary(run_dir) if not args.dry_run else {}
            # Total "detection count" includes both per-chunk issues
            # and any multi-table violations so the headline number
            # is comparable across dataset families.
            count = len(issues) + len(mt_violations)
            cells.append(
                {
                    "dataset": ds["name"],
                    "effort": ef,
                    "rc": rc,
                    "wall_s": wall,
                    "run_dir": str(run_dir),
                    "count": count,
                    "keys": [list(k) for k in keys],
                    "summary": summary,
                    "multi_table_violations": mt_violations,
                }
            )
        else:
            continue
        if failures and not args.keep_going:
            break

    # Aggregate.
    cells_path = sweep_root / "cells.jsonl"
    _write_cells_jsonl(cells, cells_path)
    rollup = _build_comparison(cells, datasets, efforts)
    sweep_meta = {
        "timestamp": _utc_stamp(),
        "model": "from config/experiment.yaml",
        "efforts": efforts,
        "datasets": [d["name"] for d in datasets],
        "dry_run": args.dry_run,
        "failures": failures,
    }
    (sweep_root / "summary.json").write_text(
        json.dumps({"meta": sweep_meta, "rollup": rollup}, indent=2),
        encoding="utf-8",
    )
    _write_tsv(rollup, sweep_root / "comparison.tsv")
    _write_markdown(
        rollup, sweep_root / "comparison.md", sweep_meta=sweep_meta
    )
    logger.info("comparison written to %s", sweep_root / "comparison.md")
    if failures:
        logger.warning("sweep had %d failure(s)", len(failures))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
