"""Minimal multi-table curation: only filter on column verbosity + English.

Unlike ``curate_pbi`` (which enforces row/column windows, empty-cell
ratios, table-count windows, etc.), this script applies just two
conditions per CSV:

1. **Drop wide-text columns.** For each column, compute the average
   character count across non-empty cells. If it exceeds
   ``--max-avg-col-chars`` (default 50) the column is dropped from the
   output.
2. **Reject non-English CSVs.** After step 1, the file is rejected
   outright if any cell contains a non-Latin letter (CJK, Cyrillic,
   Arabic, ...) OR if ``langdetect`` classifies the remaining text as
   non-English with probability ``>= --langdetect-min-prob``.

Spreadsheets where every CSV is rejected (or every CSV ends up empty
after the column drop) are skipped entirely. Surviving CSVs are
written under ``<output-dir>/<workbook>/<csv>`` with a sibling
``manifest.json`` recording per-table verdicts.

Usage::

    python -m src.data_prep.curate_pbi_min \\
        --source-dir data/bing21_grouped \\
        --output-dir data/bing21_grouped_min \\
        --max-avg-col-chars 50 \\
        --langdetect-min-prob 0.5
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .curate import (
    _cell_is_empty,
    _is_non_latin_letter,
    _langdetect_english,
    _scan as _scan_with_layout,
)


def _scan(source_dir):
    """Backwards-compat shim: returns just the {workbook: [paths]} dict."""
    groups, _layout = _scan_with_layout(source_dir)
    return groups

logger = logging.getLogger(__name__)


def _avg_col_chars(col: pd.Series) -> float:
    """Average character count across non-empty cells of ``col``."""
    s = col.astype(str)
    s = s[s.str.strip().ne("")]
    if s.empty:
        return 0.0
    return float(s.str.len().mean())


def _has_non_latin_cell(df: pd.DataFrame) -> Tuple[bool, str]:
    """Return ``(found, example)``. Scans headers + every cell."""
    for col in df.columns:
        for ch in str(col):
            if _is_non_latin_letter(ch):
                return True, ch
    for col in df.columns:
        for value in df[col].astype(str):
            for ch in value:
                if _is_non_latin_letter(ch):
                    return True, ch
    return False, ""


def _empty_cell_ratio(df: pd.DataFrame) -> float:
    """Fraction of cells that are empty/missing-sentinel."""
    total = int(df.shape[0]) * int(df.shape[1])
    if total == 0:
        return 0.0
    n_empty = 0
    for col in df.columns:
        for v in df[col].tolist():
            if _cell_is_empty(v):
                n_empty += 1
    return n_empty / total


def _process_csv(
    path: Path,
    *,
    max_avg_col_chars: float,
    langdetect_min_prob: float,
    max_empty_cell_ratio: float,
    max_rows: Optional[int],
    seed: int,
) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    """Apply the two filters to ``path``.

    Returns ``(df_or_None, detail)``. ``df_or_None`` is the cleaned
    DataFrame on acceptance, ``None`` on rejection. ``detail`` always
    carries the verdict and any diagnostics for the manifest.
    """
    detail: Dict[str, Any] = {"path": str(path)}
    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
    except Exception as exc:  # noqa: BLE001
        detail["status"] = "rejected"
        detail["reason"] = f"unreadable_csv: {exc}"
        return None, detail

    detail["orig_rows"] = int(df.shape[0])
    detail["orig_columns"] = int(df.shape[1])

    # Step 1: drop wide-text columns by average character count.
    avgs = {col: _avg_col_chars(df[col]) for col in df.columns}
    dropped_cols = [c for c, a in avgs.items() if a > max_avg_col_chars]
    detail["dropped_columns"] = [
        {"column": str(c), "avg_chars": round(avgs[c], 2)} for c in dropped_cols
    ]
    if dropped_cols:
        df = df.drop(columns=dropped_cols)

    if df.shape[1] == 0:
        detail["status"] = "rejected"
        detail["reason"] = "all_columns_dropped_as_wide_text"
        return None, detail

    # Step 1.5: cheap pre-check. Reject obviously sparse CSVs before
    # paying for subsampling / langdetect. The authoritative check is
    # repeated on the final (post-subsample) frame below.
    empty_ratio_pre = _empty_cell_ratio(df)
    if empty_ratio_pre > max_empty_cell_ratio:
        detail["empty_cell_ratio"] = round(empty_ratio_pre, 3)
        detail["status"] = "rejected"
        detail["reason"] = (
            f"empty_cell_ratio={empty_ratio_pre:.2f} (> {max_empty_cell_ratio})"
        )
        return None, detail

    # Step 1.75: cap row count by random subsample. Deterministic per
    # (seed, path). pandas .sample with a fixed random_state is stable.
    if max_rows is not None and df.shape[0] > max_rows:
        # Stable across processes: Python's built-in hash() is salted by
        # PYTHONHASHSEED, which would make subsamples non-reproducible
        # across runs. Use a SHA-256 prefix instead so the manifest's
        # recorded `subsample_seed` actually reproduces the row pick.
        path_digest = hashlib.sha256(str(path).encode("utf-8")).digest()[:4]
        path_int = int.from_bytes(path_digest, "big")
        sub_seed = (seed ^ path_int) & 0x7FFFFFFF
        df = df.sample(n=max_rows, random_state=sub_seed).sort_index().reset_index(drop=True)
        detail["subsampled"] = True
        detail["subsample_seed"] = int(sub_seed)
    else:
        detail["subsampled"] = False

    # Step 1.8: re-check empty ratio on the FINAL (possibly
    # subsampled) frame -- this is what we'll write to disk.
    empty_ratio = _empty_cell_ratio(df)
    detail["empty_cell_ratio"] = round(empty_ratio, 3)
    if empty_ratio > max_empty_cell_ratio:
        detail["status"] = "rejected"
        detail["reason"] = (
            f"empty_cell_ratio_after_subsample={empty_ratio:.2f} "
            f"(> {max_empty_cell_ratio})"
        )
        return None, detail

    # Step 2a: any non-Latin letter -> reject.
    found, example = _has_non_latin_cell(df)
    if found:
        detail["status"] = "rejected"
        detail["reason"] = f"non_latin_char={example!r}"
        return None, detail

    # Step 2b: langdetect on the surviving content.
    is_english, lang = _langdetect_english(
        df, min_english_prob=langdetect_min_prob,
    )
    detail["language"] = {
        "top": lang.get("language"),
        "english_probability": round(float(lang.get("english_probability", 0.0)), 3),
        "sample_chars": lang.get("sample_chars"),
        "reason": lang.get("reason"),
    }
    if not is_english:
        detail["status"] = "rejected"
        detail["reason"] = (
            f"langdetect_top={lang.get('language')!r} "
            f"english_prob={lang.get('english_probability', 0.0):.2f} "
            f"(< {langdetect_min_prob})"
        )
        return None, detail

    detail["status"] = "kept"
    detail["final_rows"] = int(df.shape[0])
    detail["final_columns"] = int(df.shape[1])
    return df, detail


def curate(
    source_dir: Path,
    output_dir: Path,
    *,
    max_avg_col_chars: float = 50.0,
    langdetect_min_prob: float = 0.5,
    max_empty_cell_ratio: float = 0.3,
    max_rows: Optional[int] = 100,
    num_spreadsheets: Optional[int] = None,
    num_csvs: Optional[int] = None,
    max_tables_per_workbook: Optional[int] = None,
    seed: int = 42,
    dedupe_schema_within_workbook: bool = True,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    groups = _scan(source_dir)
    logger.info("scanned %d workbook(s) in %s", len(groups), source_dir)

    # Pre-filter by table count (cheap, no CSV reads). When the caller
    # asks for "<= 4 tables per workbook" we throw out the rest before
    # we sample so the random pick lands on eligible workbooks only.
    sids = sorted(groups)
    if max_tables_per_workbook is not None:
        sids = [s for s in sids if len(groups[s]) <= max_tables_per_workbook]
        logger.info(
            "%d workbook(s) have <= %d table(s)",
            len(sids), max_tables_per_workbook,
        )

    rng = random.Random(seed)
    rng.shuffle(sids)

    selected: List[Dict[str, Any]] = []
    skip_reasons: Dict[str, int] = defaultdict(int)
    n_kept_csvs = 0
    n_rejected_csvs = 0
    n_workbooks_scanned = 0

    for sid in sids:
        # Stop once we've collected enough workbooks (or CSVs).
        # Both targets are TARGETS, not upper bounds on attempts --
        # workbooks/CSVs that are rejected don't count.
        if num_spreadsheets is not None and len(selected) >= num_spreadsheets:
            break
        if num_csvs is not None and n_kept_csvs >= num_csvs:
            break
        n_workbooks_scanned += 1

        paths = sorted(groups[sid])
        kept: List[Tuple[Path, pd.DataFrame, Dict[str, Any]]] = []
        rejected: List[Dict[str, Any]] = []
        seen_schemas: Dict[Tuple[str, ...], str] = {}
        for p in paths:
            if num_csvs is not None and n_kept_csvs >= num_csvs:
                break
            df, detail = _process_csv(
                p,
                max_avg_col_chars=max_avg_col_chars,
                langdetect_min_prob=langdetect_min_prob,
                max_empty_cell_ratio=max_empty_cell_ratio,
                max_rows=max_rows,
                seed=seed,
            )
            if df is None:
                rejected.append(detail)
                bucket = (detail.get("reason") or "unknown").split(" ", 1)[0]
                skip_reasons[bucket] += 1
                n_rejected_csvs += 1
                continue
            # Schema-dedup within the workbook: a CSV that survives the
            # filters but has the SAME (post-clean) column headers as a
            # previously-kept sibling is dropped here. Counted against
            # rejections (so it doesn't consume the --num-csvs budget).
            if dedupe_schema_within_workbook:
                schema_key = tuple(str(c) for c in df.columns)
                prev = seen_schemas.get(schema_key)
                if prev is not None:
                    detail["status"] = "rejected"
                    detail["reason"] = (
                        f"duplicate_schema_within_workbook (matches {prev})"
                    )
                    detail["duplicate_of"] = prev
                    rejected.append(detail)
                    skip_reasons["duplicate_schema_within_workbook"] += 1
                    n_rejected_csvs += 1
                    continue
                seen_schemas[schema_key] = p.name
            kept.append((p, df, detail))
            n_kept_csvs += 1

        if not kept:
            continue

        sub = output_dir / sid
        sub.mkdir(parents=True, exist_ok=True)
        table_meta: List[Dict[str, Any]] = []
        for src, df, detail in kept:
            out_path = sub / src.name
            df.to_csv(out_path, index=False)
            detail["output_path"] = str(out_path.relative_to(output_dir))
            table_meta.append(detail)
        selected.append({
            "spreadsheet_id": sid,
            "n_tables_kept": len(kept),
            "n_tables_rejected": len(rejected),
            "tables_kept": table_meta,
            "tables_rejected": rejected,
        })
        logger.info(
            "  selected %s: kept %d/%d table(s)  [%d/%d]",
            sid, len(kept), len(kept) + len(rejected),
            len(selected), num_spreadsheets or len(sids),
        )

    manifest = {
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "seed": seed,
        "filters": {
            "max_avg_col_chars": max_avg_col_chars,
            "langdetect_min_prob": langdetect_min_prob,
            "max_empty_cell_ratio": max_empty_cell_ratio,
            "max_rows": max_rows,
            "max_tables_per_workbook": max_tables_per_workbook,
            "num_spreadsheets_requested": num_spreadsheets,
            "num_csvs_requested": num_csvs,
            "dedupe_schema_within_workbook": dedupe_schema_within_workbook,
        },
        "n_workbooks_in_source": len(groups),
        "n_workbooks_after_table_count_filter": len(sids),
        "n_workbooks_scanned": n_workbooks_scanned,
        "n_workbooks_kept": len(selected),
        "n_csvs_kept": n_kept_csvs,
        "n_csvs_rejected": n_rejected_csvs,
        "skip_reason_counts": dict(skip_reasons),
        "spreadsheets": selected,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    return manifest


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Minimal curation: drop wide-text columns (avg chars > N) "
            "and reject non-English CSVs."
        )
    )
    p.add_argument("--source-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--max-avg-col-chars", type=float, default=50.0,
        help="Drop a column if its average character count per non-empty "
             "cell exceeds this value. Default: 50.",
    )
    p.add_argument(
        "--langdetect-min-prob", type=float, default=0.5,
        help="Reject the CSV if langdetect's English probability is below "
             "this value. Default: 0.5.",
    )
    p.add_argument(
        "--max-empty-cell-ratio", type=float, default=0.3,
        help="Reject the CSV if the fraction of empty/missing-sentinel "
             "cells (computed after wide-text columns are dropped) "
             "exceeds this value. Default: 0.3.",
    )
    p.add_argument(
        "--max-rows", type=int, default=100,
        help="If a CSV has more than this many rows after column "
             "filtering, randomly subsample down to this count. "
             "Subsampling is deterministic per (--seed, path). "
             "The empty-cell-ratio check is re-applied to the final "
             "subsampled frame. Set to 0 to disable. Default: 100.",
    )
    p.add_argument(
        "--num-spreadsheets", type=int, default=None,
        help="Stop once this many workbooks have been kept. Sampling is "
             "deterministic given --seed. Default: process every workbook.",
    )
    p.add_argument(
        "--num-csvs", type=int, default=None,
        help="Stop once this many CSVs have been kept (across workbooks). "
             "Combined with --num-spreadsheets, whichever is reached first "
             "ends sampling. Default: no CSV cap.",
    )
    p.add_argument(
        "--max-tables-per-workbook", type=int, default=None,
        help="Skip workbooks containing more than this many CSVs. "
             "Applied BEFORE sampling so the random pick lands on "
             "eligible workbooks. Default: no cap.",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="RNG seed for the sampling order. Default: 42.",
    )
    p.add_argument(
        "--dedupe-schema-within-workbook",
        dest="dedupe_schema_within_workbook",
        action="store_true",
        default=True,
        help="Within each workbook, drop CSVs whose post-clean column "
             "headers exactly match an already-kept sibling. Default: on.",
    )
    p.add_argument(
        "--no-dedupe-schema-within-workbook",
        dest="dedupe_schema_within_workbook",
        action="store_false",
        help="Disable per-workbook schema deduplication.",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    args = parse_args(argv)
    if not args.source_dir.is_dir():
        logger.error("source-dir not found: %s", args.source_dir)
        return 2
    manifest = curate(
        args.source_dir, args.output_dir,
        max_avg_col_chars=args.max_avg_col_chars,
        langdetect_min_prob=args.langdetect_min_prob,
        max_empty_cell_ratio=args.max_empty_cell_ratio,
        max_rows=(args.max_rows or None),
        num_spreadsheets=args.num_spreadsheets,
        num_csvs=args.num_csvs,
        max_tables_per_workbook=args.max_tables_per_workbook,
        seed=args.seed,
        dedupe_schema_within_workbook=args.dedupe_schema_within_workbook,
    )
    logger.info(
        "kept %d/%d workbooks (scanned %d), %d CSVs (rejected %d) -> %s",
        manifest["n_workbooks_kept"],
        manifest["n_workbooks_in_source"],
        manifest["n_workbooks_scanned"],
        manifest["n_csvs_kept"],
        manifest["n_csvs_rejected"],
        args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
