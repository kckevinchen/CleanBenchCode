"""
Canonicalize ``synthesis.canonical_fix`` on every cluster of a cleaning-
detection run so the labelling UI's "Fix" column shows a literal canonical
cell value instead of a natural-language directive.

Background
----------
The aggregator pipeline (``src.core.aggregator``) writes a ``synthesis``
block per cluster that includes ``canonical_fix`` -- the LLM's reconciled
suggestion. In practice that field is often INSTRUCTION-shaped::

    canonical_fix: 'Change Town Name to "Indian Stream Twp (T1 R6 BKP EKR)"'
    canonical_fix: "Change 'Great britain' to 'Great Britain' in rows 11 and 19."
    canonical_fix: "Change 'Private Sector Total' from '3.9' to '3.90'"

The labelling UI (see ``frontend/labeling/label.js::computeBeforeAfter``)
already renders ``cluster.synthesis.canonical_fix`` verbatim as the
"after" value next to the original cell. The UX is much cleaner when
that field holds the LITERAL post-fix cell value::

    canonical_fix: 'Indian Stream Twp (T1 R6 BKP EKR)'
    canonical_fix: 'Great Britain'
    canonical_fix: '3.90'

This script asks an LLM to rewrite each instruction-shaped fix into the
exact post-fix string, preserving the original directive as
``canonical_fix_raw`` so the change is reversible. Clusters where the
LLM cannot decide unambiguously are LEFT alone (the original instruction
stays in place).

Mirrors the design of ``experiments/canonicalize_typo_benchmark.py``
(same prompt skeleton, same reasoning-effort / model knobs, same dedup
strategy, same atomic-rewrite-with-.bak contract).

Usage
-----
::

    python -m experiments.canonicalize_cluster_fixes \\
        --run-dir outputs/20260508T222447Z_gpt5-family \\
        --model   gpt-5.2 \\
        --concurrency 8

By default this rewrites ``clusters.json`` AND ``clusters.jsonl`` in
place, snapshotting the originals to ``<file>.precanon.bak``.  Use
``--dry-run`` to preview the candidate set without calling the LLM, or
``--out-suffix .canonical`` to write to sibling files instead of in
place.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Make ``src`` importable when run as a script.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.llm_client import LLMClient  # noqa: E402

# Reuse the typo-benchmark heuristic so we have ONE definition of
# "instruction-shaped" across the codebase. The patterns are tuned for
# detector / synthesis prose ("Replace X with Y", "should be ...",
# "possible correction: ..."), which is the same language family that
# appears in cluster.synthesis.canonical_fix.
from experiments.canonicalize_typo_benchmark import (  # noqa: E402
    looks_like_instruction,
    parse_response,
)

logger = logging.getLogger("canonicalize_cluster_fixes")


# ---------------------------------------------------------------------------
# CSV row cache (closes the row_values coverage gap)
# ---------------------------------------------------------------------------
#
# Aggregator clusters frequently omit ``row_values`` -- in this run only
# ~15% of clusters carry it. Without the original cell value the LLM
# correctly bails on directives like "Format to 2 decimals" because
# there's no anchor. We therefore preload each source CSV once and look
# up (row, column) on demand. ``run.summary.config.csv_paths`` lists the
# absolute paths the run was launched against, so we resolve sources
# back to their on-disk file.

_CSV_FRAME_CACHE: Dict[str, Optional[Dict[str, Any]]] = {}


def _build_source_to_path(run_dir: Path) -> Dict[str, Path]:
    """Read ``summary.json::config.csv_paths`` and return
    ``{basename: absolute_path}`` for every source the run touched.
    Returns empty dict if the summary is missing or malformed - the
    caller will then skip CSV enrichment entirely."""
    summary_path = run_dir / "summary.json"
    if not summary_path.is_file():
        return {}
    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    cfg = data.get("config") or {}
    out: Dict[str, Path] = {}
    for p in cfg.get("csv_paths") or []:
        pp = Path(p)
        # Last write wins; basenames are unique in our corpora.
        out[pp.name] = pp
    return out


def _load_csv_frame(path: Path) -> Optional[Dict[str, Any]]:
    """Read ``path`` once and return ``{"headers": [...], "rows": [{col: val}, ...]}``.

    Uses pandas with everything-as-string so leading zeros / decimals
    survive verbatim. Returns None on read errors (caller falls back to
    the row_values-only path).
    """
    cached = _CSV_FRAME_CACHE.get(str(path))
    if cached is not None or str(path) in _CSV_FRAME_CACHE:
        return cached
    try:
        import pandas as pd  # local import keeps the CLI fast when CSVs aren't needed
    except ImportError:
        logger.warning("pandas not available; skipping CSV enrichment")
        _CSV_FRAME_CACHE[str(path)] = None
        return None
    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[])
    except Exception as exc:  # noqa: BLE001
        logger.warning("failed to read %s: %s", path, exc)
        _CSV_FRAME_CACHE[str(path)] = None
        return None
    headers = [str(c) for c in df.columns]
    rows = df.to_dict(orient="records")
    frame = {"headers": headers, "rows": rows}
    _CSV_FRAME_CACHE[str(path)] = frame
    return frame


def _lookup_cell(
    csv_paths: Dict[str, Path],
    source: Optional[str],
    row: Any,
    column: str,
) -> Optional[str]:
    """Look up the cell value at (source, row, column) by reading the
    source CSV. ``row`` is a 0-based data-row index (skipping the
    header), matching the aggregator's row convention. Returns None when
    the CSV isn't available, the row is out of range, or the column is
    not in the header."""
    if not source or not column:
        return None
    try:
        row_idx = int(row)
    except (TypeError, ValueError):
        return None
    path = csv_paths.get(source)
    if path is None:
        return None
    frame = _load_csv_frame(path)
    if frame is None:
        return None
    rows = frame["rows"]
    if row_idx < 0 or row_idx >= len(rows):
        return None
    rec = rows[row_idx]
    if column not in rec:
        return None
    return str(rec[column])


def _lookup_row(
    csv_paths: Dict[str, Path],
    source: Optional[str],
    row: Any,
) -> Dict[str, Any]:
    """Same as :func:`_lookup_cell` but returns the whole row dict so
    the prompt can include sibling-cell context. Empty dict on miss."""
    if not source:
        return {}
    try:
        row_idx = int(row)
    except (TypeError, ValueError):
        return {}
    path = csv_paths.get(source)
    if path is None:
        return {}
    frame = _load_csv_frame(path)
    if frame is None:
        return {}
    rows = frame["rows"]
    if row_idx < 0 or row_idx >= len(rows):
        return {}
    return dict(rows[row_idx])


# ---------------------------------------------------------------------------
# Candidate selection
# ---------------------------------------------------------------------------

def needs_canonicalization(cluster: Dict[str, Any]) -> bool:
    """A cluster is a canonicalization candidate when its
    ``synthesis.canonical_fix`` is missing, the literal string ``"null"``,
    blank, or instruction-shaped.

    Likely-false-positive clusters are deliberately INCLUDED -- the
    operator may still want to see the literal value the model proposed,
    and the LLM is allowed to return null when there is no defensible
    fix. The CLI's ``--skip-fps`` flag flips this off when batch cost
    is a concern.
    """
    syn = cluster.get("synthesis") or {}
    if not isinstance(syn, dict):
        return True
    cf = syn.get("canonical_fix")
    if cf is None:
        return True
    if not isinstance(cf, str):
        return True
    s = cf.strip()
    if not s or s.lower() == "null":
        return True
    return looks_like_instruction(s)


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You normalize a data-quality detector's reconciled 'canonical_fix' "
    "into the exact post-fix CELL VALUE for a single cell. The detector "
    "flagged a single cell in a table and proposed a fix; that fix is "
    "sometimes a natural-language directive (e.g., \"Change 'X' to 'Y'\", "
    "\"Replace ... with ...\", \"Format to N decimals\") rather than the "
    "literal new value. Your job is to output the exact string the cell "
    "would contain after applying the directive to its current value. "
    "Preserve case, punctuation, spacing, accents, abbreviations and "
    "number formatting exactly as required by the directive.\n\n"
    "Two rules that override conservative behaviour:\n"
    "  1. If the directive names a SINGLE explicitly quoted candidate "
    "value (patterns like 'should be \"X\"', 'change to \"X\"', "
    "'(e.g., \"X\" if applicable)') use that quoted value. Hedging "
    "words like 'if applicable', 'possibly', 'e.g.' do NOT make a "
    "single-candidate instruction ambiguous.\n"
    "  2. If the directive describes a deterministic transformation "
    "(\"format to two decimal places\", \"trim whitespace\", \"convert "
    "to title case\", \"normalize legal suffix\", \"strip parenthetical "
    "qualifier\") AND the original cell value is provided, APPLY that "
    "transformation to the original_value and output the result. The "
    "fact that the directive talks about 'all values in the column' "
    "does not make it ambiguous for a single known cell.\n\n"
    "Return null ONLY when the directive names no candidate, names "
    "multiple incompatible candidates, or genuinely requires external "
    "knowledge you don't have (e.g. 'pick the right city for this "
    "address' with no source of truth). NEVER invent a value. Output "
    "ONLY a JSON object."
)

# We build a verbose prompt so the LLM has every signal the cluster
# carries (original cell value + directives + observed text +
# explanation). The original_cell_value field is the most important
# one when it is non-null because applying the directive to it usually
# determines the answer uniquely.
_USER_TEMPLATE = """Source: {source_json}
Column: {column_json}
Row: {row_json}

Original cell value (verbatim, may be null if not captured):
{original_value_json}

Other cells in the same row (verbatim, may be empty):
{row_values_json}

Reconciled canonical fix (the directive to apply):
{canonical_fix_json}

Synthesis explanation (background, can be empty):
{canonical_explanation_json}

Detector observed text (background, often quotes the original value):
{observed_json}

Detector suggested_fix (alternative directive, may be empty):
{suggested_fix_json}

Output JSON exactly in this shape, nothing else:
{{"canonical_corrected_value": "<the cell value after applying the directive, or null if undetermined>"}}
"""


def _select_original_value(
    cluster: Dict[str, Any],
    column: str,
    csv_paths: Optional[Dict[str, Path]] = None,
) -> Optional[str]:
    """Best-effort lookup of the literal pre-fix cell value.

    Order of preference:
      1. ``cluster.row_values[column]`` -- the aggregator populates this
         when the chunk has enough context.
      2. ``cluster.detections[*].full_row_values[column]`` -- some
         detector arms emit a full per-row dict.
      3. ``cluster.detections[*].row_values[column]`` -- older variant.
      4. The on-disk CSV at (source, row, column) -- only consulted
         when ``csv_paths`` is provided. This closes the gap for the
         ~85% of clusters that omit ``row_values``.
    Returns ``None`` when no source carries the value (the LLM can still
    work from the directive text in that case).
    """
    rv = cluster.get("row_values")
    if isinstance(rv, dict) and column in rv:
        v = rv[column]
        if v is not None:
            return str(v)
    for det in cluster.get("detections") or []:
        for key in ("full_row_values", "row_values"):
            d = det.get(key) if isinstance(det, dict) else None
            if isinstance(d, dict) and column in d and d[column] is not None:
                return str(d[column])
    if csv_paths is not None:
        v = _lookup_cell(csv_paths, cluster.get("source"), cluster.get("row"), column)
        if v is not None:
            return v
    return None


def _select_row_values(
    cluster: Dict[str, Any],
    csv_paths: Optional[Dict[str, Path]] = None,
) -> Dict[str, Any]:
    """Return a non-empty row_values dict if any source carries one;
    otherwise consult the on-disk CSV via ``csv_paths``. Empty dict on
    miss. Used as supporting context for the LLM."""
    rv = cluster.get("row_values")
    if isinstance(rv, dict) and rv:
        return rv
    for det in cluster.get("detections") or []:
        if not isinstance(det, dict):
            continue
        for key in ("full_row_values", "row_values"):
            d = det.get(key)
            if isinstance(d, dict) and d:
                return d
    if csv_paths is not None:
        d = _lookup_row(csv_paths, cluster.get("source"), cluster.get("row"))
        if d:
            return d
    return {}


def _select_observed(cluster: Dict[str, Any]) -> str:
    """Pick a representative ``observed`` string from the cluster's
    detections (the first non-empty one). The detector arms usually
    quote the offending value(s) here, which is invaluable when
    ``row_values`` is missing."""
    for det in cluster.get("detections") or []:
        if not isinstance(det, dict):
            continue
        ob = det.get("observed")
        if isinstance(ob, str) and ob.strip():
            return ob
    return ""


def _select_suggested_fix(cluster: Dict[str, Any]) -> str:
    """Pick the first non-empty per-detection ``suggested_fix``. Often
    contains the literal target value quoted inside an instruction."""
    for det in cluster.get("detections") or []:
        if not isinstance(det, dict):
            continue
        sf = det.get("suggested_fix")
        if isinstance(sf, str) and sf.strip():
            return sf
    return ""


def build_prompt(
    cluster: Dict[str, Any],
    *,
    csv_paths: Optional[Dict[str, Path]] = None,
) -> str:
    cols = cluster.get("columns") or []
    column = cols[0] if cols else ""
    syn = cluster.get("synthesis") or {}
    return _USER_TEMPLATE.format(
        source_json=json.dumps(cluster.get("source") or ""),
        column_json=json.dumps(column),
        row_json=json.dumps(cluster.get("row")),
        original_value_json=json.dumps(
            _select_original_value(cluster, column, csv_paths=csv_paths),
        ),
        row_values_json=json.dumps(
            _select_row_values(cluster, csv_paths=csv_paths),
            ensure_ascii=False,
        ),
        canonical_fix_json=json.dumps(syn.get("canonical_fix") or ""),
        canonical_explanation_json=json.dumps(syn.get("canonical_explanation") or ""),
        observed_json=json.dumps(_select_observed(cluster)),
        suggested_fix_json=json.dumps(_select_suggested_fix(cluster)),
    )


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

@dataclass
class CanonicalizationResult:
    canonical: Optional[str]
    raw_response: str
    error: Optional[str]


def canonicalize_one(
    llm: LLMClient,
    cluster: Dict[str, Any],
    *,
    model: str,
    reasoning_effort: str,
    csv_paths: Optional[Dict[str, Path]] = None,
) -> CanonicalizationResult:
    try:
        resp = llm.query(
            model,
            build_prompt(cluster, csv_paths=csv_paths),
            reasoning_effort=reasoning_effort,
            reasoning_summary="auto",
            system_prompt=_SYSTEM_PROMPT,
        )
    except Exception as exc:  # noqa: BLE001
        return CanonicalizationResult(None, "", f"call_failed: {exc}")
    canonical = parse_response(resp.text or "")
    return CanonicalizationResult(canonical, resp.text or "", None)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def _dedupe_key(
    cluster: Dict[str, Any],
    csv_paths: Optional[Dict[str, Path]] = None,
) -> Tuple:
    """Two clusters share an LLM call iff they touch the same column on
    the same source with the same original cell value AND the same fix
    directives. This collapses e.g. the 60+ identical "Reg" -> "Regional"
    clusters down to one call. Whole-row context isn't part of the key
    because the directive + original_value already determine the
    answer uniquely for single-column clusters."""
    cols = cluster.get("columns") or []
    column = cols[0] if cols else ""
    syn = cluster.get("synthesis") or {}
    return (
        cluster.get("source") or "",
        column,
        _select_original_value(cluster, column, csv_paths=csv_paths) or "",
        (syn.get("canonical_fix") or "").strip(),
        _select_suggested_fix(cluster).strip(),
    )


def _load_clusters_json(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(f"{path} does not contain a JSON list")
    return data


def _atomic_write_json(path: Path, data: Any) -> None:
    """Write JSON to ``path`` via .tmp + os.replace, snapshotting the
    prior file to ``<path>.precanon.bak`` first. Matches the contract
    used by ``/api/external-compare/benchmark/update`` and friends."""
    bak = path.with_suffix(path.suffix + ".precanon.bak")
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        if path.exists():
            bak.write_bytes(path.read_bytes())
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def _atomic_write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    bak = path.with_suffix(path.suffix + ".precanon.bak")
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        if path.exists():
            bak.write_bytes(path.read_bytes())
        with tmp.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False))
                fh.write("\n")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def run(
    *,
    run_dir: Path,
    model: str,
    reasoning_effort: str,
    concurrency: int,
    limit: Optional[int],
    skip_fps: bool,
    in_place: bool,
    out_suffix: str,
    report_path: Optional[Path],
    dry_run: bool,
    enrich_from_csv: bool = True,
) -> dict:
    json_path = run_dir / "clusters.json"
    jsonl_path = run_dir / "clusters.jsonl"
    if not json_path.is_file():
        raise FileNotFoundError(f"clusters.json not found in {run_dir}")

    clusters = _load_clusters_json(json_path)
    logger.info("Loaded %d clusters from %s", len(clusters), json_path)

    # Build the source-to-CSV map once. When the run's CSVs are no
    # longer on disk (e.g. the dataset moved) this is empty and the
    # script silently falls back to row_values-only lookup.
    csv_paths: Dict[str, Path] = {}
    if enrich_from_csv:
        csv_paths = _build_source_to_path(run_dir)
        if csv_paths:
            missing = sum(1 for p in csv_paths.values() if not p.is_file())
            logger.info("CSV enrichment: %d source paths registered (%d missing on disk)",
                        len(csv_paths), missing)
        else:
            logger.info("CSV enrichment: no csv_paths in summary.json (disabled)")

    # Filter candidates.
    candidates: List[Tuple[int, Dict[str, Any]]] = []
    for i, c in enumerate(clusters):
        if skip_fps:
            syn = c.get("synthesis") or {}
            if isinstance(syn, dict) and syn.get("is_likely_false_positive"):
                continue
        if needs_canonicalization(c):
            candidates.append((i, c))
    if limit is not None:
        candidates = candidates[:limit]
    logger.info("Candidates for canonicalization: %d / %d", len(candidates), len(clusters))

    if dry_run:
        for i, c in candidates[:50]:
            syn = c.get("synthesis") or {}
            cols = c.get("columns") or []
            col = cols[0] if cols else ""
            print(json.dumps({
                "index": i,
                "source": c.get("source"),
                "row": c.get("row"),
                "column": col,
                "current_canonical_fix": syn.get("canonical_fix"),
                "original_value": _select_original_value(c, col, csv_paths=csv_paths),
            }, ensure_ascii=False))
        return {"candidates": len(candidates), "unique_calls": 0, "updated": 0,
                "left_unchanged": len(candidates), "errors": 0}

    # Dedupe.
    seen: Dict[Tuple, List[int]] = {}
    reps: Dict[Tuple, Dict[str, Any]] = {}
    for i, c in candidates:
        k = _dedupe_key(c, csv_paths=csv_paths)
        seen.setdefault(k, []).append(i)
        reps.setdefault(k, c)
    unique_keys = list(seen)
    logger.info("Unique LLM tasks after dedup: %d (saved %d redundant calls)",
                len(unique_keys), len(candidates) - len(unique_keys))

    llm = LLMClient()

    # Fire LLM calls in parallel; collect results.
    results: Dict[Tuple, CanonicalizationResult] = {}
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        future_to_key = {
            ex.submit(
                canonicalize_one, llm, reps[k],
                model=model, reasoning_effort=reasoning_effort,
                csv_paths=csv_paths,
            ): k for k in unique_keys
        }
        for n, fut in enumerate(as_completed(future_to_key), 1):
            k = future_to_key[fut]
            try:
                results[k] = fut.result()
            except Exception as exc:  # noqa: BLE001
                results[k] = CanonicalizationResult(None, "", f"future_failed: {exc}")
            if n % 25 == 0 or n == len(future_to_key):
                logger.info("  ... %d / %d unique tasks done", n, len(future_to_key))

    # Apply results.
    report_rows: List[Dict[str, Any]] = []
    updated = unchanged = errors_n = 0
    for k, res in results.items():
        for idx in seen[k]:
            c = clusters[idx]
            # Ensure synthesis is a writable dict attached to the cluster.
            # Three corner cases to handle: missing key, explicit None, and
            # the (rare) non-dict value. The previous version used
            # ``setdefault`` inside a ternary, which silently dropped the
            # writeback when ``synthesis`` was ``None`` or absent -- those
            # rows would log as "updated" in the report but the new fix
            # never landed in the cluster dict.
            cur_syn = c.get("synthesis")
            if isinstance(cur_syn, dict):
                syn = cur_syn
            else:
                syn = {}
                if cur_syn is not None:
                    syn["canonical_fix_raw_synthesis"] = cur_syn
                c["synthesis"] = syn
            prior = syn.get("canonical_fix")
            entry = {
                "index": idx,
                "source": c.get("source"),
                "row": c.get("row"),
                "column": (c.get("columns") or [None])[0],
                "previous_canonical_fix": prior,
                "new_canonical_fix": res.canonical,
                "error": res.error,
                "raw_response": (res.raw_response or "")[:500],
            }
            if res.error:
                errors_n += 1
                entry["status"] = "error"
            elif res.canonical is None:
                unchanged += 1
                entry["status"] = "left_unchanged"
            else:
                syn["canonical_fix_raw"] = prior
                syn["canonical_fix"] = res.canonical
                syn["canonical_fix_canonicalized"] = True
                syn["canonical_fix_canonicalized_by"] = model
                updated += 1
                entry["status"] = "updated"
            report_rows.append(entry)

    # Persist. Default: rewrite clusters.json and clusters.jsonl in
    # place with .precanon.bak snapshots. Override path with --out-suffix
    # writes sibling files (e.g., clusters.canonical.json) instead.
    if in_place:
        out_json = json_path
        out_jsonl = jsonl_path
    else:
        out_json = json_path.with_name(f"{json_path.stem}{out_suffix}{json_path.suffix}")
        out_jsonl = jsonl_path.with_name(f"{jsonl_path.stem}{out_suffix}{jsonl_path.suffix}")

    _atomic_write_json(out_json, clusters)
    logger.info("Wrote %s", out_json)

    # clusters.jsonl is the row-oriented mirror; keep it in sync. The
    # labelling app only reads clusters.json, but the relations / report
    # builders read the .jsonl, so a divergence here would be a
    # silent data hazard.
    if jsonl_path.exists():
        _atomic_write_jsonl(out_jsonl, clusters)
        logger.info("Wrote %s", out_jsonl)
    else:
        logger.info("No clusters.jsonl to mirror (skipped)")

    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as fh:
            for r in report_rows:
                fh.write(json.dumps(r, ensure_ascii=False))
                fh.write("\n")
        logger.info("Wrote change report to %s", report_path)

    return {
        "candidates": len(candidates),
        "unique_calls": len(unique_keys),
        "updated": updated,
        "left_unchanged": unchanged,
        "errors": errors_n,
        "out_json": str(out_json),
        "out_jsonl": str(out_jsonl) if jsonl_path.exists() else None,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--run-dir", required=True, type=Path,
                   help="Path to a cleaning-detection run dir (contains clusters.json).")
    p.add_argument("--model", default="gpt-5.2",
                   help="Azure OpenAI deployment (default: gpt-5.2). "
                        "Comma-separated list rotates round-robin per call.")
    p.add_argument("--reasoning-effort", default="low",
                   choices=["none", "low", "medium", "high", "xhigh"])
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--limit", type=int, default=None,
                   help="Process only the first N candidate clusters (smoke test).")
    p.add_argument("--skip-fps", action="store_true",
                   help="Skip clusters flagged as is_likely_false_positive.")
    p.add_argument("--out-suffix", default=".canonical",
                   help="When NOT writing in place, suffix to insert before .json/.jsonl "
                        "(default: .canonical -> clusters.canonical.json). "
                        "Ignored when --in-place is set.")
    p.add_argument("--in-place", action="store_true",
                   help="Overwrite clusters.json / clusters.jsonl in the run "
                        "dir, snapshotting originals to <file>.precanon.bak. "
                        "REQUIRED for the labelling UI to pick up the new fixes.")
    p.add_argument("--report", type=Path, default=None,
                   help="Optional JSONL log of every per-cluster decision "
                        "(default: <run-dir>/canonicalize_fixes.report.jsonl).")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the first 50 candidates without calling the LLM.")
    p.add_argument("--no-csv-enrich", action="store_true",
                   help="Disable on-disk CSV lookup of original cell values. "
                        "By default the script reads each source CSV listed in "
                        "summary.json::config.csv_paths once and uses it to "
                        "supply original_value/row_values when the cluster "
                        "doesn't carry them.")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    report_path = args.report
    if report_path is None and not args.dry_run:
        report_path = args.run_dir / "canonicalize_fixes.report.jsonl"

    stats = run(
        run_dir=args.run_dir,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        concurrency=args.concurrency,
        limit=args.limit,
        skip_fps=args.skip_fps,
        in_place=args.in_place,
        out_suffix=args.out_suffix,
        report_path=report_path,
        dry_run=args.dry_run,
        enrich_from_csv=not args.no_csv_enrich,
    )
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
