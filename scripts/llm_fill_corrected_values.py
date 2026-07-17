"""Best-effort LLM backfill of ``corrected_value`` for merged-benchmark error
cells that have no concrete fix (GPT-5.5).

Some confirmed error cells - mostly ``fd_violation`` and the calculation
inconsistencies - carry no usable corrected value: the labeled fix was prose
advice or nothing at all, and there is no cluster canonical to fall back on.
This script asks GPT-5.5, given the full (small) source table plus the
detected relation/rationale, for the single concrete corrected value of each
such cell. When the model cannot determine a confident single value it
returns ``null`` and the cell stays empty.

Results are cached to ``benchmark/merged_v1/llm_corrected_cache.jsonl`` keyed
by ``(source, row, column)``. The exporter reads that cache as the final
fallback for an empty ``corrected_value`` (see
``experiments.export_merged_benchmark``), so this pass is durable across
re-exports and re-running this script only fills cells not already cached.

Usage::

    az login   # required before any live call
    python scripts/llm_fill_corrected_values.py            # fill all gaps
    python scripts/llm_fill_corrected_values.py --limit 5  # try a few first
    python scripts/llm_fill_corrected_values.py --dry-run  # show prompts only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.export_merged_benchmark import OUTPUTS, REGISTRY
from src.core.llm_client import LLMClient
from src.labeling._helpers import _scalar
from src.verifier.verify import _parse_json_object

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCH = PROJECT_ROOT / "benchmark" / "merged_v1"
MODEL = "gpt-5.5"
CACHE_NAME = "llm_corrected_cache.jsonl"
MAX_TABLE_ROWS = 120  # tables in scope are <30 rows; cap is a safety net.

CellKey = Tuple[str, int, str]

SYSTEM_PROMPT = (
    "You are a meticulous data-cleaning expert. You are given a CSV table and "
    "ONE specific cell that has been confirmed to contain a data-quality error "
    "(e.g. a functional-dependency violation or a calculation inconsistency). "
    "Using the rest of the table and the stated relation/rationale, determine "
    "the single concrete corrected value for that cell. Prefer the value that "
    "makes the row consistent with the rest of the table. If there is no clear, "
    "confident single correct value, return null - do NOT guess. Respond with "
    "STRICT JSON only, no prose, no code fences."
)


def _registry_runs() -> List[str]:
    seen: List[str] = []
    for pool in REGISTRY.values():
        for _label, _types, run_id in pool:
            if run_id not in seen:
                seen.append(run_id)
    return seen


def _build_source_index() -> Tuple[Dict[str, Path], Dict[str, List[dict]]]:
    """Return (source -> csv path, source -> [clusters])."""
    csv_of: Dict[str, Path] = {}
    clusters_of: Dict[str, List[dict]] = {}
    for run_id in _registry_runs():
        run_dir = OUTPUTS / run_id
        summary = run_dir / "summary.json"
        if summary.exists():
            try:
                cfg = json.loads(summary.read_text()).get("config", {})
            except json.JSONDecodeError:
                cfg = {}
            for p in cfg.get("csv_paths") or []:
                csv_of.setdefault(os.path.basename(p), Path(p))
        cl = run_dir / "clusters.jsonl"
        if cl.exists():
            for line in cl.open():
                line = line.strip()
                if not line:
                    continue
                try:
                    c = json.loads(line)
                except json.JSONDecodeError:
                    continue
                src = c.get("source")
                if src:
                    clusters_of.setdefault(src, []).append(c)
    return csv_of, clusters_of


def _relation_context(clusters: List[dict], row: int, column: str) -> List[str]:
    """Detection explanations for clusters covering (row, column)."""
    out: List[str] = []
    for c in clusters:
        if c.get("row") != row:
            continue
        cols = c.get("columns") or []
        if column and cols and column not in cols:
            continue
        syn = c.get("synthesis") or {}
        for fld in ("canonical_explanation", "root_cause", "canonical_fix"):
            v = syn.get(fld)
            if isinstance(v, str) and v.strip():
                out.append(v.strip())
        for det in c.get("detections") or []:
            ex = det.get("explanation")
            if isinstance(ex, str) and ex.strip():
                out.append(ex.strip())
    # de-dup, keep order, cap length
    seen: set = set()
    uniq: List[str] = []
    for s in out:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq[:6]


def _table_as_text(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    head = "idx," + ",".join(json.dumps(c, ensure_ascii=False) for c in cols)
    lines = [head]
    for i in range(min(len(df), MAX_TABLE_ROWS)):
        cells = [str(i)]
        for c in cols:
            cells.append(json.dumps(_scalar(df.iat[i, df.columns.get_loc(c)]), ensure_ascii=False))
        lines.append(",".join(cells))
    if len(df) > MAX_TABLE_ROWS:
        lines.append(f"... ({len(df) - MAX_TABLE_ROWS} more rows omitted)")
    return "\n".join(lines)


def _build_prompt(src: str, df: pd.DataFrame, row: int, column: str,
                  original: str, error_types: List[str], rels: List[str]) -> str:
    rel_block = "\n".join(f"  - {r}" for r in rels) if rels else "  (none provided)"
    return (
        f"TABLE (CSV, source={src}; first column 'idx' is the 0-based row index):\n"
        f"{_table_as_text(df)}\n\n"
        f"TARGET CELL:\n"
        f"  row index (0-based): {row}\n"
        f"  column: {json.dumps(column, ensure_ascii=False)}\n"
        f"  current value: {json.dumps(original, ensure_ascii=False)}\n"
        f"  confirmed error type(s): {', '.join(error_types) or '(unspecified)'}\n"
        f"  detected relation / rationale:\n{rel_block}\n\n"
        f"TASK: Give the corrected value for the TARGET cell so the row becomes "
        f"consistent with the rest of the table and satisfies the stated "
        f"relation/constraint. If you cannot determine a single confident "
        f"correct value, set corrected_value to null.\n\n"
        f'Respond with STRICT JSON only:\n'
        f'{{"corrected_value": <string or null>, "confident": <true|false>, '
        f'"reason": "<one short sentence>"}}'
    )


def _load_cache(path: Path) -> Dict[CellKey, dict]:
    out: Dict[CellKey, dict] = {}
    if not path.exists():
        return out
    for line in path.open():
        line = line.strip()
        if not line:
            continue
        try:
            o = json.loads(line)
        except json.JSONDecodeError:
            continue
        out[(o["source"], int(o["row"]), o["column"])] = o
    return out


def _append_cache(path: Path, rec: dict) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bench-dir", default=str(DEFAULT_BENCH))
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--reasoning-effort", default="medium")
    ap.add_argument("--limit", type=int, default=None, help="cap cells this run")
    ap.add_argument("--dry-run", action="store_true", help="print prompts, no LLM")
    args = ap.parse_args()

    bench_dir = Path(args.bench_dir)
    cells_path = bench_dir / "benchmark.json"
    cache_path = bench_dir / CACHE_NAME

    rows = [json.loads(l) for l in cells_path.open() if l.strip()]
    missing = [
        r for r in rows
        if r.get("status") == "error" and r.get("corrected_value") in (None, "", [])
    ]
    cache = _load_cache(cache_path)
    todo = [r for r in missing if (r["source"], int(r["row"]), r["column"]) not in cache]
    if args.limit is not None:
        todo = todo[: args.limit]

    print(f"error cells missing corrected_value: {len(missing)}")
    print(f"already cached: {len(missing) - len([r for r in missing if (r['source'], int(r['row']), r['column']) not in cache])}")
    print(f"to process this run: {len(todo)}")
    if not todo:
        return

    csv_of, clusters_of = _build_source_index()
    client = None if args.dry_run else LLMClient()
    frames: Dict[str, pd.DataFrame] = {}

    filled = 0
    no_fix = 0
    failed = 0
    for i, r in enumerate(todo, 1):
        src, row, col = r["source"], int(r["row"]), r["column"]
        original = r.get("original_value", "")
        ets = r.get("error_types") or []
        path = csv_of.get(src)
        if path is None or not path.exists():
            print(f"  [{i}/{len(todo)}] {src} r{row} c={col!r}: CSV not resolved, skip")
            failed += 1
            continue
        if src not in frames:
            frames[src] = pd.read_csv(path, dtype=str, keep_default_na=False)
        df = frames[src]
        rels = _relation_context(clusters_of.get(src, []), row, col)
        prompt = _build_prompt(src, df, row, col, original, ets, rels)

        if args.dry_run:
            print(f"\n===== [{i}/{len(todo)}] {src} r{row} c={col!r} =====")
            print(prompt[:1600])
            continue

        try:
            resp = client.query(
                args.model, prompt,
                reasoning_effort=args.reasoning_effort,
                system_prompt=SYSTEM_PROMPT,
            )
        except Exception as exc:  # noqa: BLE001 - best effort; never crash the batch.
            print(f"  [{i}/{len(todo)}] {src} r{row} c={col!r}: LLM error {exc}")
            failed += 1
            continue

        payload, err = _parse_json_object(resp.text)
        if payload is None:
            print(f"  [{i}/{len(todo)}] {src} r{row} c={col!r}: parse fail ({err})")
            failed += 1
            continue

        cv = payload.get("corrected_value")
        confident = bool(payload.get("confident"))
        cv = cv if (isinstance(cv, str) and cv.strip() and confident) else None
        rec = {
            "source": src, "row": row, "column": col,
            "original_value": original,
            "corrected_value": cv,
            "confident": confident,
            "reason": str(payload.get("reason", ""))[:300],
            "error_types": ets,
            "model": resp.model or args.model,
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        _append_cache(cache_path, rec)
        cache[(src, row, col)] = rec
        if cv:
            filled += 1
            tag = "FILLED"
        else:
            no_fix += 1
            tag = "no-fix"
        print(f"  [{i}/{len(todo)}] {src} r{row} c={col!r}: {tag} "
              f"-> {json.dumps(cv, ensure_ascii=False)}")

    print(f"\ndone: filled={filled} no_fix={no_fix} failed={failed} "
          f"cache={cache_path.relative_to(PROJECT_ROOT)}")
    print("Re-run `python -m experiments.export_merged_benchmark` to apply.")


if __name__ == "__main__":
    main()
