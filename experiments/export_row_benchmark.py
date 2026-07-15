"""Per-ROW benchmark export with LLM-determined fix + explanation.

Builds the per-row portion of the cross-type benchmark -- the error
types whose unit of meaning is a ROW (a relationship across columns or
across rows)::

    fd_violation, row_calculation_inconsistency,
    string_calculation_inconsistency, row_duplication

It also appends ``inter_table_inconsistency`` -- a MULTI-table type
materialised from multi-table runs (``INTER_TABLE_REGISTRY``) rather than
the per-row REGISTRY pools -- as one record per involved cell (both sides
of each cross-sheet pair), with a precomputed bi-directional fix +
explanation (no LLM).

Unlike the per-cell exporter, the "error" here is a violated relation /
calculation / duplicate, which fans across SEVERAL columns of a row and
is rarely labeled with a concrete fix. So this exporter:

  * collapses the fanned per-column cells back into ONE finding per
    ``(source, anchor_row, error_type)``;
  * recovers the rule + rationale + participating rows + involved
    columns from the run's cluster detections;
  * asks GPT-5.5, given the rule + a compact table slice (involved
    columns x flagged/peer rows), to pick WHICH cell is wrong, supply
    the corrected value, and explain it in one sentence. An
    operator-labeled fix, when present, overrides the model's target +
    value (the explanation is still the model's). ``row_duplication`` is
    structural: ``corrected_value="drop_row"``, ``target_column=null``.

Outputs (default ``benchmark/merged_v2/``)::

    benchmark_row.json / .csv        per-row ground truth + fix + explanation
    llm_row_cache.jsonl              cached LLM fix+explanation (resumable)

Run::

    python -m experiments.export_row_benchmark                # all pools
    python -m experiments.export_row_benchmark --limit 8      # sample
    python -m experiments.export_row_benchmark --no-llm       # structure only
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(it, **_kw):  # type: ignore
        return it

from src.core.llm_client import LLMClient
from src.labeling.app import RunContext

from experiments.export_merged_benchmark import (
    DATASET_OF,
    LABEL_RANK,
    OUTPUTS,
    PROJECT_ROOT,
    REGISTRY,
    _materialize_run,
    build_exclusion_union,
    row_finding_is_excluded,
)

PER_ROW_TYPES: Set[str] = {
    "fd_violation",
    "row_calculation_inconsistency",
    "string_calculation_inconsistency",
    "row_duplication",
}
ROW_DUP = "row_duplication"

# Inter-table inconsistency is a MULTI-table type (the same entity disagrees
# across sheets of a workbook). It is produced by multi-table runs whose
# artifact shape differs from the per-row REGISTRY runs -- it is curated in the
# multi-table labeling UI and persisted as ``multi_table_labels.jsonl`` +
# ``multi_table_violations.jsonl`` -- so it is materialised here separately and
# appended to ``benchmark_row.json`` (one record per involved cell, both sides
# of each cross-sheet pair so a detector that flags either sheet is credited).
INTER_TABLE_TYPE = "inter_table_inconsistency"

# A multi-table label ``status`` -> benchmark verdict. ``not_a_violation`` and
# anything else are dropped (clean); only confirmed / unsure cross the bar.
_MT_VERDICT = {"confirmed_inconsistency": "error", "unsure": "ambiguous"}

# Which multi-table runs feed inter_table_inconsistency, and which workbooks to
# include (``workbooks=None`` => every workbook with a confirmed/unsure label).
# Each source CSV (a single sheet) resolves uniquely inside ``source_root`` and
# emits the ``dataset`` label below. These 15 inter-table sheets are folded into
# the ``bing21_500`` corpus (identified by their normal filename, e.g.
# ``32c26f90.csv``) rather than carrying a separate ``mt100`` dataset tag -- the
# same sheets' single-column/relation detections are merged into the excel_500
# (=bing21_500) runs, so every record for a given sheet lands in ONE dataset.
# ``source_root`` still points at the ``curated_bing21_mt100_dedup`` data
# directory (an on-disk location, not a dataset label). Source basenames are
# unique across the two roots (verified: no collision with the 502 bing21_500
# sheets).
INTER_TABLE_REGISTRY: List[Dict[str, Any]] = [
    {
        "dataset": "bing21_500",
        "run": "20260622T184510Z_gpt5-family",
        "source_root": PROJECT_ROOT / "data" / "curated_bing21_mt100_dedup",
        # ``None`` => every workbook with a confirmed/unsure label in the run.
        # As of merged_v6 this is 4 workbooks (adds b49031f16271… with 5
        # confirmed and 44b1a96909… with 2 unsure to the original two), i.e.
        # 12 records (10 error + 2 ambiguous) across 8 sheets.
        "workbooks": None,
    },
]

GroupKey = Tuple[str, Optional[int], str]  # (source, anchor_row, error_type)

EXPLAIN_MODEL = "gpt-5.5"
EXPLAIN_PROMPT_VERSION = 1
EXPLAIN_SYSTEM = (
    "You are a meticulous data-quality reviewer examining a ROW-LEVEL error in a "
    "spreadsheet: a violated relationship between columns (functional dependency), "
    "an inconsistent row calculation, or a duplicate row. You are given the rule "
    "that is violated, its rationale, and a slice of the table (the involved "
    "columns for the flagged rows -- marked with * -- plus peer rows for context). "
    "Decide the single best fix and explain it.\n\n"
    "Reply with a JSON object ONLY (no markdown, no prose around it):\n"
    '{"target_row": <int row index whose cell is wrong, or null>,\n'
    ' "target_column": <name of the column whose value is wrong, or null if the '
    'fix is to drop a duplicate row>,\n'
    ' "corrected_value": <the correct value for that cell as a string, or '
    '"drop_row" to delete a duplicate row>,\n'
    ' "explanation": <ONE concise sentence: what is wrong and why this fix is '
    'right, referencing the rule/pattern>}'
)


# ---------------------------------------------------------------------
# Rule / rationale / table-slice helpers
# ---------------------------------------------------------------------

def _rule_and_rationale(det: Dict[str, Any]) -> Tuple[str, str]:
    observed = (det.get("observed") or "").strip()
    rule = observed
    for pre in ("Row violates relation:", "Row violates calculation:",
                "Row violates:", "Violates relation:"):
        if rule.startswith(pre):
            rule = rule[len(pre):].strip()
            break
    expl = (det.get("explanation") or "")
    if "Rationale:" in expl:
        rationale = expl.split("Rationale:", 1)[1]
    else:
        rationale = expl
    rationale = rationale.split("\nDetection:", 1)[0].strip()
    return rule, rationale[:500]


def _table_slice(df: Any, columns: List[str], rows: List[int],
                 cap_rows: int = 12, cap_cell: int = 60) -> str:
    if df is None:
        return "(table unavailable)"
    cols = [c for c in columns if c in getattr(df, "columns", [])]
    if not cols:
        cols = list(df.columns)[:8]
    flagged = [r for r in rows if isinstance(r, int) and 0 <= r < len(df)]
    show = list(dict.fromkeys(flagged))
    for r in range(len(df)):
        if len(show) >= cap_rows:
            break
        if r not in show:
            show.append(r)
    out = ["row | " + " | ".join(cols)]
    for r in show[:cap_rows]:
        vals = []
        for c in cols:
            v = str(df.iat[r, df.columns.get_loc(c)])
            vals.append(v if len(v) <= cap_cell else v[:cap_cell - 1] + "…")
        mark = " *" if r in flagged else ""
        out.append(f"{r}{mark} | " + " | ".join(vals))
    return "\n".join(out)


def _cell_value(df: Any, row: Optional[int], column: Optional[str]) -> Optional[str]:
    if df is None or row is None or not column or column not in getattr(df, "columns", []):
        return None
    if not (0 <= row < len(df)):
        return None
    return str(df.iat[row, df.columns.get_loc(column)])


def _parse_json_obj(text: str) -> Optional[Dict[str, Any]]:
    t = (text or "").strip()
    m = re.search(r"```(?:json)?\s*(.+?)```", t, re.S)
    if m:
        t = m.group(1).strip()
    if not t.startswith("{"):
        i, j = t.find("{"), t.rfind("}")
        if i >= 0 and j > i:
            t = t[i:j + 1]
    try:
        o = json.loads(t)
        return o if isinstance(o, dict) else None
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------
# Merge: one finding per (source, anchor_row, error_type)
# ---------------------------------------------------------------------

def _merge_findings(pools: List[str]) -> Tuple[Dict[GroupKey, Dict[str, Any]], Dict[str, Any]]:
    findings: Dict[GroupKey, Dict[str, Any]] = {}
    frames: Dict[str, Any] = {}
    for pool in pools:
        for run_label, _types, run_id in REGISTRY[pool]:
            run_dir = OUTPUTS / run_id
            if not run_dir.is_dir():
                continue
            cells, _s, _e = _materialize_run(run_dir)
            if not any(set(c.error_types or []) & PER_ROW_TYPES for c in cells):
                continue
            ctx = RunContext(run_dir)
            cl_by_id = {c.get("cluster_id"): c for c in ctx.clusters}
            for c in cells:
                etypes = [t for t in (c.error_types or []) if t in PER_ROW_TYPES]
                if not etypes or c.label not in ("error", "ambiguous"):
                    continue
                for et in etypes:
                    key: GroupKey = (c.source, c.row, et)
                    g = findings.setdefault(key, {
                        "source": c.source, "anchor_row": c.row, "error_type": et,
                        "status": "ambiguous", "columns": [], "rows": set(),
                        "rule": "", "rationale": "", "labeled": {},
                        "llm_suggested": [], "datasets": set(),
                    })
                    g["datasets"].add(DATASET_OF[pool])
                    if LABEL_RANK.get(c.label, 0) > LABEL_RANK.get(g["status"], 0):
                        g["status"] = c.label
                    if c.column and c.column not in g["columns"]:
                        g["columns"].append(c.column)
                    if c.corrected_value not in (None, "", []) and c.column:
                        g["labeled"][c.column] = c.corrected_value
                    if c.llm_suggested_value:
                        g["llm_suggested"].append(c.llm_suggested_value)
                    # Resolve rule/rationale/rows/columns from the cluster once.
                    if not g["rule"]:
                        for cid in (c.cluster_ids or []):
                            cl = cl_by_id.get(cid)
                            if not cl:
                                continue
                            dets = cl.get("detections") or []
                            det = next((d for d in dets if d.get("error_type") == et), dets[0] if dets else None)
                            if not det:
                                continue
                            g["rule"], g["rationale"] = _rule_and_rationale(det)
                            for r in det.get("affected_rows") or []:
                                if isinstance(r, int):
                                    g["rows"].add(r)
                            for col in (cl.get("columns") or det.get("columns") or []):
                                if col and col not in g["columns"]:
                                    g["columns"].append(col)
                            break
                if c.source not in frames:
                    try:
                        frames[c.source] = ctx.load_frame(c.source)
                    except Exception:  # noqa: BLE001
                        frames[c.source] = None
    # finalise rows: anchor first, then the rest sorted
    for g in findings.values():
        rows = sorted(g["rows"])
        if isinstance(g["anchor_row"], int) and g["anchor_row"] not in rows:
            rows = [g["anchor_row"]] + rows
        elif isinstance(g["anchor_row"], int):
            rows = [g["anchor_row"]] + [r for r in rows if r != g["anchor_row"]]
        g["rows"] = rows or ([g["anchor_row"]] if g["anchor_row"] is not None else [])
    return findings, frames


def _build_records(findings: Dict[GroupKey, Dict[str, Any]]) -> List[Dict[str, Any]]:
    # The materialiser fans a row-level finding to EVERY affected row, so the
    # same violation shows up once per anchor row (e.g. fd rows [2,10] and
    # [10,2]). Dedup by violation identity = (source, error_type, rows-set,
    # rule), merging columns / labeled fixes / status across the anchors.
    by_id: Dict[Tuple[str, str, Tuple[int, ...], str], Dict[str, Any]] = {}
    for g in findings.values():
        rows_key = tuple(sorted(r for r in g["rows"] if isinstance(r, int)))
        vid = (g["source"], g["error_type"], rows_key, g["rule"])
        m = by_id.get(vid)
        if m is None:
            by_id[vid] = {
                "source": g["source"], "error_type": g["error_type"],
                "status": g["status"], "rows": list(rows_key) or list(g["rows"]),
                "columns": list(g["columns"]), "rule": g["rule"],
                "rationale": g["rationale"], "labeled": dict(g["labeled"]),
                "datasets": set(g["datasets"]),
            }
        else:
            if LABEL_RANK.get(g["status"], 0) > LABEL_RANK.get(m["status"], 0):
                m["status"] = g["status"]
            for col in g["columns"]:
                if col not in m["columns"]:
                    m["columns"].append(col)
            m["labeled"].update(g["labeled"])
            m["datasets"] |= g["datasets"]

    records: List[Dict[str, Any]] = []
    for g in by_id.values():
        datasets = sorted(g["datasets"])
        records.append({
            "source": g["source"],
            "error_type": g["error_type"],
            "status": g["status"],
            "rows": g["rows"],
            "columns": g["columns"],
            "rule": g["rule"],
            "rationale": g["rationale"],
            "target_row": None,
            "target_column": None,
            "original_value": None,
            "corrected_value": None,
            "explanation": None,
            "dataset": datasets[0] if len(datasets) == 1 else "multi",
            "_labeled": g["labeled"],  # stripped before write
        })
    return records


# ---------------------------------------------------------------------
# LLM fix + explanation (error findings only)
# ---------------------------------------------------------------------

def _cache_key(rec: Dict[str, Any]) -> str:
    blob = json.dumps({
        "v": EXPLAIN_PROMPT_VERSION, "model": EXPLAIN_MODEL,
        "source": rec["source"], "error_type": rec["error_type"],
        "rows": rec["rows"], "columns": sorted(rec["columns"]),
        "rule": rec["rule"],
    }, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _load_cache(path: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return out
    for line in path.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            o = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(o.get("key"), str) and isinstance(o.get("result"), dict):
            out[o["key"]] = o["result"]
    return out


def _apply_result(rec: Dict[str, Any], res: Dict[str, Any], frames: Dict[str, Any]) -> None:
    labeled = rec.get("_labeled") or {}
    if rec["error_type"] == ROW_DUP:
        rec["target_column"] = None
        rec["target_row"] = None
        rec["corrected_value"] = "drop_row"
    elif labeled:
        # operator-labeled fix wins for target + value
        col = next(iter(labeled))
        rec["target_column"] = col
        rec["corrected_value"] = labeled[col]
        tr = res.get("target_row")
        rec["target_row"] = tr if isinstance(tr, int) else rec["rows"][0] if rec["rows"] else None
    else:
        col = res.get("target_column")
        rec["target_column"] = col if (col in rec["columns"]) else None
        tr = res.get("target_row")
        rec["target_row"] = tr if isinstance(tr, int) else (rec["rows"][0] if rec["rows"] else None)
        cv = res.get("corrected_value")
        rec["corrected_value"] = cv if isinstance(cv, (str, list)) and cv != "" else None
    rec["explanation"] = (res.get("explanation") or "").strip() or None
    rec["original_value"] = _cell_value(frames.get(rec["source"]), rec["target_row"], rec["target_column"])


def _generate(records: List[Dict[str, Any]], frames: Dict[str, Any], cache_path: Path,
              *, limit: int, concurrency: int, use_llm: bool) -> int:
    cache = _load_cache(cache_path)
    todo: List[Dict[str, Any]] = []
    for rec in records:
        if rec["status"] != "error":
            continue
        ck = _cache_key(rec)
        rec["_ck"] = ck
        if ck in cache:
            _apply_result(rec, cache[ck], frames)
        elif use_llm:
            todo.append(rec)
    if limit and len(todo) > limit:
        todo = todo[:limit]
    if not todo or not use_llm:
        return 0
    llm = LLMClient()

    def _gen(rec: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        labeled = rec.get("_labeled") or {}
        prompt = (
            f"Error type: {rec['error_type']}\n"
            f"Rule violated: {rec['rule']}\n"
            f"Rationale: {rec['rationale']}\n"
            f"Involved columns: {rec['columns']}\n"
            f"Flagged rows (marked * below): {rec['rows']}\n"
            + (f"Operator-suggested fix: {labeled}\n" if labeled else "")
            + f"\nTable slice:\n{_table_slice(frames.get(rec['source']), rec['columns'], rec['rows'])}\n\n"
            "Return the JSON object."
        )
        try:
            resp = llm.query(EXPLAIN_MODEL, prompt, reasoning_effort="low",
                             reasoning_summary="auto", system_prompt=EXPLAIN_SYSTEM)
            return rec, _parse_json_obj(resp.text or "")
        except Exception as exc:  # noqa: BLE001
            print(f"  row-explain failed {rec['source']} {rec['error_type']} {rec['rows']}: {exc}")
            return rec, None

    n_new = 0
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=concurrency) as ex, \
            cache_path.open("a", encoding="utf-8") as fh:
        futures = [ex.submit(_gen, rec) for rec in todo]
        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc="explaining-rows", unit="finding", file=sys.stdout):
            rec, res = fut.result()
            if not res:
                continue
            _apply_result(rec, res, frames)
            fh.write(json.dumps({"key": rec["_ck"], "source": rec["source"],
                                 "error_type": rec["error_type"], "result": res},
                                ensure_ascii=False) + "\n")
            fh.flush()
            n_new += 1
    return n_new


_CSV_FIELDS = [
    "source", "error_type", "status", "rows", "columns", "rule", "rationale",
    "target_row", "target_column", "original_value", "corrected_value",
    "explanation", "dataset",
    # inter_table_inconsistency linkage (empty for the single-source row types).
    "workbook", "linked_source", "linked_row",
]


def _csv_val(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, list):
        return ";".join(str(x) for x in v) if all(isinstance(x, (int, str)) for x in v) else json.dumps(v, ensure_ascii=False)
    return str(v)


def _write_csv(path: Path, records: List[Dict[str, Any]]) -> None:
    tmp = path.with_suffix(".csv.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(_CSV_FIELDS)
        for r in records:
            w.writerow([_csv_val(r.get(k)) for k in _CSV_FIELDS])
    tmp.replace(path)


# ---------------------------------------------------------------------
# Inter-table inconsistency (multi-table runs -> benchmark_row records)
# ---------------------------------------------------------------------

def _read_mt_cell(root: Path, workbook: str, sheet: str, row: int,
                  column: str, _cache: Dict[Path, Any]) -> Optional[str]:
    """Live CSV value at ``(sheet, row, column)`` so ``original_value`` matches
    the source exactly (benchmark integrity). ``None`` if it cannot be resolved."""
    import pandas as pd

    if sheet is None or column is None or row is None:
        return None
    path = root / workbook / sheet
    if not path.exists():
        return None
    df = _cache.get(path)
    if df is None:
        try:
            df = pd.read_csv(path, dtype=str, keep_default_na=False)
        except Exception:  # noqa: BLE001
            return None
        _cache[path] = df
    if column not in df.columns or not (0 <= row < len(df)):
        return None
    return str(df.iloc[row][column])


def _materialize_inter_table() -> List[Dict[str, Any]]:
    """Build ``inter_table_inconsistency`` records from the multi-table runs in
    ``INTER_TABLE_REGISTRY``.

    A multi-table violation links the SAME entity across two sheets of a
    workbook; a reviewer confirms it in the multi-table UI
    (``multi_table_labels.jsonl``). We keep every ``confirmed_inconsistency``
    (-> ``error``) / ``unsure`` (-> ``ambiguous``) violation for the configured
    workbooks, recover the involved cells from ``multi_table_violations.jsonl``,
    and emit ONE record per involved cell (both sides of the pair). Distinct
    violation_ids that point at the SAME cell-set (e.g. the same pair flagged as
    both ``attribute_conflict`` and ``surface_drift``) are merged. The fix is the
    BI-DIRECTIONAL acceptable set: the sorted distinct surface forms across the
    sheets (``list`` when they differ), mirroring the per-cell inconsistency
    convention; ``None`` for ambiguous findings.
    """
    out: List[Dict[str, Any]] = []
    frame_cache: Dict[Path, Any] = {}
    for spec in INTER_TABLE_REGISTRY:
        run_dir = OUTPUTS / spec["run"]
        labels_p = run_dir / "multi_table_labels.jsonl"
        viol_p = run_dir / "multi_table_violations.jsonl"
        if not labels_p.exists() or not viol_p.exists():
            print(f"  WARN inter-table run missing artifacts: {spec['run']}")
            continue
        dataset = spec["dataset"]
        root = Path(spec["source_root"])
        wb_filter = spec.get("workbooks")

        # violation_id -> best verdict (error > ambiguous) among included workbooks
        verdict: Dict[str, str] = {}
        for ln in labels_p.open(encoding="utf-8"):
            ln = ln.strip()
            if not ln:
                continue
            o = json.loads(ln)
            if wb_filter is not None and o.get("workbook") not in wb_filter:
                continue
            v = _MT_VERDICT.get(o.get("status"))
            vid = o.get("violation_id")
            if v is None or vid is None:
                continue
            if LABEL_RANK.get(v, 0) > LABEL_RANK.get(verdict.get(vid, ""), 0):
                verdict[vid] = v

        # violation_id -> its sides (one per sheet) + metadata
        sides: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
        meta: Dict[str, Dict[str, Any]] = {}
        for ln in viol_p.open(encoding="utf-8"):
            ln = ln.strip()
            if not ln:
                continue
            o = json.loads(ln)
            vid = o.get("violation_id")
            if vid not in verdict:
                continue
            cols = o.get("columns") or []
            sides[vid].append({"sheet": o.get("sheet"), "row": o.get("row"),
                               "column": cols[0] if cols else None})
            m = meta.setdefault(vid, {"workbook": o.get("workbook"),
                                      "rationale": "", "kinds": set()})
            if o.get("kind"):
                m["kinds"].add(o["kind"])
            if not m["rationale"] and o.get("rationale"):
                m["rationale"] = str(o["rationale"])[:500]

        # Merge violation_ids that point at the SAME cell-set into one finding.
        findings: Dict[frozenset, Dict[str, Any]] = {}
        for vid, raw_cells in sides.items():
            wb = meta[vid]["workbook"]
            cells = []
            ok = True
            for c in raw_cells:
                val = _read_mt_cell(root, wb, c["sheet"], c["row"], c["column"],
                                    frame_cache)
                if val is None:
                    ok = False
                    break
                cells.append({**c, "value": val})
            if not ok or len(cells) < 2:
                continue
            ident = frozenset((c["sheet"], c["row"], c["column"]) for c in cells)
            f = findings.get(ident)
            if f is None:
                findings[ident] = {
                    "cells": cells, "workbook": wb, "verdict": verdict[vid],
                    "rationale": meta[vid]["rationale"],
                    "kinds": set(meta[vid]["kinds"]),
                }
            else:
                if LABEL_RANK.get(verdict[vid], 0) > LABEL_RANK.get(f["verdict"], 0):
                    f["verdict"] = verdict[vid]
                f["kinds"] |= meta[vid]["kinds"]
                if not f["rationale"]:
                    f["rationale"] = meta[vid]["rationale"]

        # Emit one record per involved cell.
        for f in findings.values():
            cells = f["cells"]
            v = f["verdict"]
            acceptable = sorted({c["value"] for c in cells})
            corrected = acceptable if (v == "error" and len(acceptable) > 1) else None
            kinds = ", ".join(sorted(f["kinds"])) or "cross-sheet disagreement"
            for c in cells:
                others = [o for o in cells if o is not c]
                linked = others[0] if others else None
                sheets = " / ".join(sorted({x["sheet"] for x in cells}))
                expl = (
                    f"The same entity appears across sheets {sheets} in workbook "
                    f"{f['workbook']}, but column '{c['column']}' disagrees "
                    f"({kinds}): "
                    + " vs ".join(repr(x["value"]) for x in cells)
                    + "; either surface form is acceptable, so the cells are an "
                    "inter-table inconsistency."
                )
                out.append({
                    "source": c["sheet"],
                    "error_type": INTER_TABLE_TYPE,
                    "status": v,
                    "rows": [c["row"]],
                    "columns": [c["column"]],
                    "rule": (f"Same entity must agree across the workbook's sheets "
                             f"on '{c['column']}'."),
                    "rationale": f["rationale"],
                    "target_row": c["row"] if v == "error" else None,
                    "target_column": c["column"] if v == "error" else None,
                    "original_value": c["value"],
                    "corrected_value": corrected,
                    "explanation": expl,
                    "dataset": dataset,
                    "workbook": f["workbook"],
                    "linked_source": linked["sheet"] if linked else None,
                    "linked_row": linked["row"] if linked else None,
                })
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pools", nargs="*", default=list(REGISTRY), choices=list(REGISTRY))
    ap.add_argument("--out", default=str(PROJECT_ROOT / "benchmark" / "merged_v2"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--no-llm", action="store_true")
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    findings, frames = _merge_findings(args.pools)
    records = _build_records(findings)

    n_new = _generate(records, frames, out_dir / "llm_row_cache.jsonl",
                      limit=args.limit, concurrency=args.concurrency, use_llm=not args.no_llm)

    for rec in records:
        rec.pop("_labeled", None)
        rec.pop("_ck", None)

    # Inter-table inconsistency findings come from multi-table runs (a separate
    # artifact shape) and carry a precomputed fix + explanation, so they are
    # appended AFTER the per-row LLM pass -- the model never re-explains them.
    inter_table = _materialize_inter_table()
    records.extend(inter_table)

    # Cross-run exclusion immunity: drop any per-row finding whose source file,
    # target column, or involved rows fall in a slice excluded by ANY run in the
    # selected pools (mirrors the per-cell rule so a row finding never survives
    # inside a region a sibling run marked too messy to label).
    excl_union = build_exclusion_union(args.pools)
    n_before = len(records)
    records = [r for r in records if not row_finding_is_excluded(excl_union, r)]
    n_excluded_rows = n_before - len(records)

    records.sort(key=lambda r: (r["source"], r["error_type"],
                                r["rows"][0] if r["rows"] else -1))

    path = out_dir / "benchmark_row.json"
    tmp = path.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)
    _write_csv(out_dir / "benchmark_row.csv", records)

    status_counts = collections.Counter(r["status"] for r in records)
    type_counts = collections.Counter(r["error_type"] for r in records if r["status"] == "error")
    n_err = status_counts.get("error", 0)
    n_expl = sum(1 for r in records if r["status"] == "error" and r["explanation"])
    try:
        shown = out_dir.resolve().relative_to(PROJECT_ROOT)
    except ValueError:
        shown = out_dir
    print(f"\nwrote {shown}/benchmark_row.json + .csv  ({len(records)} findings)")
    print(f"  cross-run exclusions dropped: {n_excluded_rows} finding(s)")
    print(f"  status: {dict(status_counts)}")
    print(f"  error findings by type: {dict(type_counts)}")
    print(f"  fix+explanation: {n_expl}/{n_err} error findings ({n_new} new LLM calls)")
    if inter_table:
        it_status = collections.Counter(r["status"] for r in inter_table)
        it_src = len({r["source"] for r in inter_table})
        print(f"  inter_table_inconsistency: {len(inter_table)} records "
              f"across {it_src} sheets {dict(it_status)}")
    print("\nsample error findings:")
    shown_n = 0
    for r in records:
        if r["status"] != "error" or not r["explanation"]:
            continue
        tgt = f"{r['target_column']}={r['corrected_value']!r}" if r["target_column"] else r["corrected_value"]
        print(f"  [{r['error_type']}] rows {r['rows']} | rule: {r['rule']}")
        print(f"      fix: {tgt}")
        print(f"      {r['explanation']}")
        shown_n += 1
        if shown_n >= 6:
            break


if __name__ == "__main__":
    main()
