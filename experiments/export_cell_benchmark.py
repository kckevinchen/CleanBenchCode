"""Per-CELL benchmark export with LLM-generated error explanations.

Builds the per-cell portion of the cross-type benchmark -- the error
types whose unit of meaning is a single cell::

    typo, inconsistency, numeric_outlier, semantic_incompatible

It reuses the SAME per-run materialisation + merge helpers as
``experiments.export_cleanbench`` (so cells/labels/fixes stay
identical to the live workbench export), keeps only the per-cell error
types, and adds an ``explanation`` field: a one-sentence, GPT-5.5
account of WHAT is wrong and WHY the proposed fix is right, generated
from the column's value distribution + the fix. Explanations are
generated for ERROR cells only (ambiguous cells carry no fix) and
cached to ``llm_explanation_cache.jsonl`` so re-runs / prompt edits are
cheap and resumable.

Outputs (default ``benchmark/CleanBench/``)::

    benchmark_cell.json          per-cell ground truth + ``explanation``
                                 (error + ambiguous; absent cell => clean)
    llm_explanation_cache.jsonl  cached LLM explanations (keyed, resumable)

Run::

    python -m experiments.export_cell_benchmark                  # all pools, all explanations
    python -m experiments.export_cell_benchmark --limit 15       # explain only 15 (sample)
    python -m experiments.export_cell_benchmark --no-llm         # skip the LLM entirely
    python -m experiments.export_cell_benchmark --pools excel_500
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is a dev dep
    def tqdm(it, **_kw):  # type: ignore
        return it

from experiments.export_cleanbench import (
    DATASET_OF,
    LABEL_RANK,
    OUTPUTS,
    PROJECT_ROOT,
    REGISTRY,
    _materialize_run,
    _resolve_concrete,
    build_exclusion_union,
    cell_is_excluded,
)
from src.core.llm_client import LLMClient
from src.labeling._helpers import _letter_core
from src.labeling.app import RunContext

# Error types whose unit of meaning is a single cell (this exporter).
# The remaining types (fd_violation, row/string_calculation_inconsistency,
# row_duplication) are per-ROW and handled by a separate exporter.
PER_CELL_TYPES: Set[str] = {
    "typo",
    "inconsistency",
    "numeric_outlier",
    "semantic_incompatible",
}

CellKey = Tuple[str, Optional[int], str]

# Typo subtype labels. A typo is split by HOW its fix changes the value:
#   * ``typo-formatting`` -- the fix differs only by spacing, punctuation, or
#     casing (the letter/digit core is identical). Reuses ``_letter_core``
#     from ``src.labeling._helpers`` (the same normalisation the low-value-fix
#     detector uses), so the rule matches the rest of the pipeline.
#   * ``typo-spelling``   -- any real letter/digit change (Propety->Property).
# These replace the bare ``typo`` label in the exported ``error_types``.
TYPO_FORMATTING = "typo-formatting"
TYPO_SPELLING = "typo-spelling"


def _typo_subtype(original: str, fix: Any) -> Optional[str]:
    """Classify a typo correction as formatting-only vs spelling.

    Returns ``TYPO_FORMATTING`` when ``original`` and EVERY candidate in
    ``fix`` share the same ``_letter_core`` (differ only by spacing /
    punctuation / casing), ``TYPO_SPELLING`` when any candidate has a real
    letter/digit change, or ``None`` when there is no usable fix string to
    compare (so the caller can leave the plain ``typo`` label in place).
    """
    if not fix:
        return None
    cands = fix if isinstance(fix, list) else [fix]
    cands = [c for c in cands if isinstance(c, str) and c]
    if not cands:
        return None

    def _core(s: str) -> str:
        # _letter_core strips whitespace + punctuation + casefolds, catching
        # spacing/punctuation/casing diffs. But Python's ``\w`` KEEPS underscore
        # (a Pc connector punctuation), so strip it too -- otherwise a
        # snake_case<->spaced/joined diff would read as spelling. (No current
        # benchmark cell exercises this; it keeps the cosmetic rule complete.)
        return _letter_core(s).replace("_", "")

    base = _core(original or "")
    for c in cands:
        if _core(c) != base:
            return TYPO_SPELLING
    return TYPO_FORMATTING

# ---------------------------------------------------------------------
# LLM explanation
# ---------------------------------------------------------------------

EXPLAIN_MODEL = "gpt-5.5"
# Bump to invalidate cached explanations after editing the prompt.
EXPLAIN_PROMPT_VERSION = 1
EXPLAIN_SYSTEM = (
    "You are a meticulous data-quality reviewer. You are shown one column "
    "from a spreadsheet, the distribution of its values, one cell value that "
    "has been flagged as an error, and the proposed correction. In ONE "
    "concise sentence, explain WHAT is wrong with the flagged value and WHY "
    "the correction is appropriate. Reference the column's pattern when it "
    "explains the fix (e.g. 'most values are capitalised', 'the column "
    "stores integers', 'the rest spell out the full name'). Be specific and "
    "factual. Reply with the single sentence only -- no preamble, no "
    "bullet points, no surrounding quotes."
)


def _column_context(df: Any, column: str, cap: int = 40) -> str:
    """Compact ``'value' (xN)`` listing of a column's distinct values."""
    if df is None or column not in getattr(df, "columns", []):
        return ""
    vc = df[column].astype(str).value_counts()
    lines: List[str] = []
    for val, cnt in vc.head(cap).items():
        v = val if len(val) <= 80 else val[:77] + "..."
        lines.append(f"  {v!r} (x{int(cnt)})")
    extra = len(vc) - cap
    if extra > 0:
        lines.append(f"  ... (+{extra} more distinct values)")
    return "\n".join(lines)


def _fix_phrase(corrected: Any) -> str:
    if isinstance(corrected, list):
        return "any of " + ", ".join(repr(x) for x in corrected)
    if isinstance(corrected, str) and corrected:
        return repr(corrected)
    return "(no specific corrected value recorded)"


def _build_prompt(
    error_types: List[str], column: str, col_ctx: str,
    original: str, corrected: Any,
) -> str:
    return (
        f"Error type(s): {', '.join(error_types)}\n"
        f"Column: {column!r}\n"
        f"Distinct values in this column (value x count):\n"
        f"{col_ctx or '  (column values unavailable)'}\n\n"
        f"Flagged cell value: {original!r}\n"
        f"Proposed correction: {_fix_phrase(corrected)}\n\n"
        "One-sentence explanation of what is wrong and why the correction "
        "is right:"
    )


def _cache_key(
    source: str, row: Optional[int], column: str,
    error_types: List[str], original: str, corrected: Any,
) -> str:
    blob = json.dumps(
        {
            "v": EXPLAIN_PROMPT_VERSION,
            "model": EXPLAIN_MODEL,
            "source": source, "row": row, "column": column,
            "error_types": sorted(error_types),
            "original": original, "corrected": corrected,
        },
        ensure_ascii=False, sort_keys=True,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _load_explanation_cache(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
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
        k, e = o.get("key"), o.get("explanation")
        if isinstance(k, str) and isinstance(e, str) and e.strip():
            out[k] = e.strip()
    return out


# ---------------------------------------------------------------------
# Merge (per-cell types only)
# ---------------------------------------------------------------------

def _merge_cells(pools: List[str]) -> Tuple[Dict[CellKey, Dict[str, Any]], Dict[str, Any]]:
    """Accumulate per-cell-type verdicts per (source, row, column) across
    every (pool x type) run, plus a per-source CSV frame cache for the
    explanation context."""
    merged: Dict[CellKey, Dict[str, Any]] = {}
    frames: Dict[str, Any] = {}
    for pool in pools:
        for run_label, run_types, run_id in REGISTRY[pool]:
            run_dir = OUTPUTS / run_id
            if not run_dir.is_dir():
                print(f"  WARN missing run dir: {pool}/{run_label} {run_id}")
                continue
            # A confirmed cell finding can arrive with an EMPTY ``error_types``:
            # operator FN cells and the trivial whitespace/punctuation tagger go
            # through the "legacy generic flow" that leaves the type blank. Such
            # a cell still belongs to the type its run targets -- a labeled FN in
            # the typo run is a typo -- so attribute it to the run's own per-cell
            # type(s) instead of dropping it. (Guard: a non-empty NON-per-cell
            # sentinel like ``["row_duplication"]`` is left to drop out as before;
            # only a truly-empty ``error_types`` falls back to the run type.)
            run_cell_types = sorted(run_types & PER_CELL_TYPES)
            cells, _skipped, _excl = _materialize_run(run_dir)
            ctx: Optional[RunContext] = None
            for c in cells:
                types = [t for t in (c.error_types or []) if t in PER_CELL_TYPES]
                if not types and not (c.error_types or []):
                    types = run_cell_types
                if not types:
                    continue
                key: CellKey = (c.source, c.row, c.column)
                g = merged.setdefault(key, {
                    "source": c.source, "row": c.row, "column": c.column,
                    "original_value": c.original_value,
                    "per_type": {}, "datasets": set(),
                })
                g["datasets"].add(DATASET_OF[pool])
                if c.original_value:
                    g["original_value"] = c.original_value
                for t in types:
                    prev = g["per_type"].get(t)
                    cand = {
                        "label": c.label,
                        "corrected_value": c.corrected_value,
                        "llm_suggested_value": c.llm_suggested_value,
                    }
                    if prev is None or LABEL_RANK.get(c.label, 0) > LABEL_RANK.get(prev["label"], 0):
                        g["per_type"][t] = cand
                if c.source not in frames:
                    if ctx is None:
                        try:
                            ctx = RunContext(run_dir)
                        except Exception:  # noqa: BLE001
                            ctx = None
                    try:
                        frames[c.source] = ctx.load_frame(c.source) if ctx else None
                    except Exception:  # noqa: BLE001
                        frames[c.source] = None
    return merged, frames


def _resolve_records(merged: Dict[CellKey, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Collapse each cell's per-cell-type verdicts into one benchmark record
    (error / ambiguous; clean cells are dropped)."""
    records: List[Dict[str, Any]] = []
    for key, g in merged.items():
        labels = {t: d["label"] for t, d in g["per_type"].items()}
        top = max((LABEL_RANK.get(lab, 0) for lab in labels.values()), default=0)
        if top == 3:
            status = "error"
        elif top == 2:
            status = "ambiguous"
        else:
            continue  # all per-cell verdicts clean -> drop (absent == clean)
        error_types = sorted(t for t, lab in labels.items() if lab == status)
        corrected: Any = None
        if status == "error":
            concretes: List[str] = []
            for t in error_types:
                for v in _resolve_concrete(g["per_type"][t], g["original_value"]):
                    if v not in concretes:
                        concretes.append(v)
            if len(concretes) == 1:
                corrected = concretes[0]
            elif concretes:
                corrected = sorted(concretes)
            # vacuous-fix guard: a scalar fix equal to the original "fixes"
            # nothing. Lists (bi-directional sets) legitimately keep the original.
            if isinstance(corrected, str) and corrected.strip() == (g["original_value"] or "").strip():
                corrected = None
        datasets = sorted(g["datasets"])
        rec: Dict[str, Any] = {
            "source": key[0], "row": key[1], "column": key[2],
            "original_value": g["original_value"],
            "status": status,
            "error_types": error_types,
            "corrected_value": corrected,
            "explanation": None,
            "dataset": datasets[0] if len(datasets) == 1 else "multi",
        }
        # Stamp a typo subtype hint from the typo type's OWN fix: the resolved
        # corrected_value for an error cell, or the LLM-proposed fix for an
        # ambiguous one. Applied to error_types only later, by
        # _apply_typo_subtypes, so the explanation cache stays keyed on "typo".
        if "typo" in error_types:
            te = g["per_type"].get("typo") or {}
            if status == "error":
                tcv = _resolve_concrete(te, g["original_value"])
                tfix: Any = tcv[0] if len(tcv) == 1 else (tcv or None)
            else:
                tfix = te.get("llm_suggested_value")
            sub = _typo_subtype(g["original_value"], tfix)
            if sub is not None:
                rec["_typo_sub"] = sub
        records.append(rec)
    return records


def _apply_typo_subtypes(records: List[Dict[str, Any]]) -> Dict[str, int]:
    """Replace the bare ``typo`` label with its subtype
    (``typo-formatting`` / ``typo-spelling``) in each record's
    ``error_types``, consuming the ``_typo_sub`` hint stamped by
    :func:`_resolve_records`. MUST run AFTER explanation generation so the
    explanation cache stays keyed on the plain ``typo`` label. Returns a
    ``{subtype: count}`` tally (``typo (unclassified)`` covers typo cells
    with no usable fix to classify, which keep the bare label)."""
    counts: collections.Counter = collections.Counter()
    for rec in records:
        sub = rec.pop("_typo_sub", None)
        if "typo" not in rec["error_types"]:
            continue
        if sub:
            rec["error_types"] = sorted(
                sub if t == "typo" else t for t in rec["error_types"]
            )
            counts[sub] += 1
        else:
            counts["typo (unclassified)"] += 1
    return dict(counts)


def _generate_explanations(
    records: List[Dict[str, Any]], frames: Dict[str, Any],
    cache_path: Path, *, limit: int, concurrency: int, use_llm: bool,
) -> int:
    """Fill ``explanation`` for ERROR records (cached + resumable). Returns
    the number of NEW LLM calls made."""
    cache = _load_explanation_cache(cache_path)
    todo: List[Dict[str, Any]] = []
    for rec in records:
        if rec["status"] != "error":
            continue
        ck = _cache_key(rec["source"], rec["row"], rec["column"],
                        rec["error_types"], rec["original_value"], rec["corrected_value"])
        rec["_ck"] = ck
        if ck in cache:
            rec["explanation"] = cache[ck]
        elif use_llm:
            todo.append(rec)
    # Stratify: interleave round-robin across the primary error type so a
    # ``--limit`` sample is representative (a few of each type) rather than
    # all-typo (which sorts first).
    if todo:
        groups: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
        for r in todo:
            groups[r["error_types"][0] if r["error_types"] else ""].append(r)
        ranked: List[Tuple[int, str, Dict[str, Any]]] = []
        for t, rs in groups.items():
            for i, r in enumerate(rs):
                ranked.append((i, t, r))
        ranked.sort(key=lambda x: (x[0], x[1]))
        todo = [r for _, _, r in ranked]
    if limit and len(todo) > limit:
        todo = todo[:limit]
    if not todo or not use_llm:
        return 0

    llm = LLMClient()

    def _gen(rec: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
        prompt = _build_prompt(
            rec["error_types"], rec["column"],
            _column_context(frames.get(rec["source"]), rec["column"]),
            rec["original_value"], rec["corrected_value"],
        )
        try:
            resp = llm.query(
                EXPLAIN_MODEL, prompt,
                reasoning_effort="low", reasoning_summary="auto",
                system_prompt=EXPLAIN_SYSTEM,
            )
            txt = (resp.text or "").strip().strip('"').strip()
        except Exception as exc:  # noqa: BLE001
            print(f"  explain failed {rec['source']} r{rec['row']} {rec['column']}: {exc}")
            txt = ""
        return rec, txt

    n_new = 0
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=concurrency) as ex, \
            cache_path.open("a", encoding="utf-8") as fh:
        futures = {ex.submit(_gen, rec): rec for rec in todo}
        for fut in tqdm(
            as_completed(futures), total=len(futures),
            desc="explaining", unit="cell", smoothing=0.1, file=sys.stdout,
        ):
            rec, txt = fut.result()
            if not txt:
                continue
            rec["explanation"] = txt
            fh.write(json.dumps({
                "key": rec["_ck"], "source": rec["source"], "row": rec["row"],
                "column": rec["column"], "explanation": txt,
                "prompt_version": EXPLAIN_PROMPT_VERSION, "model": EXPLAIN_MODEL,
            }, ensure_ascii=False) + "\n")
            fh.flush()
            n_new += 1
    return n_new


_CSV_FIELDS = [
    "source", "row", "column", "original_value", "status",
    "error_types", "corrected_value", "explanation", "dataset",
]


def _csv_corrected(cv: Any) -> str:
    """Flatten a corrected_value for CSV: scalar verbatim, bi-directional
    list as a JSON array, null as empty."""
    if cv is None:
        return ""
    if isinstance(cv, list):
        return json.dumps(cv, ensure_ascii=False)
    return str(cv)


def _write_csv(path: Path, records: List[Dict[str, Any]]) -> None:
    """Mirror benchmark_cell.json 1:1 as CSV (error_types ';'-joined,
    corrected_value JSON-array when bi-directional, proper quoting)."""
    tmp = path.with_suffix(".csv.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(_CSV_FIELDS)
        for r in records:
            w.writerow([
                r["source"], r["row"], r["column"], r["original_value"],
                r["status"], ";".join(r["error_types"]),
                _csv_corrected(r["corrected_value"]),
                r["explanation"] or "", r["dataset"],
            ])
    tmp.replace(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pools", nargs="*", default=list(REGISTRY), choices=list(REGISTRY))
    ap.add_argument("--out", default=str(PROJECT_ROOT / "benchmark" / "CleanBench"))
    ap.add_argument("--limit", type=int, default=0,
                    help="cap the number of NEW LLM explanation calls (0 = all)")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--no-llm", action="store_true", help="skip explanation generation")
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    merged, frames = _merge_cells(args.pools)
    records = _resolve_records(merged)

    # Cross-run exclusion immunity: a cell inside a slice (file / row / column)
    # excluded by ANY run in the selected pools must not be a positive label,
    # even when a sibling run labelled it. Drop those leaked cells here (before
    # spending LLM explanation calls on them). Their immunity lives in
    # ignored.jsonl, regenerated by export_cleanbench.
    excl_union = build_exclusion_union(args.pools)
    n_before = len(records)
    records = [r for r in records
               if not cell_is_excluded(excl_union, r["source"], r["row"], r["column"])]
    n_excluded = n_before - len(records)

    n_new = _generate_explanations(
        records, frames, out_dir / "llm_explanation_cache.jsonl",
        limit=args.limit, concurrency=args.concurrency, use_llm=not args.no_llm,
    )

    # Split ``typo`` into ``typo-formatting`` / ``typo-spelling`` AFTER
    # explanations are generated (keeps the cache keyed on plain "typo").
    typo_sub_counts = _apply_typo_subtypes(records)

    for rec in records:
        rec.pop("_ck", None)
    records.sort(key=lambda r: (r["source"], r["row"] if r["row"] is not None else -1, r["column"]))

    path = out_dir / "benchmark_cell.json"
    tmp = path.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)
    _write_csv(out_dir / "benchmark_cell.csv", records)

    # -------- summary --------
    status_counts = collections.Counter(r["status"] for r in records)
    type_counts: collections.Counter = collections.Counter()
    for r in records:
        if r["status"] == "error":
            for t in r["error_types"]:
                type_counts[t] += 1
    n_err = status_counts.get("error", 0)
    n_explained = sum(1 for r in records if r["status"] == "error" and r["explanation"])
    try:
        shown = out_dir.resolve().relative_to(PROJECT_ROOT)
    except ValueError:
        shown = out_dir
    print(f"\nwrote {shown}/benchmark_cell.json + .csv  ({len(records)} cells)")
    print(f"  cross-run exclusions dropped: {n_excluded} cell(s)")
    print(f"  status: {dict(status_counts)}")
    print(f"  error TP by type: {dict(type_counts)}")
    print(f"  typo subtypes: {typo_sub_counts}")
    print(f"  explanations: {n_explained}/{n_err} error cells "
          f"({n_new} new LLM calls this run)")
    print("\nsample explained error cells:")
    shown_n = 0
    for r in records:
        if r["status"] != "error" or not r["explanation"]:
            continue
        print(f"  [{','.join(r['error_types'])}] {r['column']}: "
              f"{r['original_value']!r} -> {r['corrected_value']!r}")
        print(f"      {r['explanation']}")
        shown_n += 1
        if shown_n >= 8:
            break


if __name__ == "__main__":
    main()
