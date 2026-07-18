"""Build the unified, cross-error-type CleanBench materialization.

Composes the SAME per-run materialisation the live workbench export
endpoint uses (``src.labeling.routers.exports``) for every (pool x
error_type) labeling run, then merges the per-type results into one
per-cell benchmark that is honest about the three hazards of a
cross-type merge:

1. **Different ignore sets per type** -> a cell is immune only if it is
   skipped/excluded in some type AND flagged by NO type (scoped union;
   ``error`` / ``ambiguous`` always beat ``ignore``).
2. **Conflicting labels per type** -> resolved by a precedence lattice
   ``error > ambiguous > ignore > clean``; the per-type verdicts stay
   available in the ``per_type/<type>.jsonl`` bundle so nothing is lost.
3. **Different granularity per type** -> cell-level types (typo / semantic /
   outlier / fd) key on (source, row, column); row-level findings like
   row_duplication carry an empty ``column`` so a row match is implied.

corrected_value conflicts are merged into an acceptable-set: the union
of CONCRETE (non-instruction) fixes proposed by the error-types that
flagged the cell -> ``str`` when they agree on one value, ``list`` when
they genuinely differ (bi-directional), ``null`` when only prose advice
exists (the per-type fixes stay in the ``per_type`` bundle).

Outputs (default ``benchmark/CleanBench/``)::

    benchmark.json       unified per-cell ground truth (source,row,column,
                         original_value,status,error_types,
                         corrected_value,dataset). Only error + ambiguous
                         cells; absent cell => clean.
    ignored.jsonl        scoped-union immunity set (skip + exclusion sentinels)
    per_type/<type>.jsonl   per-type bundle (each type's own verdicts; canonical).
                            A cell that is a TP under some OTHER type but not
                            this one is relabeled clean->ambiguous (FP immunity).

Read-only w.r.t. the runs. Run::

    python -m experiments.export_cleanbench            # all pools
    python -m experiments.export_cleanbench --pools excel_500
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from src.labeling._cells import (
    _apply_exclusions_to_cells,
    _deduplicate_benchmark_cells,
    _make_value_resolver,
    _materialize_inconsistency_cells_for_export,
)
from src.labeling._helpers import _derive_trivial_fix, _looks_like_instruction
from src.labeling.app import RunContext
from src.labeling.exclusions_store import ExclusionStore
from src.labeling.export import (
    materialize_label_benchmark,
    materialize_skipped_cells,
)
from src.labeling.schema import ExclusionScope

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = PROJECT_ROOT / "outputs"

# Precedence lattice for the unified per-cell verdict.
LABEL_RANK = {"error": 3, "ambiguous": 2, "clean": 1, "": 0}

# (pool -> [(run_type_label, {canonical_error_types}, run_dir_name), ...]).
# run_type_label is just for logging; the {canonical_error_types} set
# records which types a run targets (shown in the manifest registry).
REGISTRY: Dict[str, List[Tuple[str, Set[str], str]]] = {
    "pbi_500": [
        ("typo", {"typo"}, "20260604T223438Z_model_ce334f25"),
        ("inconsistency", {"inconsistency"}, "20260605T000013Z_model_06e51dd5"),
        ("semantic_incompatible", {"semantic_incompatible"}, "20260611T223250Z_model_907da1ef"),
        ("numeric_outlier", {"numeric_outlier"}, "20260612T072822Z_model_206fc9c3"),
        ("calc", {"row_calculation_inconsistency", "string_calculation_inconsistency"},
         "20260612T221140Z_gpt-5.2,gpt-5.4,gpt-5.5_relations_adaptive"),
        ("fd_violation", {"fd_violation"},
         "20260612T224555Z_gpt-5.2,gpt-5.4,gpt-5.5_relations_adaptive"),
        ("row_duplication", {"row_duplication"}, "20260613T023203Z_model_d377df5d"),
    ],
    "excel_500": [
        ("typo", {"typo"}, "20260630T012247Z_model_a60d81db"),
        ("inconsistency", {"inconsistency"}, "20260610T175821Z_model_26527bcd"),
        ("semantic_incompatible", {"semantic_incompatible"}, "20260610T204519Z_model_165101e3"),
        ("numeric_outlier", {"numeric_outlier"}, "20260612T072838Z_model_710c95a6"),
        ("fd_violation", {"fd_violation"},
         "20260612T224615Z_gpt-5.2,gpt-5.4,gpt-5.5_relations_adaptive"),
        ("row_duplication", {"row_duplication"}, "20260613T023150Z_model_44e78160"),
        ("calc", {"row_calculation_inconsistency", "string_calculation_inconsistency"},
         "20260616T220338Z_gpt-5.2,gpt-5.4,gpt-5.5_relations_adaptive"),
    ],
}

# Pool -> emitted ``dataset`` label. The benchmark contains exactly two
# corpora: the Power BI set (``curated_pbi_500_dedup``) and the Bing21
# Excel set (``curated_bing21_500_dedup``). The two calc-completing CSVs
# (25039521.csv, 4b553bb5.csv) were already ported into the excel_500
# runs, so they come in via that pool - there is NO separate excel_200
# pool. Net result: ``dataset`` is always one of {pbi_500, bing21_500}.
DATASET_OF: Dict[str, str] = {
    "pbi_500": "pbi_500",
    "excel_500": "bing21_500",
}

CellKey = Tuple[str, Optional[int], str]

# Cross-run exclusion union: (excluded whole files, rows-by-source, cols-by-source).
ExclusionUnion = Tuple[Set[str], Dict[str, Set[int]], Dict[str, Set[str]]]


def build_exclusion_union(pools: List[str]) -> ExclusionUnion:
    """Global union of ACTIVE exclusions across EVERY run in ``pools``.

    An operator marks a CSV / rows / columns as too messy to label in ANY
    one run (e.g. the typo run excludes a file); that immunity must apply to
    all of CleanBench, not just the run that recorded it -- otherwise
    a sibling run (inconsistency, semantic, ...) still emits error/ambiguous
    labels inside the excluded slice. This unions every run's ``exclusions.jsonl``
    so a later filter can drop those leaked cells / rows.

    Reads ``exclusions.jsonl`` directly (latest-wins via ``ExclusionStore``) --
    no heavy ``RunContext`` / CSV load.
    """
    files: Set[str] = set()
    rows: Dict[str, Set[int]] = collections.defaultdict(set)
    cols: Dict[str, Set[str]] = collections.defaultdict(set)
    for pool in pools:
        for _label, _types, run_id in REGISTRY.get(pool, []):
            path = OUTPUTS / run_id / "exclusions.jsonl"
            if not path.exists():
                continue
            for e in ExclusionStore(path).active():
                src = e.source or ""
                if e.scope == ExclusionScope.CSV:
                    files.add(src)
                elif e.scope == ExclusionScope.ROWS:
                    for r in e.rows or []:
                        try:
                            rows[src].add(int(r))
                        except (TypeError, ValueError):
                            continue
                elif e.scope == ExclusionScope.COLUMNS:
                    for c in e.columns or []:
                        if c is not None:
                            cols[src].add(str(c))
    return files, rows, cols


def cell_is_excluded(union: ExclusionUnion, source: str,
                     row: Optional[int], column: Optional[str]) -> bool:
    """True if ``(source, row, column)`` falls under any exclusion in ``union``.

    Mirrors the ``IgnoreRecord`` consumer rule: a cell is immune if its whole
    file, its row, or its column is excluded by any run.
    """
    files, rows, cols = union
    if source in files:
        return True
    if row is not None and row in rows.get(source, set()):
        return True
    if column and column in cols.get(source, set()):
        return True
    return False


def row_finding_is_excluded(union: ExclusionUnion, rec: Dict[str, Any]) -> bool:
    """True if a per-ROW benchmark finding lands in an excluded slice.

    A row-level finding (fd_violation, calc, row_duplication, ...) is dropped
    when its source file is excluded, its ``target_column`` is an excluded
    column, or ANY of its involved ``rows`` (or ``target_row``) is an excluded
    row -- consistent with the cell rule (exclude if any covered part is out).
    """
    files, rows, cols = union
    source = rec.get("source") or ""
    if source in files:
        return True
    tc = rec.get("target_column")
    if tc and tc in cols.get(source, set()):
        return True
    ex_rows = rows.get(source, set())
    if ex_rows:
        involved = list(rec.get("rows") or [])
        tr = rec.get("target_row")
        if tr is not None:
            involved.append(tr)
        for r in involved:
            try:
                if int(r) in ex_rows:
                    return True
            except (TypeError, ValueError):
                continue
    return False


def _load_llm_corrected_cache(out_dir: Path) -> Dict[Tuple[str, int, str], str]:
    """Optional best-effort fixes from ``llm_corrected_cache.jsonl``.

    Maps ``(source, row, column) -> corrected_value`` for cells whose fix the
    LLM could supply. Missing file / no-fix entries (``corrected_value: null``)
    are skipped, so this only ever ADDS a concrete value, never blanks one.
    """
    path = out_dir / "llm_corrected_cache.jsonl"
    out: Dict[Tuple[str, int, str], str] = {}
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
        cv = o.get("corrected_value")
        if isinstance(cv, str) and cv.strip():
            out[(o["source"], int(o["row"]), o["column"])] = cv
    return out



def _concrete_values(cv: Any) -> List[str]:
    """Concrete (non-instruction) string fixes from a corrected_value."""
    out: List[str] = []
    if cv is None:
        return out
    for v in cv if isinstance(cv, list) else [cv]:
        if isinstance(v, str) and v and not _looks_like_instruction(v):
            out.append(v)
    return out


def _resolve_concrete(entry: Dict[str, Any], original: str = "") -> List[str]:
    """Concrete fix(es) for one per-type verdict.

    Mirrors ``src.labeling.export._concrete_corrected_value`` and encodes the
    same rule: the "Original / Fix" panel value (``llm_suggested_value``) is the
    DEFAULT; the operator's ``corrected_value`` only wins when it is a genuine
    concrete MODIFICATION. A labeled ``corrected_value`` is frequently an EDIT
    INSTRUCTION ("Change 'Hotor' to 'Motor'.", "Remove the trailing comma")
    rather than a literal value, while the concrete result sits in
    ``llm_suggested_value`` (the panel Fix, from ``canonical_record`` /
    per-detection literals). So: take the concrete ``corrected_value`` items
    (an instruction is not concrete and is skipped); else the concrete
    ``llm_suggested_value``; else DERIVE the literal result of a trivial,
    unambiguous instruction (``_derive_trivial_fix``); else drop it
    (``[]`` -> ``null``). An edit-instruction sentence is never emitted verbatim.

    A tier is SKIPPED when every concrete value it offers equals the cell's
    own ``original`` (a vacuous no-op label - some annotators saved the
    original verbatim instead of the fix), so the fallback continues to
    ``llm_suggested_value`` which usually holds the genuine correction. A
    tier with at least one differing value is kept whole, so a bi-directional
    inconsistency set that legitimately contains the original survives.
    """
    orig = (original or "").strip()

    def _useful(vals: List[str]) -> bool:
        return any(v.strip() != orig for v in vals)

    cv = entry.get("corrected_value")
    concrete = _concrete_values(cv)
    if concrete and _useful(concrete):
        return concrete
    sug = _concrete_values(entry.get("llm_suggested_value"))
    if sug and _useful(sug):
        return sug
    # Deterministic derivation: when the only fix is a trivial edit INSTRUCTION
    # ("Remove the trailing comma"), emit the literal RESULT of the edit rather
    # than the sentence. Faithful by construction (``_derive_trivial_fix`` only
    # honours unambiguous classes and applies that specific edit); returns
    # nothing for out-of-scope prose, so those cells fall through to ``null``.
    for prose in (cv, entry.get("llm_suggested_value")):
        if isinstance(prose, str) and prose:
            derived = _derive_trivial_fix(original, prose)
            if derived and derived.strip() != orig:
                return [derived]
    # Last resort: keep a recorded corrected_value verbatim ONLY when it is a
    # genuine value -- not an edit instruction (those are dropped above) and not
    # a vacuous echo of the original.
    if (
        isinstance(cv, str)
        and cv
        and cv.strip() != orig
        and not _looks_like_instruction(cv)
    ):
        return [cv]
    return []


def _inconsistency_value_sets(run_dir: Path) -> Dict[Tuple[str, str, str], List[str]]:
    """Map ``(source, column, value) -> sorted cluster value-set`` from the
    run's ``column_cluster_drafts.jsonl``.

    For an inconsistency cell whose cluster has no single generated/confirmed
    canonical, the documented benchmark fix (see
    ``src.labeling.schema.BenchmarkCell``) is the BI-DIRECTIONAL acceptable
    set: the surface forms in its cluster, any of which is a valid value.
    We union the value-sets of every cluster that contains the value so a
    value appearing in multiple draft clusters carries all acceptable forms.
    """
    drafts_path = run_dir / "column_cluster_drafts.jsonl"
    out: Dict[Tuple[str, str, str], set] = collections.defaultdict(set)
    if not drafts_path.exists():
        return {}
    for line in drafts_path.open():
        line = line.strip()
        if not line:
            continue
        try:
            o = json.loads(line)
        except json.JSONDecodeError:
            continue
        src, col = o.get("source"), o.get("column")
        for c in o.get("draft_clusters") or []:
            vals = [v for v in (c.get("values") or []) if isinstance(v, str)]
            if len(vals) < 2:
                continue
            for v in vals:
                out[(src, col, v)].update(vals)
    return {k: sorted(vs) for k, vs in out.items() if len(vs) >= 2}


def _materialize_run(run_dir: Path):
    """Return (cells, skipped, exclusions) for one run via the canonical
    export pipeline (identical to the workbench /export/benchmark route)."""
    ctx = RunContext(run_dir)
    resolver = _make_value_resolver(ctx)
    cells = materialize_label_benchmark(
        ctx.clusters,
        ctx.store,
        value_resolver=resolver,
        cell_note_overrides=ctx.cell_notes_store.active_overrides(),
        row_verdict_overrides=ctx.row_verdict_store.active_overrides(),
    )
    cells.extend(_materialize_inconsistency_cells_for_export(ctx))
    cells = _apply_exclusions_to_cells(ctx, cells)
    cells = _deduplicate_benchmark_cells(cells)

    # Backfill bi-directional fixes for inconsistency error cells that the
    # materialiser left without a corrected_value (no LLM/operator canonical):
    # use the cluster's acceptable value-set so corrected_value is never empty
    # on a confirmed inconsistency error.
    value_sets = _inconsistency_value_sets(run_dir)
    if value_sets:
        for c in cells:
            if (
                c.label == "error"
                and "inconsistency" in (c.error_types or [])
                and c.corrected_value in (None, "", [])
            ):
                vs = value_sets.get((c.source, c.column, c.original_value))
                if vs:
                    c.corrected_value = vs

    bench_keys = {(c.source, c.row, c.column) for c in cells}
    skipped = materialize_skipped_cells(
        ctx.clusters, ctx.store, value_resolver=resolver, exclude_keys=bench_keys,
    )
    skipped = _apply_exclusions_to_cells(ctx, skipped)
    exclusions = list(ctx.exclusions_store.active())
    return cells, skipped, exclusions


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pools", nargs="*", default=list(REGISTRY),
                    choices=list(REGISTRY), help="which pools to include")
    ap.add_argument("--out", default=str(PROJECT_ROOT / "benchmark" / "CleanBench"))
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Global accumulators keyed by (source, row, column).
    merged: Dict[CellKey, Dict[str, Any]] = {}
    # Per-type bundle: error_type -> {cell_key -> record dict} (own verdicts).
    per_type: Dict[str, Dict[CellKey, Dict[str, Any]]] = collections.defaultdict(dict)
    skip_keys: Set[CellKey] = set()
    # exclusion sentinels reduced to matchers: file set, rows-by-source, cols-by-source
    excl_files: Set[str] = set()
    excl_rows: Dict[str, Set[int]] = collections.defaultdict(set)
    excl_cols: Dict[str, Set[str]] = collections.defaultdict(set)
    excl_records: List[Dict[str, Any]] = []

    stats = collections.Counter()
    for pool in args.pools:
        for run_label, _run_types, run_id in REGISTRY[pool]:
            run_dir = OUTPUTS / run_id
            if not run_dir.is_dir():
                print(f"  WARN missing run dir: {pool}/{run_label} {run_id}")
                continue
            cells, skipped, exclusions = _materialize_run(run_dir)
            stats[f"{pool}/{run_label}:cells"] = len(cells)

            for c in cells:
                key: CellKey = (c.source, c.row, c.column)
                types = list(c.error_types) or ["_uncategorized"]
                # ---- unified accumulator
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
                        "pool": pool,
                    }
                    if prev is None or LABEL_RANK.get(c.label, 0) > LABEL_RANK.get(prev["label"], 0):
                        g["per_type"][t] = cand

                # ---- per-type bundle (own verdicts, lossless)
                for t in types:
                    b = per_type[t]
                    rec = b.get(key)
                    if rec is None or LABEL_RANK.get(c.label, 0) > LABEL_RANK.get(rec["label"], 0):
                        b[key] = {
                            "source": c.source, "row": c.row, "column": c.column,
                            "original_value": c.original_value, "label": c.label,
                            "corrected_value": c.corrected_value,
                            "llm_suggested_value": c.llm_suggested_value,
                            "error_type": t,
                            "dataset": DATASET_OF[pool],
                        }

            for s in skipped:
                skip_keys.add((s.source, s.row, s.column))

            for e in exclusions:
                src = e.source or ""
                if e.scope == ExclusionScope.CSV:
                    excl_files.add(src)
                    excl_records.append({"source": src, "row": None, "column": None,
                                         "reason": "exclusion", "scope": "csv", "dataset": DATASET_OF[pool]})
                elif e.scope == ExclusionScope.ROWS:
                    for r in e.rows or []:
                        try:
                            ri = int(r)
                        except (TypeError, ValueError):
                            continue
                        excl_rows[src].add(ri)
                        excl_records.append({"source": src, "row": ri, "column": None,
                                             "reason": "exclusion", "scope": "rows", "dataset": DATASET_OF[pool]})
                elif e.scope == ExclusionScope.COLUMNS:
                    for col in e.columns or []:
                        if col is None:
                            continue
                        excl_cols[src].add(str(col))
                        excl_records.append({"source": src, "row": None, "column": str(col),
                                             "reason": "exclusion", "scope": "columns", "dataset": DATASET_OF[pool]})

    # -------- resolve each unified cell through the lattice -----------
    def _excluded(key: CellKey) -> bool:
        s, r, c = key
        if s in excl_files:
            return True
        if r is not None and r in excl_rows.get(s, ()):
            return True
        if c and c in excl_cols.get(s, ()):
            return True
        return False

    cells_out: List[Dict[str, Any]] = []
    ignored_skip_records: List[Dict[str, Any]] = []
    type_tp = collections.Counter()
    status_counts = collections.Counter()
    llm_cv = _load_llm_corrected_cache(out_dir)
    n_llm_filled = 0

    for key, g in merged.items():
        # Exclusion beats every label: a cell in a slice excluded by ANY run is
        # never a positive benchmark label (its immunity is already recorded as
        # a wildcard exclusion record in ignored.jsonl). Drop it before it can be
        # emitted as an error / ambiguous / clean cell.
        if _excluded(key):
            continue
        labels = {t: d["label"] for t, d in g["per_type"].items()}
        ranks = [LABEL_RANK.get(lab, 0) for lab in labels.values()]
        top = max(ranks) if ranks else 0
        if top == 3:
            status = "error"
        elif top == 2:
            status = "ambiguous"
        else:
            # all present verdicts are clean. skip beats clean: if this cell is
            # skipped under ANY type, immunise it.
            if key in skip_keys:
                ignored_skip_records.append({
                    "source": key[0], "row": key[1], "column": key[2],
                    "original_value": g["original_value"],
                    "reason": "skip", "scope": None,
                })
                continue
            status = "clean"

        # error_types = the types that drove this cell's final status:
        # for an ``error`` cell, the types that confirmed it as an error;
        # for an ``ambiguous`` cell, the types that flagged it as ambiguous.
        error_types = sorted(t for t, lab in labels.items()
                             if lab == status and t != "_uncategorized")
        # corrected_value: union of concrete fixes from the contributing
        # types (prefer literal corrected_value, fall back to the concrete
        # llm_suggested_value, then recorded prose). Only error cells carry
        # a ground-truth fix; ambiguous cells have no canonical answer.
        concretes: List[str] = []
        if status == "error":
            for t in error_types:
                for v in _resolve_concrete(g["per_type"][t], g["original_value"]):
                    if v not in concretes:
                        concretes.append(v)
        if not concretes:
            merged_cv: Any = None
        elif len(concretes) == 1:
            merged_cv = concretes[0]
        else:
            merged_cv = sorted(concretes)

        # Final fallback: a best-effort LLM-supplied fix (cached on disk by
        # llm_corrected_cache.jsonl) for an otherwise empty cell.
        from_llm = False
        if status == "error" and merged_cv in (None, "", []):
            cached = llm_cv.get((key[0], key[1], key[2]))
            if cached:
                merged_cv = cached
                from_llm = True

        # Vacuous-fix guard: a SCALAR corrected_value equal to the cell's own
        # value "fixes" nothing and wrongly implies this exact cell was the
        # error - drop it to None. Applies to every source (recorded label,
        # llm_suggested, or the LLM backfill): the FD verdict fans across all
        # relation columns and some labeled typo fixes never changed the text.
        # LIST values are exempt: a bi-directional inconsistency set is meant
        # to include the original as one acceptable form.
        if (
            status == "error"
            and isinstance(merged_cv, str)
            and merged_cv.strip() == (g["original_value"] or "").strip()
        ):
            merged_cv = None
            from_llm = False
        if from_llm:
            n_llm_filled += 1

        datasets = sorted(g["datasets"])
        rec = {
            "source": key[0], "row": key[1], "column": key[2],
            "original_value": g["original_value"],
            "status": status,
            "error_types": error_types,
            "corrected_value": merged_cv,
            "dataset": datasets[0] if len(datasets) == 1 else "multi",
        }
        cells_out.append(rec)
        status_counts[status] += 1
        if status == "error":
            for t in error_types:
                type_tp[t] += 1

    # -------- write outputs ------------------------------------------
    def _write(path: Path, rows: List[Dict[str, Any]]) -> int:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        tmp.replace(path)
        return len(rows)

    cells_out.sort(key=lambda r: (r["source"], r["row"] if r["row"] is not None else -1, r["column"]))
    # Lean benchmark: drop ``clean`` records. Only ``error`` (positives) and
    # ``ambiguous`` (explicit FP immunity) are emitted; a cell ABSENT from
    # benchmark.json is treated as clean by the scorer. Matches the repo's
    # existing typo benchmark convention ("confirmed non-errors are excluded").
    n_clean_dropped = sum(1 for r in cells_out if r["status"] == "clean")
    cells_out = [r for r in cells_out if r["status"] != "clean"]
    n_cells = _write(out_dir / "benchmark.json", cells_out)

    ignored_all = ignored_skip_records + excl_records
    n_ignored = _write(out_dir / "ignored.jsonl", ignored_all)

    # Cross-type ambiguity for the per-type bundles. A cell that is a
    # confirmed error (TP) under SOME error type but is NOT a TP under
    # error type T is genuinely dirty - just for a different reason - so
    # labeling it a confident true-negative ("clean") in T's bundle would
    # be wrong and would penalise a model that flags it. Relabel those
    # clean cells to "ambiguous" so T's eval gives them FP immunity. The
    # ``cross_type_tp`` field records which other types flagged it.
    tp_types_by_key: Dict[CellKey, Set[str]] = {}
    for key, g in merged.items():
        tps = {t for t, d in g["per_type"].items()
               if d["label"] == "error" and t != "_uncategorized"}
        if tps:
            tp_types_by_key[key] = tps

    n_reclassified = 0
    for t, b in per_type.items():
        for key, rec in b.items():
            if rec["label"] == "error":
                continue
            other_tp = tp_types_by_key.get(key, set()) - {t}
            if not other_tp:
                continue
            if rec["label"] == "clean":
                n_reclassified += 1
            rec["label"] = "ambiguous"
            rec["cross_type_tp"] = sorted(other_tp)

    bundle_dir = out_dir / "per_type"
    bundle_dir.mkdir(exist_ok=True)
    per_type_counts = {}
    for t, b in sorted(per_type.items()):
        rows = sorted(b.values(), key=lambda r: (r["source"], r["row"] if r["row"] is not None else -1, r["column"]))
        _write(bundle_dir / f"{t}.jsonl", rows)
        per_type_counts[t] = {
            "cells": len(rows),
            "tp": sum(1 for r in rows if r["label"] == "error"),
            "cross_type_ambiguous": sum(1 for r in rows if r.get("cross_type_tp")),
        }

    # -------- console summary ----------------------------------------
    try:
        shown = out_dir.resolve().relative_to(PROJECT_ROOT)
    except ValueError:
        shown = out_dir
    print(f"\nwrote {shown}/")
    print(f"  benchmark.json    {n_cells} (clean dropped: {n_clean_dropped})")
    print(f"  ignored.jsonl     {n_ignored}")
    print(f"  per_type/*.jsonl  {len(per_type_counts)} files "
          f"(cross-type clean->ambiguous: {n_reclassified})")
    if llm_cv:
        print(f"  LLM corrected_value fills applied: {n_llm_filled}")
    print(f"\nstatus: {dict(status_counts)}")
    print("TP by error_type:")
    for t, n in sorted(type_tp.items()):
        print(f"   {t:34s} {n}")


if __name__ == "__main__":
    main()
