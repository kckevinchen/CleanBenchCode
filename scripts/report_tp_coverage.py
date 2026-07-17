"""Report true-positive (TP) coverage per error type across the labeling
run folders shown in the workbench UI (PBI_500 / Excel_500 / Excel_200).

For every run that belongs to one of those virtual folders this scans the
operator verdicts (``labels.jsonl`` + ``row_verdicts.jsonl``, latest-wins)
and counts how many cells/clusters were confirmed as real errors
(``status="tp"``). Relation-pipeline runs bundle several multi-cell error
types in one run, so their TP count is broken down by the cluster's
``error_type`` (``synthesis.primary_error_type`` falling back to
``error_types[0]``).

Output: ``benchmark/TP_COVERAGE_REPORT.md`` (overwritten each run). This is
a read-only analysis - it does not modify any run or benchmark data.

Run: ``python scripts/report_tp_coverage.py``
"""

from __future__ import annotations

import collections
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
REPORT = ROOT / "benchmark" / "TP_COVERAGE_REPORT.md"

# The three workbench virtual folders we care about, in display order.
TARGET_FOLDERS = ["PBI_500", "Excel_500", "Excel_200"]

# Canonical ordering of the error types that appear in these folders.
TYPE_ORDER = [
    "typo",
    "inconsistency",
    "semantic_incompatible",
    "numeric_outlier",
    "fd_violation",
    "row_duplication",
    "row_calculation_inconsistency",
    "string_calculation_inconsistency",
]


def load_folder_names() -> dict[str, str]:
    fp = OUTPUTS / "folders.json"
    data = json.loads(fp.read_text())
    return {f["id"]: f["name"] for f in data.get("folders", [])}


def runs_by_folder() -> dict[str, list[tuple[str, str]]]:
    """folder_name -> list of (run_dir_name, display_name)."""
    id_to_name = load_folder_names()
    out: dict[str, list[tuple[str, str]]] = collections.defaultdict(list)
    for meta in OUTPUTS.glob("*/run_metadata.json"):
        try:
            d = json.loads(meta.read_text())
        except Exception:
            continue
        fid = d.get("folder_id", "")
        name = id_to_name.get(fid)
        if not name:
            continue
        out[name].append((meta.parent.name, d.get("display_name", "")))
    for name in out:
        out[name].sort(key=lambda t: t[0])
    return out


def cluster_error_types(run: Path) -> dict[str, str]:
    p = run / "clusters.jsonl"
    m: dict[str, str] = {}
    if not p.exists():
        return m
    for line in p.open():
        line = line.strip()
        if not line:
            continue
        try:
            o = json.loads(line)
        except Exception:
            continue
        syn = o.get("synthesis") or {}
        ets = o.get("error_types") or []
        m[o.get("cluster_id")] = syn.get("primary_error_type") or (ets[0] if ets else "?")
    return m


def summary_error_types(run: Path) -> list[str]:
    p = run / "summary.json"
    if not p.exists():
        return []
    try:
        cfg = json.loads(p.read_text()).get("config", {})
    except Exception:
        return []
    return cfg.get("error_types") or []


def tp_by_type_for_run(
    run_dir: str,
) -> tuple[dict[str, set], dict[str, set]]:
    """Return (tp_cellkeys_by_type, labeled_cellkeys_by_type) for one run.

    Verdicts are keyed by ``(source, row, column)`` so unioning across
    several runs of the same type (e.g. "Typo" + "Typo Strict") dedupes
    the same cell instead of double-counting. Row-level verdicts with no
    row fall back to the stable ``target_id``.

    Cluster/cell labels are attributed to the cluster's error type. When
    a run's ``summary.json`` declares no error type (relation runs), the
    run's single distinct cluster error type is used as the fallback so
    its ``row_verdicts`` are attributed correctly rather than to "?".
    """
    run = OUTPUTS / run_dir
    cl_types = cluster_error_types(run)
    decl = summary_error_types(run)
    single = decl[0] if len(decl) == 1 else None
    distinct_cl = {et for et in cl_types.values() if et and et != "?"}
    fallback = single or (next(iter(distinct_cl)) if len(distinct_cl) == 1 else "?")

    tp_by: dict[str, set] = collections.defaultdict(set)
    lab_by: dict[str, set] = collections.defaultdict(set)

    p = run / "labels.jsonl"
    latest: dict[str, dict] = {}
    if p.exists():
        for line in p.open():
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except Exception:
                continue
            latest[o.get("target_id")] = o
    for tid, o in latest.items():
        et = cl_types.get(tid) or fallback
        src, row, col = o.get("source"), o.get("row"), o.get("column")
        key = (src, row, col) if (src and row is not None) else tid
        lab_by[et].add(key)
        if o.get("status") == "tp":
            tp_by[et].add(key)

    # row_verdicts (latest-wins by cell), attributed to the run fallback type.
    rvp = run / "row_verdicts.jsonl"
    rv_latest: dict[tuple, str] = {}
    if rvp.exists():
        for line in rvp.open():
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except Exception:
                continue
            rv_latest[(o.get("source"), o.get("row"), o.get("column"))] = o.get("status")
    for key, status in rv_latest.items():
        lab_by[fallback].add(key)
        if status == "tp":
            tp_by[fallback].add(key)

    return tp_by, lab_by


def main() -> None:
    folders = runs_by_folder()
    # folder -> type -> set(cell keys)
    tp_keys = {f: collections.defaultdict(set) for f in TARGET_FOLDERS}
    lab_keys = {f: collections.defaultdict(set) for f in TARGET_FOLDERS}
    run_index: dict[str, list[str]] = {f: [] for f in TARGET_FOLDERS}

    for folder in TARGET_FOLDERS:
        for run_dir, dn in folders.get(folder, []):
            t, l = tp_by_type_for_run(run_dir)
            for et, keys in l.items():
                lab_keys[folder][et] |= keys
            for et, keys in t.items():
                tp_keys[folder][et] |= keys
            shown = (
                ", ".join(f"{et}:{len(t.get(et, set()))}tp/{len(l[et])}" for et in sorted(l))
                or "(none)"
            )
            run_index[folder].append(f"- `{dn or '—'}` → `{run_dir}` — {shown}")

    # Collapse sets to counts for reporting.
    tp = {f: {et: len(ks) for et, ks in tp_keys[f].items()} for f in TARGET_FOLDERS}
    lab = {f: {et: len(ks) for et, ks in lab_keys[f].items()} for f in TARGET_FOLDERS}

    all_types = TYPE_ORDER + [
        et for f in TARGET_FOLDERS for et in lab[f] if et not in TYPE_ORDER
    ]
    seen = set()
    all_types = [t for t in all_types if not (t in seen or seen.add(t))]
    # Drop the "?" bucket: stale label target_ids no longer present in
    # clusters.json (always non-TP); they carry no coverage signal.
    all_types = [t for t in all_types if t != "?"]

    lines: list[str] = []
    lines.append("# TP coverage by error type across run folders")
    lines.append("")
    lines.append(
        "Generated by `scripts/report_tp_coverage.py`. TP = operator verdict "
        "`tp` (latest-wins) from each run's `labels.jsonl` + `row_verdicts.jsonl`. "
        "Relation runs are split by the cluster's error type."
    )
    lines.append("")
    lines.append("## TP per error type")
    lines.append("")
    header = "| error_type | " + " | ".join(TARGET_FOLDERS) + " |"
    sep = "|" + "---|" * (len(TARGET_FOLDERS) + 1)
    lines.append(header)
    lines.append(sep)
    for et in all_types:
        cells = []
        for f in TARGET_FOLDERS:
            n = tp[f].get(et, 0)
            has_run = et in lab[f]
            if not has_run:
                cells.append("·")        # type not labeled in this folder
            elif n == 0:
                cells.append("**0**")    # labeled but zero TP
            else:
                cells.append(str(n))
        lines.append(f"| {et} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(
        "Legend: number = TP count · `**0**` = run exists but every cluster "
        "was labeled FP · `·` = no run for that type in that folder."
    )
    lines.append("")

    # Zero-TP callouts
    lines.append("## Zero-TP categories")
    lines.append("")
    for f in TARGET_FOLDERS:
        zeros = [et for et in lab[f] if et != "?" and tp[f].get(et, 0) == 0]
        if zeros:
            lines.append(f"- **{f}**: " + ", ".join(sorted(zeros)))
        else:
            lines.append(f"- **{f}**: none")
    lines.append("")

    # Recommended source pool per type (unified)
    lines.append("## Recommended source pool per error type")
    lines.append("")
    lines.append("| error_type | best pool (TP) | notes |")
    lines.append("|---|---|---|")
    for et in all_types:
        ranked = sorted(
            ((tp[f].get(et, 0), f) for f in TARGET_FOLDERS), reverse=True
        )
        best_n, best_f = ranked[0]
        pools_with = [f for f in TARGET_FOLDERS if tp[f].get(et, 0) > 0]
        if best_n == 0:
            note = "NO TP anywhere"
        elif len(pools_with) == 1:
            note = f"only {pools_with[0]} has TP"
        else:
            note = "available in: " + ", ".join(
                f"{f}({tp[f].get(et,0)})" for f in pools_with
            )
        lines.append(f"| {et} | {best_f} ({best_n}) | {note} |")
    lines.append("")

    # Run index
    lines.append("## Run index (folder → runs)")
    lines.append("")
    for f in TARGET_FOLDERS:
        lines.append(f"### {f}")
        lines.extend(run_index[f] or ["- (no runs)"])
        lines.append("")

    REPORT.write_text("\n".join(lines))
    print(f"wrote {REPORT.relative_to(ROOT)}")
    # Also echo the core table to stdout.
    print("\n".join(lines[: 6 + 2 + len(all_types) + 4]))


if __name__ == "__main__":
    main()
