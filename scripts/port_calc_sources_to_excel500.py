"""Port two source CSVs (and their already-human-labeled detections) from
the Excel_200 labeling runs into the matching Excel_500 runs.

Motivation
==========
`25039521.csv` (carries the only row_calculation_inconsistency TP) and
`4b553bb5.csv` (carries the only string_calculation_inconsistency TP) live
in the Excel_200 corpus and were already detected + human-labeled there for
every relevant error type. The Excel_500 run folder had **zero** TP for the
two calculation error types. This script adds those two CSVs to the
Excel_500 corpus and ports their existing detections + verdicts into each
affected Excel_500 run, so Excel_500 gains TP coverage for the calc types
(and stays internally consistent for the other types these CSVs touch).

It does NOT call any LLM. It copies the exact Excel_200 CSV (so the row
indices in the ported clusters/labels stay aligned) and the human verdicts
verbatim.

Authoritative artifacts for the labeling UI + benchmark export + the TP
coverage report are: ``clusters.jsonl`` / ``clusters.json`` (queue + export),
``labels.jsonl`` (verdicts), ``column_cluster_drafts.jsonl`` (inconsistency),
and ``summary.json -> config.csv_paths`` (CSV resolution). We port exactly
those. We deliberately skip ``issues.jsonl`` and the relation provenance
files: they are not read by the export/coverage path and their non
source-keyed ids (``r0255-i0000``) would risk collisions.

Idempotent: re-running first strips any prior entries for the two target
sources from each destination file, then re-appends, so counts never drift.
Backs up every mutated run dir under ``backups/`` first.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
CORPUS_500 = ROOT / "data" / "curated_bing21_500_dedup"
STAMP = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
BACKUP_DIR = ROOT / "backups" / f"port_calc_sources_{STAMP}"

# Target sources -> the exact Excel_200 CSV the labels were made against.
TARGET_SOURCES = {
    "25039521.csv": ROOT / "data" / "curated_bing21_200"
    / "26021fc0708a0b8b457d998cfa31ec9a3af33bd8.xlsx" / "25039521.csv",
    "4b553bb5.csv": ROOT / "data" / "curated_bing21_200"
    / "88a2b55bf37002a02d52b5881a7c158ced62369a.xlsx" / "4b553bb5.csv",
}
TARGETS = set(TARGET_SOURCES)

# Where each target CSV is dropped inside the Excel_500 corpus (basename
# preserved so the `source` key stays `25039521.csv` / `4b553bb5.csv`).
CORPUS_DEST = {
    name: CORPUS_500 / f"{Path(name).stem}.xlsx_ported" / name
    for name in TARGET_SOURCES
}

# Each affected Excel_500 run + its Excel_200 source run. ``etypes`` (when
# set) filters which cluster error types to port - needed for the relations
# StringNumeric run, whose Excel_200 source also contains fd clusters that
# belong to the FD run instead.
PAIRS = [
    {
        "name": "StringNumeric",
        "e500": "20260616T220338Z_gpt-5.2,gpt-5.4,gpt-5.5_relations_adaptive",
        "e200": "20260519T000148Z_gpt-5.2,gpt-5.4,gpt-5.5_relations_adaptive",
        "etypes": {
            "row_calculation_inconsistency",
            "string_calculation_inconsistency",
        },
    },
    {
        "name": "Typo",
        "e500": "20260605T195248Z_model_a60d81db",
        "e200": "20260512T213238Z_gpt5-family",
        "etypes": None,
    },
    {
        "name": "Outlier",
        "e500": "20260612T072838Z_model_710c95a6",
        "e200": "20260510T211448Z_gpt5-family",
        "etypes": None,
    },
    {
        "name": "FD",
        "e500": "20260612T224615Z_gpt-5.2,gpt-5.4,gpt-5.5_relations_adaptive",
        "e200": "20260514T063141Z_gpt5-family",
        "etypes": None,
    },
    {
        "name": "Duplication",
        "e500": "20260613T023150Z_model_44e78160",
        "e200": "20260604T223616Z_model_b0d5c850",
        "etypes": None,
    },
    {
        "name": "Inconsistency",
        "e500": "20260610T175821Z_model_26527bcd",
        "e200": "20260513T234733Z_gpt5-family",
        "etypes": None,
    },
]

MUTABLE_FILES = [
    "clusters.jsonl",
    "clusters.json",
    "labels.jsonl",
    "summary.json",
    "column_cluster_drafts.jsonl",
]


# --------------------------------------------------------------------- io


def read_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    out = []
    for line in p.open():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def write_jsonl(p: Path, rows: list[dict]) -> None:
    with p.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def cluster_etype(c: dict) -> str:
    syn = c.get("synthesis") or {}
    ets = c.get("error_types") or []
    return syn.get("primary_error_type") or (ets[0] if ets else "?")


# --------------------------------------------------------------------- steps


def backup_run(run_id: str) -> None:
    src = OUTPUTS / run_id
    dst = BACKUP_DIR / run_id
    dst.mkdir(parents=True, exist_ok=True)
    for fn in MUTABLE_FILES:
        f = src / fn
        if f.exists():
            shutil.copy2(f, dst / fn)


def copy_csvs_into_corpus() -> list[str]:
    added = []
    for name, src in TARGET_SOURCES.items():
        dst = CORPUS_DEST[name]
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not src.exists():
            raise FileNotFoundError(src)
        if not dst.exists():
            shutil.copy2(src, dst)
            added.append(str(dst.relative_to(ROOT)))
    return added


def strip_targets(rows: list[dict]) -> list[dict]:
    """Drop any record whose source is one of the target CSVs."""
    return [r for r in rows if r.get("source") not in TARGETS]


def port_pair(pair: dict) -> dict:
    e500 = OUTPUTS / pair["e500"]
    e200 = OUTPUTS / pair["e200"]
    etypes = pair["etypes"]

    # --- select Excel_200 clusters for the target sources (+ etype filter)
    src_clusters = [
        c for c in read_jsonl(e200 / "clusters.jsonl")
        if c.get("source") in TARGETS
        and (etypes is None or cluster_etype(c) in etypes)
    ]
    ported_cids = {c.get("cluster_id") for c in src_clusters}

    # --- inconsistency column-cluster drafts for the target sources
    src_drafts = [
        d for d in read_jsonl(e200 / "column_cluster_drafts.jsonl")
        if d.get("source") in TARGETS
    ]

    # --- latest label per target, restricted to what we actually ported
    latest: dict[str, dict] = {}
    for rec in read_jsonl(e200 / "labels.jsonl"):
        latest[rec.get("target_id")] = rec
    src_labels = []
    for tid, rec in latest.items():
        tgt = rec.get("target")
        if tgt == "cluster" and tid in ported_cids:
            src_labels.append(rec)
        elif tgt in ("column_cluster", "cell") and rec.get("source") in TARGETS:
            src_labels.append(rec)

    # --- write clusters.jsonl + clusters.json (strip-then-append)
    cl_jsonl = strip_targets(read_jsonl(e500 / "clusters.jsonl")) + src_clusters
    write_jsonl(e500 / "clusters.jsonl", cl_jsonl)
    cj = e500 / "clusters.json"
    if cj.exists():
        arr = json.loads(cj.read_text() or "[]")
        arr = strip_targets(arr) + src_clusters
        cj.write_text(json.dumps(arr, ensure_ascii=False, indent=2))

    # --- labels.jsonl (strip-then-append)
    lab = strip_targets(read_jsonl(e500 / "labels.jsonl")) + src_labels
    write_jsonl(e500 / "labels.jsonl", lab)

    # --- inconsistency drafts
    if src_drafts:
        dpath = e500 / "column_cluster_drafts.jsonl"
        drafts = strip_targets(read_jsonl(dpath)) + src_drafts
        write_jsonl(dpath, drafts)

    # --- summary.json: ensure CSV paths present; refresh counts
    sp = e500 / "summary.json"
    summary = json.loads(sp.read_text())
    cfg = summary.setdefault("config", {})
    for key_owner in (cfg, summary):
        paths = key_owner.get("csv_paths")
        if isinstance(paths, list):
            have = {Path(p).name for p in paths}
            for name in TARGET_SOURCES:
                if name not in have:
                    paths.append(str(CORPUS_DEST[name]))
    n_paths = len(cfg.get("csv_paths") or [])
    if "clusters_total" in summary and summary["clusters_total"] is not None:
        summary["clusters_total"] = len(cl_jsonl)
    if "source_count" in summary:
        summary["source_count"] = n_paths
    if "n_sources" in summary:
        summary["n_sources"] = n_paths
    sp.write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    # --- report TP delta for the target sources
    tp = sum(1 for r in src_labels if r.get("status") == "tp")
    return {
        "name": pair["name"],
        "clusters_ported": len(src_clusters),
        "drafts_ported": len(src_drafts),
        "labels_ported": len(src_labels),
        "tp_ported": tp,
    }


def main() -> None:
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    print(f"backup dir: {BACKUP_DIR.relative_to(ROOT)}")
    for pair in PAIRS:
        backup_run(pair["e500"])
    added = copy_csvs_into_corpus()
    print(f"CSVs copied into corpus: {added or '(already present)'}")
    print()
    for pair in PAIRS:
        r = port_pair(pair)
        print(
            f"  {r['name']:14s} clusters={r['clusters_ported']:3d} "
            f"drafts={r['drafts_ported']:3d} labels={r['labels_ported']:3d} "
            f"TP={r['tp_ported']:3d}"
        )
    print("\ndone.")


if __name__ == "__main__":
    main()
