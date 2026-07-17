#!/usr/bin/env python3
"""Merge a *finished* detection run's candidates INTO an existing (already
labeled) run directory so the new sources appear for labeling in the workbench
— without disturbing any existing labels/exclusions.

Use case: the mt100 full-coverage batch writes ``outputs/mt100_<type>/`` for
each error type. Rather than labeling those as separate runs, we splice their
clusters into the corresponding excel_500 REGISTRY run (which already has the
labeling context set up), e.g.::

    python scripts/merge_run_into.py outputs/mt100_typo outputs/20260630T012247Z_model_a60d81db

What it merges (append / additive):
  * clusters.json  (+ clusters.jsonl)   <- what the workbench displays
  * issues.json    (+ issues.jsonl)     <- evidence / detail panels
  * chunk_reports.json (+ .jsonl)       <- provenance (if present in both)
  * column_cluster_drafts.jsonl         <- inconsistency column-board drafts
  * summary.json   -> config.csv_paths extended (source resolution) + additive
                      counters (total_issues, clusters_total, *_by_source, ...)

What it NEVER touches: labels.jsonl, exclusions.jsonl, column_clusters.jsonl,
row_verdicts.jsonl, benchmark*.jsonl, raw_responses/, runs/.

Safety:
  * Aborts if the two runs share any source *basename* or any cluster_id
    (a basename clash would silently merge unrelated files). Override w/ --force.
  * Detects an already-merged state (src sources already present) and aborts.
  * Backs up every file it rewrites to ``<target>/.merge_backups/<utc>/`` first.

The live workbench auto-reloads (RunCache.is_stale watches summary/clusters/
issues mtimes) — no server restart needed.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = PROJECT_ROOT / "outputs"

# Files rewritten by the merge (backed up first). (json_array, jsonl) pairs.
_ARRAY_FILES = ["clusters", "issues", "chunk_reports"]
# Optional JSONL-only artifacts whose records are source-keyed. The source
# basename overlap guard above makes appending these safe for disjoint runs.
_JSONL_ONLY_FILES = ["column_cluster_drafts"]

# summary.json scalar counters that are additive over disjoint sources.
_SCALAR_COUNTERS = [
    "chunks", "llm_calls", "chunks_with_errors", "chunks_with_parse_errors",
    "total_issues", "clusters_total", "clusters_multi_type", "clusters_single_type",
    "clusters_synthesised", "clusters_auto_synthesised", "clusters_synthesis_failed",
    "clusters_likely_false_positives",
]

# summary.json dict counters {key -> int}, additive over disjoint sources.
_DICT_COUNTERS = [
    "issues_by_category", "issues_by_error_type", "issues_by_error_type_category",
    "clusters_by_primary_error_type", "clusters_by_confidence",
    "issues_by_source", "clusters_by_source",
]


def _resolve_run(arg: str) -> Path:
    p = Path(arg)
    if p.is_dir():
        return p.resolve()
    cand = OUTPUTS / arg
    if cand.is_dir():
        return cand.resolve()
    sys.exit(f"error: run dir not found: {arg!r} (looked at {p} and {cand})")


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _source_basenames(summary: Dict[str, Any]) -> List[str]:
    cfg = summary.get("config") or {}
    return [Path(p).name for p in (cfg.get("csv_paths") or [])]


def _add_dicts(target: Dict[str, Any], src: Dict[str, Any]) -> None:
    """In-place additive merge of {key -> number} counters."""
    for k, v in (src or {}).items():
        if isinstance(v, (int, float)) and isinstance(target.get(k, 0), (int, float)):
            target[k] = target.get(k, 0) + v
        else:
            target.setdefault(k, v)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", help="finished detection run (e.g. outputs/mt100_typo)")
    ap.add_argument("target", help="existing labeled run to merge INTO")
    ap.add_argument("--dry-run", action="store_true", help="report only; write nothing")
    ap.add_argument("--force", action="store_true",
                    help="proceed despite source/cluster-id overlap (dangerous)")
    args = ap.parse_args()

    src_dir = _resolve_run(args.src)
    tgt_dir = _resolve_run(args.target)
    if src_dir == tgt_dir:
        sys.exit("error: src and target are the same directory")

    src_summary = _read_json(src_dir / "summary.json")
    tgt_summary = _read_json(tgt_dir / "summary.json")

    if not (src_dir / "clusters.json").exists():
        sys.exit(f"error: {src_dir}/clusters.json missing — run not finished?")

    src_bn = _source_basenames(src_summary)
    tgt_bn = set(_source_basenames(tgt_summary))
    overlap_bn = sorted(set(src_bn) & tgt_bn)

    src_clusters: List[Dict[str, Any]] = _read_json(src_dir / "clusters.json")
    tgt_clusters: List[Dict[str, Any]] = _read_json(tgt_dir / "clusters.json")
    src_cids = {c.get("cluster_id") for c in src_clusters}
    tgt_cids = {c.get("cluster_id") for c in tgt_clusters}
    overlap_cid = sorted(x for x in (src_cids & tgt_cids) if x)

    print(f"src   : {src_dir}  ({len(src_bn)} sources, {len(src_clusters)} clusters)")
    print(f"target: {tgt_dir}  ({len(tgt_bn)} sources, {len(tgt_clusters)} clusters)")

    if overlap_bn:
        msg = (f"source-basename overlap ({len(overlap_bn)}): {overlap_bn[:5]}"
               f"{' ...' if len(overlap_bn) > 5 else ''} — looks ALREADY MERGED "
               f"or a name clash.")
        if not args.force:
            sys.exit(f"ABORT: {msg}\n       (re-run with --force to override)")
        print(f"WARNING (--force): {msg}")
    if overlap_cid:
        msg = f"cluster_id overlap ({len(overlap_cid)}): {overlap_cid[:5]}"
        if not args.force:
            sys.exit(f"ABORT: {msg}\n       (re-run with --force to override)")
        print(f"WARNING (--force): {msg}")

    if args.dry_run:
        print("\n[dry-run] would merge:")
        for stem in _ARRAY_FILES:
            sp = src_dir / f"{stem}.json"
            if sp.exists():
                n = len(_read_json(sp))
                print(f"  {stem}: +{n}")
        for stem in _JSONL_ONLY_FILES:
            sp = src_dir / f"{stem}.jsonl"
            tp = tgt_dir / f"{stem}.jsonl"
            if sp.exists() and tp.exists():
                n = sum(1 for line in sp.open(encoding="utf-8") if line.strip())
                print(f"  {stem}: +{n}")
        print(f"  csv_paths: +{len(src_bn)}  -> {len(tgt_bn) + len(src_bn)}")
        print(f"  total_issues: +{src_summary.get('total_issues', 0)}")
        print(f"  clusters_total: +{src_summary.get('clusters_total', 0)}")
        print("[dry-run] nothing written.")
        return 0

    # ---- backup everything we are about to rewrite -----------------------
    ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_dir = tgt_dir / ".merge_backups" / ts
    backup_dir.mkdir(parents=True, exist_ok=True)
    to_backup = ["summary.json"]
    for stem in _ARRAY_FILES:
        to_backup += [f"{stem}.json", f"{stem}.jsonl"]
    to_backup += [f"{stem}.jsonl" for stem in _JSONL_ONLY_FILES]
    for name in to_backup:
        fp = tgt_dir / name
        if fp.exists():
            shutil.copy2(fp, backup_dir / name)
    print(f"\nbacked up {len(list(backup_dir.iterdir()))} file(s) -> {backup_dir}")

    # ---- merge array files (json array + jsonl append) -------------------
    for stem in _ARRAY_FILES:
        sj = src_dir / f"{stem}.json"
        tj = tgt_dir / f"{stem}.json"
        if not (sj.exists() and tj.exists()):
            print(f"  skip {stem}: missing in src or target")
            continue
        merged = _read_json(tj) + _read_json(sj)
        with tj.open("w", encoding="utf-8") as fh:
            json.dump(merged, fh, ensure_ascii=False)
        # jsonl: byte-append src lines (guaranteeing a newline boundary)
        sl, tl = src_dir / f"{stem}.jsonl", tgt_dir / f"{stem}.jsonl"
        if sl.exists() and tl.exists():
            with tl.open("a", encoding="utf-8") as out:
                if tl.stat().st_size and not tl.read_text(encoding="utf-8").endswith("\n"):
                    out.write("\n")
                with sl.open("r", encoding="utf-8") as inp:
                    for line in inp:
                        line = line.rstrip("\n")
                        if line:
                            out.write(line + "\n")
        print(f"  {stem}: +{len(_read_json(sj))}  -> {len(merged)}")

    # ---- merge source-keyed JSONL-only artifacts -------------------------
    for stem in _JSONL_ONLY_FILES:
        sl = src_dir / f"{stem}.jsonl"
        tl = tgt_dir / f"{stem}.jsonl"
        if not (sl.exists() and tl.exists()):
            print(f"  skip {stem}: missing in src or target")
            continue
        added = 0
        with tl.open("a", encoding="utf-8") as out:
            if tl.stat().st_size and not tl.read_text(encoding="utf-8").endswith("\n"):
                out.write("\n")
            with sl.open("r", encoding="utf-8") as inp:
                for line in inp:
                    line = line.rstrip("\n")
                    if line:
                        out.write(line + "\n")
                        added += 1
        print(f"  {stem}: +{added}")

    # ---- merge summary.json ---------------------------------------------
    tgt_cfg = tgt_summary.setdefault("config", {})
    src_cfg = src_summary.get("config") or {}

    # csv_paths: extend (dedup by basename, keep order)
    existing = set(_source_basenames(tgt_summary))
    added_paths = [p for p in (src_cfg.get("csv_paths") or [])
                   if Path(p).name not in existing]
    tgt_cfg["csv_paths"] = list(tgt_cfg.get("csv_paths") or []) + added_paths
    tgt_cfg["source_count"] = len(tgt_cfg["csv_paths"])
    # Relation-discovery summaries also keep a top-level copy.
    if isinstance(tgt_summary.get("csv_paths"), list):
        top_existing = {Path(p).name for p in tgt_summary["csv_paths"]}
        tgt_summary["csv_paths"] += [
            p for p in (src_summary.get("csv_paths") or src_cfg.get("csv_paths") or [])
            if Path(p).name not in top_existing
        ]
    if isinstance(tgt_summary.get("n_sources"), int):
        tgt_summary["n_sources"] = len(tgt_cfg["csv_paths"])
    # config.sources (list of basenames) if present
    if isinstance(tgt_cfg.get("sources"), list):
        add_srcs = [s for s in (src_cfg.get("sources") or []) if s not in set(tgt_cfg["sources"])]
        tgt_cfg["sources"] = tgt_cfg["sources"] + add_srcs

    for k in _SCALAR_COUNTERS:
        if k in src_summary or k in tgt_summary:
            tgt_summary[k] = (tgt_summary.get(k, 0) or 0) + (src_summary.get(k, 0) or 0)
    for k in _DICT_COUNTERS:
        merged = dict(tgt_summary.get(k) or {})
        _add_dicts(merged, src_summary.get(k) or {})
        tgt_summary[k] = merged
    # top-level source_count = distinct sources that produced clusters/issues
    distinct = set(tgt_summary.get("clusters_by_source") or {}) | \
        set(tgt_summary.get("issues_by_source") or {})
    if distinct:
        tgt_summary["source_count"] = len(distinct)

    with (tgt_dir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(tgt_summary, fh, ensure_ascii=False, indent=2)
    print(f"  summary.json: csv_paths +{len(added_paths)} "
          f"(-> {tgt_cfg['source_count']}), counters merged")

    print(f"\nDONE. {len(added_paths)} new source(s) now labelable in {tgt_dir.name}.")
    print("The workbench auto-reloads; refresh the run in the UI.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
