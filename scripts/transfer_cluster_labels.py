#!/usr/bin/env python3
"""Carry column-cluster labels from one run to another.

Two runs over the same data may produce different cluster drafts
(different ``k_max``, different prompt variants, different LLM
sampling). Re-labelling from scratch is expensive, so we transfer
verdicts whose cluster -- a SET of surface values -- survives the
re-cluster intact.

Identity rule
-------------
A source cluster matches a target cluster iff:

* same ``source`` (CSV filename),
* same ``column``,
* same value-set (``frozenset`` of the cluster's surface strings).

The cluster's ``local_id`` (``c1``, ``c2``, ...) is positional and
will generally differ between runs -- we re-anchor to the target
run's local_id, so the saved ``target_id`` in the new
``labels.jsonl`` matches what the workbench would mint when it
surfaces the target run's drafts.

Selection rule
--------------
For each source ``target_id`` we keep only the LATEST verdict (the
labelling store's append-order semantics). ``--statuses`` (default
``tp,unsure``) restricts which verdicts get carried over; anything
else is dropped on the floor.

Mismatch reporting
------------------
Clusters whose value-set has no exact match in the target run are
NOT carried over (carrying them would re-anchor a verdict onto a
different cluster shape). The script reports their count and -- with
``--report`` -- writes the dropped target_ids to a sidecar JSONL so
you can re-label only the drift.

Safety
------
The script REFUSES to overwrite an existing target ``labels.jsonl``
unless you pass ``--force``. With ``--dry-run`` nothing is written.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Draft enumeration
# ---------------------------------------------------------------------------


def _enumerate_drafts(
    drafts_path: Path,
) -> Dict[Tuple[str, str], List[Tuple[str, FrozenSet[str]]]]:
    """Return ``(source, column) -> [(local_id, value_set), ...]``.

    Mirrors the workbench's draft-surfacing logic in
    ``src/labeling/app.py::list_column_clusters`` so the
    ``local_id`` we emit lines up with what the UI mints from the
    same drafts file:

    * Drafts with fewer than 2 distinct string values are skipped
      (singletons aren't shown as inconsistency clusters).
    * Surviving drafts are numbered ``c1``, ``c2``, ... in the
      order they appear in ``draft_clusters``.
    """
    out: Dict[Tuple[str, str], List[Tuple[str, FrozenSet[str]]]] = defaultdict(list)
    with drafts_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            src = rec.get("source")
            col = rec.get("column")
            if not isinstance(src, str) or not isinstance(col, str):
                continue
            idx = 0
            for dc in rec.get("draft_clusters") or []:
                values: List[str] = []
                for v in dc.get("values") or []:
                    if isinstance(v, dict):
                        v = v.get("value")
                    if isinstance(v, str) and v:
                        values.append(v)
                if len(values) < 2:
                    continue
                idx += 1
                out[(src, col)].append((f"c{idx}", frozenset(values)))
    return out


def _enumerate_drafts_full(
    drafts_path: Path,
) -> Tuple[List[dict], Dict[Tuple[str, str], int],
           Dict[Tuple[str, str, str], dict]]:
    """Return mutable in-memory representation of a drafts file.

    Returns three handles wired to the SAME underlying dict objects
    so callers can mutate ``records[idx]['draft_clusters']`` and
    have the mutation reflected in both ``records`` (the on-disk
    order) and ``per_cluster`` (per-local-id dict views).

    * ``records`` -- list of parsed rows, in on-disk order. The
      ``draft_clusters`` list inside each row is the real list, not
      a copy, so ``.append`` is live.
    * ``index_by_col`` -- ``(source, column) -> records[idx]`` for
      fast lookup / membership testing. New rows get appended to
      ``records`` and registered here.
    * ``per_cluster`` -- ``(source, column, local_id) -> draft
      dict`` for the live-numbering local-id assigned in
      :func:`_enumerate_drafts`. Useful for reading metadata about
      a specific cluster without re-walking the records list.
    """
    records: List[dict] = []
    index_by_col: Dict[Tuple[str, str], int] = {}
    per_cluster: Dict[Tuple[str, str, str], dict] = {}
    with drafts_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            src = rec.get("source")
            col = rec.get("column")
            if not isinstance(src, str) or not isinstance(col, str):
                continue
            records.append(rec)
            index_by_col[(src, col)] = len(records) - 1
            idx = 0
            for dc in rec.get("draft_clusters") or []:
                values: List[str] = []
                for v in dc.get("values") or []:
                    if isinstance(v, dict):
                        v = v.get("value")
                    if isinstance(v, str) and v:
                        values.append(v)
                if len(values) < 2:
                    continue
                idx += 1
                per_cluster[(src, col, f"c{idx}")] = dc
    return records, index_by_col, per_cluster


def _live_cluster_count(record: dict) -> int:
    """Count of ``draft_clusters`` entries that survive the
    ``len(values) >= 2`` filter -- i.e. would receive a ``c{idx}``
    local id under :func:`_enumerate_drafts`.
    """
    n = 0
    for dc in record.get("draft_clusters") or []:
        values: List[str] = []
        for v in dc.get("values") or []:
            if isinstance(v, dict):
                v = v.get("value")
            if isinstance(v, str) and v:
                values.append(v)
        if len(values) >= 2:
            n += 1
    return n


def _inject_cluster(
    records: List[dict],
    index_by_col: Dict[Tuple[str, str], int],
    source: str,
    column: str,
    draft: dict,
) -> str:
    """Append a copy of ``draft`` onto the target drafts file's
    in-memory representation and return the synthesized ``c{N}``
    local id.

    If the target file has no record for ``(source, column)`` yet,
    a new ``{source, column, draft_clusters: []}`` record is
    appended; otherwise the existing record's ``draft_clusters``
    list is extended in place. Either way the new local id is
    ``c{live_count_after_append}`` -- matching how the labelling
    backend renumbers drafts at surfacing time.

    The injected draft is annotated with
    ``{"injected_from_transfer": True}`` so downstream consumers /
    audits can tell synthetic drafts from LLM-proposed ones.
    """
    inj = dict(draft)
    inj["injected_from_transfer"] = True
    key = (source, column)
    if key not in index_by_col:
        new_rec = {
            "source": source,
            "column": column,
            "runs_completed": 0,
            "stopped_reason": "injected_from_transfer",
            "gt_missing_mass": 0.0,
            "chao1_missed": 0.0,
            "draft_clusters": [inj],
        }
        records.append(new_rec)
        index_by_col[key] = len(records) - 1
        return "c1"
    rec = records[index_by_col[key]]
    if not isinstance(rec.get("draft_clusters"), list):
        rec["draft_clusters"] = []
    rec["draft_clusters"].append(inj)
    return f"c{_live_cluster_count(rec)}"


def _write_drafts_atomic(drafts_path: Path, records: List[dict]) -> None:
    """Atomic rewrite of the drafts JSONL via tmp-file + rename.

    Tmp file lives next to the target so the final ``rename`` is
    a same-filesystem operation (atomic on POSIX). Caller is
    responsible for taking the ``.bak_before_inject`` snapshot
    BEFORE invoking this helper.
    """
    tmp = drafts_path.with_suffix(drafts_path.suffix + ".tmp")
    with tmp.open("w") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    os.replace(tmp, drafts_path)


# ---------------------------------------------------------------------------
# Label reading
# ---------------------------------------------------------------------------


def _latest_labels(labels_path: Path) -> Dict[str, dict]:
    """Latest-line-wins reduction keyed by ``(target, target_id)``."""
    latest: Dict[Tuple[str, str], dict] = {}
    with labels_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            key = (rec.get("target"), rec.get("target_id"))
            latest[key] = rec
    # Only column_cluster targets are in scope for this script;
    # cell-level labels can't survive a re-cluster anyway since
    # they're row-anchored, not cluster-anchored.
    return {tid: r for (tgt, tid), r in latest.items()
            if tgt == "column_cluster" and isinstance(tid, str)}


def _parse_colclust_target_id(tid: str) -> Optional[Tuple[str, str, str]]:
    """Inverse of :func:`column_cluster_target_id`.

    Returns ``(source, column, local_id)`` or ``None`` if the id
    doesn't have the expected ``colclust\\x00...`` shape.
    """
    parts = tid.split("\x00")
    if len(parts) != 4 or parts[0] != "colclust":
        return None
    return parts[1], parts[2], parts[3]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src-run", required=True, type=Path,
                    help="Path to the source run directory.")
    ap.add_argument("--tgt-run", required=True, type=Path,
                    help="Path to the target run directory.")
    ap.add_argument("--statuses", default="tp,unsure",
                    help="Comma-separated list of statuses to carry "
                         "over (default: tp,unsure). Use 'all' to "
                         "carry every status (tp, unsure, fp, skip).")
    ap.add_argument("--annotator", default=None,
                    help="Override annotator on the transferred "
                         "labels. Defaults to the source label's "
                         "own annotator.")
    ap.add_argument("--note-prefix", default="[transferred] ",
                    help="String prepended to each transferred "
                         "label's note. Set to empty to disable.")
    ap.add_argument("--report", type=Path, default=None,
                    help="Optional JSONL path: write one line per "
                         "DROPPED source label (no value-set match) "
                         "so you can re-label only the drift.")
    ap.add_argument("--force", action="store_true",
                    help="Allow appending to a non-empty target "
                         "labels.jsonl.")
    ap.add_argument("--create-missing", action="store_true",
                    help="For source labels whose value-set has no "
                         "match in the target's drafts, inject the "
                         "source draft into the target's "
                         "column_cluster_drafts.jsonl and carry the "
                         "label using the newly-minted target "
                         "local_id. The target drafts file is "
                         "backed up to "
                         "``column_cluster_drafts.jsonl"
                         ".bak_before_inject`` before the rewrite.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Don't write; just report counts.")
    args = ap.parse_args(argv)

    src_run: Path = args.src_run
    tgt_run: Path = args.tgt_run
    if not src_run.is_dir():
        print(f"error: src-run not found: {src_run}", file=sys.stderr)
        return 2
    if not tgt_run.is_dir():
        print(f"error: tgt-run not found: {tgt_run}", file=sys.stderr)
        return 2

    src_drafts_path = src_run / "column_cluster_drafts.jsonl"
    tgt_drafts_path = tgt_run / "column_cluster_drafts.jsonl"
    src_labels_path = src_run / "labels.jsonl"
    tgt_labels_path = tgt_run / "labels.jsonl"
    for p in (src_drafts_path, tgt_drafts_path, src_labels_path):
        if not p.exists():
            print(f"error: missing required file: {p}", file=sys.stderr)
            return 2

    if args.statuses.strip().lower() == "all":
        statuses = {"tp", "unsure", "fp", "skip"}
    else:
        statuses = {s.strip().lower() for s in args.statuses.split(",")
                    if s.strip()}
    if not statuses:
        print("error: --statuses produced an empty set", file=sys.stderr)
        return 2

    src_by_col = _enumerate_drafts(src_drafts_path)
    # When ``--create-missing`` is on we need the full draft dicts
    # for the SOURCE side (to copy the metadata-rich entry into the
    # target) AND a mutable in-memory view of the TARGET side (so
    # we can append new draft_clusters and rewrite atomically).
    src_full: Dict[Tuple[str, str, str], dict] = {}
    if args.create_missing:
        _, _, src_full = _enumerate_drafts_full(src_drafts_path)
        tgt_records, tgt_index_by_col, _ = _enumerate_drafts_full(
            tgt_drafts_path,
        )
    else:
        tgt_records = []
        tgt_index_by_col = {}

    tgt_by_col = _enumerate_drafts(tgt_drafts_path)

    # Source: (source, column, local_id) -> value_set
    src_lookup: Dict[Tuple[str, str, str], FrozenSet[str]] = {}
    for (src, col), lst in src_by_col.items():
        for lid, vs in lst:
            src_lookup[(src, col, lid)] = vs

    # Target: (source, column, value_set) -> [local_id, ...]
    tgt_lookup: Dict[Tuple[str, str, FrozenSet[str]],
                     List[str]] = defaultdict(list)
    for (src, col), lst in tgt_by_col.items():
        for lid, vs in lst:
            tgt_lookup[(src, col, vs)].append(lid)

    latest = _latest_labels(src_labels_path)
    candidates = [r for r in latest.values()
                  if (r.get("status") or "").lower() in statuses]

    # Sort for deterministic output ordering.
    candidates.sort(key=lambda r: (
        r.get("source") or "",
        r.get("column") or "",
        r.get("target_id") or "",
    ))

    transferred: List[dict] = []
    injected: List[dict] = []
    dropped: List[dict] = []
    ambiguous: List[dict] = []
    bad_id: List[dict] = []
    src_cluster_missing: List[dict] = []

    now = datetime.now(timezone.utc).isoformat(timespec="microseconds")

    for src_label in candidates:
        tid = src_label.get("target_id") or ""
        parsed = _parse_colclust_target_id(tid)
        if parsed is None:
            bad_id.append(src_label)
            continue
        src_csv, col, src_lid = parsed
        vs = src_lookup.get((src_csv, col, src_lid))
        if vs is None:
            # The source label points at a cluster that no longer
            # exists in the source's drafts (e.g. clustering was
            # re-run mid-labelling). Without the source value-set
            # we can't safely match a target cluster.
            src_cluster_missing.append(src_label)
            continue
        tgt_lids = tgt_lookup.get((src_csv, col, vs)) or []
        if not tgt_lids and args.create_missing:
            # No target cluster for this value-set: inject the
            # source's draft into the target drafts so the labelling
            # UI surfaces a cluster the new label can anchor to.
            # The source draft dict carries kind / canonical /
            # target_style / agreement, all preserved verbatim.
            src_draft = src_full.get((src_csv, col, src_lid))
            if src_draft is None:
                dropped.append(src_label)
                continue
            new_lid = _inject_cluster(
                tgt_records, tgt_index_by_col, src_csv, col, src_draft,
            )
            # Keep tgt_by_col / tgt_lookup in sync so subsequent
            # source labels pointing at the SAME value-set re-anchor
            # cleanly without a second injection.
            tgt_by_col[(src_csv, col)].append((new_lid, vs))
            tgt_lookup[(src_csv, col, vs)].append(new_lid)
            tgt_lids = [new_lid]
            injected.append({
                "source": src_csv,
                "column": col,
                "new_local_id": new_lid,
                "n_values": len(vs),
            })
        if not tgt_lids:
            dropped.append(src_label)
            continue
        if len(tgt_lids) > 1:
            # Two target clusters share the same value-set -- the
            # target's clustering pass split (or duplicated) the
            # column in a way exact-set matching can't disambiguate.
            # We refuse rather than guessing.
            ambiguous.append(src_label)
            continue
        new_lid = tgt_lids[0]
        new_tid = f"colclust\x00{src_csv}\x00{col}\x00{new_lid}"
        note = src_label.get("note") or ""
        # Idempotent note tagging. The source label may itself be a
        # transferred label (note already starts with the prefix's
        # core token, with or without trailing whitespace) -- if so,
        # we leave the note alone rather than producing a chain
        # like ``[transferred] [transferred] ...``.
        if args.note_prefix:
            prefix_token = args.note_prefix.rstrip()
            if prefix_token and not note.startswith(prefix_token):
                note = f"{args.note_prefix}{note}".rstrip()
        annotator = args.annotator or src_label.get("annotator") or ""
        new_label = {
            "target": "column_cluster",
            "target_id": new_tid,
            "source": src_csv,
            "row": None,
            "column": col,
            "columns": src_label.get("columns") or [],
            "status": src_label.get("status"),
            "corrected_value": src_label.get("corrected_value"),
            "corrected_values": src_label.get("corrected_values") or {},
            "note": note,
            "annotator": annotator,
            "ts": now,
        }
        # Preserve confirmed_canonical only if it still appears in
        # the (identical) value-set -- exact-set match guarantees
        # this, but we check defensively.
        cc = src_label.get("confirmed_canonical")
        if isinstance(cc, str) and cc in vs:
            new_label["confirmed_canonical"] = cc
        transferred.append(new_label)

    # ---------------------------------------------------------- report

    total = len(candidates)
    print(f"source run:  {src_run}")
    print(f"target run:  {tgt_run}")
    print(f"statuses:    {sorted(statuses)}")
    print(f"create-missing: {args.create_missing}")
    print(f"candidates (latest, in scope): {total}")
    print(f"  -> transferred:        {len(transferred)}")
    if injected:
        print(f"     of which injected: {len(injected)} "
              f"(new target clusters created)")
    print(f"  -> dropped (no match): {len(dropped)}")
    if ambiguous:
        print(f"  -> ambiguous (>=2 target candidates): {len(ambiguous)}")
    if src_cluster_missing:
        print(f"  -> src cluster vanished: {len(src_cluster_missing)}")
    if bad_id:
        print(f"  -> malformed target_id: {len(bad_id)}")

    if args.dry_run:
        print("\n(dry run: nothing written)")
        return 0

    # Existing labels guard
    if tgt_labels_path.exists() and tgt_labels_path.stat().st_size > 0:
        if not args.force:
            print(
                f"\nerror: {tgt_labels_path} already has content; "
                f"re-run with --force to append.",
                file=sys.stderr,
            )
            return 1

    # Rewrite the target drafts file IFF we injected anything. The
    # original is preserved as ``*.bak_before_inject`` (single
    # snapshot; do not clobber an existing backup so repeated
    # transfers can still recover the pristine version).
    if injected:
        bak = tgt_drafts_path.with_suffix(
            tgt_drafts_path.suffix + ".bak_before_inject",
        )
        if not bak.exists():
            bak.write_bytes(tgt_drafts_path.read_bytes())
            print(f"backed up target drafts to {bak}")
        else:
            print(f"backup already exists at {bak} (kept)")
        _write_drafts_atomic(tgt_drafts_path, tgt_records)
        print(
            f"rewrote {tgt_drafts_path} with "
            f"{len(injected)} injected cluster(s)"
        )

    if transferred:
        tgt_labels_path.parent.mkdir(parents=True, exist_ok=True)
        with tgt_labels_path.open("a") as fh:
            for lbl in transferred:
                fh.write(json.dumps(lbl, ensure_ascii=False) + "\n")
        print(f"\nappended {len(transferred)} labels to {tgt_labels_path}")
    else:
        print("\nno labels to write")

    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("w") as fh:
            for lbl in dropped:
                fh.write(json.dumps({
                    "reason": "no_target_value_set_match",
                    **lbl,
                }, ensure_ascii=False) + "\n")
            for lbl in ambiguous:
                fh.write(json.dumps({
                    "reason": "ambiguous_target_value_set",
                    **lbl,
                }, ensure_ascii=False) + "\n")
            for lbl in src_cluster_missing:
                fh.write(json.dumps({
                    "reason": "src_cluster_missing",
                    **lbl,
                }, ensure_ascii=False) + "\n")
            for lbl in bad_id:
                fh.write(json.dumps({
                    "reason": "malformed_target_id",
                    **lbl,
                }, ensure_ascii=False) + "\n")
        print(f"wrote drift report to {args.report}")

    return 0


if __name__ == "__main__":
    sys.exit(_main())
