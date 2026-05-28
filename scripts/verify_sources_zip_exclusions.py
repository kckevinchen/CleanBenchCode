"""Verify ``benchmark/sources_<run>.zip`` honours every active UI exclusion.

Cross-references the live ``/api/runs/{run_id}/exclusions`` view against
the on-disk zip:

* scope=csv     -> the source's expected arcname must NOT appear in the zip.
* scope=rows    -> the zipped CSV's row count must equal
                   ``original_rows - len(drop_rows)``.
* scope=columns -> none of the dropped columns may appear in the zipped
                   file's header.
"""
from __future__ import annotations

import io
import json
import os
import sys
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

import pandas as pd

RUN_ID = "20260512T213238Z_gpt5-family"
ZIP_PATH = Path(f"benchmark/sources_{RUN_ID}.zip")
API = f"http://127.0.0.1:8000/api/runs/{RUN_ID}"


def _fetch(path: str):
    with urllib.request.urlopen(f"{API}{path}") as r:
        return json.loads(r.read())


def _common_prefix(paths: list[Path]) -> Path | None:
    if len(paths) < 2:
        return None
    try:
        cp = Path(os.path.commonpath([str(p) for p in paths]))
    except ValueError:
        return None
    if cp.is_dir() and not any(p == cp for p in paths):
        return cp
    return None


def main() -> int:
    # 1) Live exclusion state
    actives = _fetch("/exclusions")["items"]
    csv_excl = {e["source"] for e in actives if e["scope"] == "csv"}
    rows_excl: dict[str, set[int]] = defaultdict(set)
    cols_excl: dict[str, set[str]] = defaultdict(set)
    for e in actives:
        if e["scope"] == "rows":
            for r in e.get("rows") or []:
                try:
                    rows_excl[e["source"]].add(int(r))
                except (TypeError, ValueError):
                    pass
        elif e["scope"] == "columns":
            for c in e.get("columns") or []:
                if c is not None:
                    cols_excl[e["source"]].add(str(c))

    # 2) Sources + paths
    run = _fetch("")
    src_path = {s["name"]: Path(s["path"]) for s in run["sources"]}

    # 3) Arcname map (mirrors export_sources)
    included = sorted(
        (name, p) for name, p in src_path.items() if name not in csv_excl
    )
    try:
        resolved = [p.resolve() for _, p in included]
    except OSError:
        resolved = [Path(p) for _, p in included]
    cp = _common_prefix(resolved)
    arcname = {}
    for (name, _), rp in zip(included, resolved):
        if cp is not None:
            try:
                rel = rp.relative_to(cp).as_posix()
                if rel and not rel.startswith(".."):
                    arcname[name] = rel
                    continue
            except (OSError, ValueError):
                pass
        arcname[name] = name

    # 4) Inspect the zip
    if not ZIP_PATH.exists():
        print(f"FAIL: zip not found: {ZIP_PATH}")
        return 2
    print(f"Zip               : {ZIP_PATH}")
    print(f"Size              : {ZIP_PATH.stat().st_size:,} bytes")
    print(f"Common prefix     : {cp}")
    print(f"Total sources     : {len(src_path)}")
    print(f"Excluded CSVs     : {len(csv_excl)}")
    print(f"Row-excl sources  : {len(rows_excl)}  (positional indices)")
    print(f"Col-excl sources  : {len(cols_excl)}")
    print()

    with zipfile.ZipFile(ZIP_PATH) as zf:
        names_in_zip = set(zf.namelist())
        # 4a) scope=csv -> arcname absent
        miss = []
        leaked = []
        for src in sorted(csv_excl):
            arc = arcname.get(src, src)  # may not be in arcname (was excluded)
            # Recompute arcname as if it were included, just to know the
            # path we DON'T want to see. Use the same prefix.
            try:
                rp = src_path[src].resolve()
            except OSError:
                rp = src_path[src]
            arc_candidate = (
                rp.relative_to(cp).as_posix()
                if cp is not None and str(rp).startswith(str(cp))
                else src
            )
            present = (
                arc_candidate in names_in_zip
                or src in names_in_zip
                or any(n.endswith("/" + Path(arc_candidate).name) for n in names_in_zip if n.endswith(Path(arc_candidate).name))
            )
            # Stricter: any zip entry whose basename equals the CSV name
            # AND whose path matches arc_candidate is a leak.
            basename = Path(arc_candidate).name
            siblings_with_same_basename = [
                n for n in names_in_zip if Path(n).name == basename
            ]
            if siblings_with_same_basename:
                # Compare against exact expected arcname
                if arc_candidate in names_in_zip:
                    leaked.append((src, arc_candidate))
                else:
                    # Different file with the same basename — note but
                    # don't flag as a leak (different workbook).
                    pass
            else:
                pass

        print(f"[scope=csv]  excluded files                                : {len(csv_excl)}")
        print(f"             leaked entries in zip (basename + path match): {len(leaked)}")
        for src, arc in leaked:
            print(f"             LEAK: {src} -> {arc}")

        # 4b) scope=rows -> count check
        row_fails = []
        row_oks = 0
        for src, drop in rows_excl.items():
            arc = arcname.get(src)
            if arc is None or arc not in names_in_zip:
                row_fails.append((src, "arcname missing from zip", arc))
                continue
            with zf.open(arc) as fh:
                data = fh.read()
            sanitised_rows = len(pd.read_csv(io.BytesIO(data)))
            # Load the source frame the same way the labelling backend does
            try:
                src_frame_rows = len(pd.read_csv(src_path[src]))
            except Exception as ex:
                row_fails.append((src, f"cannot load source: {ex}", arc))
                continue
            expected = src_frame_rows - len(drop)
            if sanitised_rows == expected:
                row_oks += 1
            else:
                row_fails.append(
                    (src, f"rows={sanitised_rows} expected={expected} (orig={src_frame_rows} drop={len(drop)})", arc)
                )
        print()
        print(f"[scope=rows] checked                                       : {len(rows_excl)}")
        print(f"             ok (row count matches orig - drop)            : {row_oks}")
        print(f"             mismatched                                    : {len(row_fails)}")
        for src, reason, arc in row_fails:
            print(f"             FAIL: {src} ({arc}): {reason}")

        # 4c) scope=columns -> header check
        # If a source is ALSO CSV-excluded, its file is (correctly) absent
        # from the zip — the CSV exclusion supersedes column exclusions.
        col_fails = []
        col_superseded = 0
        col_oks = 0
        for src, drop_cols in cols_excl.items():
            if src in csv_excl:
                col_superseded += 1
                continue
            arc = arcname.get(src)
            if arc is None or arc not in names_in_zip:
                col_fails.append(
                    (src, sorted(drop_cols), ["<arcname missing from zip>"], arc)
                )
                continue
            with zf.open(arc) as fh:
                head = pd.read_csv(fh, nrows=0)
            header_cols = set(head.columns)
            leaked_cols = header_cols & drop_cols
            if leaked_cols:
                col_fails.append(
                    (src, sorted(drop_cols), sorted(leaked_cols), arc)
                )
            else:
                col_oks += 1
        print()
        print(f"[scope=cols] checked                                       : {len(cols_excl)}")
        print(f"             superseded by scope=csv (file absent)         : {col_superseded}")
        print(f"             ok (no dropped col in header)                 : {col_oks}")
        print(f"             leaked                                        : {len(col_fails)}")
        for entry in col_fails:
            src, drop_cols, leaked_cols, arc = entry
            print(
                f"             LEAK: {src} ({arc}): excluded={drop_cols} "
                f"still present={leaked_cols}"
            )

    # 5) Cross-check: every entry in zip belongs to an included source
    expected_arcs = set(arcname.values())
    zip_set = set(names_in_zip)
    unexpected = zip_set - expected_arcs
    missing = expected_arcs - zip_set
    print()
    print(f"[zip]        entries                                          : {len(zip_set)}")
    print(f"             expected included sources                        : {len(expected_arcs)}")
    print(f"             entries with no matching included source         : {len(unexpected)}")
    if unexpected:
        for n in sorted(unexpected)[:20]:
            print(f"             EXTRA: {n}")
    print(f"             included sources missing from zip                : {len(missing)}")
    if missing:
        for n in sorted(missing)[:20]:
            print(f"             MISSING: {n}")

    # Overall verdict
    csv_leaks = len(leaked)
    fails = csv_leaks + len(row_fails) + len(col_fails) + len(unexpected) + len(missing)
    print()
    print("=" * 60)
    print(f"VERDICT: {'PASS' if fails == 0 else f'FAIL ({fails} issues)'}")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
