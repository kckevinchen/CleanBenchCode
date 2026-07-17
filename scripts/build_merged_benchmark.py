"""Build a merged cross-type benchmark folder in one command.

Thin orchestrator that composes the three canonical exporters into a single
output directory (e.g. ``benchmark/merged_v3``) -- the exact workflow used to
refresh the benchmark after re-labeling a pool in the workbench:

  1. ``experiments.export_cell_benchmark``   -> ``benchmark_cell.json`` / ``.csv``
     (+ ``llm_explanation_cache.jsonl``)         per-CELL types: ``typo``
                                                 (split into ``typo-formatting`` /
                                                 ``typo-spelling``), ``inconsistency``,
                                                 ``numeric_outlier``, ``semantic_incompatible``
  2. ``experiments.export_row_benchmark``    -> ``benchmark_row.json`` / ``.csv``
     (+ ``llm_row_cache.jsonl``)                 per-ROW types: ``fd_violation``,
                                                 ``row_calculation_inconsistency``,
                                                 ``string_calculation_inconsistency``,
                                                 ``row_duplication``
  3. ``experiments.export_merged_benchmark`` -> ``ignored.jsonl``  (the scoped
                                                 skip + exclusion immunity set; the
                                                 step's ``benchmark.json`` / ``per_type/``
                                                 are written to a scratch dir and
                                                 discarded -- only ``ignored.jsonl`` is
                                                 copied into the output)

All three re-derive their labels LIVE from the runs in the shared ``REGISTRY``
(``experiments.export_merged_benchmark.REGISTRY``), so re-labeling ``pbi_500`` /
``excel_500`` in the workbench and re-running this script is all it takes to
rebuild the benchmark -- no ``REGISTRY`` edit is needed while the run dirs are
unchanged. ``az login`` must be active for any new LLM calls.

Caches: with ``--from <prev_dir>`` the two LLM caches are copied forward before
the run. Both are CONTENT-keyed (sha256 over source/row/column/error_types/
original/corrected), so only cells whose label or fix actually changed trigger a
new LLM call -- every unchanged explanation is reused for free. ``--from`` also
enables a ``prev -> new`` diff (added / removed / changed cells, row findings,
ignored records) at the end.

Run::

    # rebuild v3 from v2: reuse caches, regenerate ignored.jsonl, then diff
    python -m scripts.build_merged_benchmark --out benchmark/merged_v3 --from benchmark/merged_v2

    # restrict to one pool / cap new explanation calls / skip the LLM entirely
    python -m scripts.build_merged_benchmark --out benchmark/merged_v3 --from benchmark/merged_v2 --pools excel_500
    python -m scripts.build_merged_benchmark --out /tmp/bench --no-llm

    # structure only (skip the ignored.jsonl regeneration step)
    python -m scripts.build_merged_benchmark --out /tmp/bench --no-ignored
"""

from __future__ import annotations

import argparse
import collections
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# The two content-keyed LLM caches carried forward from the previous version so
# only changed cells re-call the model.
CACHE_FILES = ("llm_explanation_cache.jsonl", "llm_row_cache.jsonl")

CellKey = tuple[str, int | None, str]


# ---------------------------------------------------------------------------
# subprocess driver
# ---------------------------------------------------------------------------

def _run_module(module: str, args: list[str], log_path: Path, stream_stdout: bool) -> None:
    """Run ``python -m <module> <args>`` from the project root.

    ``stderr`` (the benign stale-label / duplicate-header noise the exporters
    emit) is always redirected to ``log_path``. ``stdout`` is inherited when
    ``stream_stdout`` so the user sees the tqdm progress bar + summary; for the
    ignored-set step it is sent to the log too (only the file is wanted). A
    non-zero exit aborts the build after printing the log tail.
    """
    cmd = [sys.executable, "-m", module, *args]
    print(f"\n$ {' '.join(cmd)}", flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as errlog:
        proc = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=None if stream_stdout else errlog,
            stderr=errlog, check=False,
        )
    if proc.returncode != 0:
        tail = "".join(log_path.read_text(encoding="utf-8").splitlines(keepends=True)[-25:])
        raise SystemExit(
            f"\n{module} failed (exit {proc.returncode}). Last log lines "
            f"({log_path}):\n{tail}"
        )


def _seed_caches(prev: Path, out: Path) -> list[str]:
    """Copy the LLM caches from ``prev`` into ``out`` (best effort)."""
    seeded: list[str] = []
    for name in CACHE_FILES:
        src = prev / name
        if src.exists():
            shutil.copy2(src, out / name)
            seeded.append(name)
    return seeded


# ---------------------------------------------------------------------------
# diff (prev -> new)
# ---------------------------------------------------------------------------

def _load(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def _cell_key(r: dict[str, Any]) -> CellKey:
    return (r["source"], r["row"], r["column"])


def _short_src(s: str) -> str:
    return s[-26:]


def _list_cells(tag: str, keys: list[CellKey], m: dict[CellKey, dict[str, Any]],
                cap: int = 25) -> None:
    if not keys:
        return
    print(f"  {tag} ({len(keys)}):")
    ordered = sorted(keys, key=lambda k: (k[0], k[1] if k[1] is not None else -1, k[2]))
    for k in ordered[:cap]:
        r = m[k]
        print(f"    {tag[0]} {_short_src(k[0])} r{k[1]} {k[2]!r} "
              f"[{r['status']}/{','.join(r['error_types'])}] "
              f"{r['original_value']!r} -> {r.get('corrected_value')!r}")
    if len(keys) > cap:
        print(f"    ... and {len(keys) - cap} more")


def _diff_cell(prev: Path, out: Path) -> None:
    a = _load(prev / "benchmark_cell.json")
    b = _load(out / "benchmark_cell.json")
    if not a and not b:
        return
    ka = {_cell_key(r): r for r in a}
    kb = {_cell_key(r): r for r in b}
    added = [k for k in kb if k not in ka]
    removed = [k for k in ka if k not in kb]
    status_chg = etype_chg = cv_chg = 0
    for k in ka.keys() & kb.keys():
        x, y = ka[k], kb[k]
        status_chg += x["status"] != y["status"]
        etype_chg += x["error_types"] != y["error_types"]
        cv_chg += x.get("corrected_value") != y.get("corrected_value")
    sa = dict(collections.Counter(r["status"] for r in a))
    sb = dict(collections.Counter(r["status"] for r in b))
    print(f"benchmark_cell.json: {len(a)} -> {len(b)} ({len(b) - len(a):+d})  "
          f"added {len(added)}  removed {len(removed)}  "
          f"changed[status={status_chg} types={etype_chg} fix={cv_chg}]")
    print(f"  status {sa} -> {sb}")
    _list_cells("added", added, kb)
    _list_cells("removed", removed, ka)


def _diff_row(prev: Path, out: Path) -> None:
    a = _load(prev / "benchmark_row.json")
    b = _load(out / "benchmark_row.json")
    if not a and not b:
        return

    def rk(r: dict[str, Any]) -> tuple[str, str, tuple[int, ...]]:
        return (r["source"], r["error_type"], tuple(sorted(r.get("rows") or [])))

    ka = {rk(r): r for r in a}
    kb = {rk(r): r for r in b}
    added = [k for k in kb if k not in ka]
    removed = [k for k in ka if k not in kb]
    chg = 0
    for k in ka.keys() & kb.keys():
        for f in ("status", "target_row", "target_column", "corrected_value"):
            if ka[k].get(f) != kb[k].get(f):
                chg += 1
                break
    print(f"benchmark_row.json:  {len(a)} -> {len(b)} ({len(b) - len(a):+d})  "
          f"added {len(added)}  removed {len(removed)}  changed {chg}")


def _diff_ignored(prev: Path, out: Path) -> None:
    pa, pb = prev / "ignored.jsonl", out / "ignored.jsonl"
    if not pa.exists() and not pb.exists():
        return
    A = {json.dumps(x, sort_keys=True, ensure_ascii=False) for x in _load(pa)}
    B = {json.dumps(x, sort_keys=True, ensure_ascii=False) for x in _load(pb)}
    print(f"ignored.jsonl:       {len(A)} -> {len(B)}  "
          f"added {len(B - A)}  removed {len(A - B)}")


def _diff(prev: Path, out: Path) -> None:
    print(f"\n=== diff {prev.name} -> {out.name} ===")
    _diff_cell(prev, out)
    _diff_row(prev, out)
    _diff_ignored(prev, out)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True,
                    help="output benchmark dir (e.g. benchmark/merged_v3)")
    ap.add_argument("--from", dest="prev", default=None,
                    help="previous version dir (e.g. benchmark/merged_v2): seeds "
                         "the LLM caches and enables a prev->new diff")
    ap.add_argument("--pools", nargs="*", default=None,
                    help="restrict to these REGISTRY pools (default: all)")
    ap.add_argument("--concurrency", type=int, default=8,
                    help="parallel LLM calls for the cell/row exporters")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap the number of NEW LLM calls (0 = all)")
    ap.add_argument("--no-llm", action="store_true",
                    help="skip new LLM calls (cached explanations still apply)")
    ap.add_argument("--no-ignored", action="store_true",
                    help="skip the ignored.jsonl regeneration step")
    ap.add_argument("--no-diff", action="store_true",
                    help="skip the prev->new diff even when --from is given")
    ap.add_argument("--fresh", action="store_true",
                    help="do not seed caches from --from (force full regeneration)")
    args = ap.parse_args()

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    prev: Path | None = None
    if args.prev:
        prev = Path(args.prev).resolve()
        if prev == out:
            raise SystemExit("--from and --out must differ")
        if not prev.is_dir():
            print(f"WARN: --from {prev} is not a directory; skipping cache seed + diff")
            prev = None

    logs = Path(tempfile.mkdtemp(prefix="benchbuild_"))

    # 1. seed caches so only changed cells re-call the LLM
    if prev and not args.fresh:
        seeded = _seed_caches(prev, out)
        print(f"seeded caches from {prev.name}: {seeded or 'none found'}")

    # forwarded args
    pool_args = (["--pools", *args.pools] if args.pools else [])
    llm_args: list[str] = ["--concurrency", str(args.concurrency)]
    if args.limit:
        llm_args += ["--limit", str(args.limit)]
    if args.no_llm:
        llm_args.append("--no-llm")

    # 2. per-cell benchmark (+ explanation cache)
    _run_module(
        "experiments.export_cell_benchmark",
        ["--out", str(out), *pool_args, *llm_args],
        logs / "cell.stderr.log", stream_stdout=True,
    )

    # 3. per-row benchmark (+ row cache)
    _run_module(
        "experiments.export_row_benchmark",
        ["--out", str(out), *pool_args, *llm_args],
        logs / "row.stderr.log", stream_stdout=True,
    )

    # 4. ignored.jsonl -- regenerate via the canonical merged exporter into a
    #    scratch dir and copy only ignored.jsonl out (drop benchmark.json/per_type)
    if not args.no_ignored:
        with tempfile.TemporaryDirectory(prefix="benchbuild_ignored_") as scratch:
            _run_module(
                "experiments.export_merged_benchmark",
                ["--out", scratch, *pool_args],
                logs / "ignored.log", stream_stdout=False,
            )
            src = Path(scratch) / "ignored.jsonl"
            if src.exists():
                shutil.copy2(src, out / "ignored.jsonl")
                n = sum(1 for _ in (out / "ignored.jsonl").open(encoding="utf-8"))
                print(f"\nignored.jsonl: {n} records "
                      f"(regenerated via export_merged_benchmark)")
            else:
                print("\nWARN: export_merged_benchmark produced no ignored.jsonl")

    # 5. diff prev -> new
    if prev and not args.no_diff:
        _diff(prev, out)

    built = sorted(p.name for p in out.iterdir() if p.is_file())
    print(f"\ndone: built {out}")
    print(f"  files: {built}")
    print(f"  logs : {logs}")


if __name__ == "__main__":
    main()
