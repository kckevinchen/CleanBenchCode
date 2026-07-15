#!/usr/bin/env python3
"""Adapter for ADR'24 repair methods over one benchmark table.

The upstream repository under ``external/adr/adr_repo`` is never modified.  This
adapter builds a per-table working directory outside the clone, discovers rules
from the dirty CSV only, invokes an upstream method, and interprets repairs as
detection cells (repaired value != dirty value).
"""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import pandas as pd

from discover_rules import discover_fds, write_holoclean_dc, write_horizon_fd

ADR_DIR = Path(__file__).resolve().parent
REPO_ROOT = ADR_DIR.parents[1]
UPSTREAM_ROOT = ADR_DIR / "adr_repo"
PINNED_COMMIT_PATH = ADR_DIR / "PINNED_COMMIT.txt"
MAMBA_ENV = ADR_DIR / "mamba_env"
PYTHON = MAMBA_ENV / "bin" / "python"
CACHE_ROOT = ADR_DIR / "cache" / "v8"
DETECTOR_PREFIX = "adr"


@dataclasses.dataclass
class MethodResult:
    detector: str
    dataset: str
    cells: set[tuple[int, int]]
    repaired_path: str | None
    rules: int
    elapsed: float
    error: str | None = None


def pinned_commit() -> str:
    if PINNED_COMMIT_PATH.exists():
        return PINNED_COMMIT_PATH.read_text(encoding="utf-8").strip()
    return subprocess.check_output(
        ["git", "-C", str(UPSTREAM_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()


def detector_name(method: str) -> str:
    return f"{DETECTOR_PREFIX}_{method.lower()}"


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False, low_memory=False).fillna("")


def _safe_copy_dirty(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


# Every ADR'24 Python repair method shares the same CLI
# (--dirty_path/--clean_path/--rule_path/--task_name/--onlyed/--perfected) and
# writes ./Repaired_res/<out_dir>/<task_prefix>/repaired_*.csv.  They differ only
# in the script path, the output-directory name, and whether they consume a
# functional-dependency file ("fd") or a denial-constraint file ("dc").
_METHOD_SPECS: dict[str, tuple[str, str, str]] = {
    # method          (script relative to adr_repo,           Repaired_res subdir, rule kind)
    "horizon": ("horizon/horizon.py", "horizon", "fd"),
    "relative_trust": ("Relative_Trust/relative.py", "Relative_trust", "fd"),
    "unified": ("Unified/Unified.py", "Unified", "fd"),
    "holistic": ("BigDansing_Holistic/holistic.py", "holistic", "dc"),
    "bigdansing": ("BigDansing_Holistic/bigdansing.py", "bigdansing", "dc"),
    "daisy": ("Daisy/daisy.py", "daisy", "dc"),
}

_TASK_NAME = "adrbench1"
_TASK_PREFIX = "adrbench"


def prepare_inputs(ds_dir: Path, slug: str, work_dir: Path) -> dict:
    """Materialize the upstream-shaped inputs for one table (unsupervised).

    Discovers approximate FDs from the dirty CSV only and writes both the Horizon
    FD file and the HoloClean DC file so any method can be driven from the same
    working directory.  Returns the paths plus per-format rule counts.
    """
    dirty_src = ds_dir / "dirty.csv"
    dirty_named = work_dir / "input" / f"{slug}-inner_outer_error-01.csv"
    _safe_copy_dirty(dirty_src, dirty_named)

    df = read_csv(dirty_named)
    fds = discover_fds(df)
    rules_dir = work_dir / "rules"
    horizon_fd = rules_dir / "dc_rules-validate-fd-horizon.txt"
    holoclean_dc = rules_dir / "dc_rules_holoclean.txt"
    write_horizon_fd(fds, horizon_fd)
    write_holoclean_dc(fds, holoclean_dc)
    (rules_dir / "discovered_rules.json").write_text(
        json.dumps([dataclasses.asdict(fd) for fd in fds], indent=2), encoding="utf-8"
    )
    pseudo_clean = work_dir / "input" / f"{slug}-metric_only_clean.csv"
    metric_df = df.copy()
    if metric_df.shape[0] > 0 and metric_df.shape[1] > 0:
        original = str(metric_df.iat[0, 0])
        metric_df.iat[0, 0] = original + "__ADR_METRIC_SENTINEL__"
    metric_df.to_csv(pseudo_clean, index=False)
    n_fd = len([fd for fd in fds if len(fd.lhs) == 1])
    n_dc = len(list(fds))
    return {
        "dirty_named": dirty_named,
        "pseudo_clean": pseudo_clean,
        "fd_path": horizon_fd,
        "dc_path": holoclean_dc,
        "n_fd": n_fd,
        "n_dc": n_dc,
    }


def _diff_cells(dirty_path: Path, repaired_path: Path) -> set[tuple[int, int]]:
    dirty = read_csv(dirty_path)
    repaired = read_csv(repaired_path)
    rows = min(len(dirty), len(repaired))
    cols = min(dirty.shape[1], repaired.shape[1])
    cells: set[tuple[int, int]] = set()
    for i in range(rows):
        for j in range(cols):
            if str(repaired.iat[i, j]) != str(dirty.iat[i, j]):
                cells.add((int(i), int(j)))
    return cells


def run_method(method: str, ds_dir: Path, slug: str, *, timeout: int = 120, force: bool = False) -> MethodResult:
    """Run one ADR'24 repair method on a single table and return detection cells.

    Generic over every entry in ``_METHOD_SPECS``: all share the same CLI and
    write ``Repaired_res/<out_dir>/<task>/repaired_*.csv``.  Detection = cells the
    repaired table changes vs. the dirty input.  Faithful/unsupervised: rules come
    only from the dirty CSV, and ``perfected=0`` means repairs use dirty+rules.
    """
    method = method.lower()
    if method not in _METHOD_SPECS:
        raise ValueError(f"Unknown ADR method {method!r}; known: {sorted(_METHOD_SPECS)}")
    script_rel, out_dir_name, rule_kind = _METHOD_SPECS[method]

    started = time.time()
    detector = detector_name(method)
    work_dir = CACHE_ROOT / method / slug
    meta_path = work_dir / "meta.json"
    repaired_path = work_dir / "repaired.csv"
    commit = pinned_commit()

    if not force and meta_path.exists() and repaired_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("upstream_commit") == commit and meta.get("detector") == detector:
                return MethodResult(
                    detector,
                    slug,
                    _diff_cells(ds_dir / "dirty.csv", repaired_path),
                    str(repaired_path),
                    int(meta.get("rules", 0)),
                    0.0,
                    meta.get("error"),
                )
        except Exception:
            pass

    if work_dir.exists() and force:
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    out_base = work_dir / "Repaired_res" / out_dir_name / _TASK_PREFIX
    (work_dir / "Exp_result" / out_dir_name / _TASK_PREFIX).mkdir(parents=True, exist_ok=True)
    out_base.mkdir(parents=True, exist_ok=True)

    n_rules = 0
    try:
        prepared = prepare_inputs(ds_dir, slug, work_dir)
        dirty_named = prepared["dirty_named"]
        rule_path = prepared["fd_path"] if rule_kind == "fd" else prepared["dc_path"]
        n_rules = prepared["n_fd"] if rule_kind == "fd" else prepared["n_dc"]
        if n_rules == 0:
            shutil.copyfile(dirty_named, repaired_path)
            error = f"no {rule_kind.upper()} rules discovered; emitted zero detections"
        else:
            cmd = [
                str(PYTHON),
                str(UPSTREAM_ROOT / script_rel),
                "--task_name",
                _TASK_NAME,
                "--rule_path",
                str(rule_path),
                "--onlyed",
                "0",
                "--perfected",
                "0",
                "--dirty_path",
                str(dirty_named),
                # clean_path is only read for metrics; a dirty-derived one-cell
                # sentinel avoids upstream divide-by-zero on clean tables while
                # keeping inference unsupervised (perfected=0 => dirty+rules only).
                "--clean_path",
                str(prepared["pseudo_clean"]),
            ]
            env = os.environ.copy()
            env.setdefault("OMP_NUM_THREADS", "1")
            env.setdefault("OPENBLAS_NUM_THREADS", "1")
            proc = subprocess.run(
                cmd,
                cwd=work_dir,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
                check=False,
            )
            # Output filename suffix varies by method (check_string vs path slice);
            # glob is robust to both conventions.
            produced = sorted(out_base.glob("repaired_*.csv"))
            # ignore any "perfect_repaired_*" variant (perfected mode; unused here)
            produced = [p for p in produced if not p.name.startswith("perfect_repaired_")]
            if proc.returncode != 0 or not produced:
                shutil.copyfile(dirty_named, repaired_path)
                error = (
                    f"{method} failed rc={proc.returncode}: "
                    f"{(proc.stderr or proc.stdout)[-1000:]}"
                )
            else:
                shutil.copyfile(produced[-1], repaired_path)
                error = None
        cells = _diff_cells(ds_dir / "dirty.csv", repaired_path)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        cells = set()
        if (ds_dir / "dirty.csv").exists():
            shutil.copyfile(ds_dir / "dirty.csv", repaired_path)

    elapsed = time.time() - started
    meta_path.write_text(
        json.dumps(
            {
                "detector": detector,
                "dataset": slug,
                "upstream_commit": commit,
                "method": method,
                "rules": n_rules,
                "elapsed": elapsed,
                "predictions": len(cells),
                "repaired_path": str(repaired_path),
                "error": error,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return MethodResult(detector, slug, cells, str(repaired_path), n_rules, elapsed, error)
