#!/usr/bin/env python3
"""Adapter for upstream HoloClean over one REIN benchmark table.

The upstream clone in ``holoclean_repo`` is unmodified.  Rules are mined from
``dirty.csv`` only by reusing ``external/adr/discover_rules.py``; HoloClean then
repairs the dirty table, and changed cells are emitted as detections.
"""

from __future__ import print_function

import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from collections import namedtuple
from pathlib import Path

import pandas as pd

HOLOCLEAN_DIR = Path(__file__).resolve().parent
REPO_ROOT = HOLOCLEAN_DIR.parents[1]
UPSTREAM_ROOT = HOLOCLEAN_DIR / "holoclean_repo"
PINNED_COMMIT_PATH = HOLOCLEAN_DIR / "PINNED_COMMIT.txt"
MAMBA_ENV = HOLOCLEAN_DIR / "mamba_env"
PYTHON = MAMBA_ENV / "bin" / "python"
ADR_DISCOVER = REPO_ROOT / "external" / "adr" / "discover_rules.py"
ADR_PYTHON = REPO_ROOT / "external" / "adr" / "mamba_env" / "bin" / "python"
CACHE_ROOT = HOLOCLEAN_DIR / "cache" / "v8"
DETECTOR = "holoclean"

MethodResult = namedtuple(
    "MethodResult", ["detector", "dataset", "cells", "repaired_path", "rules", "elapsed", "error"]
)


def pinned_commit():
    if PINNED_COMMIT_PATH.exists():
        return PINNED_COMMIT_PATH.read_text(encoding="utf-8").strip()
    return subprocess.check_output(
        ["git", "-C", str(UPSTREAM_ROOT), "rev-parse", "HEAD"], universal_newlines=True
    ).strip()


def read_csv(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False, low_memory=False).fillna("")


def _diff_cells(dirty_path, repaired_path):
    dirty = read_csv(dirty_path)
    repaired = read_csv(repaired_path)
    rows = min(len(dirty), len(repaired))
    cols = min(dirty.shape[1], repaired.shape[1])
    cells = set()
    for i in range(rows):
        for j in range(cols):
            if str(repaired.iat[i, j]) != str(dirty.iat[i, j]):
                cells.add((int(i), int(j)))
    return cells


def _cache_dir(slug):
    return CACHE_ROOT / ("%s__%s" % (slug, pinned_commit()))


def prepare_inputs(ds_dir, work_dir):
    dirty_src = ds_dir / "dirty.csv"
    input_dir = work_dir / "input"
    rules_dir = work_dir / "rules"
    input_dir.mkdir(parents=True, exist_ok=True)
    rules_dir.mkdir(parents=True, exist_ok=True)

    dirty = read_csv(dirty_src)
    original_columns = [str(c) for c in dirty.columns]
    safe_columns = ["C%d" % i for i in range(len(original_columns))]
    sanitized = dirty.copy()
    sanitized.columns = safe_columns
    sanitized_path = input_dir / "dirty_sanitized.csv"
    sanitized.to_csv(str(sanitized_path), index=False)

    dc_path = rules_dir / "dc_rules_holoclean.txt"
    metadata = rules_dir / "discovered_rules.json"
    py = ADR_PYTHON if ADR_PYTHON.exists() else Path(sys.executable)
    cmd = [
        str(py),
        str(ADR_DISCOVER),
        str(sanitized_path),
        "--holoclean-dc",
        str(dc_path),
        "--metadata",
        str(metadata),
        "--horizon-fd",
        str(rules_dir / "unused_horizon_fd.txt"),
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError("rule discovery failed rc=%s: %s" % (proc.returncode, proc.stderr[-1000:]))
    rules = []
    if metadata.exists():
        rules = json.loads(metadata.read_text(encoding="utf-8"))
    (work_dir / "columns.json").write_text(
        json.dumps({"original": original_columns, "safe": safe_columns}, indent=2), encoding="utf-8"
    )
    return {
        "dirty_sanitized": sanitized_path,
        "dc_path": dc_path,
        "rules": len(rules),
        "original_columns": original_columns,
    }


def _write_original_copy(ds_dir, repaired_path):
    repaired_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(str(ds_dir / "dirty.csv"), str(repaired_path))


def run_method(ds_dir, slug, timeout=600, force=False):
    started = time.time()
    work_dir = _cache_dir(slug)
    meta_path = work_dir / "meta.json"
    repaired_path = work_dir / "repaired.csv"
    commit = pinned_commit()

    if not force and meta_path.exists() and repaired_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("upstream_commit") == commit and meta.get("detector") == DETECTOR:
                return MethodResult(
                    DETECTOR,
                    slug,
                    _diff_cells(ds_dir / "dirty.csv", repaired_path),
                    str(repaired_path),
                    int(meta.get("rules", 0)),
                    0.0,
                    meta.get("error"),
                )
        except Exception:
            pass

    if force and work_dir.exists():
        shutil.rmtree(str(work_dir))
    work_dir.mkdir(parents=True, exist_ok=True)

    error = None
    n_rules = 0
    try:
        prepared = prepare_inputs(ds_dir, work_dir)
        n_rules = prepared["rules"]
        if n_rules == 0:
            error = "no DC rules discovered from dirty.csv; emitted zero detections"
            _write_original_copy(ds_dir, repaired_path)
        else:
            worker_out = work_dir / "worker_stdout.log"
            worker_err = work_dir / "worker_stderr.log"
            cmd = [
                str(PYTHON),
                str(Path(__file__).resolve()),
                "_worker",
                str(prepared["dirty_sanitized"]),
                str(prepared["dc_path"]),
                str(repaired_path),
                str(work_dir / "columns.json"),
            ]
            env = os.environ.copy()
            env["TMPDIR"] = str(HOLOCLEAN_DIR / "tmp")
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            env.setdefault("OMP_NUM_THREADS", "1")
            env.setdefault("OPENBLAS_NUM_THREADS", "1")
            with worker_out.open("w", encoding="utf-8") as out, worker_err.open(
                "w", encoding="utf-8"
            ) as err:
                proc = subprocess.run(
                    cmd,
                    cwd=str(HOLOCLEAN_DIR),
                    env=env,
                    stdout=out,
                    stderr=err,
                    universal_newlines=True,
                    timeout=timeout,
                    check=False,
                )
            if proc.returncode != 0 or not repaired_path.exists():
                err_txt = worker_err.read_text(encoding="utf-8") if worker_err.exists() else ""
                out_txt = worker_out.read_text(encoding="utf-8") if worker_out.exists() else ""
                error = "holoclean failed rc=%s: %s" % (proc.returncode, (err_txt or out_txt)[-1500:])
                _write_original_copy(ds_dir, repaired_path)
        cells = _diff_cells(ds_dir / "dirty.csv", repaired_path)
    except subprocess.TimeoutExpired:
        error = "TimeoutExpired after %ss" % timeout
        cells = set()
        _write_original_copy(ds_dir, repaired_path)
    except Exception as exc:
        error = "%s: %s" % (type(exc).__name__, exc)
        cells = set()
        if (ds_dir / "dirty.csv").exists():
            _write_original_copy(ds_dir, repaired_path)

    elapsed = time.time() - started
    meta_path.write_text(
        json.dumps(
            {
                "detector": DETECTOR,
                "dataset": slug,
                "upstream_commit": commit,
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
    return MethodResult(DETECTOR, slug, cells, str(repaired_path), n_rules, elapsed, error)


def _worker(dirty_sanitized, dc_path, repaired_path, columns_json):
    sys.path.insert(0, str(UPSTREAM_ROOT))
    import holoclean  # noqa: E402
    from detect import NullDetector, ViolationDetector  # noqa: E402
    from repair.featurize import (  # noqa: E402
        ConstraintFeaturizer,
        FreqFeaturizer,
        InitAttrFeaturizer,
        OccurAttrFeaturizer,
    )

    columns = json.loads(Path(columns_json).read_text(encoding="utf-8"))
    hc = holoclean.HoloClean(
        db_name="holo",
        domain_thresh_1=0,
        domain_thresh_2=0,
        weak_label_thresh=0.99,
        max_domain=10000,
        cor_strength=0.6,
        nb_cor_strength=0.8,
        epochs=10,
        weight_decay=0.01,
        learning_rate=0.001,
        threads=1,
        batch_size=1,
        verbose=False,
        timeout=3 * 60000,
        feature_norm=False,
        weight_norm=False,
        print_fw=False,
    ).session
    table_name = "hc_%s" % str(abs(hash(str(dirty_sanitized))))[:12]
    hc.load_data(table_name, str(dirty_sanitized))
    hc.load_dcs(str(dc_path))
    hc.ds.set_constraints(hc.get_dcs())
    hc.detect_errors([NullDetector(), ViolationDetector()])
    hc.setup_domain()
    hc.repair_errors(
        [InitAttrFeaturizer(), OccurAttrFeaturizer(), FreqFeaturizer(), ConstraintFeaturizer()]
    )
    repaired = hc.ds.repaired_data.df.copy()
    if "_tid_" in repaired.columns:
        repaired = repaired.drop(columns=["_tid_"])
    repaired = repaired.iloc[:, : len(columns["original"])]
    repaired.columns = columns["original"]
    Path(repaired_path).parent.mkdir(parents=True, exist_ok=True)
    repaired.to_csv(str(repaired_path), index=False)
    print("wrote %s shape=%s" % (repaired_path, repaired.shape))
    return 0


def main(argv):
    if len(argv) >= 2 and argv[1] == "_worker":
        try:
            return _worker(argv[2], argv[3], argv[4], argv[5])
        except Exception:
            traceback.print_exc()
            return 1
    print("adapter.py is used by run_holoclean.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
