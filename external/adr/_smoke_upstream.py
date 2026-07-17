#!/usr/bin/env python3
"""Signal-integrity smoke: run each ADR'24 method on bundled hospital with the
SHIPPED rule files and confirm it writes a repaired CSV that DIFFERS from dirty.

This isolates upstream-method health from our rule-discovery quality.  Not a
scored run; uses hospital's real clean.csv only for the methods' internal metric
prints (perfected=0 => repair itself is dirty+rules).
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

ADR = Path(__file__).resolve().parent
REPO = ADR / "adr_repo"
PY = ADR / "mamba_env" / "bin" / "python"
HOSP = REPO / "data_with_rules" / "hospital"
DIRTY = HOSP / "noise" / "hospital-inner_outer_error-01.csv"
CLEAN = HOSP / "clean.csv"
FD = HOSP / "dc_rules-validate-fd-horizon.txt"
DC = HOSP / "dc_rules_holoclean.txt"

SPECS = {
    "horizon": ("horizon/horizon.py", "horizon", FD),
    "relative_trust": ("Relative_Trust/relative.py", "Relative_trust", FD),
    "unified": ("Unified/Unified.py", "Unified", FD),
    "holistic": ("BigDansing_Holistic/holistic.py", "holistic", DC),
    "bigdansing": ("BigDansing_Holistic/bigdansing.py", "bigdansing", DC),
    "daisy": ("Daisy/daisy.py", "daisy", DC),
}


def diff_cells(a: Path, b: Path) -> int:
    da = pd.read_csv(a, dtype=str, keep_default_na=False).fillna("")
    db = pd.read_csv(b, dtype=str, keep_default_na=False).fillna("")
    if da.shape != db.shape:
        return -1
    return int((da.values != db.values).sum())


def main() -> int:
    only = sys.argv[1:] or list(SPECS)
    print(f"dirty={DIRTY.name} rows/cols check ...", flush=True)
    print(f"{'method':16} {'rc':>3} {'produced':>9} {'changed':>8}  note")
    print("-" * 70)
    for method in only:
        script_rel, out_dir, rule = SPECS[method]
        work = Path(tempfile.mkdtemp(prefix=f"adrsmoke_{method}_"))
        (work / "Repaired_res" / out_dir / "hospital").mkdir(parents=True, exist_ok=True)
        (work / "Exp_result" / out_dir / "hospital").mkdir(parents=True, exist_ok=True)
        cmd = [
            str(PY), str(REPO / script_rel),
            "--task_name", "hospital1",
            "--rule_path", str(rule),
            "--onlyed", "0", "--perfected", "0",
            "--dirty_path", str(DIRTY),
            "--clean_path", str(CLEAN),
        ]
        try:
            proc = subprocess.run(
                cmd, cwd=work, text=True, capture_output=True, timeout=900, check=False
            )
            rc = proc.returncode
            produced = [
                p for p in (work / "Repaired_res" / out_dir / "hospital").glob("repaired_*.csv")
                if not p.name.startswith("perfect_repaired_")
            ]
            if produced:
                changed = diff_cells(DIRTY, produced[-1])
                note = "OK" if changed > 0 else "wrote-but-no-change"
                print(f"{method:16} {rc:>3} {'yes':>9} {changed:>8}  {note}")
            else:
                tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-1:] or [""]
                print(f"{method:16} {rc:>3} {'NO':>9} {'-':>8}  {tail[0][:40]}")
        except subprocess.TimeoutExpired:
            print(f"{method:16} {'-':>3} {'timeout':>9} {'-':>8}  >300s")
        finally:
            shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
