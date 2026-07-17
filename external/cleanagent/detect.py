"""CleanAgent phase 2 — dataprep standardize + diff detector (run in PY3.10 venv).

Replicates CleanAgent's ``coder`` step: for each column annotated as one of the
five standardizable types, run the matching ``dataprep.clean.clean_<type>``
function and flag cells that standardization **changes or cannot parse** -- i.e.
cells whose raw value is non-standard. That is the cell-level error signal a
standardization tool exposes (mostly value *inconsistency* / bad format).

Reads ``annotations.json`` from ``annotate.py`` and the materialized REIN
datasets; writes predictions JSONL into the shared results dir so the existing
``external/rein/score.py`` (+ neutral set) grades it next to ``raha``.

Modes (``--mode``):
* ``changed``     flag cells where normalized standardized != original OR unparseable (default)
* ``unparseable`` flag only cells dataprep set to NaN (higher precision)

Run (py3.10 dataprep venv; datasets already materialized by the REIN adapter):
    external/cleanagent/venv/bin/python external/cleanagent/detect.py
    external/cleanagent/venv/bin/python external/cleanagent/detect.py --mode unparseable
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

_REIN_DIR = Path(__file__).resolve().parents[1] / "rein"
sys.path.insert(0, str(_REIN_DIR))
import common as C  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prompts as P  # noqa: E402

_WS = re.compile(r"\s+")


def _norm(v) -> str:
    """Loose normalization for the original-vs-standardized comparison."""
    if v is None:
        return ""
    return _WS.sub(" ", str(v).strip()).casefold()


def _cleaner(name: str):
    import dataprep.clean as dc  # type: ignore

    return getattr(dc, name)


def detect_column(df, col: str, cleaner_key: str, mode: str) -> set[int]:
    """Return row indices in ``col`` that standardization changes / fails."""
    fn = _cleaner(P.TYPE_TO_CLEANER[cleaner_key])
    work = df[[col]].copy()
    try:
        res = fn(work, col, errors="coerce")
    except Exception:
        return set()
    clean_cols = [c for c in res.columns if c.endswith("_clean")]
    if not clean_cols:
        return set()
    cleaned = res[clean_cols[0]]
    flagged: set[int] = set()
    for i in range(len(df)):
        orig = df.iat[i, df.columns.get_loc(col)]
        if str(orig).strip() == "":
            continue  # empty -> missing value, not this detector's job
        cval = cleaned.iat[i] if i < len(cleaned) else None
        is_nan = cval is None or (isinstance(cval, float) and cval != cval)
        if is_nan:
            flagged.add(i)  # unparseable in both modes
        elif mode == "changed" and _norm(cval) != _norm(orig):
            flagged.add(i)
    return flagged


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--annotations", default=str(Path(__file__).resolve().parent / "annotations.json"))
    ap.add_argument("--datasets-dir", default=str(C.DATASETS_DIR))
    ap.add_argument("--results-dir", default=str(C.RESULTS_DIR))
    ap.add_argument("--mode", default="changed", choices=["changed", "unparseable"])
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    ann_path = Path(args.annotations)
    if not ann_path.exists():
        raise SystemExit(f"Annotations not found: {ann_path}. Run annotate.py first.")
    annotations = json.loads(ann_path.read_text())
    ds_root = Path(args.datasets_dir)
    ds_names = sorted(annotations)
    if args.limit:
        ds_names = ds_names[: args.limit]

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    tag = f"cleanagent_{args.mode}"
    out_path = results_dir / f"predictions_{tag}.jsonl"

    ran: list[str] = []
    n_pred = 0
    n_typed_cols = 0
    with open(out_path, "w", encoding="utf-8") as fout:
        for k, ds in enumerate(ds_names, 1):
            typed = {c: t for c, t in annotations[ds]["columns"].items() if t}
            try:
                df = C.load_rein_df(ds_root / ds / "dirty.csv")
            except Exception as e:  # noqa: BLE001
                print(f"[skip] {ds}: {e}")
                continue
            ran.append(ds)
            for col, key in typed.items():
                if col not in df.columns:
                    continue
                n_typed_cols += 1
                j = df.columns.get_loc(col)
                for i in detect_column(df, col, key, args.mode):
                    fout.write(json.dumps({"dataset": ds, "detector": tag, "row": i, "col": j}) + "\n")
                    n_pred += 1
            if k % 25 == 0:
                print(f"  ...{k}/{len(ds_names)} datasets")

    ran_obj = {"per_detector": {tag: ran}, "all_loaded": ran}
    Path(str(out_path) + ".ran.json").write_text(json.dumps(ran_obj))

    print("=== cleanagent detect summary ===")
    print(f"datasets processed : {len(ran)}")
    print(f"typed columns used : {n_typed_cols}")
    print(f"cells flagged      : {n_pred}")
    print(f"predictions        : {out_path}")
    print(f"\nScore with:\n  python external/rein/score.py --predictions {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
