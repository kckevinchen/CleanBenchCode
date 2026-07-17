"""CleanAgent phase 1 — LLM column-type annotation (run in the MAIN venv).

Replicates CleanAgent's ``col_annotator`` agent programmatically: for each
materialized dataset it sends CleanAgent's exact ``COL_ANNOTATOR_SYSTEM_MESSAGE``
(+ ``CANDIDATE_COLUMN_TYPES``) to the LLM and parses the ``columnName: class``
result into one of the five standardizable types (email/phone/date/country/IP).

This phase needs the repo's Azure ``LLMClient`` (Python 3.12 main venv); the
``dataprep`` standardization runs separately in ``detect.py`` (Python 3.10),
decoupled through the ``annotations.json`` this script writes.

Output: ``annotations.json`` =
    {dataset: {"source": ..., "columns": {colname: cleaner_key|null}}}

Examples (repo root, main venv, after ``az login``):
    python external/cleanagent/annotate.py
    python external/cleanagent/annotate.py --model gpt-5.4 --limit 5
    python external/cleanagent/annotate.py --heuristic     # no LLM (dataprep parse-rate)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# REIN harness helpers (paths, REIN-faithful loader).
_REIN_DIR = Path(__file__).resolve().parents[1] / "rein"
sys.path.insert(0, str(_REIN_DIR))
import common as C  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prompts as P  # noqa: E402

# Repo root for src.core.llm_client import.
sys.path.insert(0, str(C.REPO_ROOT))

ANNOTATION_RE = re.compile(r"^[\s*>-]*(.+?)\s*[:：]\s*\*{0,2}([^*]+?)\*{0,2}\s*$")


def parse_annotations(text: str, columns: list[str]) -> dict[str, str | None]:
    """Parse ``columnName: class`` lines into {column: cleaner_key|None}."""
    by_lower = {c.lower(): c for c in columns}
    out: dict[str, str | None] = {c: None for c in columns}
    for line in text.splitlines():
        m = ANNOTATION_RE.match(line)
        if not m:
            continue
        raw_col, raw_cls = m.group(1).strip(), m.group(2).strip()
        col = by_lower.get(raw_col.lower())
        if col is None:
            for lc, orig in by_lower.items():
                if lc and (lc in raw_col.lower() or raw_col.lower() in lc):
                    col = orig
                    break
        if col is None:
            continue
        out[col] = P.normalize_type(raw_cls)
    return out


def annotate_llm(df, columns, client, model: str) -> dict[str, str | None]:
    sys_msg = P.COL_ANNOTATOR_SYSTEM_MESSAGE.format(
        candidate_column_types=P.CANDIDATE_COLUMN_TYPES,
        df=df.head().to_markdown(),
    )
    user = (
        "Annotate every column with exactly one of: "
        f"{P.CANDIDATE_COLUMN_TYPES}. Columns: {', '.join(columns)}.\n"
        "Reply only with lines 'columnName: class'."
    )
    resp = client.query(model, user, system_prompt=sys_msg, reasoning_effort="low")
    if not resp.ok:
        return {c: None for c in columns}
    return parse_annotations(resp.text, columns)


def annotate_heuristic(df, columns) -> dict[str, str | None]:
    """LLM-free fallback: assign a type if dataprep parses >=60% of a column.

    Lets the baseline run without API access; mirrors what standardization would
    keep. Requires the py3.10 dataprep venv (so only used when run there).
    """
    import warnings

    warnings.filterwarnings("ignore")
    from dataprep.clean import (  # type: ignore
        clean_country, clean_date, clean_email, clean_ip, clean_phone,
    )

    fns = {"email": clean_email, "phone": clean_phone, "date": clean_date,
           "country": clean_country, "ip": clean_ip}
    out: dict[str, str | None] = {}
    for col in columns:
        ser = df[[col]].copy()
        nonempty = (ser[col].astype(str).str.strip() != "").sum()
        best, best_rate = None, 0.0
        if nonempty:
            for key, fn in fns.items():
                try:
                    res = fn(ser.copy(), col, errors="coerce")
                except Exception:
                    continue
                clean_col = [c for c in res.columns if c.endswith("_clean")]
                if not clean_col:
                    continue
                parsed = res[clean_col[0]].notna().sum()
                rate = parsed / nonempty
                if rate > best_rate:
                    best, best_rate = key, rate
        out[col] = best if best_rate >= 0.6 else None
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets-dir", default=str(C.DATASETS_DIR))
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "annotations.json"))
    ap.add_argument("--model", default="gpt-4o",
                    help="LLM deployment (default gpt-4o, matching the CleanAgent paper)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=10, help="parallel LLM calls")
    ap.add_argument("--heuristic", action="store_true",
                    help="LLM-free dataprep parse-rate annotation (run in py3.10 venv)")
    args = ap.parse_args()

    ds_root = Path(args.datasets_dir)
    manifest = json.loads((ds_root / "_manifest.json").read_text())["datasets"]
    ds_names = sorted(manifest)
    if args.limit:
        ds_names = ds_names[: args.limit]

    client = None
    if not args.heuristic:
        from src.core.llm_client import LLMClient  # noqa: E402

        client = LLMClient()

    def _annotate_one(ds):
        try:
            df = C.load_rein_df(ds_root / ds / "dirty.csv")
        except Exception as e:  # noqa: BLE001
            print(f"[skip] {ds}: {e}")
            return ds, None
        columns = list(df.columns)
        ann = (annotate_heuristic(df, columns) if args.heuristic
               else annotate_llm(df, columns, client, args.model))
        return ds, {"source": manifest[ds]["source"], "columns": ann}

    result: dict[str, dict] = {}
    type_counts: dict[str, int] = {}
    if args.heuristic or args.concurrency <= 1:
        pairs = (_annotate_one(ds) for ds in ds_names)
    else:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            pairs = list(ex.map(_annotate_one, ds_names))
    done = 0
    for ds, entry in pairs:
        done += 1
        if entry is None:
            continue
        result[ds] = entry
        for v in entry["columns"].values():
            if v:
                type_counts[v] = type_counts.get(v, 0) + 1
        if done % 50 == 0:
            print(f"  ...{done}/{len(ds_names)} datasets annotated")

    Path(args.out).write_text(json.dumps(result, indent=2))
    n_typed_cols = sum(1 for d in result.values() for v in d["columns"].values() if v)
    print("=== annotate summary ===")
    print(f"datasets annotated : {len(result)}")
    print(f"typed columns      : {n_typed_cols}  by type: {type_counts}")
    print(f"annotations        : {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
