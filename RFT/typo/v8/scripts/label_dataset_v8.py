"""Label a CSV dataset with the RFT v8 typo model, emitting the standard
detection-run output the labeling workbench reads.

The v8 FT model (`o4-mini-v8-r1-low`) is a per-COLUMN typo detector trained on
PROMPT_V8 + the v5 typo response schema. We run it that way (one call per text
column, distinct values + counts), turn its typo predictions into the pipeline's
issue dicts, then reuse the real post-detection machinery
(`flatten_issues` -> `cluster_issues` -> `build_run_artifacts`) so the run dir is
byte-shape-identical to a normal pipeline run and the workbench auto-discovers it.

    python -m RFT.typo.v8.scripts.label_dataset_v8 --data-dir data/pbi_1000 [--limit N]
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from RFT._common._rft_common import build_client  # noqa: E402
from RFT.typo.v5 import _task as t  # noqa: E402
from RFT.typo.v8._prompt import PROMPT_V8  # noqa: E402
from src.core.aggregator import cluster_issues  # noqa: E402
from src.core.detector import ChunkReport  # noqa: E402
from src.core.llm_client import configured_endpoint_pool  # noqa: E402
from src.core.report_builder import build_run_artifacts, flatten_issues  # noqa: E402

API_VERSION = "2025-04-01-preview"
_CONFIGURED_ENDPOINTS = configured_endpoint_pool()
ENDPOINT = _CONFIGURED_ENDPOINTS[0] if _CONFIGURED_ENDPOINTS else ""
DEPLOYMENT = "o4-mini-v8-r1-low"
MAX_DISTINCT = 500


def _norm(s) -> str:
    return " ".join(str(s).split())


def _textual_columns(df: pd.DataFrame) -> list[str]:
    """Columns worth sending to a typo detector: >=30% alphabetic, >=2 distinct."""
    out = []
    for c in df.columns:
        vals = [v for v in df[c].astype(str).tolist() if v.strip()][:200]
        if not vals or df[c].nunique(dropna=True) < 2:
            continue
        alpha = sum(1 for v in vals if any(ch.isalpha() for ch in v))
        if alpha / len(vals) >= 0.30:
            out.append(c)
    return out


def _values_with_counts(df: pd.DataFrame, col: str):
    vc = df[col].astype(str).value_counts()
    return [(str(v), int(c)) for v, c in vc.items() if str(v).strip()][:MAX_DISTINCT]


def _parse_typos(raw: str | None) -> list[dict]:
    if not raw:
        return []
    s = raw.strip()
    if s.startswith("```"):
        import re
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```\s*$", "", s)
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        return []
    out = []
    for it in obj.get("typos", []) or []:
        if isinstance(it, dict) and isinstance(it.get("value"), str):
            out.append(it)
    return out


def predict_column(client, rf, cache_dir, source, column, values, effort="medium"):
    key = hashlib.sha256(
        f"{PROMPT_V8}|{DEPLOYMENT}|{effort}|{source}|{column}".encode()
    ).hexdigest()[:16]
    cp = cache_dir / f"{key}.json"
    if cp.exists():
        return _parse_typos(json.loads(cp.read_text()).get("raw"))
    user = t.build_user_message(column=column, source=source, values_with_counts=values)
    raw = None
    try:
        resp = client.chat.completions.create(
            model=DEPLOYMENT,
            messages=[{"role": "developer", "content": PROMPT_V8},
                      {"role": "user", "content": user}],
            response_format=rf, reasoning_effort=effort,
        )
        raw = resp.choices[0].message.content
    except Exception as exc:  # noqa: BLE001
        cp.write_text(json.dumps({"raw": None, "error": f"{type(exc).__name__}: {exc}"}))
        return []
    cp.write_text(json.dumps({"raw": raw}))
    return _parse_typos(raw)


def _issues_for_csv(df: pd.DataFrame, col_typos: dict[str, list[dict]]) -> list[dict]:
    issues = []
    for col, typos in col_typos.items():
        norm_series = df[col].astype(str).map(_norm)
        for typ in typos:
            val = typ["value"]
            rows = [int(i) for i in df.index[norm_series == _norm(val)].tolist()]
            if not rows:  # value not actually in the column -> hallucination, drop
                continue
            corr = typ.get("correction")
            flav = (typ.get("flavour") or "").strip()
            rat = (typ.get("rationale") or "").strip()
            issues.append({
                "row": rows[0], "columns": [col],
                "affected_rows": rows, "affected_rows_total": len(rows),
                "observed": val,
                "explanation": (f"typo:{flav} - {rat}" if flav else rat) or "typo",
                "suggested_fix": corr,
                "corrected_value": {col: corr} if isinstance(corr, str) else None,
            })
    return issues


def main():
    if not ENDPOINT:
        raise SystemExit("configure AZURE_OPENAI_ENDPOINTS before labeling")
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/pbi_1000")
    ap.add_argument("--limit", type=int, default=0, help="First N CSVs (0=all).")
    ap.add_argument("--concurrency", type=int, default=10)
    ap.add_argument("--reasoning", default="medium")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    data_dir = ROOT / args.data_dir
    csvs = sorted(data_dir.glob("*.csv")) or sorted(data_dir.rglob("*.csv"))
    if args.limit:
        csvs = csvs[: args.limit]
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tag = f"_{args.tag}" if args.tag else ""
    run_dir = ROOT / "outputs" / f"{ts}_{DEPLOYMENT}_{data_dir.name}{tag}"
    cache_dir = run_dir / "v8_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    client = build_client(ENDPOINT, API_VERSION)
    rf = t.build_response_format()
    print(f"labeling {len(csvs)} CSVs from {data_dir} with {DEPLOYMENT} -> {run_dir}", flush=True)

    reports, frames = [], {}
    t0 = time.time()
    for i, csv in enumerate(csvs, 1):
        try:
            df = pd.read_csv(csv, dtype=str, keep_default_na=False)
        except Exception as exc:  # noqa: BLE001
            print(f"  skip {csv.name}: {exc}", flush=True)
            continue
        cols = _textual_columns(df)
        col_typos: dict[str, list[dict]] = {}
        if cols:
            with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
                futs = {ex.submit(predict_column, client, rf, cache_dir, csv.name, c,
                                  _values_with_counts(df, c), args.reasoning): c for c in cols}
                for f in as_completed(futs):
                    col_typos[futs[f]] = f.result()
        issues = _issues_for_csv(df, col_typos)
        if issues:
            frames[csv.name] = df
        reports.append(ChunkReport(
            source=csv.name, start_row=0, end_row=max(0, len(df) - 1),
            model=DEPLOYMENT, prompt_name="typo_v8.PROMPT_V8",
            error_type="typo", error_type_name="Typo",
            error_type_category="Single-Column / Typo", issues=issues,
        ))
        if i % 50 == 0:
            n_iss = sum(len(r.issues) for r in reports)
            print(f"  {i}/{len(csvs)} csvs, {n_iss} issues ({time.time()-t0:.0f}s)", flush=True)

    flat = flatten_issues(reports, frames, include_full_row=True)
    clusters = cluster_issues(flat)
    config = {
        "model": DEPLOYMENT, "endpoint": ENDPOINT, "mode": "single",
        "reasoning_effort": args.reasoning,
        "error_types": ["typo"], "source_dir": str(data_dir),
        "csv_paths": [str(c) for c in csvs],
        "labeler": "RFT typo v8 (o4-mini-v8-r1-low), PROMPT_V8, per-column",
    }
    build_run_artifacts(
        reports, run_dir, flat_issues=flat, clusters=clusters, config=config,
        run_metadata={"display_name": f"{data_dir.name} typo — RFT {DEPLOYMENT}"},
        html_template_path=None, source_frames=frames,
    )
    (run_dir / "RUN_STATUS.json").write_text(json.dumps(
        {"status": "completed", "updated_at": dt.datetime.now(dt.timezone.utc).isoformat()}))
    print(f"\nDONE: {len(reports)} csvs, {len(flat)} issues, {len(clusters)} clusters -> {run_dir}", flush=True)


if __name__ == "__main__":
    main()
