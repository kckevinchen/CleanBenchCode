"""Label a CSV dataset with an RFT v9 typo model, emitting the standard
detection-run output the labeling workbench reads.

Mirror of ``RFT.typo.v8.scripts.label_dataset_v8`` (same per-column assembly,
same reuse of flatten_issues -> cluster_issues -> build_run_artifacts) but with
a v9 model + PROMPT_V9 (v5 response schema). Defaults to the BEST v9 RFT config,
``v9-r1-e2`` (micro-F1 0.750, §4b), deployed on EU2; override with --deployment
/--endpoint to label with any other v9 deployment. o4-mini RFT models run at
reasoning_effort=medium.

    python -m RFT.typo.v9.scripts.label_dataset_v9 --data-dir data/pbi_1000 [--limit N]
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
from RFT.typo.v9._prompt import PROMPT_V9  # noqa: E402
from RFT.typo.v8.scripts.label_dataset_v8 import (  # noqa: E402  reuse helpers
    _issues_for_csv,
    _parse_typos,
    _textual_columns,
    _values_with_counts,
)
from src.core.aggregator import cluster_issues  # noqa: E402
from src.core.detector import ChunkReport  # noqa: E402
from src.core.report_builder import build_run_artifacts, flatten_issues  # noqa: E402

API_VERSION = "2025-04-01-preview"
# Best v9 RFT config (§4b: micro-F1 0.750), deployed on dmxaz-test-aoai-eu2.
DEFAULT_ENDPOINT = "https://dmxaz-test-aoai-eu2.openai.azure.com/"
DEFAULT_DEPLOYMENT = "o4-mini-v9-r1-e2"


def predict_column(client, rf, cache_dir, source, column, values, deployment, effort):
    key = hashlib.sha256(
        f"{PROMPT_V9}|{deployment}|{effort}|{source}|{column}".encode()
    ).hexdigest()[:16]
    cp = cache_dir / f"{key}.json"
    if cp.exists():
        return _parse_typos(json.loads(cp.read_text()).get("raw"))
    user = t.build_user_message(column=column, source=source, values_with_counts=values)
    raw = None
    try:
        resp = client.chat.completions.create(
            model=deployment,
            messages=[{"role": "developer", "content": PROMPT_V9},
                      {"role": "user", "content": user}],
            response_format=rf, reasoning_effort=effort,
        )
        raw = resp.choices[0].message.content
    except Exception as exc:  # noqa: BLE001
        cp.write_text(json.dumps({"raw": None, "error": f"{type(exc).__name__}: {exc}"}))
        return []
    cp.write_text(json.dumps({"raw": raw}))
    return _parse_typos(raw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/pbi_1000")
    ap.add_argument("--limit", type=int, default=0, help="First N CSVs (0=all).")
    ap.add_argument("--concurrency", type=int, default=40)
    ap.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    ap.add_argument("--deployment", default=DEFAULT_DEPLOYMENT)
    ap.add_argument("--reasoning", default="medium")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    data_dir = ROOT / args.data_dir
    csvs = sorted(data_dir.glob("*.csv")) or sorted(data_dir.rglob("*.csv"))
    if args.limit:
        csvs = csvs[: args.limit]
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tag = f"_{args.tag}" if args.tag else ""
    run_dir = ROOT / "outputs" / f"{ts}_{args.deployment}_{data_dir.name}{tag}"
    cache_dir = run_dir / "v9_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    client = build_client(args.endpoint, API_VERSION)
    rf = t.build_response_format()
    print(f"labeling {len(csvs)} CSVs from {data_dir} with {args.deployment} "
          f"(effort={args.reasoning}) -> {run_dir}", flush=True)

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
                                  _values_with_counts(df, c), args.deployment,
                                  args.reasoning): c for c in cols}
                for f in as_completed(futs):
                    col_typos[futs[f]] = f.result()
        issues = _issues_for_csv(df, col_typos)
        if issues:
            frames[csv.name] = df
        reports.append(ChunkReport(
            source=csv.name, start_row=0, end_row=max(0, len(df) - 1),
            model=args.deployment, prompt_name="typo_v9.PROMPT_V9",
            error_type="typo", error_type_name="Typo",
            error_type_category="Single-Column / Typo", issues=issues,
        ))
        if i % 50 == 0:
            n_iss = sum(len(r.issues) for r in reports)
            print(f"  {i}/{len(csvs)} csvs, {n_iss} issues ({time.time()-t0:.0f}s)", flush=True)

    flat = flatten_issues(reports, frames, include_full_row=True)
    clusters = cluster_issues(flat)
    config = {
        "model": args.deployment, "endpoint": args.endpoint, "mode": "single",
        "reasoning_effort": args.reasoning,
        "error_types": ["typo"], "source_dir": str(data_dir),
        "csv_paths": [str(c) for c in csvs],
        "labeler": f"RFT typo v9 ({args.deployment}), PROMPT_V9, per-column",
    }
    build_run_artifacts(
        reports, run_dir, flat_issues=flat, clusters=clusters, config=config,
        run_metadata={"display_name": f"{data_dir.name} typo — RFT {args.deployment}"},
        html_template_path=None, source_frames=frames,
    )
    (run_dir / "RUN_STATUS.json").write_text(json.dumps(
        {"status": "completed", "updated_at": dt.datetime.now(dt.timezone.utc).isoformat()}))
    print(f"\nDONE: {len(reports)} csvs, {len(flat)} issues, {len(clusters)} clusters -> {run_dir}",
          flush=True)


if __name__ == "__main__":
    main()
