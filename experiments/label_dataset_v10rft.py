"""Label a CSV dir with a deployed typo RFT model (o4-mini FT) + its trainee prompt.

Runs an RFT model per-column with its trained developer prompt (PROMPT_V9 or
PROMPT_V10) at reasoning_effort (its trained effort), emitting the same
workbench-loadable ``issues.jsonl`` as the gpt-5.5 label runs, so
``experiments.score_typo_dir_benchmark`` scores it directly.

    # v10-r1 (default)
    python -m experiments.label_dataset_v10rft --data-dir data/_curated_bing21_500_flat
    # v9-r1-e3
    python -m experiments.label_dataset_v10rft --model o4-mini-v9-r1-e3 --prompt v9 \
        --endpoint https://sc-rf-m8udnqmj-swedencentral.cognitiveservices.azure.com/ --tag v9r1e3_med
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from RFT._common._rft_common import build_client  # noqa: E402
from RFT.typo.v5 import _task as t  # noqa: E402
from RFT.typo.v5.eval.compare_ft_vs_base import (  # noqa: E402
    _chat_response_format_to_responses_text_format,
    _extract_responses_output_text,
)
from RFT.typo.v9._prompt import PROMPT_V9  # noqa: E402
from RFT.typo.v10._prompt import PROMPT_V10  # noqa: E402
from RFT.typo.v8.scripts.label_dataset_v8 import (  # noqa: E402
    _issues_for_csv,
    _parse_typos,
    _textual_columns,
    _values_with_counts,
)
from src.core.aggregator import cluster_issues  # noqa: E402
from src.core.detector import ChunkReport  # noqa: E402
from src.core.report_builder import build_run_artifacts, flatten_issues  # noqa: E402

API_VERSION = "2025-04-01-preview"
PROMPTS = {"v9": PROMPT_V9, "v10": PROMPT_V10}


def predict_column(client, tf, cache_dir, source, column, values, *, model, effort, prompt):
    phash = hashlib.sha256(prompt.encode()).hexdigest()[:8]
    key = hashlib.sha256(f"{model}|{effort}|{phash}|{source}|{column}".encode()).hexdigest()[:16]
    cp = cache_dir / f"{key}.json"
    if cp.exists():
        return _parse_typos(json.loads(cp.read_text()).get("raw"))
    user = t.build_user_message(column=column, source=source, values_with_counts=values)
    try:
        resp = client.responses.create(
            model=model, instructions=prompt, input=user,
            text={"format": tf}, reasoning={"effort": effort},
        )
        raw = _extract_responses_output_text(resp)
    except Exception as exc:  # noqa: BLE001
        cp.write_text(json.dumps({"raw": None, "error": f"{type(exc).__name__}: {exc}"}))
        return []
    cp.write_text(json.dumps({"raw": raw}))
    return _parse_typos(raw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/_curated_bing21_500_flat")
    ap.add_argument("--model", default="o4-mini-v10-r1")
    ap.add_argument("--endpoint", default="https://dmxaz-test-aoai-eu2.openai.azure.com/")
    ap.add_argument("--prompt", choices=sorted(PROMPTS), default="v10")
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=12)
    ap.add_argument("--tag", default="rft_med")
    args = ap.parse_args()
    prompt = PROMPTS[args.prompt]

    data_dir = ROOT / args.data_dir
    csvs = sorted(data_dir.glob("*.csv")) or sorted(data_dir.rglob("*.csv"))
    if args.limit:
        csvs = csvs[: args.limit]
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = ROOT / "outputs" / f"{ts}_{args.model}_{data_dir.name}_{args.tag}"
    cache_dir = run_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    client = build_client(args.endpoint, API_VERSION)
    tf = _chat_response_format_to_responses_text_format(t.build_response_format())
    print(f"labeling {len(csvs)} CSVs with {args.model} (prompt={args.prompt}, "
          f"effort={args.effort}) -> {run_dir}", flush=True)

    reports, frames = [], {}
    t0 = time.time()
    for i, csv in enumerate(csvs, 1):
        try:
            df = pd.read_csv(csv, dtype=str, keep_default_na=False)
        except Exception as exc:  # noqa: BLE001
            print(f"  skip {csv.name}: {exc}", flush=True)
            continue
        cols = _textual_columns(df)
        col_typos = {}
        if cols:
            with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
                futs = {ex.submit(predict_column, client, tf, cache_dir, csv.name, c,
                                  _values_with_counts(df, c), model=args.model,
                                  effort=args.effort, prompt=prompt): c for c in cols}
                for f in as_completed(futs):
                    col_typos[futs[f]] = f.result()
        issues = _issues_for_csv(df, col_typos)
        if issues:
            frames[csv.name] = df
        reports.append(ChunkReport(
            source=csv.name, start_row=0, end_row=max(0, len(df) - 1),
            model=args.model, prompt_name=f"{args.prompt}.rft",
            error_type="typo", error_type_name="Typo",
            error_type_category="Single-Column / Typo", issues=issues,
        ))
        if i % 50 == 0:
            print(f"  {i}/{len(csvs)} csvs, {sum(len(r.issues) for r in reports)} issues "
                  f"({time.time()-t0:.0f}s)", flush=True)

    flat = flatten_issues(reports, frames, include_full_row=True)
    clusters = cluster_issues(flat)
    config = {
        "model": args.model, "endpoint": args.endpoint, "mode": "single",
        "reasoning_effort": args.effort, "error_types": ["typo"],
        "source_dir": str(data_dir),
        "labeler": f"{args.model} + PROMPT_{args.prompt.upper()}, per-column, effort={args.effort}",
    }
    build_run_artifacts(
        reports, run_dir, flat_issues=flat, clusters=clusters, config=config,
        run_metadata={"display_name": f"{data_dir.name} typo — {args.model} ({args.effort})"},
        html_template_path=None, source_frames=frames,
    )
    (run_dir / "RUN_STATUS.json").write_text(json.dumps(
        {"status": "completed", "updated_at": dt.datetime.now(dt.timezone.utc).isoformat()}))
    print(f"\nDONE: {len(reports)} csvs, {len(flat)} issues, {len(clusters)} clusters -> {run_dir}", flush=True)


if __name__ == "__main__":
    main()
