"""Label a CSV dataset with GPT-5.5 + one of the two baseline typo prompts.

For every (CSV, textual column) the model is called once via the Responses API
(reasoning_effort=none by default) with the selected prompt as ``instructions``
and the per-column distinct-value listing as ``input``. Output is a standard,
workbench-loadable detection-run directory under ``outputs/``.

    # with-rulebook (default)
    python -m baseline.label_gpt55 --data-dir data/curated_bing21_500_dedup

    # without-rulebook (generalized prompt)
    python -m baseline.label_gpt55 --data-dir data/curated_bing21_500_dedup \
        --prompt without-rulebook

Score the resulting run dir with the shared scorer (not part of baseline/):

    TYPO_BENCH_DIR=benchmark/typo_excel_500 \
        python -m experiments.score_typo_dir_benchmark <run_dir>
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baseline import _task as t  # noqa: E402
from baseline._common import (  # noqa: E402
    build_client,
    chat_rf_to_responses_text,
    extract_responses_output_text,
    issues_for_csv,
    load_csv,
    parse_typos,
    textual_columns,
    values_with_counts,
)
from baseline.prompt import PROMPT_SHA, PROMPTS  # noqa: E402
from src.core.aggregator import cluster_issues  # noqa: E402
from src.core.detector import ChunkReport  # noqa: E402
from src.core.llm_client import configured_endpoint_pool  # noqa: E402
from src.core.report_builder import build_run_artifacts, flatten_issues  # noqa: E402

API_VERSION = "2025-04-01-preview"
_CONFIGURED_ENDPOINTS = configured_endpoint_pool()
DEFAULT_ENDPOINT = _CONFIGURED_ENDPOINTS[0] if _CONFIGURED_ENDPOINTS else ""
DEFAULT_MODEL = "gpt-5.5"


def predict_column(
    client, text_format, cache_dir, source, column, values, *, model, effort, prompt, prompt_sha
):
    key = hashlib.sha256(f"{prompt_sha}|{model}|{effort}|{source}|{column}".encode()).hexdigest()[
        :16
    ]
    cp = cache_dir / f"{key}.json"
    if cp.exists():
        return parse_typos(json.loads(cp.read_text()).get("raw"))
    user = t.build_user_message(column=column, source=source, values_with_counts=values)
    try:
        resp = client.responses.create(
            model=model,
            instructions=prompt,
            input=user,
            text={"format": text_format},
            reasoning={"effort": effort},
        )
        raw = extract_responses_output_text(resp)
    except Exception as exc:
        cp.write_text(json.dumps({"raw": None, "error": f"{type(exc).__name__}: {exc}"}))
        return []
    cp.write_text(json.dumps({"raw": raw}))
    return parse_typos(raw)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data-dir", default="data/curated_bing21_500_dedup")
    ap.add_argument(
        "--prompt",
        choices=sorted(PROMPTS),
        default="with-rulebook",
        help="with-rulebook = clean spec-derived rulebook; "
        "without-rulebook = compact generalized rule.",
    )
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument(
        "--endpoint",
        default=DEFAULT_ENDPOINT,
        required=not bool(DEFAULT_ENDPOINT),
        help="Azure OpenAI endpoint (default: configured environment endpoint).",
    )
    ap.add_argument(
        "--effort", default="none", choices=["none", "minimal", "low", "medium", "high", "xhigh"]
    )
    ap.add_argument("--limit", type=int, default=0, help="Process at most N CSVs.")
    ap.add_argument("--concurrency", type=int, default=12)
    ap.add_argument(
        "--tag", default="", help="Run-id suffix. Default: <prompt-with-underscores>_<effort>."
    )
    args = ap.parse_args()

    prompt = PROMPTS[args.prompt]
    prompt_sha = PROMPT_SHA[args.prompt]
    tag = args.tag or f"{args.prompt.replace('-', '_')}_{args.effort}"

    data_dir = ROOT / args.data_dir
    csvs = sorted(data_dir.glob("*.csv")) or sorted(data_dir.rglob("*.csv"))
    if args.limit:
        csvs = csvs[: args.limit]
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = ROOT / "outputs" / f"{ts}_{args.model}_{data_dir.name}_{tag}"
    cache_dir = run_dir / "gpt55_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "RUN_STATUS.json").write_text(
        json.dumps(
            {"status": "running", "updated_at": dt.datetime.now(dt.timezone.utc).isoformat()}
        )
    )

    client = build_client(args.endpoint, API_VERSION)
    text_format = chat_rf_to_responses_text(t.build_response_format())
    print(
        f"labeling {len(csvs)} CSVs with {args.model} (prompt={args.prompt} "
        f"sha={prompt_sha[:12]}, effort={args.effort}) -> {run_dir}",
        flush=True,
    )

    reports, frames = [], {}
    t0 = time.time()
    for i, csv in enumerate(csvs, 1):
        df = load_csv(csv)
        if df is None:
            print(f"  skip {csv.name}", flush=True)
            continue
        cols = textual_columns(df)
        col_typos = {}
        if cols:
            with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
                futs = {
                    ex.submit(
                        predict_column,
                        client,
                        text_format,
                        cache_dir,
                        csv.name,
                        c,
                        values_with_counts(df, c),
                        model=args.model,
                        effort=args.effort,
                        prompt=prompt,
                        prompt_sha=prompt_sha,
                    ): c
                    for c in cols
                }
                for f in as_completed(futs):
                    col_typos[futs[f]] = f.result()
        issues = issues_for_csv(df, col_typos)
        if issues:
            frames[csv.name] = df
        reports.append(
            ChunkReport(
                source=csv.name,
                start_row=0,
                end_row=max(0, len(df) - 1),
                model=args.model,
                prompt_name=f"baseline.{args.prompt}",
                error_type="typo",
                error_type_name="Typo",
                error_type_category="Single-Column / Typo",
                issues=issues,
            )
        )
        if i % 50 == 0:
            print(
                f"  {i}/{len(csvs)} csvs, {sum(len(r.issues) for r in reports)} issues "
                f"({time.time()-t0:.0f}s)",
                flush=True,
            )

    flat = flatten_issues(reports, frames, include_full_row=True)
    clusters = cluster_issues(flat)
    config = {
        "model": args.model,
        "endpoint": args.endpoint,
        "mode": "single",
        "reasoning_effort": args.effort,
        "error_types": ["typo"],
        "source_dir": str(data_dir),
        "csv_paths": [str(c) for c in csvs],
        "prompt": args.prompt,
        "prompt_sha256": prompt_sha,
        "labeler": f"{args.model} + baseline {args.prompt} prompt, per-column, "
        f"effort={args.effort}",
    }
    build_run_artifacts(
        reports,
        run_dir,
        flat_issues=flat,
        clusters=clusters,
        config=config,
        run_metadata={
            "display_name": f"{data_dir.name} typo — {args.model} " f"{args.prompt} ({args.effort})"
        },
        html_template_path=None,
        source_frames=frames,
    )
    (run_dir / "RUN_STATUS.json").write_text(
        json.dumps(
            {"status": "completed", "updated_at": dt.datetime.now(dt.timezone.utc).isoformat()}
        )
    )
    print(
        f"\nDONE: {len(reports)} csvs, {len(flat)} issues, {len(clusters)} clusters "
        f"-> {run_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
