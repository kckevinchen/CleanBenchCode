"""Run the inconsistency RULEBOOK prompt over a whole curated CSV dataset.

Unlike ``run_baseline.py`` (which is keyed to the inconsistency benchmark
gold), this driver simply *labels* every auditable column of every CSV in a
dataset directory with the v1 rulebook prompt and dumps the raw findings. No
scoring happens here -- the comparison against the human-labeled reference
runs is a separate downstream step.

Dataset layout (curated_*_dedup):  ``<csv-dir>/<source>/<source>.csv``.
The emitted ``source`` field is the CSV filename (e.g.
``110888756_party_wise_sentiment.csv``) so it lines up 1:1 with the
``source`` key in the reference runs' ``labels.jsonl``.

GPT-5.5 settings mirror the proven ``run_baseline`` path:
``reasoning_effort=none`` ([[feedback_gpt55_reasoning_none]]) and
``temperature=0`` ([[feedback_gpt55_temp0]]); strict ``response_format``.
The endpoint pool round-robins and marks any ``(endpoint, model)`` that
404s as dead so it is skipped for the rest of the run.

Usage::

    python -m RFT.inconsistency.v1.eval.run_on_dataset \
        --csv-dir data/curated_pbi_500_dedup \
        --model gpt-5.5 --reasoning-effort none --temperature 0 \
        --tag pbi500
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from RFT._common._rft_common import build_client  # noqa: E402
from RFT.inconsistency.v1 import (  # noqa: E402
    PROMPT_INCONSISTENCY,
    build_response_format,
    build_user_message,
)
from RFT.inconsistency.v1.eval.run_baseline import (  # noqa: E402
    _auditable_columns,
    _parse_findings,
    _prompt_hash,
)
from src.core.llm_client import configured_endpoint_pool  # noqa: E402

API_VERSION = "2025-04-01-preview"


class EndpointPool:
    """Thread-safe round-robin over clients with 404 dead-marking."""

    def __init__(self, endpoints, api_version):
        self._clients = [(build_client(ep, api_version), ep) for ep in endpoints]
        self._dead: set[str] = set()
        self._i = 0
        self._lock = threading.Lock()

    def live_count(self) -> int:
        with self._lock:
            return len(self._clients) - len(self._dead)

    def next_live(self):
        """Return (client, endpoint) for the next non-dead endpoint, or None."""
        with self._lock:
            n = len(self._clients)
            for _ in range(n):
                client, ep = self._clients[self._i % n]
                self._i += 1
                if ep not in self._dead:
                    return client, ep
            return None

    def mark_dead(self, endpoint: str) -> None:
        with self._lock:
            self._dead.add(endpoint)


def _discover_csvs(csv_dir: Path):
    """Map source-filename -> csv Path for ``<dir>/<name>/<name>.csv`` and
    any CSVs sitting directly in ``<dir>``."""
    out: dict[str, Path] = {}
    for p in sorted(csv_dir.glob("*/*.csv")):
        out.setdefault(p.name, p)
    for p in sorted(csv_dir.glob("*.csv")):
        out.setdefault(p.name, p)
    return out


def run_column(pool, model, effort, temperature, response_format,
               source, column, values, cache_dir):
    cache_key = hashlib.sha256(
        f"{_prompt_hash(model, effort, temperature)}|{source}|{column}".encode("utf-8")
    ).hexdigest()[:16]
    cache_path = cache_dir / f"{cache_key}.json"
    if cache_path.exists():
        raw = json.loads(cache_path.read_text()).get("raw")
        err = None
    else:
        messages = [
            {"role": "developer", "content": PROMPT_INCONSISTENCY},
            {"role": "user", "content": build_user_message(column, source, values)},
        ]
        kwargs = dict(model=model, messages=messages, response_format=response_format)
        if effort:
            kwargs["reasoning_effort"] = effort
        if temperature is not None:
            kwargs["temperature"] = temperature
        raw, err = None, None
        transient = 0
        # Visit endpoints; 404 -> dead+skip (free), 429/5xx -> backoff retry.
        while True:
            picked = pool.next_live()
            if picked is None:
                err = "no live endpoint for model"
                break
            client, ep = picked
            try:
                resp = client.chat.completions.create(**kwargs)
                raw = resp.choices[0].message.content
                err = None
                break
            except Exception as exc:  # noqa: BLE001
                msg = f"{type(exc).__name__}: {exc}"
                if "404" in msg or "DeploymentNotFound" in msg:
                    pool.mark_dead(ep)
                    err = msg
                    continue
                if ("429" in msg or "RateLimit" in msg) and transient < 5:
                    time.sleep(min(60, 5 * 2 ** transient))
                    transient += 1
                    err = msg
                    continue
                err = msg
                break
        if raw is not None:
            cache_path.write_text(json.dumps({"raw": raw, "error": None}))

    pred = _parse_findings(raw)  # {norm_value: corrected_value}
    return {
        "source": source, "column": column, "error": err,
        "n_distinct": len(values), "n_flagged": len(pred),
        "findings": _parse_findings_full(raw),
    }


def _parse_findings_full(raw):
    """Full finding dicts (value/corrected_value/kind/rationale), tolerant of
    fenced or bare JSON."""
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
    for it in obj.get("findings", []) or []:
        if isinstance(it, dict) and isinstance(it.get("value"), str):
            out.append({
                "value": it.get("value"),
                "corrected_value": it.get("corrected_value"),
                "kind": it.get("kind"),
                "rationale": it.get("rationale"),
            })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-dir", required=True)
    ap.add_argument("--model", default="gpt-5.5")
    ap.add_argument("--endpoints", default="",
                    help="Comma-separated endpoint pool (default: environment).")
    ap.add_argument("--reasoning-effort", default="none",
                    help="'' to omit; 'none' for gpt-5.5 baselines.")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0,
                    help="Cap number of (source,column) jobs (smoke test).")
    ap.add_argument("--out", default="",
                    help="Output dir (default outputs/inconsistency_rulebook_"
                         "gpt55/<dataset>).")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    csv_dir = (_PROJECT_ROOT / args.csv_dir).resolve() if not Path(args.csv_dir).is_absolute() \
        else Path(args.csv_dir)
    dataset = csv_dir.name
    out_dir = Path(args.out) if args.out else (
        _PROJECT_ROOT / "outputs" / "inconsistency_rulebook_gpt55" / dataset)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "cache"
    cache_dir.mkdir(exist_ok=True)

    eps = [e.strip() for e in args.endpoints.split(",") if e.strip()]
    if not eps:
        eps = configured_endpoint_pool()
    if not eps:
        raise SystemExit("configure --endpoints or AZURE_OPENAI_ENDPOINTS")
    pool = EndpointPool(eps, API_VERSION)
    response_format = build_response_format()

    csvs = _discover_csvs(csv_dir)
    jobs = []  # (source, column, values)
    for source, path in csvs.items():
        for column, values in _auditable_columns(path).items():
            jobs.append((source, column, values))
    if args.limit:
        jobs = jobs[: args.limit]

    ph = _prompt_hash(args.model, args.reasoning_effort, args.temperature)
    print(f"dataset={dataset} csvs={len(csvs)} jobs={len(jobs)} "
          f"endpoints={len(eps)} model={args.model} "
          f"effort={args.reasoning_effort!r} temp={args.temperature} "
          f"prompt_hash={ph}", flush=True)

    results = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = {
            ex.submit(run_column, pool, args.model, args.reasoning_effort,
                      args.temperature, response_format, s, c, v, cache_dir): (s, c)
            for (s, c, v) in jobs
        }
        done = 0
        for fut in as_completed(futs):
            results.append(fut.result())
            done += 1
            if done % 25 == 0 or done == len(jobs):
                errs = sum(1 for r in results if r["error"])
                flagged = sum(r["n_flagged"] for r in results)
                print(f"  {done}/{len(jobs)} cols  flagged={flagged}  "
                      f"errors={errs}  live_eps={pool.live_count()}  "
                      f"({time.time()-t0:.0f}s)", flush=True)

    tag = (args.tag + "_") if args.tag else ""
    preds_path = out_dir / f"{tag}predictions_{ph}.jsonl"
    with preds_path.open("w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    findings_path = out_dir / f"{tag}findings_{ph}.jsonl"
    with findings_path.open("w") as f:
        for r in results:
            for fd in r["findings"]:
                f.write(json.dumps({"source": r["source"], "column": r["column"],
                                    **fd}) + "\n")

    n_err = sum(1 for r in results if r["error"])
    n_flag = sum(r["n_flagged"] for r in results)
    cols_flagged = sum(1 for r in results if r["n_flagged"] > 0)
    summary = {
        "dataset": dataset, "csv_dir": str(csv_dir), "prompt_hash": ph,
        "model": args.model, "reasoning_effort": args.reasoning_effort,
        "temperature": args.temperature,
        "n_csvs": len(csvs), "n_columns": len(results),
        "columns_flagged": cols_flagged, "total_findings": n_flag,
        "api_errors": n_err,
        "predictions_file": str(preds_path), "findings_file": str(findings_path),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (out_dir / f"{tag}summary_{ph}.json").write_text(json.dumps(summary, indent=2))
    print(f"\nDONE  columns={len(results)} flagged_cols={cols_flagged} "
          f"findings={n_flag} errors={n_err}\nartifacts -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
