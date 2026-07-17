"""GPT-5.5 single-prompt cell-error detector (run in the MAIN venv).

For each materialized REIN dataset, render the table in row-chunks (all columns)
and issue ONE GPT-5.5 call per chunk using the unified multi-type prompt
(``prompt.py``). Parse the JSON ``issues`` into (dataset, row, col) predictions,
cache raw responses per chunk (resumable), and write predictions JSONL into the
shared results dir so ``external/rein/score.py`` (+ neutral set) grades it next to
``raha`` / the typo specialists / CleanAgent.

Faithful alignment: the table is loaded with the REIN-faithful loader
(``common.load_rein_df``) -- the exact frame the scorer/labels assume -- and
``CsvChunk.to_markdown`` emits the ``__row__`` index the prompt references, so the
LLM's ``row`` values map straight back to benchmark rows.

Examples (repo root, main venv, after ``az login``):
    python external/gpt5_detect/run.py                       # all 215, gpt-5.5
    python external/gpt5_detect/run.py --limit 5             # smoke test
    python external/gpt5_detect/run.py --model gpt-5.4 --chunk-size 50
    python external/gpt5_detect/run.py --min-confidence medium
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# REIN harness helpers (paths + REIN-faithful loader).
_REIN_DIR = Path(__file__).resolve().parents[1] / "rein"
sys.path.insert(0, str(_REIN_DIR))
import common as C  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prompt as PROMPT  # noqa: E402

# Repo root for src.* imports.
sys.path.insert(0, str(C.REPO_ROOT))
from src.core.data_loader import iter_chunks  # noqa: E402

DEFAULT_CACHE_DIR = Path(__file__).resolve().parent / "cache"
CONF_RANK = {"low": 1, "medium": 2, "high": 3}
VALID_TYPES = set(PROMPT.CELL_TYPES)


def _extract_json(text: str) -> dict | None:
    """Best-effort parse of a JSON object from an LLM response."""
    if not text:
        return None
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if 0 <= start < end:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return None
    return None


def _cache_path(
    cache_dir: Path,
    dataset: str,
    start_row: int,
    request_sha: str,
    model: str,
) -> Path:
    safe_model = re.sub(r"[^A-Za-z0-9._-]", "_", model)
    return cache_dir / dataset / f"r{start_row}_{safe_model}_{request_sha[:12]}.json"


def _request_sha(system_prompt: str, user_prompt: str, reasoning_effort: str) -> str:
    """Hash every request component that can change the model response."""
    payload = json.dumps(
        {
            "schema": 2,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "reasoning_effort": reasoning_effort,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_dataset_list(path: Path, available: set[str]) -> list[str]:
    """Load an exact dataset allowlist from a JSON list or ``{"datasets": [...]}``."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    values = raw.get("datasets") if isinstance(raw, dict) else raw
    if not isinstance(values, list) or not all(isinstance(v, str) for v in values):
        raise ValueError(f"{path} must contain a JSON string list or an object with 'datasets'")
    if len(values) != len(set(values)):
        raise ValueError(f"{path} contains duplicate dataset identifiers")
    unknown = sorted(set(values) - available)
    if unknown:
        preview = ", ".join(unknown[:5])
        raise ValueError(f"{path} contains {len(unknown)} unknown datasets: {preview}")
    return list(values)


def run_chunk(
    client,
    model,
    system_prompt,
    user_prompt,
    reasoning_effort,
    dataset,
    start_row,
    request_sha,
    cache_dir,
    no_cache,
):
    """Return parsed issues list (cached)."""
    cache_p = _cache_path(cache_dir, dataset, start_row, request_sha, model)
    if not no_cache and cache_p.exists():
        try:
            return json.loads(cache_p.read_text())["issues"], True
        except Exception:
            pass
    resp = client.query(model, user_prompt, system_prompt=system_prompt,
                        reasoning_effort=reasoning_effort)
    parsed = _extract_json(resp.text) if resp.ok else None
    issues = parsed.get("issues", []) if isinstance(parsed, dict) else []
    cache_p.parent.mkdir(parents=True, exist_ok=True)
    cache_p.write_text(json.dumps({
        "dataset": dataset, "start_row": start_row, "model": resp.model or model,
        "request_sha256": request_sha,
        "reasoning_effort": reasoning_effort,
        "ok": resp.ok, "error": resp.error, "issues": issues,
        "raw": resp.text if resp.ok else "",
    }, ensure_ascii=False, indent=2))
    return issues, False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets-dir", default=str(C.DATASETS_DIR))
    ap.add_argument("--results-dir", default=str(C.RESULTS_DIR))
    ap.add_argument("--model", default="gpt-5.5")
    ap.add_argument("--chunk-size", type=int, default=40)
    ap.add_argument("--reasoning-effort", default="medium")
    ap.add_argument("--min-confidence", default="low", choices=["low", "medium", "high"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=8, help="parallel LLM calls")
    ap.add_argument("--dataset-list", default="",
                    help="JSON list of exact dataset IDs to run (e.g. a frozen split)")
    ap.add_argument("--system-prompt-file", default="",
                    help="use this UTF-8 system prompt instead of prompt.py")
    ap.add_argument("--tag", default="",
                    help="output detector tag (default: gpt5_detect_<model>)")
    ap.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    ds_root = Path(args.datasets_dir)
    manifest = json.loads((ds_root / "_manifest.json").read_text())["datasets"]
    ds_names = sorted(manifest)
    if args.dataset_list:
        if args.limit:
            ap.error("--dataset-list and --limit are mutually exclusive")
        try:
            ds_names = _load_dataset_list(Path(args.dataset_list), set(manifest))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            ap.error(str(exc))
    if args.limit:
        ds_names = ds_names[: args.limit]

    from src.core.llm_client import LLMClient

    client = LLMClient()
    system_prompt = (
        Path(args.system_prompt_file).read_text(encoding="utf-8")
        if args.system_prompt_file
        else PROMPT.build_system_prompt()
    )
    if not system_prompt.strip():
        ap.error("system prompt must not be empty")
    min_conf = CONF_RANK[args.min_confidence]
    cache_dir = Path(args.cache_dir)

    tag = args.tag or f"gpt5_detect_{re.sub(r'[^A-Za-z0-9]', '', args.model)}"
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", tag):
        ap.error("--tag may contain only letters, numbers, underscore, dot, and hyphen")
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"predictions_{tag}.jsonl"
    rich_path = results_dir / f"issues_{tag}.jsonl"  # full issues for analysis

    # --- Phase 1: enumerate chunk jobs (render prompts up front) -----------
    ran: list[str] = []
    col_index_by_ds: dict[str, dict[str, int]] = {}
    jobs: list[dict] = []  # {ds, start_row, end_row, user_prompt}
    for ds in ds_names:
        try:
            df = C.load_rein_df(ds_root / ds / "dirty.csv")
        except Exception as e:
            print(f"[skip] {ds}: {e}")
            continue
        ran.append(ds)
        col_index_by_ds[ds] = {c: i for i, c in enumerate(df.columns)}
        for chunk in iter_chunks(df, Path(ds), args.chunk_size, column_mode="all"):
            jobs.append({
                "ds": ds, "start_row": chunk.start_row, "end_row": chunk.end_row,
                "user_prompt": PROMPT.build_user_prompt(
                    headers=list(df.columns), rows_markdown=chunk.to_markdown(),
                    start_row=chunk.start_row, end_row=chunk.end_row),
            })
            jobs[-1]["request_sha"] = _request_sha(
                system_prompt,
                jobs[-1]["user_prompt"],
                args.reasoning_effort,
            )

    print(f"enumerated {len(jobs)} chunks over {len(ran)} datasets; "
          f"calling {args.model} with concurrency={args.concurrency}...")

    # --- Phase 2: parallel LLM calls (fill cache) --------------------------
    results: dict[tuple[str, int], list] = {}
    n_cached = 0
    done = 0
    lock = threading.Lock()

    def _work(job):
        return job, run_chunk(
            client, args.model, system_prompt, job["user_prompt"],
            args.reasoning_effort, job["ds"], job["start_row"], job["request_sha"],
            cache_dir, args.no_cache,
        )

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = [ex.submit(_work, j) for j in jobs]
        for fut in as_completed(futures):
            job, (issues, cached) = fut.result()
            with lock:
                results[(job["ds"], job["start_row"])] = issues
                n_cached += int(cached)
                done += 1
                if done % 20 == 0:
                    print(f"  ...{done}/{len(jobs)} chunks done")

    # --- Phase 3: deterministic assembly from results ----------------------
    n_pred = 0
    type_counts: dict[str, int] = {}
    with open(out_path, "w", encoding="utf-8") as fout, \
            open(rich_path, "w", encoding="utf-8") as frich:
        for job in jobs:
            ds = job["ds"]
            col_index = col_index_by_ds[ds]
            issues = results.get((ds, job["start_row"]), [])
            seen: set[tuple[int, int]] = set()
            for it in issues:
                if not isinstance(it, dict):
                    continue
                et = str(it.get("error_type", "")).strip()
                if et not in VALID_TYPES:
                    continue
                if CONF_RANK.get(str(it.get("confidence", "low")).lower(), 1) < min_conf:
                    continue
                col = it.get("column")
                row = it.get("row")
                if col not in col_index or not isinstance(row, int):
                    continue
                if not (job["start_row"] <= row < job["end_row"]):
                    continue
                j = col_index[col]
                frich.write(json.dumps({"dataset": ds, "row": row, "col": j,
                                        "error_type": et, "confidence": it.get("confidence"),
                                        "observed": it.get("observed"),
                                        "explanation": it.get("explanation"),
                                        "corrected_value": it.get("corrected_value")},
                                       ensure_ascii=False) + "\n")
                key = (row, j)
                if key in seen:
                    continue
                seen.add(key)
                fout.write(json.dumps({"dataset": ds, "detector": tag,
                                       "row": row, "col": j}) + "\n")
                n_pred += 1
                type_counts[et] = type_counts.get(et, 0) + 1

    ran_obj = {"per_detector": {tag: ran}, "all_loaded": ran}
    Path(str(out_path) + ".ran.json").write_text(json.dumps(ran_obj))

    print("=== gpt5_detect summary ===")
    print(f"model              : {args.model} (effort={args.reasoning_effort}, "
          f"chunk={args.chunk_size}, min_conf={args.min_confidence})")
    print(f"datasets processed : {len(ran)}")
    print(f"chunks (LLM calls) : {len(jobs)}  ({n_cached} from cache)")
    print(f"cells flagged      : {n_pred}  by type: {type_counts}")
    print(f"predictions        : {out_path}")
    print(f"\nScore with:\n  python external/rein/score.py --predictions {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
