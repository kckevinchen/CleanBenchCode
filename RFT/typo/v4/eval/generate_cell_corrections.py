"""Generate realistic correction candidates for clean + ambiguous cells.

Background
----------
The cell-level grader probe (``test_grader_sm_cell_level``) needs a
``correction`` candidate to pair with each labelled cell so the score
model has something to grade. For ``error`` rows the benchmark already
carries a canonical ``corrected_value``. For ``clean`` and
``ambiguous`` rows it does not -- the historic fallback was either a
synthetic perturbation (``fake_correction``: lowercase / swap-last-two
/ append 's') or the noisy ``llm_suggested_value`` produced upstream
by various models at various times.

A synthetic perturbation is a problem because the grader (an LLM) can
trivially recognise the pattern and give 0.0 for the wrong reason --
biasing the clean-bucket result. We want a candidate that *looks like
a real LLM-proposed correction* would, so a 0.0 from the grader
reflects "the original value is fine", not "this candidate is
obviously a synthetic mutation".

This script asks the GPT-5 family to play the role of a typo-cleaning
LLM and propose its single best-guess correction for each cell, using
the column's distinct values as context (and the existing
``llm_suggested_value`` from the benchmark as a hint when present).
Results are cached in a JSONL file keyed by
``(source, column, original_value)`` so re-runs are free.

Usage
-----
    python -m RFT.typo.v4.eval.generate_cell_corrections \
        --benchmark benchmark/benchmark_typo_20260512T213238Z_gpt5-family.jsonl \
        --csv-root data/source_inconsistency_bing21 \
        --model "gpt-5.2,gpt-5.4,gpt-5.5" \
        --concurrency 20 --max-retries 3 \
        --out outputs/cell_level_corrections.jsonl
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import logging
import re
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.core.llm_client import LLMClient  # noqa: E402

logger = logging.getLogger("generate_cell_corrections")


SYSTEM_PROMPT = (
    "You are an expert data-cleaning assistant. When shown a single "
    "value from a column you propose your single best-guess canonical "
    "correction for it, using the column's other values as context. "
    "You always reply with strict JSON only."
)

USER_TEMPLATE = """\
Column name: {column}
Source file: {source}

Sample of distinct values currently in this column (value, count):
{value_listing}

Target value to consider: {original!r}
{prior_hint}
Pretend a typo-detection system has flagged the target value as a
possible typo, even if you are not fully sure it is one. Propose your
single best guess at the canonical / corrected form for this value,
based on the column context and any prior hint above.

Constraints:
  * The correction must be a plausible replacement value of the same
    shape (same kind of entity / format) as the other column values.
  * The correction should be DIFFERENT from the original value
    (do not echo it verbatim); pick the nearest realistic
    alternative. If you truly cannot think of any plausible
    different form, return the original.
  * Do not return explanations, markdown, or code fences.

Reply with strict JSON of exactly the form:
{{"correction": "<your single best-guess corrected string>"}}
"""


_JSON_RE = re.compile(r"\{[^{}]*\"correction\"[^{}]*\}", re.DOTALL)


def _parse_correction(text: str) -> str | None:
    """Pull the ``correction`` field out of the model's reply.

    Tolerates code fences / leading prose by scanning for the first
    JSON object that contains a ``"correction"`` key.
    """
    if not isinstance(text, str):
        return None
    # Fast path: full reply is JSON.
    s = text.strip()
    for candidate in (s, *(m.group(0) for m in _JSON_RE.finditer(s))):
        try:
            obj = json.loads(candidate)
        except Exception:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("correction"), str):
            return obj["correction"]
    return None


def _csv_index(csv_root: Path) -> dict[str, Path]:
    return {p.name: p for p in csv_root.rglob("*.csv")}


def _load_csv_distinct_values(
    csv_path: Path, column: str, *, max_values: int = 60,
) -> list[tuple[str, int]]:
    counts: Counter[str] = Counter()
    try:
        with csv_path.open(encoding="utf-8", errors="replace") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames is None or column not in reader.fieldnames:
                return []
            for row in reader:
                val = row.get(column)
                if val is None:
                    continue
                val = str(val)
                if val.strip() == "":
                    continue
                counts[val] += 1
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed reading %s: %s", csv_path, exc)
        return []
    return counts.most_common(max_values)


def _format_listing(values_with_counts: list[tuple[str, int]]) -> str:
    return "\n".join(f"  - {v!r}  (count={c})" for v, c in values_with_counts)


def _cache_key(job: dict) -> tuple[str, str, str]:
    return (job["source"], job["column"], job["original"])


def _load_cache(path: Path) -> dict[tuple[str, str, str], dict]:
    if not path.exists():
        return {}
    out: dict[tuple[str, str, str], dict] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            key = (row.get("source"), row.get("column"), row.get("original"))
            if all(isinstance(k, str) for k in key):
                out[key] = row
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        default="benchmark/benchmark_typo_20260512T213238Z_gpt5-family.jsonl",
        type=Path,
    )
    parser.add_argument(
        "--csv-root",
        default="data/source_inconsistency_bing21",
        type=Path,
    )
    parser.add_argument(
        "--out",
        default=Path("outputs/cell_level_corrections.jsonl"),
        type=Path,
    )
    parser.add_argument("--model", default="gpt-5.2,gpt-5.4,gpt-5.5")
    parser.add_argument(
        "--reasoning-effort", default="low",
        help="GPT-5 reasoning effort; low is fine for short structured replies.",
    )
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument(
        "--labels", nargs="+", default=["clean", "ambiguous"],
        help="Benchmark labels to generate corrections for.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Optional cap on number of cells (for smoke tests).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)

    # Build column value listings once per (source, column).
    csv_paths = _csv_index(args.csv_root)
    listings: dict[tuple[str, str], str] = {}

    # Enumerate jobs.
    jobs: list[dict] = []
    needed_cols: set[tuple[str, str]] = set()
    with args.benchmark.open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("label") not in args.labels:
                continue
            value = r.get("original_value")
            if not isinstance(value, str) or not value:
                continue
            src = r["source"]
            col = r["column"]
            needed_cols.add((src, col))
            jobs.append({
                "source": src,
                "column": col,
                "original": value,
                "label": r["label"],
                "prior": r.get("llm_suggested_value")
                         or r.get("corrected_value") or "",
                "note": r.get("note") or "",
            })

    # De-dup jobs by (source, column, original) -- different rows can
    # cover the same value (e.g. cluster-expanded clean cells).
    seen_keys: set[tuple[str, str, str]] = set()
    deduped: list[dict] = []
    for j in jobs:
        k = _cache_key(j)
        if k in seen_keys:
            continue
        seen_keys.add(k)
        deduped.append(j)
    jobs = deduped
    logger.info("Distinct cells needing a correction: %d", len(jobs))

    # Load existing cache and skip already-done jobs.
    cache = _load_cache(args.out)
    pending = [j for j in jobs if _cache_key(j) not in cache]
    logger.info(
        "Cache hits: %d  pending: %d",
        len(jobs) - len(pending), len(pending),
    )
    if args.limit is not None:
        pending = pending[: args.limit]
        logger.info("Capped pending to --limit=%d", args.limit)

    # Build column listings only for the columns we still need.
    pending_cols = {(j["source"], j["column"]) for j in pending}
    for src, col in pending_cols:
        csv_path = csv_paths.get(src)
        if csv_path is None:
            listings[(src, col)] = "  (CSV not found)"
            continue
        listings[(src, col)] = _format_listing(
            _load_csv_distinct_values(csv_path, col)
        )

    client = LLMClient()
    out_lock = threading.Lock()
    t0 = time.monotonic()

    def _build_prompt(job: dict) -> str:
        listing = listings.get((job["source"], job["column"]), "")
        prior = ""
        if job["prior"]:
            prior = (
                f"\nA prior automated suggestion for this value was: "
                f"{job['prior']!r}. Consider it as a hint but feel free "
                "to override it.\n"
            )
        return USER_TEMPLATE.format(
            column=job["column"], source=job["source"],
            value_listing=listing or "  (no values available)",
            original=job["original"],
            prior_hint=prior,
        )

    def _run_one(job: dict) -> dict:
        prompt = _build_prompt(job)
        last_err = ""
        last_reply = ""
        for attempt in range(args.max_retries + 1):
            try:
                resp = client.query(
                    args.model, prompt,
                    reasoning_effort=args.reasoning_effort,
                    system_prompt=SYSTEM_PROMPT,
                )
                last_reply = resp.text
                if resp.error:
                    last_err = resp.error
                else:
                    corr = _parse_correction(resp.text)
                    if corr is not None:
                        return {
                            "source": job["source"],
                            "column": job["column"],
                            "original": job["original"],
                            "label": job["label"],
                            "correction": corr,
                            "model": resp.model,
                            "attempts": attempt + 1,
                        }
                    last_err = "parse_failed"
            except Exception as exc:  # noqa: BLE001
                last_err = f"exception: {exc}"
            if attempt < args.max_retries:
                time.sleep(2 ** attempt)
        logger.warning(
            "Failed cell src=%s col=%r value=%r err=%s reply=%s",
            job["source"], job["column"], job["original"],
            last_err, (last_reply or "")[:160],
        )
        return {
            "source": job["source"],
            "column": job["column"],
            "original": job["original"],
            "label": job["label"],
            "correction": None,
            "error": last_err,
            "reply_head": (last_reply or "")[:240],
            "attempts": args.max_retries + 1,
        }

    # Stream results to disk as they complete so a crash mid-run
    # still yields useful cache content.
    out_fh = args.out.open("a", encoding="utf-8")
    try:
        done = 0
        ok = 0
        fail = 0
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {pool.submit(_run_one, j): j for j in pending}
            for fut in as_completed(futures):
                row = fut.result()
                with out_lock:
                    out_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                    out_fh.flush()
                    done += 1
                    if row.get("correction") is not None:
                        ok += 1
                    else:
                        fail += 1
                if done % 100 == 0 or done == len(pending):
                    logger.info(
                        "[%d/%d] ok=%d fail=%d elapsed=%.1fs",
                        done, len(pending), ok, fail,
                        time.monotonic() - t0,
                    )
    finally:
        out_fh.close()

    # Re-load and report final coverage.
    cache_final = _load_cache(args.out)
    covered = sum(
        1 for j in jobs
        if cache_final.get(_cache_key(j), {}).get("correction") is not None
    )
    by_label = collections.Counter(j["label"] for j in jobs)
    covered_by_label: collections.Counter = collections.Counter()
    for j in jobs:
        row = cache_final.get(_cache_key(j))
        if row and row.get("correction") is not None:
            covered_by_label[j["label"]] += 1

    print("\n" + "=" * 60)
    print("CELL CORRECTION GENERATION")
    print("=" * 60)
    print(f"Distinct cells:    {len(jobs)}")
    print(f"Covered (success): {covered}")
    print(f"By label:          {dict(by_label)}")
    print(f"Covered by label:  {dict(covered_by_label)}")
    print(f"Cache file:        {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
