"""Build v11 train/valid/eval splits (gold-free training, benchmark eval).

v11 trains the typo trainee on UNLABELLED columns so the WHOLE human benchmark
can be reserved for eval. This builder is therefore unlike the v9/v10 builders
(which only *retargeted* the prompt on benchmark-derived rows): it constructs
brand-new training rows from raw CSV tables and has NO gold for them.

What it writes to ``RFT/typo/v11/data/``:

  * ``train.jsonl`` / ``valid.jsonl`` -- one column-audit row per selected
    column drawn from ``data/curated_bing21_dedup_training/`` (997 unlabelled
    tables). NO ``gold_typos`` field -- the v11 grader is gold-free (two LLM
    judges; see ``RFT/typo/v11/_grader.py``). Split BY SOURCE TABLE so no table
    straddles train/valid (zero column leakage). ``PROMPT_V11`` is baked into
    every row's developer message.

  * ``eval.jsonl`` -- the FULL human benchmark, reused verbatim from the
    already-curated v10 benchmark rows (v10 train+valid+eval == the whole
    benchmark, with ``gold_typos`` + ``ambiguous_values`` baked in), with the
    developer message retargeted to ``PROMPT_V11`` (identical to ``PROMPT_V10``,
    so this is a no-op today but keeps the provenance explicit). The benchmark
    is 100% reserved for eval because v11 never trains on it -- it is a true
    held-out test set. Only ``rft run`` consumes this file; the engine uploads
    only train+valid for the Azure job.

Why a text-column heuristic (``--min-distinct`` / letter-fraction filter):
the raw tables are dominated by numeric / date / code columns that have no
typos. Training mostly on those would make the recall judge's truth set G
empty almost everywhere, so the empty prediction would score ~1.0 and the
model would collapse to proposing nothing (the degeneracy v11's recall judge
exists to prevent). Filtering to free-text columns raises the typo prior so
the two judges get real signal, while a minority of clean text columns still
teach correct abstention. See RFT.md D11.

Usage::

    # default: richest text column per table, all qualifying tables used
    python -m RFT.typo.v11.scripts.build_v11_splits

    # denser (up to 2 cols/table) and capped to ~600 train rows
    python -m RFT.typo.v11.scripts.build_v11_splits --max-per-file 2 --max-train 600

    # inspect only -- print what would be selected, write nothing
    python -m RFT.typo.v11.scripts.build_v11_splits --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import random
import re
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from RFT._common import typo_task as _v5  # noqa: E402
from RFT.typo.v11 import PROMPT_V11  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("build_v11_splits")

# CSV's own field-size limit is too small for some wide curated cells.
csv.field_size_limit(10_000_000)

DEFAULT_POOL = _PROJECT_ROOT / "data" / "curated_bing21_dedup_training"
DEFAULT_BENCHMARK_DIR = _PROJECT_ROOT / "RFT" / "typo" / "v10" / "data"
DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / "RFT" / "typo" / "v11" / "data"
BENCHMARK_SPLITS = ("train.jsonl", "valid.jsonl", "eval.jsonl")

_LETTER = re.compile(r"[A-Za-z]")
_NUMERICISH = re.compile(r"^[\s\d.,:;/\\()%$+*#&_-]*$")


# ---------------------------------------------------------------------------
# Column profiling -- the text-column heuristic (raises the typo prior)
# ---------------------------------------------------------------------------

def _tokens(s: str) -> list[str]:
    return [t for t in re.split(r"\s+", s.strip()) if t]


def _profile(distinct: list[str]) -> dict:
    n = len(distinct)
    if n == 0:
        return dict(n=0, letter_frac=0.0, numeric_frac=1.0, multiword_frac=0.0,
                    avg_len=0.0, avg_tokens=0.0)
    letter = sum(1 for v in distinct if _LETTER.search(v))
    numeric = sum(1 for v in distinct if _NUMERICISH.match(v))
    multiword = sum(1 for v in distinct if len(_tokens(v)) >= 2)
    avg_len = sum(len(v) for v in distinct) / n
    avg_tokens = sum(len(_tokens(v)) for v in distinct) / n
    return dict(
        n=n,
        letter_frac=letter / n,
        numeric_frac=numeric / n,
        multiword_frac=multiword / n,
        avg_len=avg_len,
        avg_tokens=avg_tokens,
    )


def _qualifies(p: dict, *, min_distinct: int, letter_frac: float,
               numeric_frac: float, avg_len: float) -> bool:
    return (
        p["n"] >= min_distinct
        and p["letter_frac"] >= letter_frac
        and p["numeric_frac"] <= numeric_frac
        and p["avg_len"] >= avg_len
    )


def _richness(p: dict) -> float:
    """Higher == more free-text == higher prior of containing a real typo."""
    return (
        p["letter_frac"]
        + p["multiword_frac"]
        + min(p["avg_tokens"] / 5.0, 1.0)
        + min(math.log10(max(p["n"], 1)) / 2.0, 1.0)
    )


# ---------------------------------------------------------------------------
# CSV reading
# ---------------------------------------------------------------------------

def _read_columns(csv_path: Path) -> list[tuple[str, list[tuple[str, int]]]]:
    """Return ``[(header, [(value, count), ...]), ...]`` for one CSV.

    Robust to ragged rows and odd encodings; silently skips an unreadable file.
    """
    try:
        text = csv_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    reader = csv.reader(text.splitlines())
    try:
        header = next(reader)
    except StopIteration:
        return []
    counters: list[dict[str, int]] = [{} for _ in header]
    for row in reader:
        for ci, cell in enumerate(row):
            if ci >= len(header):
                break
            cell = cell.strip()
            if not cell:
                continue
            counters[ci][cell] = counters[ci].get(cell, 0) + 1
    out = []
    for ci, head in enumerate(header):
        vals = list(counters[ci].items())
        out.append((str(head), vals))
    return out


# ---------------------------------------------------------------------------
# Row builders
# ---------------------------------------------------------------------------

def _build_train_row(
    source: str, column: str, values_with_counts: list[tuple[str, int]],
    *, max_distinct: int,
) -> dict:
    """One gold-free column-audit row (PROMPT_V11 baked in)."""
    capped = sorted(values_with_counts, key=lambda kv: (-int(kv[1]), str(kv[0])))[
        :max_distinct
    ]
    user = _v5.build_user_message(
        column=column, source=source, values_with_counts=capped,
        max_distinct=max_distinct,
    )
    return {
        "messages": [
            {"role": "developer", "content": PROMPT_V11},
            {"role": "user", "content": user},
        ],
        "column": column,
        "source": source,
        "column_values_json": json.dumps(capped, ensure_ascii=False),
    }


def _retarget_prompt(rec: dict) -> dict:
    msgs = rec.get("messages")
    if not isinstance(msgs, list) or not msgs or msgs[0].get("role") != "developer":
        raise ValueError("benchmark row missing a developer message at messages[0]")
    msgs = list(msgs)
    msgs[0] = {**msgs[0], "content": PROMPT_V11}
    return {**rec, "messages": msgs}


def _build_eval_rows(benchmark_dir: Path) -> list[dict]:
    """The full benchmark (v10's curated rows) retargeted to PROMPT_V11."""
    rows: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for fname in BENCHMARK_SPLITS:
        path = benchmark_dir / fname
        if not path.exists():
            logger.warning("benchmark split %s missing -- skipped", path)
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            key = (rec.get("source", ""), rec.get("column", ""))
            if key in seen:
                continue
            seen.add(key)
            rows.append(_retarget_prompt(rec))
    return rows


def _gold_cell_count(rows: list[dict]) -> int:
    total = 0
    for r in rows:
        g = r.get("gold_typos")
        if isinstance(g, str) and g:
            try:
                total += len(json.loads(g))
            except json.JSONDecodeError:
                pass
        elif isinstance(g, list):
            total += len(g)
    return total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--pool", type=Path, default=DEFAULT_POOL,
                   help="Dir of unlabelled training CSVs (recursive).")
    p.add_argument("--benchmark-data-dir", type=Path, default=DEFAULT_BENCHMARK_DIR,
                   help="Dir holding the curated benchmark rows reused as eval.")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--valid-frac", type=float, default=0.12,
                   help="Fraction of training SOURCE TABLES held out for valid.")
    p.add_argument("--max-per-file", type=int, default=1,
                   help="Keep at most this many richest text columns per table.")
    p.add_argument("--max-train", type=int, default=0,
                   help="Optional cap on #train rows (0 = use all). Caps by "
                        "dropping whole sources, never splitting a table.")
    p.add_argument("--min-distinct", type=int, default=5)
    p.add_argument("--max-distinct", type=int, default=_v5.TYPO_V5_MAX_DISTINCT)
    p.add_argument("--min-letter-frac", type=float, default=0.6)
    p.add_argument("--max-numeric-frac", type=float, default=0.4)
    p.add_argument("--min-avg-len", type=float, default=3.0)
    p.add_argument("--dry-run", action="store_true",
                   help="Print selection stats; write nothing.")
    args = p.parse_args(argv)

    # --- 1. select training columns from the unlabelled pool ---------------
    csvs = sorted(args.pool.rglob("*.csv"))
    if not csvs:
        raise SystemExit(f"No CSVs under {args.pool}")
    logger.info("scanning %d CSVs under %s", len(csvs), args.pool)

    by_source: dict[str, list[dict]] = {}
    scanned_cols = 0
    qualified_cols = 0
    for csv_path in csvs:
        source = csv_path.name
        candidates = []
        for column, vals in _read_columns(csv_path):
            scanned_cols += 1
            distinct = [v for v, _ in vals]
            prof = _profile(distinct)
            if not _qualifies(
                prof, min_distinct=args.min_distinct,
                letter_frac=args.min_letter_frac,
                numeric_frac=args.max_numeric_frac, avg_len=args.min_avg_len,
            ):
                continue
            candidates.append((_richness(prof), column, vals))
        candidates.sort(key=lambda t: (-t[0], t[1]))
        for _score, column, vals in candidates[: args.max_per_file]:
            qualified_cols += 1
            row = _build_train_row(
                source, column, vals, max_distinct=args.max_distinct,
            )
            by_source.setdefault(source, []).append(row)

    sources = sorted(by_source)
    logger.info(
        "selected %d columns from %d/%d tables (scanned %d columns)",
        qualified_cols, len(sources), len(csvs), scanned_cols,
    )

    # --- 2. by-source train/valid split (zero table leakage) ---------------
    rng = random.Random(args.seed)
    rng.shuffle(sources)
    n_valid = int(len(sources) * args.valid_frac)
    valid_sources = set(sources[:n_valid])
    train_sources = sources[n_valid:]

    # optional cap on train rows: drop whole sources from the tail
    if args.max_train and args.max_train > 0:
        kept, total = [], 0
        for s in train_sources:
            k = len(by_source[s])
            if total + k > args.max_train and kept:
                break
            kept.append(s)
            total += k
        dropped = len(train_sources) - len(kept)
        if dropped:
            logger.info("max-train=%d -> kept %d train tables, dropped %d",
                        args.max_train, len(kept), dropped)
        train_sources = kept

    train_rows = [r for s in train_sources for r in by_source[s]]
    valid_rows = [r for s in sorted(valid_sources) for r in by_source[s]]

    # --- 3. eval = full benchmark (reused, retargeted) ---------------------
    eval_rows = _build_eval_rows(args.benchmark_data_dir)
    eval_sources = sorted({r.get("source", "") for r in eval_rows})

    # --- 4. leakage guard: a training table must not share a source id -----
    leak = set(train_sources + sorted(valid_sources)) & set(eval_sources)
    if leak:
        raise SystemExit(
            f"Source-id leakage between training pool and benchmark eval: {sorted(leak)[:5]}"
        )

    logger.info(
        "train=%d rows / %d tables | valid=%d rows / %d tables | eval=%d rows / "
        "%d tables (%d gold cells)",
        len(train_rows), len(train_sources), len(valid_rows), len(valid_sources),
        len(eval_rows), len(eval_sources), _gold_cell_count(eval_rows),
    )

    if args.dry_run:
        logger.info("dry-run: no files written")
        return 0

    if not train_rows or not valid_rows:
        raise SystemExit(
            "Refusing to write empty train/valid -- loosen the column filters."
        )

    # --- 5. write splits + manifest ----------------------------------------
    args.output_dir.mkdir(parents=True, exist_ok=True)

    def _write(name: str, rows: list[dict]) -> None:
        dst = args.output_dir / name
        with dst.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        logger.info("wrote %d rows -> %s", len(rows), dst)

    _write("train.jsonl", train_rows)
    _write("valid.jsonl", valid_rows)
    _write("eval.jsonl", eval_rows)

    manifest = {
        "builder": "RFT.typo.v11.scripts.build_v11_splits",
        "prompt_module": "RFT.typo.v11._prompt.PROMPT_V11",
        "prompt_chars": len(PROMPT_V11),
        "gold_free_training": True,
        "train_valid_source_pool": str(args.pool),
        "eval_source": (
            f"{args.benchmark_data_dir} (full human benchmark; "
            f"train+valid+eval merged, retargeted to PROMPT_V11)"
        ),
        "params": {
            "seed": args.seed,
            "valid_frac": args.valid_frac,
            "max_per_file": args.max_per_file,
            "max_train": args.max_train,
            "min_distinct": args.min_distinct,
            "max_distinct": args.max_distinct,
            "min_letter_frac": args.min_letter_frac,
            "max_numeric_frac": args.max_numeric_frac,
            "min_avg_len": args.min_avg_len,
        },
        "counts": {
            "csvs_scanned": len(csvs),
            "columns_scanned": scanned_cols,
            "columns_selected": qualified_cols,
            "train_rows": len(train_rows),
            "train_tables": len(train_sources),
            "valid_rows": len(valid_rows),
            "valid_tables": len(valid_sources),
            "eval_rows": len(eval_rows),
            "eval_tables": len(eval_sources),
            "eval_gold_cells": _gold_cell_count(eval_rows),
        },
        "caveats": [
            "Training columns are UNLABELLED; the reward is the v11 gold-free "
            "two-judge grader (precision + recall LLM judges + trivial/cap "
            "python gates).",
            "Eval is the human benchmark and is never seen in training (true "
            "held-out test). Source ids are disjoint from the training pool.",
            "Value-level leakage between the bing21-derived training pool and "
            "the benchmark sources is NOT audited here; run a value-overlap "
            "check before trusting eval deltas under ~0.05.",
        ],
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    logger.info("manifest -> %s", args.output_dir / "manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
