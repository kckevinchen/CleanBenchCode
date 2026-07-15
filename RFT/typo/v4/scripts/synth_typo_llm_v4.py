"""Synthesise typo-RFT training data for the v4 column-audit task.

Mirrors ``RFT/typo/v3/scripts/synth_typo_llm_v3.py`` but with:

* v4 task assets (``experiments._typo_v4_task``).
* Three flavours: spelling, punctuation, **spacing** (new).
* v4-trivial filter (keeps spacing-tokenisation edits as substantive).

Two modes:

* ``--mode deterministic`` -- no LLM calls. Picks an existing value
  per column, applies one of ``_V4_MUTATORS`` (spelling / punctuation /
  spacing), and appends the mutated form as a count=1 distinct value.
* ``--mode llm`` -- ``src.core.llm_client.LLMClient`` is asked to
  invent K realistic typos including the new spacing flavour. Results
  validated locally and cached on disk.

Outputs under ``RFT/typo/v4/data/``:

* ``train.jsonl``      -- training records (v4 ``build_record`` shape).
* ``valid.jsonl``      -- validation records.
* ``injections.jsonl`` -- per-column provenance (which CSV, which
                          values were mutated, which flavour).
* ``manifest.json``    -- run config + counts.

Usage
-----
::

    python RFT/typo/v4/scripts/synth_typo_llm_v4.py --mode deterministic
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import random
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from RFT.typo.v4 import _task as _typo_v4_task  # noqa: E402
from RFT._common.synth_typo_llm import (  # noqa: E402
    _TYPO_MUTATORS,
    _column_is_textual,
    _looks_numeric_column,
    ColumnSample,
)

logger = logging.getLogger("synth_typo_llm_v4")


# ---------------------------------------------------------------------------
# v4-only spacing mutators (added on top of v2/v3 spelling+punctuation set)
# ---------------------------------------------------------------------------


def _mut_split_word(s: str, rng: random.Random) -> tuple[str, str]:
    """Insert a space inside a long alphabetic run -- ``DuPage`` -> ``Du Page``.

    Picks an internal lowercase->uppercase boundary if one exists
    (CamelCase split), otherwise a roughly-middle position inside a
    >=6-char alphabetic run.
    """
    # Strategy 1: CamelCase boundary -- aBc...XyZ -> a Bc...XyZ
    camel = [
        i for i in range(1, len(s))
        if s[i - 1].isalpha() and s[i].isalpha()
        and s[i - 1].islower() and s[i].isupper()
    ]
    if camel:
        i = rng.choice(camel)
        return (s[:i] + " " + s[i:], "spacing")
    # Strategy 2: long alphabetic run, split near the middle.
    runs = [
        (m.start(), m.end()) for m in re.finditer(r"[A-Za-z]{6,}", s)
    ]
    if not runs:
        raise ValueError("no long alpha run for split_word")
    start, end = rng.choice(runs)
    mid = start + (end - start) // 2
    return (s[:mid] + " " + s[mid:], "spacing")


def _mut_fuse_words(s: str, rng: random.Random) -> tuple[str, str]:
    """Remove a space between two adjacent alpha tokens -- ``Mid Town`` -> ``MidTown``.

    Distinct from the v2 trivial ``_mut_double_space`` (which adds a
    redundant space): this REMOVES a real word-boundary space, which
    under the v4 rulebook is a substantive ``spacing`` typo when the
    canonical form is the spaced variant.
    """
    matches = [
        m.start() for m in re.finditer(r"(?<=[A-Za-z]) (?=[A-Za-z])", s)
    ]
    if not matches:
        raise ValueError("no inter-word space")
    i = rng.choice(matches)
    return (s[:i] + s[i + 1:], "spacing")


# v2's _TYPO_MUTATORS gives us spelling + punctuation. We add the
# two spacing mutators here. Order is informational only -- the
# injector shuffles before trying.
_V4_MUTATORS = tuple(_TYPO_MUTATORS) + (_mut_split_word, _mut_fuse_words)


# ---------------------------------------------------------------------------
# Configuration / defaults
# ---------------------------------------------------------------------------

DEFAULT_DEMO_PATH = Path("RFT/typo/v4/data/demo.jsonl")
DEFAULT_MANIFEST_PATH = Path("RFT/typo/v4/data/manifest.json")
DEFAULT_CORPUS_DIR = Path("data/Excel50KTables")
DEFAULT_OUTPUT_DIR = Path("RFT/typo/v4/data")

MAX_ROWS_PER_CSV = 600
MAX_COLS_PER_CSV = 20
MIN_DISTINCT_PER_COL = 6
MAX_DISTINCT_PER_COL = 600


@dataclass
class ColumnSampleV4:
    source: str
    column: str
    values_with_counts: list[tuple[str, int]]

    @property
    def distinct_values(self) -> list[str]:
        return [v for v, _ in self.values_with_counts]


def _read_columns_from_csv_v4(path: Path) -> list[ColumnSampleV4]:
    try:
        with path.open(
            encoding="utf-8", errors="replace", newline="",
        ) as fh:
            reader = csv.reader(fh)
            try:
                header = next(reader)
            except StopIteration:
                return []
            header = header[:MAX_COLS_PER_CSV]
            per_col: list[Counter[str]] = [Counter() for _ in header]
            for row_idx, row in enumerate(reader):
                if row_idx >= MAX_ROWS_PER_CSV:
                    break
                for ci, val in enumerate(row[:MAX_COLS_PER_CSV]):
                    val = (val or "").strip()
                    if not val:
                        continue
                    per_col[ci][val] += 1
    except (OSError, UnicodeError) as exc:
        logger.debug("Skipping %s (%s)", path, exc)
        return []
    out: list[ColumnSampleV4] = []
    for col, counter in zip(header, per_col):
        if not col or len(counter) < MIN_DISTINCT_PER_COL:
            continue
        out.append(ColumnSampleV4(
            source=path.name, column=col.strip(),
            values_with_counts=counter.most_common(MAX_DISTINCT_PER_COL),
        ))
    return out


@dataclass
class SynthV4Config:
    mode: str
    n_columns: int
    p_inject_typos: float
    min_typos_per_column: int
    max_typos_per_column: int
    seed: int
    train_ratio: float
    llm_model: str
    llm_few_shot_columns: int
    corpus_dir: str
    demo_path: str
    max_distinct: int
    english_only: bool = True
    english_threshold: float = (
        _typo_v4_task.TYPO_V4_ENGLISH_VALUE_THRESHOLD
    )
    denylist_sources: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Demo few-shot loader
# ---------------------------------------------------------------------------


def _load_v4_demo_columns(demo_path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not demo_path.exists():
        return out
    with demo_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out.append({
                "source": row.get("source"),
                "column": row.get("column"),
                "gold_typos": json.loads(
                    row.get("gold_typos", "[]") or "[]"
                ),
                "ambiguous_values": json.loads(
                    row.get("ambiguous_values", "[]") or "[]"
                ),
                "user_content": (
                    row.get("messages", [{}, {}])[1].get("content", "")
                )[:1200],
            })
    return out


def _load_denylist(manifest_path: Path) -> set[str]:
    if not manifest_path.exists():
        return set()
    try:
        m = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return set()
    return set(m.get("demo_sources", []) or [])


# ---------------------------------------------------------------------------
# Deterministic injection
# ---------------------------------------------------------------------------


def _inject_typos_deterministic_v4(
    sample: ColumnSampleV4,
    cfg: SynthV4Config,
    rng: random.Random,
) -> tuple[list[tuple[str, int]], list[dict[str, str]]]:
    if rng.random() >= cfg.p_inject_typos:
        return list(sample.values_with_counts), []
    k = rng.randint(cfg.min_typos_per_column, cfg.max_typos_per_column)
    candidates = [v for v, _ in sample.values_with_counts if len(v) >= 5]
    if len(candidates) < k:
        return list(sample.values_with_counts), []
    rng.shuffle(candidates)
    chosen = candidates[: k * 3]
    gold_typos: list[dict[str, str]] = []
    typo_strings: set[str] = set()
    distinct_set = {v for v, _ in sample.values_with_counts}
    for original in chosen:
        if len(gold_typos) >= k:
            break
        mutators = list(_V4_MUTATORS)
        rng.shuffle(mutators)
        for mut in mutators:
            try:
                mutated, flavour = mut(original, rng)
            except ValueError:
                continue
            if not mutated or mutated == original:
                continue
            if mutated in distinct_set or mutated in typo_strings:
                continue
            if _typo_v4_task.is_v4_trivial_correction(mutated, original):
                continue
            if flavour not in _typo_v4_task.TYPO_V4_FLAVOURS:
                # v2 mutators may emit "whitespace"; we don't accept
                # that flavour in v4 -- it's CLEAN per \u00a72.
                continue
            gold_typos.append({
                "value": mutated,
                "correction": original,
                "flavour": flavour,
            })
            typo_strings.add(mutated)
            break
    if not gold_typos:
        return list(sample.values_with_counts), []
    augmented = list(sample.values_with_counts) + [
        (g["value"], 1) for g in gold_typos
    ]
    augmented.sort(key=lambda vc: -vc[1])
    return augmented, gold_typos


# ---------------------------------------------------------------------------
# LLM injection
# ---------------------------------------------------------------------------


def _few_shot_block(
    demo_columns: list[dict[str, Any]], rng: random.Random, k: int,
) -> str:
    if not demo_columns or k <= 0:
        return ""
    typo_cols = [c for c in demo_columns if c["gold_typos"]]
    other_cols = [c for c in demo_columns if not c["gold_typos"]]
    rng.shuffle(typo_cols)
    rng.shuffle(other_cols)
    picked: list[dict[str, Any]] = []
    while len(picked) < k and (typo_cols or other_cols):
        if typo_cols:
            picked.append(typo_cols.pop())
        if len(picked) >= k:
            break
        if other_cols:
            picked.append(other_cols.pop())
    blocks: list[str] = []
    for c in picked:
        blocks.append(
            f"--- Example column ---\n"
            f"{c['user_content']}\n\n"
            f"Gold typos (set-valued): "
            f"{json.dumps(c['gold_typos'], ensure_ascii=False)}"
        )
    return "\n\n".join(blocks)


def _llm_synth_prompt(
    sample: ColumnSampleV4, few_shot_text: str, k: int, max_distinct: int,
) -> str:
    listing_text, _shown, _trunc = _typo_v4_task.format_distinct_values(
        sample.values_with_counts, max_distinct=max_distinct,
    )
    rulebook = _typo_v4_task.TYPO_V4_RULEBOOK
    return (
        "You are generating ONE training example for a column-audit "
        "typo model. The model will be shown a single CSV column "
        "and must return the FULL set of typos in it.\n\n"
        f"{rulebook}\n\n"
        + (f"Here are real held-out examples:\n\n{few_shot_text}\n\n"
           if few_shot_text else "")
        + f"NEW column to synthesise typos for:\n"
        f"Column header: {sample.column!r}\n"
        f"Source CSV: {sample.source!r}\n"
        f"Distinct values currently in the column:\n{listing_text}\n\n"
        f"Invent EXACTLY {k} substantive typo(s) for this column. "
        f"Each typo MUST be one of three flavours: spelling, punctuation, "
        f"or spacing (word-boundary tokenisation defect). Do NOT emit "
        f"padding-whitespace or pure-casing edits. The typo string must "
        f"be NEW (not already in the column listing) and the correction "
        f"must reference one of the existing distinct values verbatim.\n\n"
        f"Reply with a single JSON object only, no Markdown fences:\n"
        '{\n'
        '  "typos": [\n'
        '    {"value":      "<the typo string, NEW>",\n'
        '     "correction": "<the canonical form, from existing values>",\n'
        '     "flavour":    "spelling" | "punctuation" | "spacing",\n'
        '     "rationale":  "<one short sentence>"},\n'
        '    ...\n'
        '  ]\n'
        '}'
    )


def _cache_key(
    sample: ColumnSampleV4, few_shot_text: str, k: int, model: str,
) -> str:
    payload = json.dumps({
        "source": sample.source,
        "column": sample.column,
        "values": sample.values_with_counts,
        "few_shot": few_shot_text,
        "k": k,
        "model": model,
        "task_version": "v4",
    }, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(payload).hexdigest()[:24]


def _parse_llm_typos(text: str) -> list[dict[str, Any]]:
    if not text:
        return []
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```\s*$", "", s)
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        return []
    if not isinstance(obj, dict):
        return []
    typos = obj.get("typos") or []
    return typos if isinstance(typos, list) else []


def _validate_llm_typo(
    entry: dict[str, Any],
    *,
    existing_values: set[str],
    existing_typo_values: set[str],
) -> dict[str, str] | None:
    value = entry.get("value")
    correction = entry.get("correction")
    flavour = entry.get("flavour")
    if not isinstance(value, str) or not value.strip():
        return None
    if not isinstance(correction, str) or not correction.strip():
        return None
    value = value.strip()
    correction = correction.strip()
    if value == correction:
        return None
    if _typo_v4_task.is_v4_trivial_correction(value, correction):
        return None
    if value in existing_values or value in existing_typo_values:
        return None
    if correction not in existing_values:
        return None
    if flavour not in _typo_v4_task.TYPO_V4_FLAVOURS:
        flavour = "spelling"
    return {"value": value, "correction": correction, "flavour": flavour}


def _inject_typos_llm(
    sample: ColumnSampleV4,
    cfg: SynthV4Config,
    demo_columns: list[dict[str, Any]],
    rng: random.Random,
    cache_dir: Path,
    client: Any,
) -> tuple[list[tuple[str, int]], list[dict[str, str]]]:
    if rng.random() >= cfg.p_inject_typos:
        return list(sample.values_with_counts), []
    k = rng.randint(cfg.min_typos_per_column, cfg.max_typos_per_column)
    few_shot = _few_shot_block(
        demo_columns, rng, cfg.llm_few_shot_columns,
    )
    key = _cache_key(sample, few_shot, k, cfg.llm_model)
    cache_path = cache_dir / f"{key}.json"
    cached: dict[str, Any] | None = None
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning("Corrupt cache %s -- regenerating.", cache_path)
    if cached is None:
        prompt = _llm_synth_prompt(
            sample, few_shot, k, cfg.max_distinct,
        )
        resp = client.query(
            cfg.llm_model, prompt, reasoning_effort="medium",
        )
        if not resp.ok:
            logger.warning(
                "LLM call failed for %s/%s: %s",
                sample.source, sample.column, resp.error,
            )
            return list(sample.values_with_counts), []
        raw_typos = _parse_llm_typos(resp.text)
        cached = {"k_requested": k, "typos": raw_typos}
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(cached, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    existing_values = {v for v, _ in sample.values_with_counts}
    accepted: list[dict[str, str]] = []
    accepted_values: set[str] = set()
    for raw in cached["typos"]:
        cleaned = _validate_llm_typo(
            raw,
            existing_values=existing_values,
            existing_typo_values=accepted_values,
        )
        if cleaned is None:
            continue
        accepted.append(cleaned)
        accepted_values.add(cleaned["value"])
        if len(accepted) >= k:
            break
    if not accepted:
        return list(sample.values_with_counts), []
    augmented = list(sample.values_with_counts) + [
        (g["value"], 1) for g in accepted
    ]
    augmented.sort(key=lambda vc: -vc[1])
    return augmented, accepted


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _iter_corpus_csvs(
    corpus_dir: Path, denylist: set[str], rng: random.Random,
) -> list[Path]:
    if not corpus_dir.exists():
        return []
    candidates: list[Path] = []
    with os.scandir(corpus_dir) as it:
        for entry in it:
            if not entry.is_file() or not entry.name.endswith(".csv"):
                continue
            if entry.name in denylist:
                continue
            candidates.append(Path(entry.path))
    rng.shuffle(candidates)
    return candidates


def synthesize_v4(
    cfg: SynthV4Config,
    *,
    demo_path: Path,
    manifest_path: Path,
    corpus_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    if not demo_path.exists():
        logger.warning(
            "v4 demo split not found at %s; few-shot exemplars disabled.",
            demo_path,
        )
    demo_columns = _load_v4_demo_columns(demo_path)
    denylist = _load_denylist(manifest_path)
    cfg.denylist_sources = sorted(denylist)
    logger.info(
        "Loaded %d demo columns; %d denylisted CSVs.",
        len(demo_columns), len(denylist),
    )

    rng = random.Random(cfg.seed)
    candidates = _iter_corpus_csvs(corpus_dir, denylist, rng)
    if not candidates:
        raise RuntimeError(
            f"No usable CSVs in {corpus_dir} after applying denylist."
        )

    client = None
    cache_dir = output_dir / "llm_cache"
    if cfg.mode == "llm":
        from src.core.llm_client import LLMClient  # noqa: E402
        client = LLMClient()
        cache_dir.mkdir(parents=True, exist_ok=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.jsonl"
    valid_path = output_dir / "valid.jsonl"
    injections_path = output_dir / "injections.jsonl"
    synth_manifest_path = output_dir / "synth_manifest.json"

    rows: list[dict[str, Any]] = []
    csvs_scanned = 0
    for path in candidates:
        if len(rows) >= cfg.n_columns:
            break
        csvs_scanned += 1
        for sample in _read_columns_from_csv_v4(path):
            if len(rows) >= cfg.n_columns:
                break
            shim = ColumnSample(
                source=sample.source, column=sample.column,
                distinct_values=sample.distinct_values,
            )
            if _looks_numeric_column(shim):
                continue
            if not _column_is_textual(shim):
                continue
            if cfg.english_only and not _typo_v4_task.is_english_column(
                sample.column,
                (v for v, _ in sample.values_with_counts),
                value_threshold=cfg.english_threshold,
            ):
                continue
            if cfg.mode == "deterministic":
                vcs, gold_typos = _inject_typos_deterministic_v4(
                    sample, cfg, rng,
                )
            else:
                vcs, gold_typos = _inject_typos_llm(
                    sample, cfg, demo_columns, rng, cache_dir, client,
                )
            rows.append({
                "source": sample.source,
                "column": sample.column,
                "values_with_counts": vcs,
                "gold_typos": gold_typos,
                "ambiguous_values": [],
                "max_distinct": cfg.max_distinct,
                "_provenance": {
                    "csv_path": str(path),
                    "n_original_distinct": len(sample.values_with_counts),
                    "n_injected_typos": len(gold_typos),
                    "mode": cfg.mode,
                },
            })

    logger.info(
        "Scanned %d CSVs, produced %d column rows (%d with typos).",
        csvs_scanned, len(rows),
        sum(1 for r in rows if r["gold_typos"]),
    )

    rng.shuffle(rows)
    n_train = int(len(rows) * cfg.train_ratio)
    train_rows = rows[:n_train]
    valid_rows = rows[n_train:]

    label_counts_train: Counter[str] = Counter()
    label_counts_valid: Counter[str] = Counter()
    flavour_counts: Counter[str] = Counter()

    def _bucket_of(r: dict[str, Any]) -> str:
        return "has_typo" if r["gold_typos"] else "clean"

    with train_path.open("w", encoding="utf-8") as fh_train, \
            valid_path.open("w", encoding="utf-8") as fh_valid, \
            injections_path.open("w", encoding="utf-8") as fh_inj:
        for bucket, fh, counter, split_name in [
            (train_rows, fh_train, label_counts_train, "train"),
            (valid_rows, fh_valid, label_counts_valid, "valid"),
        ]:
            for r in bucket:
                item = {
                    "source": r["source"],
                    "column": r["column"],
                    "values_with_counts": r["values_with_counts"],
                    "gold_typos": r["gold_typos"],
                    "ambiguous_values": r["ambiguous_values"],
                    "max_distinct": r["max_distinct"],
                }
                try:
                    rec = _typo_v4_task.build_record(item)
                except ValueError as exc:
                    logger.warning(
                        "Dropping malformed row %s/%s: %s",
                        r["source"], r["column"], exc,
                    )
                    continue
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                counter[_bucket_of(r)] += 1
                for g in r["gold_typos"]:
                    flavour_counts[g.get("flavour", "?")] += 1
                fh_inj.write(json.dumps({
                    "split": split_name,
                    "source": r["source"],
                    "column": r["column"],
                    "gold_typos": r["gold_typos"],
                    "n_distinct_after_injection": len(
                        r["values_with_counts"]
                    ),
                    **r["_provenance"],
                }, ensure_ascii=False) + "\n")

    manifest = {
        "config": asdict(cfg),
        "csvs_scanned": csvs_scanned,
        "rows_total": len(rows),
        "n_train": sum(label_counts_train.values()),
        "n_valid": sum(label_counts_valid.values()),
        "train_label_counts": dict(label_counts_train),
        "valid_label_counts": dict(label_counts_valid),
        "injected_flavour_counts": dict(flavour_counts),
        "demo_path": str(demo_path),
        "corpus_dir": str(corpus_dir),
        "output_paths": {
            "train": str(train_path),
            "valid": str(valid_path),
            "injections": str(injections_path),
        },
    }
    synth_manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Wrote synth manifest -> %s", synth_manifest_path)
    return manifest


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode", choices=["deterministic", "llm"],
        default="deterministic",
    )
    parser.add_argument("--demo-path", type=Path, default=DEFAULT_DEMO_PATH)
    parser.add_argument(
        "--manifest-path", type=Path, default=DEFAULT_MANIFEST_PATH,
    )
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-columns", type=int, default=120)
    parser.add_argument("--p-inject-typos", type=float, default=0.6)
    parser.add_argument("--min-typos-per-column", type=int, default=1)
    parser.add_argument("--max-typos-per-column", type=int, default=3)
    parser.add_argument("--train-ratio", type=float, default=0.80)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--llm-model", default="gpt-5.2,gpt-5.4,gpt-5.5")
    parser.add_argument("--llm-few-shot-columns", type=int, default=3)
    parser.add_argument(
        "--max-distinct", type=int,
        default=_typo_v4_task.TYPO_V4_MAX_DISTINCT,
    )
    parser.add_argument("--no-english-only", action="store_true")
    parser.add_argument(
        "--english-threshold", type=float,
        default=_typo_v4_task.TYPO_V4_ENGLISH_VALUE_THRESHOLD,
    )
    args = parser.parse_args(argv)

    cfg = SynthV4Config(
        mode=args.mode,
        n_columns=args.n_columns,
        p_inject_typos=args.p_inject_typos,
        min_typos_per_column=args.min_typos_per_column,
        max_typos_per_column=args.max_typos_per_column,
        seed=args.seed,
        train_ratio=args.train_ratio,
        llm_model=args.llm_model,
        llm_few_shot_columns=args.llm_few_shot_columns,
        corpus_dir=str(args.corpus_dir),
        demo_path=str(args.demo_path),
        max_distinct=args.max_distinct,
        english_only=not args.no_english_only,
        english_threshold=args.english_threshold,
    )
    manifest = synthesize_v4(
        cfg,
        demo_path=args.demo_path,
        manifest_path=args.manifest_path,
        corpus_dir=args.corpus_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
