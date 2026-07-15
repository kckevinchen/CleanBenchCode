"""Synthesize typo-RFT training data for the v3 column-audit task.

Each output row is ONE column's view: the trainee sees the whole
column's distinct values and must return the full list of typos in
that column. So synthesis works at the column level, not the cell
level (contrast with ``experiments.synth_typo_llm`` v2):

* We sample a real CSV column from the corpus.
* With probability ``cfg.p_inject_typos`` we inject 1-3 synthetic
  typos: pick that many existing values, mutate each with one of
  ``_TYPO_MUTATORS``, and append the mutated string as a new
  ``count=1`` distinct value. The gold typo set is the list of
  ``{value, correction, flavour}`` for those mutated entries.
* Otherwise the column is left as-is (gold typos = []). This
  produces "clean column" rows that teach the trainee to emit an
  empty list when nothing is wrong.

Why this matters:

* The v3 grader is a set-F1 over (value, correction) pairs. A model
  that hallucinates typos on clean columns is heavily penalised.
  Including clean-column examples in training is essential.
* The mutators are reused from v2 (``experiments.synth_typo_llm``)
  so synthesised typos match the distribution v2 trained on; this
  isolates "per-column vs per-cell" as the only changed variable
  in a v2-vs-v3 head-to-head.

Two modes (same names as v2):

* ``--mode deterministic`` -- no LLM calls. Mutators chosen by RNG.
* ``--mode llm`` -- ``src.core.llm_client.LLMClient`` is asked to
  invent 1-3 realistic typos for the column; results validated and
  cached to disk.

Usage
-----
::

    # 1. v3 split (once, separate script)
    python -m experiments.split_typo_benchmark_v3

    # 2a. cheap, deterministic
    python -m experiments.synth_typo_llm_v3 --mode deterministic

    # 2b. LLM-driven
    python -m experiments.synth_typo_llm_v3 --mode llm \\
        --n-columns 40
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

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from RFT._common import _typo_task, _typo_v3_task
from RFT._common.synth_typo_llm import (  # noqa: E402
    _TYPO_MUTATORS,
    _column_is_textual,
    _looks_numeric_column,
    ColumnSample,
)

logger = logging.getLogger("synth_typo_llm_v3")


# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------

DEFAULT_DEMO_PATH = Path("outputs/typo_rft/benchmark_split_v3/demo.jsonl")
DEFAULT_MANIFEST_PATH = Path(
    "outputs/typo_rft/benchmark_split_v3/manifest.json"
)
DEFAULT_CORPUS_DIR = Path("data/Excel50KTables")
DEFAULT_OUTPUT_DIR = Path("outputs/typo_rft/synth_v3")

# Per-CSV scan budget. v3 needs the FULL distinct-value list per
# column (it's the trainee's input), so we cap the column count for
# memory but allow more rows per CSV than v2 (which only needed
# enough samples for per-cell labels).
MAX_ROWS_PER_CSV = 600
MAX_COLS_PER_CSV = 20
MIN_DISTINCT_PER_COL = 6
# Hard ceiling on distinct values stored per column. The trainee
# sees at most ``TYPO_V3_MAX_DISTINCT`` (500) at prompt-build time;
# storing more here lets the cap be exercised honestly.
MAX_DISTINCT_PER_COL = 600


# ---------------------------------------------------------------------------
# CSV reading (v3-specific: stores full count map per column)
# ---------------------------------------------------------------------------


@dataclass
class ColumnSampleV3:
    """One column's full distinct-value count map, ready for synthesis.

    Contrast with ``ColumnSample`` (v2): we keep per-value counts so
    that injected typos appear at count=1 and the column listing the
    trainee sees has frequency information.
    """

    source: str
    column: str
    values_with_counts: list[tuple[str, int]]

    @property
    def distinct_values(self) -> list[str]:
        return [v for v, _ in self.values_with_counts]


def _read_columns_from_csv_v3(path: Path) -> list[ColumnSampleV3]:
    """Read up to ``MAX_COLS_PER_CSV`` columns; count distinct values.

    Skips null / empty cells (per v3 grader semantics: blanks aren't
    candidates for typos).
    """
    try:
        with path.open(
            encoding="utf-8", errors="replace", newline=""
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

    out: list[ColumnSampleV3] = []
    for col, counter in zip(header, per_col):
        if not col:
            continue
        if len(counter) < MIN_DISTINCT_PER_COL:
            continue
        # Most-common-first; clip to MAX_DISTINCT_PER_COL.
        vcs = counter.most_common(MAX_DISTINCT_PER_COL)
        out.append(ColumnSampleV3(
            source=path.name, column=col.strip(),
            values_with_counts=vcs,
        ))
    return out


# ---------------------------------------------------------------------------
# Synthesis configuration
# ---------------------------------------------------------------------------


@dataclass
class SynthV3Config:
    """Knobs for a v3 synthesis run; serialised into manifest.json."""

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
    # v3 default: train and eval are English-only so the model isn't
    # asked to flag legitimate non-English glyphs as typos.
    english_only: bool = True
    english_threshold: float = (
        _typo_v3_task.TYPO_V3_ENGLISH_VALUE_THRESHOLD
    )
    denylist_sources: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Demo-driven exemplars: load v3 demo columns for LLM few-shot
# ---------------------------------------------------------------------------


def _load_v3_demo_columns(demo_path: Path) -> list[dict[str, Any]]:
    """Read the v3 demo split: each row IS already a per-column record.

    For LLM few-shot we pull out (source, column, gold_typos,
    ambiguous_values, sample of distinct values).
    """
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
                "gold_typos": json.loads(row.get("gold_typos", "[]") or "[]"),
                "ambiguous_values": json.loads(
                    row.get("ambiguous_values", "[]") or "[]"
                ),
                "user_content": row.get("messages", [{}, {}])[1].get(
                    "content", ""
                )[:1200],
            })
    return out


def _load_denylist(manifest_path: Path) -> set[str]:
    """v3 manifest's demo_sources double as the deny-list."""
    if not manifest_path.exists():
        return set()
    try:
        m = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return set()
    return set(m.get("demo_sources", []) or [])


# ---------------------------------------------------------------------------
# Deterministic mode
# ---------------------------------------------------------------------------


def _inject_typos_deterministic(
    sample: ColumnSampleV3,
    cfg: SynthV3Config,
    rng: random.Random,
) -> tuple[list[tuple[str, int]], list[dict[str, str]]]:
    """Mutate K existing values; return augmented value list + gold typos.

    The mutated values are appended to ``sample.values_with_counts``
    as ``count=1`` entries (one synthetic row carrying the typo).
    Original entries are left intact.

    Returns ``(values_with_counts, gold_typos)``.
    """
    if rng.random() >= cfg.p_inject_typos:
        # "clean column" example -- no injections.
        return list(sample.values_with_counts), []

    k = rng.randint(cfg.min_typos_per_column, cfg.max_typos_per_column)
    # Prefer values long enough to support the mutators (avoid
    # 'NY' -> 'NN' which neither swap nor delete can produce cleanly).
    candidates = [v for v, _ in sample.values_with_counts if len(v) >= 5]
    if len(candidates) < k:
        return list(sample.values_with_counts), []

    rng.shuffle(candidates)
    chosen = candidates[: k * 3]  # 3x oversample for mutator-rejection slack
    gold_typos: list[dict[str, str]] = []
    typo_strings: set[str] = set()
    distinct_set = {v for v, _ in sample.values_with_counts}
    for original in chosen:
        if len(gold_typos) >= k:
            break
        mutators = list(_TYPO_MUTATORS)
        rng.shuffle(mutators)
        produced = False
        for mut in mutators:
            try:
                mutated, flavour = mut(original, rng)
            except ValueError:
                continue
            if not mutated or mutated == original:
                continue
            if mutated in distinct_set or mutated in typo_strings:
                # Don't inject a "typo" that already exists in the column
                # under another label.
                continue
            if _typo_task.is_trivial_typo_correction(mutated, original):
                continue
            gold_typos.append({
                "value": mutated,
                "correction": original,
                "flavour": flavour,
            })
            typo_strings.add(mutated)
            produced = True
            break
        if not produced:
            continue

    if not gold_typos:
        return list(sample.values_with_counts), []

    augmented = list(sample.values_with_counts) + [
        (g["value"], 1) for g in gold_typos
    ]
    # Re-sort by count desc so the listing presented to the trainee
    # mirrors the eval-time distribution.
    augmented.sort(key=lambda vc: -vc[1])
    return augmented, gold_typos


# ---------------------------------------------------------------------------
# LLM mode
# ---------------------------------------------------------------------------


def _few_shot_v3_block(
    demo_columns: list[dict[str, Any]],
    rng: random.Random,
    k: int,
) -> str:
    """Render ``k`` v3 demo columns as exemplar blocks."""
    if not demo_columns or k <= 0:
        return ""
    # Prefer columns that contain a typo so the LLM gets a positive
    # signal; fall back to whatever's available.
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
            f"Gold typos (set-valued): {json.dumps(c['gold_typos'], ensure_ascii=False)}"
        )
    return "\n\n".join(blocks)


def _llm_synth_prompt_v3(
    sample: ColumnSampleV3,
    few_shot_text: str,
    k: int,
    max_distinct: int,
) -> str:
    """Build the LLM synthesis prompt for one column.

    The mode-decision (inject vs clean) and K-decision are made
    client-side; here we only ask the LLM to produce ``k`` realistic
    typos. Empirically the LLM ignores soft probability hints like
    "60% inject" in the prompt -- it always chose 'inject' and the
    upper bound K -- so we now hard-constrain both before calling.
    """
    listing_text, _shown, _trunc = _typo_v3_task.format_distinct_values(
        sample.values_with_counts, max_distinct=max_distinct,
    )
    rulebook = _typo_v3_task.TYPO_V3_RULEBOOK
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
        f"For each: pick an existing value as the canonical form, then "
        f"introduce a SUBSTANTIVE defect (spelling distortion -- one or "
        f"two character swap / drop / insertion -- OR missing-comma+space "
        f"punctuation). Do NOT use whitespace, casing, or pure punctuation-"
        f"rewrite edits. The typo string must be NEW (not already in the "
        f"column listing) and the correction must reference one of the "
        f"existing distinct values verbatim.\n\n"
        f"Reply with a single JSON object only, no Markdown fences:\n"
        '{\n'
        '  "typos": [\n'
        '    {"value":      "<the typo string, NEW>",\n'
        '     "correction": "<the canonical form, from existing values>",\n'
        '     "flavour":    "spelling" | "punctuation",\n'
        '     "rationale":  "<one short sentence>"},\n'
        '    ...\n'
        '  ]\n'
        '}'
    )


def _cache_key_v3(
    sample: ColumnSampleV3,
    few_shot_text: str,
    k: int,
    model: str,
) -> str:
    payload = json.dumps({
        "source": sample.source,
        "column": sample.column,
        "values": sample.values_with_counts,
        "few_shot": few_shot_text,
        "k": k,
        "model": model,
    }, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(payload).hexdigest()[:24]


def _parse_llm_typos(text: str) -> list[dict[str, Any]]:
    """Parse a strict JSON reply. Tolerate ```json fences. Returns typos list."""
    if not text:
        return []
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```\s*$", "", s)
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        logger.debug("LLM reply was not JSON: %r", text[:200])
        return []
    if not isinstance(obj, dict):
        return []
    typos = obj.get("typos") or []
    if not isinstance(typos, list):
        typos = []
    return typos


def _validate_llm_typo(
    entry: dict[str, Any],
    *,
    existing_values: set[str],
    existing_typo_values: set[str],
) -> dict[str, str] | None:
    """Drop LLM typos that violate the rulebook or duplicate existing values."""
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
    if _typo_task.is_trivial_typo_correction(value, correction):
        return None
    if value in existing_values or value in existing_typo_values:
        # Typo string must be a NEW distinct entry.
        return None
    if correction not in existing_values:
        # Correction must point at an existing canonical value
        # (otherwise the trainee has no way to ground it).
        return None
    if flavour not in _typo_v3_task.TYPO_V3_FLAVOURS:
        # Default rather than reject -- LLMs sometimes return "typo"
        # or capitalise it.
        flavour = "spelling"
    return {"value": value, "correction": correction, "flavour": flavour}


def _inject_typos_llm(
    sample: ColumnSampleV3,
    cfg: SynthV3Config,
    demo_columns: list[dict[str, Any]],
    rng: random.Random,
    cache_dir: Path,
    client: Any,
) -> tuple[list[tuple[str, int]], list[dict[str, str]]]:
    """LLM-driven typo injection with on-disk caching.

    Client-side decisions (made BEFORE the LLM call so the LLM can
    only influence content, not distribution):

      * Whether to inject at all -- ``rng.random() < p_inject_typos``.
        Soft probability hints in the prompt ("60% inject") were
        empirically ignored; the LLM always chose inject. Deciding
        here also saves an API call for clean-column rows.
      * Exact K (number of typos) -- uniform in [min, max].
    """
    if rng.random() >= cfg.p_inject_typos:
        return list(sample.values_with_counts), []

    k = rng.randint(cfg.min_typos_per_column, cfg.max_typos_per_column)
    few_shot = _few_shot_v3_block(
        demo_columns, rng, cfg.llm_few_shot_columns
    )
    key = _cache_key_v3(sample, few_shot, k, cfg.llm_model)
    cache_path = cache_dir / f"{key}.json"
    cached: dict[str, Any] | None = None
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning("Corrupt cache %s -- regenerating.", cache_path)

    if cached is None:
        prompt = _llm_synth_prompt_v3(
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
            # Cap at requested K even if LLM returned more.
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
) -> Iterable[Path]:
    if not corpus_dir.exists():
        return
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


def synthesize_v3(
    cfg: SynthV3Config,
    *,
    demo_path: Path,
    manifest_path: Path,
    corpus_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Run a full v3 synthesis pass; return manifest dict."""
    if not demo_path.exists():
        logger.warning(
            "v3 demo split not found at %s; few-shot exemplars disabled.",
            demo_path,
        )
    demo_columns = _load_v3_demo_columns(demo_path)
    denylist = _load_denylist(manifest_path)
    cfg.denylist_sources = sorted(denylist)
    logger.info(
        "Loaded %d demo columns; %d denylisted CSVs.",
        len(demo_columns), len(denylist),
    )

    rng = random.Random(cfg.seed)
    candidates = list(_iter_corpus_csvs(corpus_dir, denylist, rng))
    if not candidates:
        raise RuntimeError(
            f"No usable CSVs in {corpus_dir} after applying denylist."
        )

    client = None
    cache_dir = output_dir / "llm_cache"
    if cfg.mode == "llm":
        from src.core.llm_client import LLMClient  # local import: optional dep
        client = LLMClient()
        cache_dir.mkdir(parents=True, exist_ok=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.jsonl"
    valid_path = output_dir / "valid.jsonl"
    injections_path = output_dir / "injections.jsonl"
    manifest_out_path = output_dir / "manifest.json"

    rows: list[dict[str, Any]] = []
    csvs_scanned = 0
    for path in candidates:
        if len(rows) >= cfg.n_columns:
            break
        csvs_scanned += 1
        for sample in _read_columns_from_csv_v3(path):
            if len(rows) >= cfg.n_columns:
                break
            # Reuse v2's column-quality gates (they live in
            # synth_typo_llm and operate on ColumnSample, but only
            # access ``distinct_values``, which both samples expose).
            shim = ColumnSample(
                source=sample.source, column=sample.column,
                distinct_values=sample.distinct_values,
            )
            if _looks_numeric_column(shim):
                continue
            if not _column_is_textual(shim):
                continue
            if cfg.english_only and not _typo_v3_task.is_english_column(
                sample.column,
                (v for v, _ in sample.values_with_counts),
                value_threshold=cfg.english_threshold,
            ):
                continue

            if cfg.mode == "deterministic":
                vcs, gold_typos = _inject_typos_deterministic(
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

    # Stable shuffle then split.
    rng.shuffle(rows)
    n_train = int(len(rows) * cfg.train_ratio)
    train_rows = rows[:n_train]
    valid_rows = rows[n_train:]

    label_counts_train: Counter[str] = Counter()
    label_counts_valid: Counter[str] = Counter()

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
                    rec = _typo_v3_task.build_record(item)
                except ValueError as exc:
                    logger.warning(
                        "Dropping malformed row %s/%s: %s",
                        r["source"], r["column"], exc,
                    )
                    continue
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                counter[_bucket_of(r)] += 1
                fh_inj.write(json.dumps({
                    "split": split_name,
                    "source": r["source"],
                    "column": r["column"],
                    "gold_typos": r["gold_typos"],
                    "n_distinct_after_injection": len(r["values_with_counts"]),
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
        "demo_path": str(demo_path),
        "corpus_dir": str(corpus_dir),
        "output_paths": {
            "train": str(train_path),
            "valid": str(valid_path),
            "injections": str(injections_path),
        },
    }
    manifest_out_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Wrote manifest -> %s", manifest_out_path)
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
        "--mode", choices=["deterministic", "llm"], default="deterministic",
    )
    parser.add_argument("--demo-path", type=Path, default=DEFAULT_DEMO_PATH)
    parser.add_argument(
        "--manifest-path", type=Path, default=DEFAULT_MANIFEST_PATH,
    )
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-columns", type=int, default=40)
    parser.add_argument("--p-inject-typos", type=float, default=0.6)
    parser.add_argument("--min-typos-per-column", type=int, default=1)
    parser.add_argument("--max-typos-per-column", type=int, default=3)
    parser.add_argument("--train-ratio", type=float, default=0.80)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--llm-model", default="gpt-5.2,gpt-5.4,gpt-5.5",
    )
    parser.add_argument(
        "--llm-few-shot-columns", type=int, default=3,
    )
    parser.add_argument(
        "--max-distinct", type=int,
        default=_typo_v3_task.TYPO_V3_MAX_DISTINCT,
    )
    parser.add_argument(
        "--no-english-only", action="store_true",
        help=(
            "Disable the English-only filter. v3 default is to skip "
            "columns whose header or values are not predominantly "
            "ASCII Latin alphabet."
        ),
    )
    parser.add_argument(
        "--english-threshold", type=float,
        default=_typo_v3_task.TYPO_V3_ENGLISH_VALUE_THRESHOLD,
        help=(
            "Minimum fraction of ASCII alphabetic characters across "
            "a column's distinct values for it to be considered "
            "English. Default: %(default).2f."
        ),
    )
    args = parser.parse_args(argv)

    cfg = SynthV3Config(
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
    manifest = synthesize_v3(
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
