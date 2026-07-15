"""Synthesize typo-RFT training data from real CSV columns.

Pipeline:
    benchmark  --split-->  demo + eval
    demo                  ----> seed few-shot exemplars for the LLM
    data/<corpus>         ----> source of fresh columns (deny-listed
                                against any source in the benchmark)
    -------------------------->  train.jsonl + valid.jsonl
                                 injections.jsonl (provenance sidecar)

Two synthesis modes:

* ``--mode deterministic`` (default, cheap, no LLM calls):
    For each sampled column we draw values directly from the CSV and
    apply rule-defined mutations to a fraction of them
    (double-space, char swap/insert/delete, missing comma+space,
    truncation). Labels are derived from the mutation we chose, so
    they are by construction consistent with the rulebook. Use this
    for first-pass iteration and as a sanity baseline.

* ``--mode llm`` (uses ``src.core.llm_client.LLMClient``):
    For each sampled column we prompt the LLM with the demo split as
    few-shot exemplars (errors / clean / ambiguous in proportion) and
    ask it to generate N candidate (value, label, correction) triples
    matching the rulebook. LLM responses are cached on disk keyed by
    ``(column_signature, exemplar_subset_hash)`` so re-runs are free.
    Use this when you want the training set to cover the same shape
    of typos the real annotators flagged in the benchmark.

Both modes share the same output schema (RFT JSONL via
``experiments._typo_task.build_record``) so downstream
``demo_rft prepare --task typo`` works against either.

Usage
-----
::

    # 1. split (once)
    python -m experiments.split_typo_benchmark

    # 2a. cheap, deterministic
    python -m experiments.synth_typo_llm --mode deterministic

    # 2b. LLM-driven, realistic
    python -m experiments.synth_typo_llm --mode llm \\
        --n-columns 50 --examples-per-column 4
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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from RFT._common import _typo_task  # noqa: E402

logger = logging.getLogger("synth_typo_llm")


# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------

DEFAULT_DEMO_PATH = Path("outputs/typo_rft/benchmark_split/demo.jsonl")
DEFAULT_MANIFEST_PATH = Path("outputs/typo_rft/benchmark_split/manifest.json")
DEFAULT_CORPUS_DIR = Path("data/Excel50KTables")
DEFAULT_OUTPUT_DIR = Path("outputs/typo_rft/synth")

# Bounded read budget per CSV: we don't need the whole file, just a
# representative sample of distinct values per column. Wider files
# hit the file-handle ceiling faster, so we also cap columns.
MAX_ROWS_PER_CSV = 200
MAX_COLS_PER_CSV = 20
MIN_DISTINCT_PER_COL = 4
MAX_DISTINCT_PER_COL = 60

# Rulebook-aware mutation toolbox. Each mutator returns (mutated, flavour)
# OR raises ValueError if the value isn't a good substrate (e.g., too
# short to drop a comma from).
#
# v2 NOTE: ``_mut_double_space`` is intentionally NOT in ``_TYPO_MUTATORS``
# below. Its output (a duplicated internal space) is a *trivial* edit
# that v2 excludes from training -- a one-line regex catches it and
# teaching the model to flag it would just bias label predictions
# upward.

_PUNCT_RE = re.compile(r"[,;]")


def _mut_double_space(s: str, rng: random.Random) -> tuple[str, str]:
    """Insert a double space between two adjacent tokens."""
    tokens = s.split(" ")
    if len(tokens) < 2:
        raise ValueError("not multi-token")
    i = rng.randrange(len(tokens) - 1)
    tokens[i] += " "
    return (" ".join(tokens), "whitespace")


def _mut_char_swap(s: str, rng: random.Random) -> tuple[str, str]:
    """Transpose two adjacent characters in a word-y region."""
    if len(s) < 4:
        raise ValueError("too short")
    # Find a position where both chars are letters (avoid trashing
    # digit-only fragments that the rulebook says shouldn't be typo'd).
    candidates = [
        i for i in range(len(s) - 1)
        if s[i].isalpha() and s[i + 1].isalpha() and s[i] != s[i + 1]
    ]
    if not candidates:
        raise ValueError("no letter-letter pair")
    i = rng.choice(candidates)
    return (s[:i] + s[i + 1] + s[i] + s[i + 2:], "spelling")


def _mut_char_delete(s: str, rng: random.Random) -> tuple[str, str]:
    """Drop a non-first non-last letter."""
    if len(s) < 5:
        raise ValueError("too short")
    candidates = [i for i in range(1, len(s) - 1) if s[i].isalpha()]
    if not candidates:
        raise ValueError("no internal letter")
    i = rng.choice(candidates)
    return (s[:i] + s[i + 1:], "spelling")


def _mut_char_insert(s: str, rng: random.Random) -> tuple[str, str]:
    """Duplicate a letter."""
    if len(s) < 4:
        raise ValueError("too short")
    candidates = [i for i in range(len(s)) if s[i].isalpha()]
    if not candidates:
        raise ValueError("no letter")
    i = rng.choice(candidates)
    return (s[:i + 1] + s[i] + s[i + 1:], "spelling")


def _mut_missing_comma_space(s: str, rng: random.Random) -> tuple[str, str]:
    """Replace ', ' with ',' OR drop the space after a comma/semicolon."""
    matches = [m.start() for m in re.finditer(r"[,;]\s", s)]
    if not matches:
        raise ValueError("no ', ' pattern")
    i = rng.choice(matches)
    return (s[:i + 1] + s[i + 2:], "punctuation")


def _mut_truncate(s: str, rng: random.Random) -> tuple[str, str]:
    """Drop a 1-3 char tail. Used for the 'ambiguous' bucket only."""
    if len(s) < 6:
        raise ValueError("too short")
    n = rng.randint(1, 3)
    return (s[:-n], "ambiguous")


# Typo mutators (label=='typo'). v2 EXCLUDES ``_mut_double_space`` --
# that mutator produces whitespace-only corrections, which the trivial-
# correction filter strips out anyway. The remaining mutators all
# produce substantive spelling / punctuation edits.
_TYPO_MUTATORS = (
    _mut_missing_comma_space,
    _mut_char_swap,
    _mut_char_delete,
    _mut_char_insert,
)


# ---------------------------------------------------------------------------
# CSV reading
# ---------------------------------------------------------------------------


@dataclass
class ColumnSample:
    """A bag of distinct values from one column of one CSV."""

    source: str
    column: str
    distinct_values: list[str]


def _read_columns_from_csv(path: Path) -> list[ColumnSample]:
    """Read up to MAX_COLS_PER_CSV columns; capture distinct values."""
    try:
        # Open in text mode with a permissive error handler. Some files
        # in data/ have stray bytes; we silently drop them rather than
        # failing the entire synthesis batch.
        with path.open(encoding="utf-8", errors="replace", newline="") as fh:
            reader = csv.reader(fh)
            try:
                header = next(reader)
            except StopIteration:
                return []
            header = header[:MAX_COLS_PER_CSV]
            # Use ordered dicts (by insertion) to preserve original
            # value order while de-duplicating; the rulebook cares
            # about distinct values, not row order.
            per_col: list[dict[str, None]] = [dict() for _ in header]
            for row_idx, row in enumerate(reader):
                if row_idx >= MAX_ROWS_PER_CSV:
                    break
                for ci, val in enumerate(row[:MAX_COLS_PER_CSV]):
                    val = (val or "").strip()
                    if not val:
                        continue
                    per_col[ci][val] = None
    except (OSError, UnicodeError) as exc:
        logger.debug("Skipping %s (%s)", path, exc)
        return []

    out: list[ColumnSample] = []
    for col, distincts in zip(header, per_col):
        if not col:
            continue
        if len(distincts) < MIN_DISTINCT_PER_COL:
            continue
        values = list(distincts.keys())[:MAX_DISTINCT_PER_COL]
        out.append(ColumnSample(
            source=path.name,
            column=col.strip(),
            distinct_values=values,
        ))
    return out


def _looks_numeric_column(sample: ColumnSample) -> bool:
    """Reject columns whose values are mostly numeric / date-shaped.

    The rulebook says numeric and date values are NEVER typos, so a
    column dominated by them gives us no useful training signal in
    either direction.
    """
    n = len(sample.distinct_values)
    if n == 0:
        return True
    numeric = 0
    date_like = 0
    for v in sample.distinct_values:
        if re.fullmatch(r"-?\d+(?:\.\d+)?", v):
            numeric += 1
            continue
        if re.search(r"\d{1,4}[/\-]\d{1,2}[/\-]\d{1,4}", v):
            date_like += 1
    return (numeric + date_like) / n > 0.7


def _column_is_textual(sample: ColumnSample) -> bool:
    """Require at least one value of length >= 5 with a letter pair."""
    return any(
        len(v) >= 5 and re.search(r"[A-Za-z]{2,}", v)
        for v in sample.distinct_values
    )


# ---------------------------------------------------------------------------
# Synthesis state
# ---------------------------------------------------------------------------


@dataclass
class SynthConfig:
    """Knobs for a synthesis run; serialised into manifest.json."""

    mode: str
    n_columns: int
    examples_per_column: int
    p_typo: float
    p_ambiguous: float
    p_clean: float
    seed: int
    train_ratio: float
    llm_model: str
    llm_few_shot_per_label: int
    corpus_dir: str
    demo_path: str
    denylist_sources: list[str] = field(default_factory=list)


def _load_demo_exemplars(demo_path: Path) -> dict[str, list[dict[str, Any]]]:
    """Group demo rows by label for downstream few-shot sampling."""
    out: dict[str, list[dict[str, Any]]] = {
        "typo": [], "clean": [], "ambiguous": [],
    }
    with demo_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            # Benchmark uses 'error' instead of 'typo' in the label
            # field. Normalise it here so the rest of the script can
            # think in trainee-vocab.
            label = row.get("label")
            if label == "error":
                label = "typo"
            if label in out:
                out[label].append(row)
    return out


def _load_denylist(manifest_path: Path) -> set[str]:
    """Source filenames the synth must NOT pick a column from."""
    if not manifest_path.exists():
        return set()
    try:
        m = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return set()
    return set(m.get("benchmark_sources", []) or [])


# ---------------------------------------------------------------------------
# Deterministic mode
# ---------------------------------------------------------------------------


def _gen_examples_deterministic(
    sample: ColumnSample,
    cfg: SynthConfig,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Produce ``cfg.examples_per_column`` (value, label, correction) rows.

    Each example is drawn from ``sample.distinct_values``:
      * Probability ``cfg.p_typo``: pick a value, apply the first
        applicable typo mutator (in a random order); label=='typo',
        gold_correction is the original (un-mutated) value.
      * Probability ``cfg.p_ambiguous``: pick a value, truncate it
        slightly; label=='ambiguous', gold_correction=None.
      * Probability ``cfg.p_clean``: use the value as-is;
        label=='clean', gold_correction=None.
    """
    examples: list[dict[str, Any]] = []
    # v2: binary task. ``cfg.p_ambiguous`` is preserved in the config
    # for backward-compat / manifest hygiene but treated as 0 here --
    # the model never trains on ambiguous rows.
    weights = (cfg.p_typo, 0.0, cfg.p_clean)
    kinds = ("typo", "ambiguous", "clean")
    if abs(sum(weights) - 1.0) > 1e-6:
        # Normalise instead of failing -- the CLI defaults already sum
        # to 1, but allow operator overrides without a careful add-up.
        total = sum(weights) or 1.0
        weights = tuple(w / total for w in weights)

    for _ in range(cfg.examples_per_column):
        kind = rng.choices(kinds, weights=weights, k=1)[0]
        # Take the value to mutate from a copy so we can fall back if
        # all mutators reject it.
        candidates = sample.distinct_values[:]
        rng.shuffle(candidates)

        example: dict[str, Any] | None = None
        for original in candidates:
            if kind == "clean":
                example = {
                    "candidate_value": original,
                    "gold_label": "clean",
                    "gold_correction": None,
                    "flavour": None,
                    "mutator": "identity",
                    "original_value": original,
                }
                break
            if kind == "ambiguous":
                try:
                    mutated, flav = _mut_truncate(original, rng)
                except ValueError:
                    continue
                example = {
                    "candidate_value": mutated,
                    "gold_label": "ambiguous",
                    "gold_correction": None,
                    "flavour": None,  # rulebook: ambiguous has no flavour
                    "mutator": "truncate",
                    "original_value": original,
                }
                break
            # kind == "typo": try mutators in random order
            mutators = list(_TYPO_MUTATORS)
            rng.shuffle(mutators)
            for mut in mutators:
                try:
                    mutated, flav = mut(original, rng)
                except ValueError:
                    continue
                if mutated == original:
                    continue
                # v2: reject mutators that produced a trivial-edit pair
                # (the filter at output time would drop these anyway,
                # but catching them here lets us try another mutator
                # so we don't waste a slot).
                if _typo_task.is_trivial_typo_correction(mutated, original):
                    continue
                example = {
                    "candidate_value": mutated,
                    "gold_label": "typo",
                    "gold_correction": original,
                    "flavour": flav,
                    "mutator": mut.__name__,
                    "original_value": original,
                }
                break
            if example is not None:
                break

        if example is not None:
            examples.append(example)

    return examples


# ---------------------------------------------------------------------------
# LLM mode
# ---------------------------------------------------------------------------


def _few_shot_block(
    exemplars: dict[str, list[dict[str, Any]]],
    rng: random.Random,
    per_label: int,
) -> str:
    """Render N exemplars per label as a labelled text block.

    v2: only 'typo' and 'clean' labels are used as few-shot exemplars.
    Ambiguous benchmark rows are not shown because the trainee can
    never emit 'ambiguous' and we don't want to teach it that label.
    """
    out_lines: list[str] = []
    label_order = ("typo", "clean")
    for label in label_order:
        pool = exemplars.get(label, [])
        if not pool:
            continue
        picked = rng.sample(pool, min(per_label, len(pool)))
        for ex in picked:
            out_lines.append(
                "Example:\n"
                f"  column: {ex.get('column')!r}\n"
                f"  source: {ex.get('source')!r}\n"
                f"  candidate_value: {ex.get('original_value')!r}\n"
                f"  gold_label: {label!r}\n"
                f"  gold_correction: {ex.get('corrected_value')!r}\n"
                f"  note: {(ex.get('note') or '').strip()!r}"
            )
    return "\n\n".join(out_lines)


def _llm_synth_prompt(
    sample: ColumnSample,
    few_shot_text: str,
    k_per_call: int,
) -> str:
    """Compose the synthesis prompt for one column (binary task)."""
    sibling_lines = "\n".join(
        f"  - {v!r}" for v in sample.distinct_values[:15]
    )
    return (
        "You are generating training examples for a data-quality model "
        "that classifies CSV cell values as 'typo' or 'clean' (BINARY).\n\n"
        f"{_typo_task.TYPO_RULEBOOK}\n\n"
        "Below are real annotated examples from a held-out benchmark. "
        "Use them to calibrate which mutations are typos under our "
        "rulebook:\n\n"
        f"{few_shot_text}\n\n"
        "Now you are given a NEW column from a different CSV. Produce "
        f"exactly {k_per_call} new training examples for this column. "
        "Cover BOTH labels (typo, clean) in a mix appropriate to what "
        "the column would realistically contain. DO NOT emit "
        "'ambiguous' -- the model is binary.\n\n"
        "For each TYPO example, start from an existing value in the "
        "column and introduce a SUBSTANTIVE defect consistent with the "
        "rulebook: a SPELLING typo (one- or two-character distortion: "
        "swap, drop, insert) or a PUNCTUATION typo (missing comma+space, "
        "doubled comma, stray hyphen). DO NOT introduce purely "
        "whitespace, casing, or pure-spacing-rewrite edits -- those "
        "are caught by upstream normalisation and are NOT typos here.\n\n"
        "For CLEAN examples, pick a value as-is from the column.\n\n"
        f"Column header: {sample.column!r}\n"
        f"Source CSV: {sample.source!r}\n"
        f"Distinct values in column (sample):\n{sibling_lines}\n\n"
        "Reply with a single JSON object only, no Markdown fences:\n"
        '{"examples": [\n'
        '    {"candidate_value": "<string>",\n'
        '     "gold_label":      "typo" | "clean",\n'
        '     "gold_correction": "<string>" | null,\n'
        '     "flavour":         "spelling" | "punctuation" | null,\n'
        '     "rationale":       "<one short sentence>"},\n'
        "    ...\n"
        "]}"
    )


# JSON cache key derived from inputs that determine the LLM output.
def _cache_key(
    sample: ColumnSample, few_shot_text: str, k_per_call: int, model: str,
) -> str:
    payload = json.dumps({
        "source": sample.source,
        "column": sample.column,
        "values": sample.distinct_values,
        "few_shot": few_shot_text,
        "k": k_per_call,
        "model": model,
    }, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(payload).hexdigest()[:24]


def _parse_llm_examples(
    text: str,
) -> list[dict[str, Any]]:
    """Strict-ish parser for the LLM's response.

    Strips a leading ``json``-fenced block if the model insists on
    wrapping the object; the rest of the parser is plain JSON.
    """
    if not text:
        return []
    s = text.strip()
    if s.startswith("```"):
        # ```json\n{...}\n```  OR  ```\n{...}\n```
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```\s*$", "", s)
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        logger.debug("LLM reply was not JSON: %r", text[:200])
        return []
    if not isinstance(obj, dict):
        return []
    examples = obj.get("examples")
    if not isinstance(examples, list):
        return []
    return examples


def _validate_llm_example(ex: dict[str, Any]) -> dict[str, Any] | None:
    """Drop examples that violate rulebook invariants (v2 binary).

    The LLM occasionally:
      * emits ``gold_label=='ambiguous'`` -- drop (binary task).
      * labels something 'typo' without a correction
      * labels a typo where candidate_value == correction
      * emits non-string candidate_value
      * emits a 'typo' whose correction is a pure whitespace / casing /
        punctuation rewrite -- drop (trivial-edit filter).
    Filter those out so we don't poison the training set.
    """
    label = ex.get("gold_label")
    cv = ex.get("candidate_value")
    if label not in _typo_task.TYPO_OUTPUT_LABELS:
        # v2: drops both 'ambiguous' (no longer a trainee label) and any
        # unrecognised string the LLM might emit.
        return None
    if not isinstance(cv, str) or not cv:
        return None
    corr = ex.get("gold_correction")
    if label == "typo":
        if not isinstance(corr, str) or not corr or corr == cv:
            return None
        if _typo_task.is_trivial_typo_correction(cv, corr):
            return None
    else:
        # Non-typo rows should have null correction; coerce to None
        # rather than rejecting (the LLM sometimes emits "" here).
        ex["gold_correction"] = None
    return ex


def _gen_examples_llm(
    sample: ColumnSample,
    cfg: SynthConfig,
    exemplars: dict[str, list[dict[str, Any]]],
    rng: random.Random,
    cache_dir: Path,
    client: Any,
) -> list[dict[str, Any]]:
    """Generate examples via the LLM, with on-disk caching."""
    few_shot_text = _few_shot_block(
        exemplars, rng, cfg.llm_few_shot_per_label,
    )
    key = _cache_key(
        sample, few_shot_text, cfg.examples_per_column, cfg.llm_model,
    )
    cache_path = cache_dir / f"{key}.json"
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if isinstance(cached, list):
                return cached
        except json.JSONDecodeError:
            logger.warning("Corrupt cache file %s -- regenerating.", cache_path)

    prompt = _llm_synth_prompt(sample, few_shot_text, cfg.examples_per_column)
    resp = client.query(
        cfg.llm_model,
        prompt,
        reasoning_effort="medium",
    )
    if not resp.ok:
        logger.warning(
            "LLM call failed for %s/%s: %s",
            sample.source, sample.column, resp.error,
        )
        return []

    raw_examples = _parse_llm_examples(resp.text)
    examples: list[dict[str, Any]] = []
    for ex in raw_examples:
        cleaned = _validate_llm_example(ex)
        if cleaned is None:
            continue
        cleaned.setdefault("flavour", None)
        cleaned.setdefault("mutator", "llm")
        cleaned.setdefault("original_value", None)
        examples.append(cleaned)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(examples, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return examples


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _iter_corpus_csvs(
    corpus_dir: Path, denylist: set[str], rng: random.Random,
) -> Iterable[Path]:
    """Yield CSV paths from corpus_dir in random order, skipping denylist."""
    if not corpus_dir.exists():
        return
    # ``scandir`` is much faster than ``listdir`` + ``stat`` for
    # 50K-file directories. We don't materialise the full list because
    # most corpora are >>50K files; reservoir-shuffle the head.
    candidates: list[Path] = []
    with os.scandir(corpus_dir) as it:
        for entry in it:
            if not entry.is_file():
                continue
            if not entry.name.endswith(".csv"):
                continue
            if entry.name in denylist:
                continue
            candidates.append(Path(entry.path))
    rng.shuffle(candidates)
    return candidates


def synthesize(
    cfg: SynthConfig,
    *,
    demo_path: Path,
    manifest_path: Path,
    corpus_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Run a full synthesis pass; return manifest dict."""
    demo_path = Path(demo_path)
    if not demo_path.exists():
        raise FileNotFoundError(
            f"Demo split not found: {demo_path}. Run "
            f"experiments.split_typo_benchmark first."
        )

    exemplars = _load_demo_exemplars(demo_path)
    denylist = _load_denylist(manifest_path)
    cfg.denylist_sources = sorted(denylist)
    logger.info(
        "Loaded %d typo, %d clean, %d ambiguous exemplars; %d denylisted sources.",
        len(exemplars["typo"]), len(exemplars["clean"]),
        len(exemplars["ambiguous"]), len(denylist),
    )

    rng = random.Random(cfg.seed)
    candidates = list(_iter_corpus_csvs(corpus_dir, denylist, rng))
    if not candidates:
        raise RuntimeError(
            f"No usable CSVs in {corpus_dir} after applying denylist "
            f"({len(denylist)} sources)."
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

    all_examples: list[dict[str, Any]] = []
    columns_used = 0
    csvs_scanned = 0
    for path in candidates:
        if columns_used >= cfg.n_columns:
            break
        csvs_scanned += 1
        for sample in _read_columns_from_csv(path):
            if columns_used >= cfg.n_columns:
                break
            if _looks_numeric_column(sample):
                continue
            if not _column_is_textual(sample):
                continue
            if cfg.mode == "deterministic":
                gens = _gen_examples_deterministic(sample, cfg, rng)
            else:
                gens = _gen_examples_llm(
                    sample, cfg, exemplars, rng, cache_dir, client,
                )
            if not gens:
                continue
            for ex in gens:
                ex["_source"] = sample.source
                ex["_column"] = sample.column
                ex["_sibling_values"] = sample.distinct_values
            all_examples.extend(gens)
            columns_used += 1

    logger.info(
        "Scanned %d CSVs, used %d columns, produced %d examples.",
        csvs_scanned, columns_used, len(all_examples),
    )

    # Stable shuffle + 80/20 split.
    rng.shuffle(all_examples)
    n_train = int(len(all_examples) * cfg.train_ratio)
    train_examples = all_examples[:n_train]
    valid_examples = all_examples[n_train:]

    label_counts_train: Counter[str] = Counter()
    label_counts_valid: Counter[str] = Counter()

    with train_path.open("w", encoding="utf-8") as fh_train, \
            valid_path.open("w", encoding="utf-8") as fh_valid, \
            injections_path.open("w", encoding="utf-8") as fh_inj:
        for bucket, fh, counter in [
            (train_examples, fh_train, label_counts_train),
            (valid_examples, fh_valid, label_counts_valid),
        ]:
            for ex in bucket:
                item = {
                    "column": ex["_column"],
                    "source": ex["_source"],
                    "candidate_value": ex["candidate_value"],
                    "sibling_values": [
                        v for v in ex["_sibling_values"]
                        if v != ex["candidate_value"]
                    ],
                    "gold_label": ex["gold_label"],
                    "gold_correction": ex.get("gold_correction"),
                }
                try:
                    rec = _typo_task.build_record(item)
                except ValueError as exc:
                    logger.warning("Dropping malformed example: %s", exc)
                    continue
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                counter[ex["gold_label"]] += 1

                fh_inj.write(json.dumps({
                    "split": "train" if fh is fh_train else "valid",
                    **{k: v for k, v in ex.items() if not k.startswith("_")},
                    "source": ex["_source"],
                    "column": ex["_column"],
                }, ensure_ascii=False) + "\n")

    manifest = {
        "config": cfg.__dict__,
        "csvs_scanned": csvs_scanned,
        "columns_used": columns_used,
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
        help="Path to the split manifest (provides the deny-list of "
        "benchmark source filenames).",
    )
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--n-columns", type=int, default=40,
        help="Number of distinct columns to harvest examples from.",
    )
    parser.add_argument(
        "--examples-per-column", type=int, default=5,
        help="How many training examples to derive per chosen column.",
    )
    parser.add_argument("--p-typo", type=float, default=0.55)
    parser.add_argument(
        "--p-ambiguous", type=float, default=0.0,
        help="v2: ambiguous is not a trainee label and this is ignored.",
    )
    parser.add_argument("--p-clean", type=float, default=0.45)
    parser.add_argument("--train-ratio", type=float, default=0.80)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--llm-model", default="gpt-5.2,gpt-5.4,gpt-5.5",
        help="LLM model for --mode llm (single name or comma-separated pool).",
    )
    parser.add_argument(
        "--llm-few-shot-per-label", type=int, default=3,
        help="Number of demo exemplars per label included in each LLM prompt.",
    )
    args = parser.parse_args(argv)

    cfg = SynthConfig(
        mode=args.mode,
        n_columns=args.n_columns,
        examples_per_column=args.examples_per_column,
        p_typo=args.p_typo,
        p_ambiguous=args.p_ambiguous,
        p_clean=args.p_clean,
        seed=args.seed,
        train_ratio=args.train_ratio,
        llm_model=args.llm_model,
        llm_few_shot_per_label=args.llm_few_shot_per_label,
        corpus_dir=str(args.corpus_dir),
        demo_path=str(args.demo_path),
    )
    manifest = synthesize(
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
