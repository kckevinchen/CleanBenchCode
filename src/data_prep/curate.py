"""Curate a reproducible subset of CSVs from a corpus.

This module is the unified replacement for the old ``curate_tables`` and
``curate_pbi`` scripts. It transparently handles BOTH input layouts:

**Flat layout** (e.g. ``data/Excel50KTables/*.csv``)
    A directory of independent CSVs. Each CSV is its own "target".
    Output is written flat: ``<output-dir>/<name>.csv``.

**Grouped layout** (e.g. ``data/bing21_grouped/<workbook>.xlsx/*.csv``)
    A directory of *subdirectories*, where each subdirectory holds
    several CSVs that came from the same source workbook. The
    workbook is accepted only if **every** one of its tables passes
    the gates (all-or-nothing per workbook). Output is written
    nested: ``<output-dir>/<workbook>/<name>.csv``.

The layout is auto-detected from the source directory structure: if
any non-hidden, non-underscored entry is a subdirectory we treat the
whole tree as grouped; otherwise as flat. Pre-grouping is the job of
``src.data_prep.bing21_group_by_workbook`` (or any equivalent
preprocessing step).

Selection rules per table
-------------------------

A candidate table is kept if:

1. it parses as a well-formed CSV,
2. its column count is within ``[--min-columns, --max-columns]``,
3. it has at least ``--min-rows`` data rows,
4. its empty-cell ratio is <= ``--max-empty-ratio``,
5. its non-Latin-letter count is <= ``--max-non-latin-chars``,
6. a language identifier confirms it is English (modes:
   ``off`` / ``script`` / ``langdetect`` / ``llm``).

Cleaning pipeline
-----------------

Every kept table is cleaned:

1. drop columns that are entirely empty (NaN / "" / whitespace-only /
   missing-value sentinels like ``-`` / ``N/A``);
2. drop rows containing any empty cell or any oversized cell
   (``--max-cell-chars`` / ``--max-cell-words``);
3. if the cleaned table still has more than ``--max-rows`` rows,
   either deterministically subsample down (``--oversize subsample``,
   default) or reject the table (``--oversize reject``).

The final CSV written to ``--output-dir`` is asserted to contain
zero empty cells.

Usage
-----

::

    # Flat (legacy curate_tables behaviour):
    python -m experiments.data_prep.curate \
        --source-dir data/Excel50KTables \
        --output-dir data/curated \
        --num-targets 5 \
        --min-columns 3 --max-columns 10 \
        --min-rows 30 --max-rows 100

    # Grouped (legacy curate_pbi behaviour):
    python -m experiments.data_prep.curate \
        --source-dir data/PBI.allDataCSV \
        --output-dir data/curated_pbi \
        --num-targets 20 \
        --min-tables 2 --max-tables 6 \
        --min-rows 10 --max-rows 100 \
        --oversize reject

Outputs
-------

* Per-table CSVs written under ``<output-dir>/`` (flat) or
  ``<output-dir>/<workbook>/`` (grouped).
* ``<output-dir>/manifest.json`` records the layout, all filter
  parameters, every selected target, and the per-target reject
  reasons (bucketed for grouped mode).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import re
import sys
import unicodedata
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class CuratedTable:
    source_path: str
    output_path: str
    columns_original: int
    rows_original: int
    columns: int
    rows_final: int
    columns_dropped_empty: int
    rows_dropped_empty: int
    rows_dropped_oversized: int = 0
    empty_ratio_original: float = 0.0
    sampled: bool = False
    seed: int = 0


@dataclass
class ScanSkip:
    source_path: str
    reason: str


# ---------------------------------------------------------------------------
# Cleaning helpers
# ---------------------------------------------------------------------------


_MISSING_SENTINELS = frozenset({
    "-", "--", "---",
    "*", "**",
    "?", "??",
    ".", "..",
    "n/a", "n.a.", "n/a.", "na", "null", "none", "nan", "#n/a",
    "#null!", "#na",
})


def _cell_is_empty(v: object) -> bool:
    """Return True if ``v`` represents an empty cell (NaN, blank, or
    a missing-value sentinel like ``-`` / ``N/A`` / ``NULL``).
    """
    if v is None:
        return True
    try:
        if pd.isna(v):
            return True
    except (TypeError, ValueError):
        pass
    if isinstance(v, str):
        s = v.strip()
        if s == "":
            return True
        return s.lower() in _MISSING_SENTINELS
    return False


def _cell_is_oversized(
    v: object,
    *,
    max_chars: Optional[int],
    max_words: Optional[int],
) -> bool:
    """True if ``v`` exceeds either size budget (chars or whitespace tokens)."""
    if max_chars is None and max_words is None:
        return False
    if v is None:
        return False
    s = v if isinstance(v, str) else str(v)
    if max_chars and len(s) > max_chars:
        return True
    if max_words and len(s.split()) > max_words:
        return True
    return False


def _empty_cell_ratio(df: pd.DataFrame) -> float:
    """Fraction of cells in ``df`` that are empty. 0.0 if df has no cells."""
    if df.empty or df.shape[1] == 0:
        return 0.0
    total = df.shape[0] * df.shape[1]
    mask = df.apply(lambda col: col.map(_cell_is_empty))
    return float(mask.sum().sum()) / float(total)


def _clean_dataframe(
    df: pd.DataFrame,
    *,
    max_cell_chars: Optional[int] = None,
    max_cell_words: Optional[int] = None,
) -> tuple[pd.DataFrame, int, int, int]:
    """Drop all-empty columns, then drop rows with any empty / oversized cell.

    Returns ``(clean_df, dropped_cols, dropped_rows_empty,
    dropped_rows_oversized)``.
    """
    if df.empty:
        return df, 0, 0, 0
    col_empty_mask = df.apply(lambda col: col.map(_cell_is_empty).all())
    kept_cols = df.columns[~col_empty_mask]
    dropped_cols = int(col_empty_mask.sum())
    if dropped_cols:
        df = df.loc[:, kept_cols]
    if df.empty or df.shape[1] == 0:
        return df, dropped_cols, 0, 0
    row_has_empty = df.apply(lambda col: col.map(_cell_is_empty)).any(axis=1)
    dropped_rows_empty = int(row_has_empty.sum())
    dropped_rows_oversized = 0
    if max_cell_chars or max_cell_words:
        row_has_big = df.apply(
            lambda col: col.map(
                lambda v: _cell_is_oversized(
                    v, max_chars=max_cell_chars, max_words=max_cell_words,
                )
            )
        ).any(axis=1)
        only_oversized = row_has_big & ~row_has_empty
        dropped_rows_oversized = int(only_oversized.sum())
        drop_mask = row_has_empty | row_has_big
    else:
        drop_mask = row_has_empty
    kept_df = df.loc[~drop_mask].copy()
    kept_df.reset_index(drop=True, inplace=True)
    return kept_df, dropped_cols, dropped_rows_empty, dropped_rows_oversized


# ---------------------------------------------------------------------------
# Language gate
# ---------------------------------------------------------------------------


def _is_non_latin_letter(ch: str) -> bool:
    """True iff ``ch`` is alphabetic and outside the Latin script."""
    if ord(ch) < 128:
        return False
    if not ch.isalpha():
        return False
    try:
        name = unicodedata.name(ch)
    except ValueError:
        return True
    return not name.startswith("LATIN")


def _scan_non_latin(df: pd.DataFrame, sample_chars: int = 50_000) -> Tuple[int, str]:
    """Return ``(count, example)`` of non-Latin letters in headers + cells."""
    total = 0
    example = ""
    for col in df.columns:
        for ch in str(col):
            if _is_non_latin_letter(ch):
                total += 1
                if not example:
                    example = ch
    chars_seen = 0
    for col in df.columns:
        if chars_seen >= sample_chars:
            break
        for value in df[col].astype(str):
            if chars_seen >= sample_chars:
                break
            chars_seen += len(value)
            for ch in value:
                if _is_non_latin_letter(ch):
                    total += 1
                    if not example:
                        example = ch
    return total, example


_TOKEN_RE = re.compile(r"[A-Za-z\u00C0-\u024F][A-Za-z\u00C0-\u024F'\u2019\-]*")


def _looks_like_text_token(tok: str) -> bool:
    if len(tok) < 2:
        return False
    return any(ch.isalpha() for ch in tok)


def _langdetect_english(
    df: pd.DataFrame,
    *,
    min_english_prob: float,
    max_chars: int = 4000,
) -> Tuple[bool, dict]:
    """Use offline ``langdetect`` to decide if the table is English."""
    try:
        from langdetect import DetectorFactory, detect_langs
        from langdetect.lang_detect_exception import LangDetectException
    except ImportError:
        return True, {"language": "unknown", "probability": 0.0,
                      "error": "langdetect_not_installed"}

    DetectorFactory.seed = 0

    sample = _sample_text_for_lang(df, max_chars=max_chars)
    if not sample.strip() or len(sample) < 30:
        has_diacritic = any(
            ord(ch) > 127 and ch.isalpha() for ch in sample
        )
        if has_diacritic:
            return False, {
                "language": "non_english_latin",
                "probability": 0.0,
                "sample_chars": len(sample),
                "sample_preview": sample[:120],
                "reason": "short_sample_with_diacritics",
            }
        if sample.strip():
            try:
                short_scores = detect_langs(sample)
            except LangDetectException:
                short_scores = []
            if short_scores:
                top = short_scores[0]
                if top.lang != "en" and float(top.prob) >= 0.9:
                    return False, {
                        "language": top.lang,
                        "probability": float(top.prob),
                        "sample_chars": len(sample),
                        "sample_preview": sample[:120],
                        "reason": "short_sample_high_conf_non_english",
                    }
        return True, {"language": "und", "probability": 0.0,
                      "sample_chars": len(sample),
                      "reason": "insufficient_text"}

    try:
        scores = detect_langs(sample)
    except LangDetectException as exc:
        return True, {"language": "unknown", "probability": 0.0,
                      "error": f"langdetect_failed: {exc}"}

    top = scores[0]
    detail = {
        "language": top.lang,
        "probability": float(top.prob),
        "runner_up": (
            {"language": scores[1].lang, "probability": float(scores[1].prob)}
            if len(scores) > 1 else None
        ),
        "sample_chars": len(sample),
    }
    en_prob = next(
        (float(s.prob) for s in scores if s.lang == "en"), 0.0
    )
    detail["english_probability"] = en_prob
    is_english = en_prob >= min_english_prob
    return is_english, detail


def _lingua_english(
    df: pd.DataFrame,
    *,
    min_english_prob: float,
    max_chars: int = 4000,
) -> Tuple[bool, dict]:
    """Use ``lingua-language-detector`` to decide if the table is English.

    Lingua is materially more accurate than ``langdetect`` on the
    short, header-heavy samples we extract from CSVs - it uses
    rule-based filtering plus per-language n-gram models trained on
    larger corpora and exposes per-language probabilities even for
    inputs of just a few words. The detector itself is heavyweight
    (~170 MB wheel, lazy-loaded n-gram models) but is built once
    per process and reused across every table.

    Decision rule: REJECT iff the top non-English language has a
    confidence above the language-specific noise floor AND English's
    confidence is below ``min_english_prob``. We do NOT require
    English to have the highest confidence - a CSV of ASCII column
    headers ("Region", "Total", ...) often has no language with high
    confidence, and we do not want to reject those.
    """
    try:
        from lingua import LanguageDetectorBuilder
    except ImportError:
        return True, {"language": "unknown", "probability": 0.0,
                      "error": "lingua_not_installed"}

    detector = _get_lingua_detector()
    sample = _sample_text_for_lang(df, max_chars=max_chars)
    if not sample.strip() or len(sample) < 8:
        # Diacritic guard - useful when the sample is empty / tiny:
        # any Latin-script alpha char outside ASCII is suspicious.
        has_diacritic = any(
            ord(ch) > 127 and ch.isalpha() for ch in sample
        )
        if has_diacritic:
            return False, {
                "language": "non_english_latin",
                "probability": 0.0,
                "sample_chars": len(sample),
                "sample_preview": sample[:120],
                "reason": "short_sample_with_diacritics",
            }
        return True, {"language": "und", "probability": 0.0,
                      "sample_chars": len(sample),
                      "reason": "insufficient_text"}

    confidences = detector.compute_language_confidence_values(sample)
    if not confidences:
        return True, {"language": "und", "probability": 0.0,
                      "sample_chars": len(sample),
                      "reason": "no_confidence_scores"}

    en_prob = 0.0
    top_non_en: Optional[Tuple[str, float]] = None
    top_three: List[Tuple[str, float]] = []
    for c in confidences[:3]:
        top_three.append((c.language.name, float(c.value)))
    for c in confidences:
        name = c.language.name
        prob = float(c.value)
        if name == "ENGLISH":
            en_prob = prob
        elif top_non_en is None:
            top_non_en = (name, prob)

    detail: Dict[str, Any] = {
        "english_probability": en_prob,
        "top_three": top_three,
        "sample_chars": len(sample),
        "sample_preview": sample[:120],
    }
    if top_non_en is not None:
        detail["top_non_english"] = {
            "language": top_non_en[0],
            "probability": top_non_en[1],
        }

    # Reject when the top non-English candidate dominates English by
    # a large factor and clears an absolute floor. Tuned on the Bing21
    # corpus: a 2x dominance ratio plus a 0.30 floor catches Dutch /
    # French / German / Spanish CSVs whose headers all score around
    # 0.4-0.7 in their language and <0.1 in English, while leaving
    # bare-ASCII tables (every score <0.15, no clear winner) accepted.
    if top_non_en is not None:
        nen_lang, nen_prob = top_non_en
        if nen_prob >= 0.30 and nen_prob > 2.0 * max(en_prob, 0.05):
            detail["language"] = nen_lang
            detail["probability"] = nen_prob
            detail["reason"] = "non_english_dominant"
            return False, detail

    is_english = en_prob >= min_english_prob or (
        top_non_en is None or top_non_en[1] < 0.30
    )
    detail["language"] = "ENGLISH" if is_english else (
        top_non_en[0] if top_non_en else "und"
    )
    detail["probability"] = en_prob if is_english else (
        top_non_en[1] if top_non_en else 0.0
    )
    return is_english, detail


_LINGUA_DETECTOR_CACHE: list = []


def _get_lingua_detector():
    """Lazy-build and cache the lingua detector (full-accuracy mode).

    The high-accuracy build pre-loads n-gram models on first call to
    ``compute_language_confidence_values`` (~1-2 s per language the
    first time it sees that language). The cache keeps the warmed
    detector alive across calls so per-table cost is amortized.
    """
    if _LINGUA_DETECTOR_CACHE:
        return _LINGUA_DETECTOR_CACHE[0]
    from lingua import LanguageDetectorBuilder
    detector = LanguageDetectorBuilder.from_all_languages().build()
    _LINGUA_DETECTOR_CACHE.append(detector)
    return detector


def _sample_text_for_lang(df: pd.DataFrame, max_chars: int = 1500) -> str:
    """Return a compact sample of natural-language text for the LID."""
    seen: set = set()
    parts: List[str] = []
    total = 0
    for col in df.columns:
        c = str(col)
        if c not in seen:
            seen.add(c)
            parts.append(c)
            total += len(c) + 1
            if total >= max_chars:
                return " ".join(parts)
    for col in df.columns:
        if total >= max_chars:
            break
        for value in df[col].astype(str):
            if total >= max_chars:
                break
            tokens = [t for t in _TOKEN_RE.findall(value) if _looks_like_text_token(t)]
            if not tokens:
                continue
            tok = max(tokens, key=len)
            if tok in seen:
                continue
            seen.add(tok)
            parts.append(tok)
            total += len(tok) + 1
    return " ".join(parts)


_LANG_SYSTEM_PROMPT = (
    "You are a strict language identifier. Given a sample of words drawn "
    "from a CSV (column headers + cell tokens), decide whether the table's "
    "natural-language content is English. Personal/place names, scientific "
    "terms, units, and ASCII-only identifiers do NOT make a table 'English' "
    "on their own - look at the actual word stems, not just the alphabet. "
    "Return ONLY a single JSON object on one line: "
    '{"english": true|false, "language": "<iso 639-1 or short name>", '
    '"confidence": <0..1>}.'
)


def _build_lang_prompt(sample: str) -> str:
    return (
        "Sample tokens (whitespace-separated, deduplicated):\n"
        f"{sample}\n\n"
        "Respond with the JSON object only, no prose."
    )


_LANG_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_lang_json(text: str) -> Optional[dict]:
    if not text:
        return None
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = candidate.strip("`").lstrip("json").strip()
    m = _LANG_RE.search(candidate)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _llm_is_english(
    df: pd.DataFrame,
    source: Path,
    *,
    cache_dir: Optional[Path],
    client_factory,
    model: str,
    reasoning_effort: str,
) -> Tuple[bool, dict]:
    """Return ``(is_english, detail)`` for the LLM language gate."""
    sample = _sample_text_for_lang(df)
    if not sample.strip():
        return True, {"language": "und", "confidence": 0.0, "cached": False,
                      "sample_chars": 0, "reason": "no_text_sample"}

    cache_path: Optional[Path] = None
    if cache_dir is not None:
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            file_hash = _hash_file(source)
            cache_key = hashlib.sha256(
                f"{file_hash}|{model}|{reasoning_effort}".encode()
            ).hexdigest()[:32]
            cache_path = cache_dir / f"{cache_key}.json"
            if cache_path.is_file():
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                cached["cached"] = True
                return bool(cached.get("english")), cached
        except OSError as exc:
            logger.debug("Language cache unavailable (%s); proceeding without.", exc)

    try:
        client = client_factory()
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM client unavailable for language check: %s", exc)
        return True, {"language": "unknown", "confidence": 0.0, "cached": False,
                      "error": f"client_init_failed: {exc}"}

    try:
        resp = client.query(
            model,
            _build_lang_prompt(sample),
            reasoning_effort=reasoning_effort,
            reasoning_summary="auto",
            system_prompt=_LANG_SYSTEM_PROMPT,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Language-ID LLM call failed for %s: %s", source.name, exc)
        return True, {"language": "unknown", "confidence": 0.0, "cached": False,
                      "error": f"call_failed: {exc}"}

    parsed = _parse_lang_json(resp.text or "")
    if not parsed:
        logger.warning(
            "Language-ID parse failed for %s; raw=%r", source.name,
            (resp.text or "")[:200],
        )
        return True, {"language": "unknown", "confidence": 0.0, "cached": False,
                      "error": "parse_failed", "raw": (resp.text or "")[:200]}

    is_english = bool(parsed.get("english"))
    detail = {
        "language": str(parsed.get("language", "unknown")),
        "confidence": float(parsed.get("confidence", 0.0) or 0.0),
        "english": is_english,
        "cached": False,
        "sample_chars": len(sample),
        "model": model,
    }

    if cache_path is not None:
        try:
            cache_path.write_text(
                json.dumps({**detail, "sample_preview": sample[:300]},
                           ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as exc:
            logger.debug("Failed to write lang cache %s: %s", cache_path, exc)

    return is_english, detail


def _assert_no_empty_cells(df: pd.DataFrame, path: Path) -> None:
    """Defensive: the curated output must not contain ANY empty cell."""
    mask = df.apply(lambda col: col.map(_cell_is_empty))
    bad = int(mask.sum().sum())
    if bad:
        raise AssertionError(
            f"Final curated CSV {path} still contains {bad} empty cell(s); "
            "cleaning logic is broken."
        )


# ---------------------------------------------------------------------------
# Layout detection
# ---------------------------------------------------------------------------


def _spreadsheet_id(name: str) -> str:
    """Everything before the first underscore in the file stem.

    Used as a grouping key for **flat** corpora that follow the
    ``<workbook-id>_<table-name>.csv`` naming convention (e.g. PBI).
    Files without an underscore become their own group of size 1.
    """
    stem = Path(name).stem
    return stem.split("_", 1)[0] if "_" in stem else stem


def _scan(source_dir: Path) -> Tuple[Dict[str, List[Path]], str]:
    """Group CSVs from ``source_dir`` and report the detected layout.

    Returns ``(groups, layout)`` where layout is ``"grouped"`` (when the
    source dir contains workbook subdirectories) or ``"flat"`` (when it
    contains CSVs directly).

    * Grouped layout: ``<source-dir>/<workbook-dir>/*.csv``. Group key
      is the workbook directory name; subdirectories whose names start
      with ``.`` or ``_`` are skipped (e.g. ``_unmatched/``,
      ``.lang_cache/``).
    * Flat layout: ``<source-dir>/<file>.csv``. CSVs that share an
      underscore-separated prefix are co-grouped (PBI convention);
      CSVs without an underscore become their own group of size 1.
    """
    groups: Dict[str, List[Path]] = defaultdict(list)
    entries = list(source_dir.iterdir())
    pre_grouped = any(
        p.is_dir() and not p.name.startswith(".") and not p.name.startswith("_")
        for p in entries[:32]
    )
    if pre_grouped:
        for d in entries:
            if not d.is_dir():
                continue
            if d.name.startswith(".") or d.name.startswith("_"):
                continue
            for p in sorted(d.iterdir()):
                if p.is_file() and p.suffix.lower() == ".csv":
                    groups[d.name].append(p)
        return dict(groups), "grouped"
    for p in entries:
        if p.is_file() and p.suffix.lower() == ".csv":
            groups[_spreadsheet_id(p.name)].append(p)
    return dict(groups), "flat"


# ---------------------------------------------------------------------------
# CSV inspection helpers
# ---------------------------------------------------------------------------


def _peek_columns(path: Path) -> Optional[int]:
    try:
        header = pd.read_csv(
            path,
            dtype=str,
            keep_default_na=False,
            nrows=0,
            encoding_errors="replace",
        )
    except Exception:  # noqa: BLE001
        return None
    return len(header.columns)


def _count_rows(path: Path) -> Optional[int]:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            total = sum(1 for _ in fh)
    except OSError:
        return None
    return max(total - 1, 0)


def _load_full(path: Path) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(
            path,
            dtype=str,
            keep_default_na=False,
            encoding_errors="replace",
            on_bad_lines="skip",
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to load %s: %s", path, exc)
        return None


# ---------------------------------------------------------------------------
# Per-table predicate (shared by flat + grouped paths)
# ---------------------------------------------------------------------------


@dataclass
class _TableEval:
    clean_df: pd.DataFrame
    rows: int
    columns: int
    orig_rows: int
    orig_columns: int
    dropped_empty_columns: int
    dropped_rows_with_empties: int
    dropped_rows_oversized: int
    empty_ratio_original: float
    sampled: bool


def _evaluate_table(
    path: Path,
    *,
    min_rows: int,
    max_rows: int,
    min_cols: int,
    max_cols: int,
    max_non_latin_chars: int,
    max_empty_ratio: float,
    language_check: str,
    language_min_english_prob: float,
    language_min_confidence: float,
    language_llm_model: str,
    language_llm_effort: str,
    language_cache_dir: Optional[Path],
    llm_client_factory,
    max_cell_chars: Optional[int],
    max_cell_words: Optional[int],
    max_oversized_row_ratio: float,
    oversize: str,
    seed: int,
) -> Tuple[Optional[_TableEval], Optional[str]]:
    """Run the 9-step gate stack on ``path``.

    Returns ``(eval, None)`` on accept or ``(None, reason)`` on reject.
    """
    df = _load_full(path)
    if df is None or df.empty:
        return None, "empty_or_unreadable"

    orig_rows = len(df)
    orig_cols = len(df.columns)
    if orig_cols < min_cols or orig_cols > max_cols:
        return None, f"columns={orig_cols} outside [{min_cols},{max_cols}]"
    if orig_rows < min_rows:
        return None, f"rows={orig_rows} < min_rows={min_rows}"
    # In `reject` mode the raw row count is also an upper bound, since
    # we won't subsample.
    if oversize == "reject" and orig_rows > max_rows:
        return None, f"rows={orig_rows} > max_rows={max_rows}"

    if language_check != "off":
        non_latin_count, non_latin_example = _scan_non_latin(df)
        if non_latin_count > max_non_latin_chars:
            return None, (
                f"non_latin_chars={non_latin_count} > "
                f"max_non_latin_chars={max_non_latin_chars} "
                f"(example={non_latin_example!r})"
            )
        if language_check == "langdetect":
            is_english, detail = _langdetect_english(
                df, min_english_prob=language_min_english_prob,
            )
            if not is_english:
                return None, (
                    f"langdetect_top={detail.get('language')!r} "
                    f"english_prob={detail.get('english_probability', 0.0):.2f} "
                    f"(< {language_min_english_prob})"
                )
        elif language_check == "lingua":
            is_english, detail = _lingua_english(
                df, min_english_prob=language_min_english_prob,
            )
            if not is_english:
                return None, (
                    f"lingua_top={detail.get('language')!r} "
                    f"prob={detail.get('probability', 0.0):.2f} "
                    f"english_prob={detail.get('english_probability', 0.0):.2f} "
                    f"reason={detail.get('reason', 'low_english_prob')}"
                )
        elif language_check == "llm":
            is_english, detail = _llm_is_english(
                df,
                path,
                cache_dir=language_cache_dir,
                client_factory=llm_client_factory,
                model=language_llm_model,
                reasoning_effort=language_llm_effort,
            )
            confident = detail.get("confidence", 0.0) >= language_min_confidence
            if not is_english and confident:
                return None, (
                    f"llm_lang={detail.get('language')!r} "
                    f"confidence={detail.get('confidence', 0.0):.2f} "
                    f"(>= {language_min_confidence})"
                )

    empty_ratio = _empty_cell_ratio(df)
    if empty_ratio > max_empty_ratio:
        return None, (
            f"empty_ratio={empty_ratio:.3f} > max_empty_ratio={max_empty_ratio}"
        )

    clean_df, dropped_cols, dropped_rows, dropped_oversized = _clean_dataframe(
        df,
        max_cell_chars=max_cell_chars,
        max_cell_words=max_cell_words,
    )
    if clean_df.empty or clean_df.shape[1] == 0:
        return None, "all_rows_or_cols_empty_after_clean"

    total_orig_rows = max(orig_rows, 1)
    oversized_ratio = dropped_oversized / total_orig_rows
    if oversized_ratio > max_oversized_row_ratio:
        return None, (
            f"oversized_row_ratio={oversized_ratio:.3f} "
            f"> max_oversized_row_ratio={max_oversized_row_ratio} "
            f"(dropped {dropped_oversized}/{total_orig_rows} oversized row(s))"
        )

    clean_cols = int(clean_df.shape[1])
    clean_rows = int(clean_df.shape[0])
    if not (min_cols <= clean_cols <= max_cols):
        return None, (
            f"clean_columns={clean_cols} outside [{min_cols},{max_cols}] "
            f"(dropped {dropped_cols} empty col(s))"
        )
    if clean_rows < min_rows:
        return None, (
            f"clean_rows={clean_rows} < min_rows={min_rows} "
            f"(dropped {dropped_rows} row(s) with empties)"
        )
    if oversize == "reject" and clean_rows > max_rows:
        return None, (
            f"clean_rows={clean_rows} > max_rows={max_rows} "
            f"(oversize=reject)"
        )

    sampled = False
    if clean_rows > max_rows:
        # oversize == "subsample": deterministically downsample.
        clean_df = (
            clean_df.sample(n=max_rows, random_state=seed)
                    .sort_index()
                    .reset_index(drop=True)
        )
        clean_rows = max_rows
        sampled = True
    else:
        clean_df = clean_df.reset_index(drop=True)

    return (
        _TableEval(
            clean_df=clean_df,
            rows=clean_rows,
            columns=clean_cols,
            orig_rows=orig_rows,
            orig_columns=orig_cols,
            dropped_empty_columns=int(dropped_cols),
            dropped_rows_with_empties=int(dropped_rows),
            dropped_rows_oversized=int(dropped_oversized),
            empty_ratio_original=round(float(empty_ratio), 4),
            sampled=sampled,
        ),
        None,
    )


# ---------------------------------------------------------------------------
# Unified curate()
# ---------------------------------------------------------------------------


def curate(
    source_dir: Path,
    output_dir: Path,
    *,
    num_targets: int,
    min_rows: int = 10,
    max_rows: int = 100,
    min_columns: int = 2,
    max_columns: int = 10,
    min_tables: int = 1,
    max_tables: int = 10_000_000,
    seed: int = 42,
    max_scan: Optional[int] = None,
    overwrite: bool = False,
    oversize: str = "subsample",
    max_non_latin_chars: int = 0,
    max_empty_ratio: float = 0.1,
    language_check: str = "langdetect",
    language_min_english_prob: float = 0.5,
    language_min_confidence: float = 0.6,
    language_llm_model: str = "gpt-5.2",
    language_llm_effort: str = "minimal",
    language_cache_dir: Optional[Path] = None,
    max_cell_chars: Optional[int] = 200,
    max_cell_words: Optional[int] = 40,
    max_oversized_row_ratio: float = 0.1,
    dedupe_schema_within_group: bool = True,
) -> Dict[str, Any]:
    """Curate up to ``num_targets`` source units (CSVs or workbooks).

    Parameters
    ----------
    source_dir
        Directory containing the corpus. Layout is auto-detected:
        if it has subdirectories, the corpus is treated as grouped
        (one workbook per subdir); otherwise as flat (one table per
        CSV).
    num_targets
        Maximum number of source units to accept (tables in flat mode,
        workbooks in grouped mode).
    min_tables, max_tables
        In grouped mode, only consider workbooks with this many tables.
        Ignored in flat mode (every group has exactly 1 table).
    oversize
        ``"subsample"`` (default): if a cleaned table has more than
        ``max_rows`` rows, deterministically sample down to
        ``max_rows``. ``"reject"``: drop the table (and, in grouped
        mode, the whole workbook) instead.
    max_scan
        Cap on the number of source units inspected before giving up.
        ``None`` = no cap.
    overwrite
        Overwrite existing output CSVs. Defaults to skipping pre-existing.

    Returns the manifest dict (also written to
    ``<output-dir>/manifest.json``).
    """
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)

    if not source_dir.is_dir():
        raise FileNotFoundError(f"source-dir not found: {source_dir}")
    if num_targets <= 0:
        raise ValueError("num_targets must be positive")
    if min_columns > max_columns:
        raise ValueError("min_columns must be <= max_columns")
    if max_rows <= 0:
        raise ValueError("max_rows must be positive")
    if oversize not in ("subsample", "reject"):
        raise ValueError("oversize must be 'subsample' or 'reject'")
    if language_check not in ("off", "script", "langdetect", "lingua", "llm"):
        raise ValueError(
            "language_check must be one of: off, script, langdetect, lingua, llm"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    if language_cache_dir is None:
        language_cache_dir = output_dir / ".lang_cache"

    groups, layout = _scan(source_dir)
    logger.info(
        "scanned %d source unit(s) in %s (layout=%s)",
        len(groups), source_dir, layout,
    )

    # Apply per-workbook table-count window only in grouped mode.
    if layout == "grouped":
        eligible = [
            sid for sid, paths in groups.items()
            if min_tables <= len(paths) <= max_tables
        ]
        logger.info(
            "%d/%d workbook(s) have between %d and %d tables",
            len(eligible), len(groups), min_tables, max_tables,
        )
    else:
        eligible = list(groups.keys())

    rng = random.Random(seed)
    rng.shuffle(eligible)

    # Lazy LLM client.
    llm_client_singleton: List[Any] = []

    def _client_factory():
        if not llm_client_singleton:
            from ..core.llm_client import LLMClient
            llm_client_singleton.append(LLMClient())
        return llm_client_singleton[0]

    selected: List[Dict[str, Any]] = []
    skip_reasons: Dict[str, int] = defaultdict(int)
    skipped_examples: List[ScanSkip] = []
    scanned = 0

    eval_kwargs = dict(
        min_rows=min_rows, max_rows=max_rows,
        min_cols=min_columns, max_cols=max_columns,
        max_non_latin_chars=max_non_latin_chars,
        max_empty_ratio=max_empty_ratio,
        language_check=language_check,
        language_min_english_prob=language_min_english_prob,
        language_min_confidence=language_min_confidence,
        language_llm_model=language_llm_model,
        language_llm_effort=language_llm_effort,
        language_cache_dir=language_cache_dir,
        llm_client_factory=_client_factory,
        max_cell_chars=max_cell_chars,
        max_cell_words=max_cell_words,
        max_oversized_row_ratio=max_oversized_row_ratio,
        oversize=oversize,
        seed=seed,
    )

    for sid in eligible:
        if len(selected) >= num_targets:
            break
        if max_scan is not None and scanned >= max_scan:
            logger.warning(
                "Hit max_scan=%d before collecting %d target(s) (have %d)",
                max_scan, num_targets, len(selected),
            )
            break
        scanned += 1
        paths = sorted(groups[sid])

        # Decide output paths up front so we can honour --no-overwrite cheaply.
        if layout == "grouped":
            out_subdir = output_dir / sid
            out_paths = [out_subdir / p.name for p in paths]
        else:
            out_subdir = output_dir
            out_paths = [output_dir / paths[0].name]
        if not overwrite and any(p.exists() for p in out_paths):
            skip_reasons["output_exists"] += 1
            if len(skipped_examples) < 20:
                skipped_examples.append(ScanSkip(str(paths[0]), "output_exists"))
            continue

        # Evaluate every table; in grouped mode any reject kills the workbook.
        evals: List[Tuple[Path, _TableEval]] = []
        reject_reason: Optional[str] = None
        seen_schemas: Dict[Tuple[str, ...], str] = {}
        dropped_dupes: List[Tuple[str, str]] = []  # (dropped_name, kept_name)
        for p in paths:
            ev, reason = _evaluate_table(p, **eval_kwargs)
            if ev is None:
                reject_reason = f"{p.name}: {reason}"
                break
            if dedupe_schema_within_group and layout == "grouped" and len(paths) > 1:
                schema_key = tuple(str(c) for c in ev.clean_df.columns)
                prev = seen_schemas.get(schema_key)
                if prev is not None:
                    dropped_dupes.append((p.name, prev))
                    logger.debug(
                        "  dedupe %s/%s: same schema as %s, dropping",
                        sid, p.name, prev,
                    )
                    continue
                seen_schemas[schema_key] = p.name
            evals.append((p, ev))

        if reject_reason is not None:
            bucket = (
                reject_reason.split(":", 1)[1].strip().split(" ", 1)[0]
                if ":" in reject_reason else reject_reason
            )
            skip_reasons[bucket] += 1
            if len(skipped_examples) < 20:
                skipped_examples.append(ScanSkip(sid, reject_reason))
            logger.debug("skip %s: %s", sid, reject_reason)
            continue

        # Post-dedup, the workbook may have fallen below the table-count window.
        if (
            layout == "grouped"
            and dropped_dupes
            and len(evals) < min_tables
        ):
            reason = (
                f"too_few_tables_after_dedup: {len(evals)} < min_tables={min_tables} "
                f"(dropped {len(dropped_dupes)} duplicate-schema table(s))"
            )
            skip_reasons["too_few_tables_after_dedup"] += 1
            if len(skipped_examples) < 20:
                skipped_examples.append(ScanSkip(sid, reason))
            logger.debug("skip %s: %s", sid, reason)
            continue
        if dropped_dupes:
            skip_reasons["duplicate_schema_within_group"] += len(dropped_dupes)

        # Write the (cleaned) outputs. Re-derive out_paths from the surviving
        # evals so that schema-dedup drops are honoured cleanly.
        out_subdir.mkdir(parents=True, exist_ok=True)
        table_meta: List[Dict[str, Any]] = []
        for src_path, ev in evals:
            if layout == "grouped":
                out_path = out_subdir / src_path.name
            else:
                out_path = output_dir / src_path.name
            _assert_no_empty_cells(ev.clean_df, out_path)
            ev.clean_df.to_csv(out_path, index=False)
            try:
                rel_out = str(out_path.relative_to(output_dir))
            except ValueError:
                rel_out = str(out_path)
            table_meta.append({
                "source_path": str(src_path),
                "filename": src_path.name,
                "output_path": rel_out,
                "rows": ev.rows,
                "columns": ev.columns,
                "orig_rows": ev.orig_rows,
                "orig_columns": ev.orig_columns,
                "dropped_empty_columns": ev.dropped_empty_columns,
                "dropped_rows_with_empties": ev.dropped_rows_with_empties,
                "dropped_rows_oversized": ev.dropped_rows_oversized,
                "empty_ratio_original": ev.empty_ratio_original,
                "sampled": ev.sampled,
            })

        selected.append({
            "id": sid,
            "n_tables": len(table_meta),
            "tables": table_meta,
            "dropped_duplicate_schema": [
                {"filename": dropped, "duplicate_of": kept}
                for dropped, kept in dropped_dupes
            ],
        })
        logger.info(
            "  selected %s (%d table(s), %d row(s) total)",
            sid, len(table_meta), sum(t["rows"] for t in table_meta),
        )

    manifest = {
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "layout": layout,
        "seed": seed,
        "num_targets_requested": num_targets,
        "num_targets_selected": len(selected),
        "scanned": scanned,
        "filters": {
            "min_rows": min_rows, "max_rows": max_rows,
            "min_columns": min_columns, "max_columns": max_columns,
            "min_tables": min_tables if layout == "grouped" else None,
            "max_tables": max_tables if layout == "grouped" else None,
            "oversize": oversize,
            "max_scan": max_scan,
            "overwrite": overwrite,
            "max_non_latin_chars": max_non_latin_chars,
            "max_empty_ratio": max_empty_ratio,
            "language_check": language_check,
            "language_min_english_prob": language_min_english_prob,
            "language_min_confidence": language_min_confidence,
            "language_llm_model": language_llm_model if language_check == "llm" else None,
            "language_llm_effort": language_llm_effort if language_check == "llm" else None,
            "max_cell_chars": max_cell_chars,
            "max_cell_words": max_cell_words,
            "max_oversized_row_ratio": max_oversized_row_ratio,
            "dedupe_schema_within_group": (
                dedupe_schema_within_group if layout == "grouped" else None
            ),
        },
        "skip_reason_counts": dict(skip_reasons),
        "skipped_examples": [asdict(s) for s in skipped_examples],
        "selected": selected,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(
        "Curated %d/%d target(s) (scanned %d, layout=%s) -> %s",
        len(selected), num_targets, scanned, layout, output_dir,
    )
    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Curate a reproducible subset of small, hand-auditable CSVs "
            "from a flat or pre-grouped corpus. Layout is auto-detected."
        ),
    )
    p.add_argument("--source-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--num-targets", type=int, default=5,
        help=(
            "How many source units to accept. In flat layouts this is "
            "tables; in grouped layouts it's workbooks."
        ),
    )
    p.add_argument("--min-columns", type=int, default=2)
    p.add_argument("--max-columns", type=int, default=10)
    p.add_argument("--min-rows", type=int, default=10)
    p.add_argument("--max-rows", type=int, default=100)
    p.add_argument(
        "--min-tables", type=int, default=1,
        help="Grouped layouts only: minimum tables per workbook.",
    )
    p.add_argument(
        "--max-tables", type=int, default=10_000_000,
        help="Grouped layouts only: maximum tables per workbook.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--max-scan", type=int, default=None,
        help="Cap on source units inspected. Default: no cap.",
    )
    p.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing curated files in --output-dir.",
    )
    p.add_argument(
        "--oversize", choices=("subsample", "reject"), default="subsample",
        help=(
            "If a cleaned table has more than --max-rows: 'subsample' "
            "(default) deterministically samples down; 'reject' drops "
            "the table (and the whole workbook in grouped mode)."
        ),
    )
    p.add_argument(
        "--language-check",
        choices=("off", "script", "langdetect", "lingua", "llm"),
        default="lingua",
        help=(
            "'lingua' (default) uses lingua-language-detector - more "
            "accurate than langdetect on short / header-heavy samples. "
            "'langdetect' is the previous default. 'llm' calls a model. "
            "'script' only checks for non-Latin chars. 'off' disables."
        ),
    )
    p.add_argument("--language-min-english-prob", type=float, default=0.5)
    p.add_argument("--language-min-confidence", type=float, default=0.6)
    p.add_argument("--language-llm-model", default="gpt-5.2")
    p.add_argument("--language-llm-effort", default="minimal")
    p.add_argument("--language-cache-dir", type=Path, default=None)
    p.add_argument("--max-non-latin-chars", type=int, default=0)
    p.add_argument("--max-empty-ratio", type=float, default=0.1)
    p.add_argument(
        "--max-cell-chars", type=int, default=200,
        help="Drop rows containing any cell with > N chars. 0 disables.",
    )
    p.add_argument(
        "--max-cell-words", type=int, default=40,
        help="Drop rows containing any cell with > N whitespace tokens. 0 disables.",
    )
    p.add_argument("--max-oversized-row-ratio", type=float, default=0.1)
    p.add_argument(
        "--dedupe-schema-within-group",
        dest="dedupe_schema_within_group",
        action="store_true",
        default=True,
        help="In grouped mode, drop tables whose column headers exactly match "
             "another table already kept from the same workbook (default: on).",
    )
    p.add_argument(
        "--no-dedupe-schema-within-group",
        dest="dedupe_schema_within_group",
        action="store_false",
        help="Disable per-workbook schema deduplication.",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    if not args.source_dir.is_dir():
        logger.error("source directory not found: %s", args.source_dir)
        return 2
    manifest = curate(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        num_targets=args.num_targets,
        min_rows=args.min_rows, max_rows=args.max_rows,
        min_columns=args.min_columns, max_columns=args.max_columns,
        min_tables=args.min_tables, max_tables=args.max_tables,
        seed=args.seed,
        max_scan=args.max_scan,
        overwrite=args.overwrite,
        oversize=args.oversize,
        max_non_latin_chars=args.max_non_latin_chars,
        max_empty_ratio=args.max_empty_ratio,
        language_check=args.language_check,
        language_min_english_prob=args.language_min_english_prob,
        language_min_confidence=args.language_min_confidence,
        language_llm_model=args.language_llm_model,
        language_llm_effort=args.language_llm_effort,
        language_cache_dir=args.language_cache_dir,
        max_cell_chars=args.max_cell_chars or None,
        max_cell_words=args.max_cell_words or None,
        max_oversized_row_ratio=args.max_oversized_row_ratio,
        dedupe_schema_within_group=args.dedupe_schema_within_group,
    )
    if manifest["num_targets_selected"] == 0:
        print(
            "No target(s) matched the filters. Loosen the gates or raise "
            "--max-scan / --num-targets."
        )
        return 1
    print(
        f"Curated {manifest['num_targets_selected']}/{manifest['num_targets_requested']} "
        f"target(s) (layout={manifest['layout']}) -> {args.output_dir}"
    )
    print(f"Manifest: {Path(args.output_dir) / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
