"""Typo-detection RFT task assets (v4 -- per-column audit, per-cell judgement).

v4 changes relative to v3 (``experiments/_typo_v3_task.py``):

* **Rulebook loaded from disk.** The canonical rulebook lives at
  ``RFT/typo/v4/prompts/rulebook.md`` so prompt edits don't require
  a Python release.
* **Three flavours.** ``spelling`` and ``punctuation`` from v3,
  plus a new ``spacing`` flavour for word-boundary tokenisation
  defects (``Mid Town`` <-> ``MidTown``, ``data base`` <-> ``database``).
* **Per-cell judgement.** v4 does NOT require an in-column anchor;
  each value is judged on its own merits as a string.
* **v4-specific trivial filter.** Pure padding-whitespace and pure
  casing edits stay trivial (CLEAN under v4 §2). Spacing-tokenisation
  edits (which v3's filter would drop) are now SUBSTANTIVE and are
  retained by the v4 filter.
* **Grader.** Multigrader combining a ``python`` sub-grader for
  deterministic gold-set bookkeeping (TP / wrong-fix / missed,
  returned as a [0, 1] gold score) and a ``score_model`` sub-grader
  that judges ONLY the model's out-of-gold predictions against the
  v4 rulebook (returned as a [0, 1] extras precision score). The
  multigrader's ``calculate_output`` formula combines them as
  ``5 * (0.7 * gold + 0.3 * extras)`` to land in [0, 5] with
  ``pass_threshold = 4``. The python leg is intentionally tiny and
  stdlib-only to avoid the v3 "Job failed in postprocessing" path.

The set-F1 scoring helper (``compute_column_f1``) is kept compatible
with v3's so eval scripts can switch between v3 and v4 cleanly.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from RFT._common._typo_task import is_trivial_typo_correction  # noqa: F401


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TYPO_V4_TRAIN_FILE = "typo_v4_train.jsonl"
TYPO_V4_VALID_FILE = "typo_v4_valid.jsonl"

# Three flavours -- v3's two plus the new ``spacing`` family.
TYPO_V4_FLAVOURS: tuple[str, ...] = ("spelling", "punctuation", "spacing")

# Cap on distinct values shown to the trainee. Same as v3.
TYPO_V4_MAX_DISTINCT = 500

# English-only column filter threshold (fraction of ASCII alphabetic
# characters across distinct values). Same default as v3.
TYPO_V4_ENGLISH_VALUE_THRESHOLD = 0.6

# Azure RFT score_model whitelist (as of 2026-05). Anything outside
# this set is rejected at grader-upload time.
TYPO_V4_GRADER_MODEL_WHITELIST: tuple[str, ...] = (
    "gpt-4o-2024-08-06",
    "o3-mini-2025-01-31",
)
TYPO_V4_DEFAULT_GRADER_MODEL = "gpt-4o-2024-08-06"


# ---------------------------------------------------------------------------
# Rulebook -- loaded from disk so prompt edits don't require code changes
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
TYPO_V4_RULEBOOK_PATH = (
    _PROJECT_ROOT / "RFT" / "typo" / "v4" / "prompts" / "rulebook.md"
)


def _load_rulebook() -> str:
    if not TYPO_V4_RULEBOOK_PATH.exists():
        raise FileNotFoundError(
            f"v4 rulebook missing at {TYPO_V4_RULEBOOK_PATH}; the v4 task "
            f"module cannot be imported without it."
        )
    return TYPO_V4_RULEBOOK_PATH.read_text(encoding="utf-8")


TYPO_V4_RULEBOOK = _load_rulebook()


# ---------------------------------------------------------------------------
# v4-specific trivial-correction filter
# ---------------------------------------------------------------------------


def is_v4_trivial_correction(original: Any, correction: Any) -> bool:
    """Return True iff (original -> correction) is a v4-trivial edit.

    A correction is v4-trivial when it differs from the original
    ONLY by:

    * padding whitespace (leading/trailing or repeated runs collapsed
      to a single space), OR
    * casing.

    Both classes are explicitly CLEAN under v4 §2 -- caught by
    deterministic regexes upstream of the LLM.

    Critically, **spacing-tokenisation edits are NOT v4-trivial**.
    A change like ``DuPage`` -> ``Du Page`` or ``data base`` ->
    ``database`` alters how the value tokenises into words and is
    therefore a substantive ``spacing`` typo per v4 §1. v3's
    ``is_trivial_typo_correction`` would have dropped these; the
    v4 filter retains them.

    Punctuation-only edits are likewise NOT v4-trivial -- they fall
    under the v4 ``punctuation`` flavour.
    """
    if not isinstance(original, str) or not isinstance(correction, str):
        return False
    if original == correction:
        return False
    # Collapse internal whitespace runs and strip padding from both.
    o_norm = re.sub(r"\s+", " ", original).strip()
    c_norm = re.sub(r"\s+", " ", correction).strip()
    if o_norm == c_norm:
        # Only difference was padding whitespace.
        return True
    if o_norm.casefold() == c_norm.casefold():
        # Only difference was casing (after padding normalisation).
        return True
    return False


# ---------------------------------------------------------------------------
# Column English-only filter (re-exported from v3 for split-script parity)
# ---------------------------------------------------------------------------


def _ascii_alpha_fraction(text: str) -> float:
    if not text:
        return 1.0
    alpha = [c for c in text if c.isalpha()]
    if not alpha:
        return 1.0
    ascii_alpha = sum(1 for c in alpha if ord(c) < 128)
    return ascii_alpha / len(alpha)


def is_english_column(
    header: str,
    values: Iterable[str],
    *,
    value_threshold: float = TYPO_V4_ENGLISH_VALUE_THRESHOLD,
) -> bool:
    """Heuristic: column header is ASCII alphabetic AND its values
    are predominantly ASCII Latin.

    Mirrors v3's filter so v3 and v4 evaluate the same column set.
    """
    if not isinstance(header, str) or not header:
        return False
    for c in header:
        if c.isalpha() and ord(c) >= 128:
            return False
    sample = list(values)
    if not sample:
        return True
    joined = "".join(sample)
    return _ascii_alpha_fraction(joined) >= value_threshold


# ---------------------------------------------------------------------------
# Trainee prompt assembly
# ---------------------------------------------------------------------------

TYPO_V4_SYSTEM_INSTRUCTION = (
    "You are a data-quality reviewer auditing ONE CSV column for "
    "typos. Apply the rulebook below to the column's distinct values "
    "and return the FULL list of typos you find as a single JSON "
    "object. Reply with the JSON object only -- no Markdown fences, "
    "no surrounding text.\n\n"
    f"{TYPO_V4_RULEBOOK}\n\n"
    "Output JSON shape (strict):\n"
    "  typos: list of objects, each with keys:\n"
    "    value      -- the typo-bearing distinct value, copied "
    "verbatim from the input.\n"
    "    correction -- the intended canonical form (string).\n"
    "    flavour    -- 'spelling' | 'punctuation' | 'spacing'.\n"
    "    rationale  -- one short sentence naming the defect.\n"
    "Return an EMPTY list if you find no typos. Every value NOT in "
    "the list is treated as CLEAN."
)


def format_distinct_values(
    values_with_counts: Sequence[tuple[str, int]],
    *,
    max_distinct: int = TYPO_V4_MAX_DISTINCT,
) -> tuple[str, int, bool]:
    """Render distinct values as a numbered listing, count-descending.

    Returns ``(listing_text, n_shown, truncated)``. Same shape as v3.
    """
    items = sorted(
        values_with_counts,
        key=lambda kv: (-int(kv[1]), str(kv[0])),
    )
    truncated = len(items) > max_distinct
    shown = items[:max_distinct]
    width = len(str(len(shown))) if shown else 1
    cw = max((len(str(c)) for _, c in shown), default=1)
    lines = [
        f"  {i:>{width}}.  {int(c):>{cw}}x  {str(v)!r}"
        for i, (v, c) in enumerate(shown, start=1)
    ]
    return "\n".join(lines), len(shown), truncated


def build_user_message(
    *,
    column: str,
    source: str,
    values_with_counts: Sequence[tuple[str, int]],
    max_distinct: int = TYPO_V4_MAX_DISTINCT,
) -> str:
    listing, n_shown, truncated = format_distinct_values(
        values_with_counts, max_distinct=max_distinct,
    )
    trunc_note = (
        f"\n  (showing the {n_shown} most-frequent values; the column "
        f"has more)"
        if truncated else ""
    )
    return (
        f"Column header: {column}\n"
        f"Source CSV: {source}\n"
        f"Distinct values in this column ({n_shown} shown"
        f"{trunc_note}):\n"
        f"{listing}\n\n"
        "Return the JSON object listing every typo you find."
    )


# ---------------------------------------------------------------------------
# Training record assembly
# ---------------------------------------------------------------------------


def _validate_gold_typo(entry: Mapping[str, Any]) -> dict[str, str]:
    """Coerce / validate one gold typo entry against v4 rules."""
    value = entry.get("value")
    correction = entry.get("correction")
    if not isinstance(value, str) or not value:
        raise ValueError(f"gold typo missing/empty 'value': {entry!r}")
    if not isinstance(correction, str) or not correction:
        raise ValueError(
            f"gold typo missing/empty 'correction': {entry!r}"
        )
    if is_v4_trivial_correction(value, correction):
        raise ValueError(
            f"gold typo is v4-trivial (padding/casing only): "
            f"value={value!r}, correction={correction!r}"
        )
    flavour = entry.get("flavour") or "spelling"
    if flavour not in TYPO_V4_FLAVOURS:
        flavour = "spelling"
    return {"value": value, "correction": correction, "flavour": flavour}


def build_record(item: Mapping[str, Any]) -> dict[str, Any]:
    """Shape one (column, distinct-values, gold) example as an RFT JSONL row.

    Required keys in ``item``:
      column                -- CSV column header.
      source                -- CSV filename.
      values_with_counts    -- Sequence of (value, row_count) tuples.
      gold_typos            -- List of {value, correction, [flavour]} dicts.

    Optional keys:
      ambiguous_values      -- List of column values whose benchmark
                               label was ``ambiguous``.
      max_distinct          -- Override TYPO_V4_MAX_DISTINCT.

    Emits a flat dict with ``messages`` plus the item.* namespace
    fields the v4 grader reads:

      * column, source                       -- identification
      * gold_typos          (JSON string)    -- gold set for Python leg
      * ambiguous_values    (JSON string)    -- neutral set
      * column_values_json  (JSON string)    -- [[value, count], ...]
                                                so the score_model
                                                grader can apply
                                                rulebook context to
                                                out-of-gold extras.
    """
    column = item["column"]
    source = item.get("source", "")
    values_with_counts = [
        (str(v), int(c)) for v, c in item["values_with_counts"]
    ]
    if not values_with_counts:
        raise ValueError(
            f"build_record: empty values_with_counts for column={column!r}"
        )

    gold_typos = [
        _validate_gold_typo(g) for g in (item.get("gold_typos") or [])
    ]
    ambiguous_values = [
        str(v) for v in (item.get("ambiguous_values") or []) if v
    ]
    max_distinct = int(item.get("max_distinct", TYPO_V4_MAX_DISTINCT))

    distinct_set = {v for v, _ in values_with_counts}
    for g in gold_typos:
        if g["value"] not in distinct_set:
            raise ValueError(
                f"gold typo value not in column's distinct values: "
                f"column={column!r}, value={g['value']!r}"
            )

    user_msg = build_user_message(
        column=column,
        source=source,
        values_with_counts=values_with_counts,
        max_distinct=max_distinct,
    )
    # Cap stored column listing to max_distinct for grader cost control
    # (the trainee already sees only this slice).
    sorted_values = sorted(
        values_with_counts, key=lambda kv: (-kv[1], kv[0]),
    )[:max_distinct]
    return {
        "messages": [
            {"role": "developer", "content": TYPO_V4_SYSTEM_INSTRUCTION},
            {"role": "user", "content": user_msg},
        ],
        "column": column,
        "source": source,
        "gold_typos": json.dumps(gold_typos, ensure_ascii=False),
        "ambiguous_values": json.dumps(
            ambiguous_values, ensure_ascii=False
        ),
        "column_values_json": json.dumps(
            sorted_values, ensure_ascii=False
        ),
    }


# ---------------------------------------------------------------------------
# Response format (strict JSON schema with spacing flavour)
# ---------------------------------------------------------------------------


def build_response_format() -> dict[str, Any]:
    typo_entry = {
        "type": "object",
        "properties": {
            "value": {
                "type": "string",
                "description": (
                    "The typo-bearing distinct value, copied "
                    "verbatim from the input listing."
                ),
            },
            "correction": {
                "type": "string",
                "description": "Intended canonical form.",
            },
            "flavour": {
                "type": "string",
                "enum": list(TYPO_V4_FLAVOURS),
            },
            "rationale": {
                "type": "string",
                "description": "One short sentence naming the defect.",
            },
        },
        "required": ["value", "correction", "flavour", "rationale"],
        "additionalProperties": False,
    }
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "typo_column_audit_v4",
            "schema": {
                "type": "object",
                "properties": {
                    "typos": {
                        "type": "array",
                        "items": typo_entry,
                        "description": (
                            "Every typo found in the column. Empty "
                            "if the column has no typos."
                        ),
                    },
                },
                "required": ["typos"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }


# ---------------------------------------------------------------------------
# Grader -- multigrader: Python (gold bookkeeping) + score_model (extras)
# ---------------------------------------------------------------------------
#
# Design (per discussion):
#   * Python sub-grader ``gold`` -- deterministic bookkeeping over the
#     gold set. Computes TP / wrong-fix / missed and returns a single
#     float in [0, 1] capturing recall + correction correctness on the
#     gold-labelled typos.
#   * Score-model sub-grader ``extras`` -- judges ONLY the model's
#     out-of-gold predictions (the things the python grader cannot
#     classify) using the v4 rulebook + column-value listing. Returns
#     a single float in [0, 1] capturing precision on extras
#     (valid_extra > ambiguous_extra > fp).
#   * ``calculate_output`` combines them as 5*(0.7*gold + 0.3*extras),
#     landing in [0, 5] with pass_threshold = 4.
#
# Why multi (not the single score_model from the first v4 draft):
# the score-model rubric is unstable on gold bookkeeping -- it
# repeatedly mis-counts TP when the typo list is long. Deterministic
# Python handles that leg perfectly. The score_model is reserved for
# the truly subjective leg (is this out-of-gold flag actually a
# typo per the rulebook?).
#
# v3 python-grader incident: the v3 launch hit "Job failed in
# postprocessing" with a python grader that, locally, ran fine. The
# defensive guards below are written specifically to avoid the
# suspected root causes:
#   1. Return is ALWAYS a plain ``float`` -- never numpy, never dict,
#      never ``int``. Azure's postprocessor coerces strictly.
#   2. Bare ``except Exception`` wrapping the whole grade() body, so
#      no exception escapes. Returns 0.0 on any failure.
#   3. No third-party imports -- only ``json`` and ``re`` from stdlib.
#      (pandas/rapidfuzz/etc are available but we don't need them and
#      every import widens the cold-start failure surface.)
#   4. The source is dedented and embedded as a regular Python string
#      so the JSON serialisation is straightforward and we can unit-
#      test it inline (``python_grade(sample, item)`` below).


# Python grader source. KEEP IN SYNC with ``python_grade`` below --
# the function below is a verbatim copy so we can sanity-test
# locally. The constant is what actually gets shipped to Azure.
TYPO_V4_PYTHON_GRADER_SOURCE = '''
"""Python grader for the v4 typo column-audit task.

Returns a single float in [0, 1] capturing gold-set recall + per-
correction agreement. Out-of-gold predictions are IGNORED here --
the score_model leg of the multigrader judges those.

Scoring (G = |gold| after dropping ambiguous values):
  gold_match   = predictions whose value+correction match gold.
  gold_wrong_fix = predictions on a gold value with disagreeing
                   correction.
  missed       = G - gold_match - gold_wrong_fix.
  raw = 1.0 * gold_match + 0.3 * gold_wrong_fix - 1.0 * missed.
  Map raw in [-G, +G] linearly to [0, 1] (G == 0 -> 1.0, perfect).

Predictions whose (value -> correction) edit is v4-trivial (pure
padding-whitespace or pure casing) are dropped silently -- those
are CLEAN per v4 \u00a72 and must not count as predictions.

Defensive contract:
  * Never raises -- any exception returns 0.0.
  * Returns a plain ``float`` in [0.0, 1.0].
  * Uses only ``json`` and ``re`` from the standard library.
"""

import json
import re


def _norm(text):
    if not isinstance(text, str):
        return ""
    return re.sub(r"\\s+", " ", text).strip().casefold()


def _is_v4_trivial(original, correction):
    if not isinstance(original, str) or not isinstance(correction, str):
        return False
    if original == correction:
        return False
    o_norm = re.sub(r"\\s+", " ", original).strip()
    c_norm = re.sub(r"\\s+", " ", correction).strip()
    if o_norm == c_norm:
        return True
    if o_norm.casefold() == c_norm.casefold():
        return True
    return False


def _load_json_string(value, default):
    if isinstance(value, (list, dict)):
        return value
    if not isinstance(value, str) or not value:
        return default
    try:
        return json.loads(value)
    except Exception:
        return default


def _extract_pred_typos(sample):
    if isinstance(sample, dict):
        oj = sample.get("output_json")
        if isinstance(oj, dict):
            tl = oj.get("typos")
            if isinstance(tl, list):
                return tl
        ot = sample.get("output_text") or ""
        if isinstance(ot, str) and ot.strip():
            try:
                parsed = json.loads(ot)
            except Exception:
                return []
            if isinstance(parsed, dict):
                tl = parsed.get("typos")
                if isinstance(tl, list):
                    return tl
    return []


def grade(sample, item):
    try:
        pred_list = _extract_pred_typos(sample)
        gold_list = _load_json_string(item.get("gold_typos"), [])
        ambiguous = _load_json_string(item.get("ambiguous_values"), [])
        ambiguous_keys = set()
        for a in ambiguous:
            k = _norm(a)
            if k:
                ambiguous_keys.add(k)

        gold_map = {}
        for entry in gold_list:
            if not isinstance(entry, dict):
                continue
            v = _norm(entry.get("value"))
            c = _norm(entry.get("correction"))
            if not v or not c or v in ambiguous_keys:
                continue
            gold_map[v] = c

        pred_map = {}
        for entry in pred_list:
            if not isinstance(entry, dict):
                continue
            v_raw = entry.get("value")
            c_raw = entry.get("correction")
            v = _norm(v_raw)
            c = _norm(c_raw)
            if not v or not c or v in ambiguous_keys:
                continue
            if _is_v4_trivial(v_raw, c_raw):
                continue
            pred_map[v] = c

        G = len(gold_map)
        if G == 0:
            return 1.0

        gold_match = 0
        gold_wrong_fix = 0
        for k, gc in gold_map.items():
            pc = pred_map.get(k)
            if pc is None:
                continue
            if pc == gc:
                gold_match += 1
            else:
                gold_wrong_fix += 1
        missed = G - gold_match - gold_wrong_fix

        raw = 1.0 * gold_match + 0.3 * gold_wrong_fix - 1.0 * missed
        norm = (raw + float(G)) / (2.0 * float(G))
        if norm < 0.0:
            norm = 0.0
        if norm > 1.0:
            norm = 1.0
        return float(norm)
    except Exception:
        return 0.0
'''


def python_grade(sample: Mapping[str, Any], item: Mapping[str, Any]) -> float:
    """Local verbatim copy of the embedded python grader for sanity tests.

    Kept structurally identical to ``TYPO_V4_PYTHON_GRADER_SOURCE``
    so we can validate the grader against eval data BEFORE upload.
    """
    try:
        pred_list = _local_extract_pred_typos(sample)
        gold_list = _local_load_json_string(item.get("gold_typos"), [])
        ambiguous = _local_load_json_string(item.get("ambiguous_values"), [])
        ambiguous_keys = {_local_norm(a) for a in ambiguous if _local_norm(a)}

        gold_map: dict[str, str] = {}
        for entry in gold_list:
            if not isinstance(entry, Mapping):
                continue
            v = _local_norm(entry.get("value"))
            c = _local_norm(entry.get("correction"))
            if not v or not c or v in ambiguous_keys:
                continue
            gold_map[v] = c

        pred_map: dict[str, str] = {}
        for entry in pred_list:
            if not isinstance(entry, Mapping):
                continue
            v_raw = entry.get("value")
            c_raw = entry.get("correction")
            v = _local_norm(v_raw)
            c = _local_norm(c_raw)
            if not v or not c or v in ambiguous_keys:
                continue
            if is_v4_trivial_correction(v_raw, c_raw):
                continue
            pred_map[v] = c

        G = len(gold_map)
        if G == 0:
            return 1.0
        gold_match = 0
        gold_wrong_fix = 0
        for k, gc in gold_map.items():
            pc = pred_map.get(k)
            if pc is None:
                continue
            if pc == gc:
                gold_match += 1
            else:
                gold_wrong_fix += 1
        missed = G - gold_match - gold_wrong_fix
        raw = 1.0 * gold_match + 0.3 * gold_wrong_fix - 1.0 * missed
        norm = (raw + float(G)) / (2.0 * float(G))
        return float(max(0.0, min(1.0, norm)))
    except Exception:
        return 0.0


def _local_norm(text: Any) -> str:
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text).strip().casefold()


def _local_load_json_string(value: Any, default: Any) -> Any:
    if isinstance(value, (list, dict)):
        return value
    if not isinstance(value, str) or not value:
        return default
    try:
        return json.loads(value)
    except Exception:
        return default


def _local_extract_pred_typos(sample: Any) -> list:
    if isinstance(sample, Mapping):
        oj = sample.get("output_json")
        if isinstance(oj, Mapping):
            tl = oj.get("typos")
            if isinstance(tl, list):
                return tl
        ot = sample.get("output_text") or ""
        if isinstance(ot, str) and ot.strip():
            try:
                parsed = json.loads(ot)
            except Exception:
                return []
            if isinstance(parsed, Mapping):
                tl = parsed.get("typos")
                if isinstance(tl, list):
                    return tl
    return []


def build_python_grader() -> dict[str, Any]:
    """Python sub-grader: deterministic gold-set bookkeeping."""
    return {
        "type": "python",
        "name": "typo_v4_python_gold",
        "source": TYPO_V4_PYTHON_GRADER_SOURCE,
    }


def build_score_model_extras_grader(
    grader_model: str = TYPO_V4_DEFAULT_GRADER_MODEL,
) -> dict[str, Any]:
    """Score-model sub-grader: judges OUT-OF-GOLD predictions only.

    Returns a float in [0, 1] reflecting how rulebook-justified the
    model's extras are. Empty extras list -> 1.0 (perfect precision).
    """
    if grader_model not in TYPO_V4_GRADER_MODEL_WHITELIST:
        raise ValueError(
            f"grader_model {grader_model!r} not in Azure RFT score_model "
            f"whitelist {TYPO_V4_GRADER_MODEL_WHITELIST!r}"
        )

    developer = (
        "You are evaluating the PRECISION of typo predictions from a "
        "column-audit model. You judge each predicted typo on its own "
        "merits, using ONLY the v4 rulebook and the column data you "
        "are shown. You are NOT given any gold set, label set, or "
        "list of known-clean values, and you must not assume one "
        "exists -- every prediction is evaluated against the rulebook "
        "and the column evidence alone.\n\n"
        f"{TYPO_V4_RULEBOOK}\n\n"
        "Inputs (plain-text labelled fields):\n"
        "  - column:             CSV column header.\n"
        "  - source:             source table id.\n"
        "  - column_values:      JSON array of [value, count] pairs "
        "for the column (your only evidence about what values "
        "actually appear and how often).\n"
        "  - model_output:       the audit model's raw JSON reply "
        "(an object with a 'typos' array of "
        "{value, correction, flavour, rationale}).\n\n"
        "Rulebook reminders that govern every classification below:\n"
        "  * Casing-only edits, padding-whitespace-only edits "
        "(adding or stripping leading/trailing whitespace only), and "
        "no-op edits (correction equals value) are NOT typos. A "
        "prediction whose edit reduces to one of these violates the "
        "rulebook and must be scored as a false positive -- it must "
        "NOT be excused on the grounds that the rulebook would "
        "ignore such an edit. The trivial-edit rule disqualifies "
        "such edits from being typos; it does not exempt the model "
        "from being scored when it proposes one.\n"
        "  * Spelling, punctuation, and spacing-tokenisation typos "
        "ARE in scope: inserting a missing comma in a list, "
        "splitting a run-together compound, joining a wrongly-split "
        "compound, fixing a misspelled word inside a multi-word "
        "value, and similar substantive edits ARE typos under the "
        "rulebook. Do not confuse these with padding-whitespace.\n"
        "  * You may and should use your own knowledge of standard "
        "language and established domain terminology (general "
        "dictionary spelling, well-known proper nouns of any kind, "
        "and recognised technical/scientific terms in any field) "
        "when judging whether a proposed correction is grounded. "
        "Proper nouns and domain-specific terms are NOT exempt from "
        "being typos: if the value is a misspelling of an "
        "established form and the proposed correction is that "
        "established form, it counts as a valid typo.\n"
        "  * A proposed correction is ungrounded when the resulting "
        "form is not an attested standard form -- i.e. the edit "
        "produces another arbitrary surface variant rather than a "
        "word/term recognisable from general or domain knowledge. "
        "Ungrounded corrections are false positives even when the "
        "rationale sounds confident and even when the original "
        "value also looks unusual.\n"
        "  * Pure numeric values, dates, and fixed-length identifier "
        "codes do not have spelling typos; substantive edits to them "
        "(changing a digit, reformatting a date) are normally false "
        "positives.\n\n"
        "Using column convention as evidence:\n"
        "  The column_values list is your primary source of evidence "
        "about what this column normally looks like. When the other "
        "values in the column share a consistent punctuation, "
        "spacing, casing, or formatting pattern, a single value "
        "that breaks that pattern is a signal for a possible real "
        "typo (typically spacing-tokenisation or punctuation), "
        "even when the textual difference looks small. If the "
        "proposed correction brings the value INTO line with the "
        "prevailing convention of the column, you may score it as "
        "valid. Conversely, if a proposed correction would move a "
        "value AWAY from the prevailing convention, or if no clear "
        "prevailing convention is visible in column_values, the "
        "convention argument does NOT apply and the prediction "
        "must be judged on the rulebook reminders above alone. "
        "Casing-only, padding-whitespace-only, and no-op edits "
        "remain false positives regardless of column convention.\n\n"
        "Scoring procedure:\n"
        "  Step 1 -- Parse model_output as JSON. If parsing fails, "
        "or 'typos' is missing or not an array, return Score: 0.0.\n"
        "  Step 2 -- If the 'typos' array is empty, return "
        "Score: 1.0 (the model made no positive claims, so there is "
        "nothing to penalise on precision).\n"
        "  Step 3 -- For each entry in model_output.typos, classify "
        "into ONE of:\n"
        "    * trivial_fp -- (value -> correction) is a no-op (value "
        "equals correction), or differs only in letter case, or "
        "differs only in leading/trailing whitespace. The rulebook "
        "forbids these as typo proposals; treat as a FALSE POSITIVE. "
        "Reward 0.0.\n"
        "    * valid     -- the value IS a typo per the v4 rulebook "
        "(spelling / punctuation / spacing-tokenisation) AND the "
        "proposed correction is a plausible, grounded fix. "
        "Groundedness means the corrected form is attested in "
        "standard usage -- standard English spelling, well-known "
        "proper-noun spelling, established scientific nomenclature, "
        "or an unambiguous formatting fix the rulebook recognises. "
        "Use your world knowledge for this; do not require evidence "
        "of the corrected form inside the column. Reward 1.0.\n"
        "    * ambiguous -- the value plausibly might be a typo but "
        "the proposed correction is not uniquely determined (two or "
        "more equally defensible fixes exist), OR the value is a "
        "marginal case where rulebook scope is genuinely unclear. "
        "Reward 0.5.\n"
        "    * fp        -- per the rulebook the value is CLEAN "
        "(in-scope content with no typo), OR the proposed "
        "correction is ungrounded (the resulting form is not an "
        "attested standard term in general or domain usage), OR "
        "the value is out of scope for typo detection (pure number, "
        "date, identifier code, etc.) and the edit is substantive. "
        "Reward 0.0.\n"
        "  Step 4 -- Return Score: mean(rewards) over all 'typos' "
        "entries. Clamp to [0.0, 1.0].\n\n"
        "Respond ONLY with two lines:\n"
        "Score: <0.0-1.0>\n"
        "Reasoning: <one short sentence per typos entry, comma-"
        "separated>"
    )

    # NOTE: no whitespace inside {{ ... }} -- Azure parses
    # ``{{ item.x }}`` as a variable named " item.x ".
    # The grader is NOT given gold_typos or ambiguous_values: it must
    # judge every prediction against the rulebook + column evidence.
    user = (
        "Column: {{item.column}}\n"
        "Source: {{item.source}}\n"
        "Column values (JSON): {{item.column_values_json}}\n"
        "\n"
        "Model output:\n"
        "{{sample.output_text}}"
    )

    return {
        "name": "typo_v4_score_model_extras",
        "type": "score_model",
        "model": grader_model,
        "input": [
            {"role": "developer", "content": developer},
            {"role": "user", "content": user},
        ],
        "range": [0.0, 1.0],
    }


# Weight on the gold (Python) leg in the multigrader.  The remainder
# 1 - GOLD_WEIGHT goes to the extras (score_model) leg.  0.7/0.3
# follows the design: gold is the primary objective; extras precision
# is a secondary regulariser.
TYPO_V4_GOLD_WEIGHT = 0.7


def build_grader(
    grader_model: str = TYPO_V4_DEFAULT_GRADER_MODEL,
) -> dict[str, Any]:
    """Multigrader = Python (gold bookkeeping) + score_model (extras).

    ``grader_model`` only configures the score_model sub-grader; the
    python sub-grader runs in Azure's sandbox.
    """
    gold = build_python_grader()
    extras = build_score_model_extras_grader(grader_model=grader_model)
    gw = TYPO_V4_GOLD_WEIGHT
    ew = 1.0 - gw
    # calculate_output expression: subscore identifiers come from the
    # keys in ``graders``. We map to a [0, 5] range with pass=4.
    formula = (
        f"5 * ({gw:.2f} * typo_v4_python_gold + "
        f"{ew:.2f} * typo_v4_score_model_extras)"
    )
    return {
        "type": "multi",
        "name": "typo_v4_multigrader",
        "graders": {
            "typo_v4_python_gold": gold,
            "typo_v4_score_model_extras": extras,
        },
        "calculate_output": formula,
        "pass_threshold": 4.0,
        "range": [0.0, 5.0],
    }


# ---------------------------------------------------------------------------
# Offline scoring helper -- compatible signature with v3.compute_column_f1
# ---------------------------------------------------------------------------


def compute_column_f1(
    pred_typos: Iterable[Mapping[str, Any]],
    gold_typos: Iterable[Mapping[str, Any]],
    ambiguous_values: Iterable[str] = (),
) -> dict[str, float | int]:
    """Set-F1 + TP/FP/FN with v4 ambiguous-value handling.

    Match key: (normalised value, normalised correction), case-folded
    with collapsed whitespace. Predictions whose (value -> correction)
    edit is v4-trivial are dropped silently (they cost nothing).
    Predictions on ambiguous values are credited neither way.
    """

    def _norm(text: Any) -> str:
        if not isinstance(text, str):
            return ""
        return re.sub(r"\s+", " ", text).strip().casefold()

    amb = {_norm(a) for a in ambiguous_values if a}

    pred_map: dict[str, str] = {}
    for entry in pred_typos:
        if not isinstance(entry, Mapping):
            continue
        v = entry.get("value")
        c = entry.get("correction")
        if not isinstance(v, str) or not isinstance(c, str):
            continue
        if is_v4_trivial_correction(v, c):
            continue
        nv = _norm(v)
        nc = _norm(c)
        if not nv or not nc:
            continue
        if nv in amb:
            continue
        pred_map[nv] = nc

    gold_map: dict[str, str] = {}
    for entry in gold_typos:
        if not isinstance(entry, Mapping):
            continue
        v = entry.get("value")
        c = entry.get("correction")
        if not isinstance(v, str) or not isinstance(c, str):
            continue
        nv = _norm(v)
        nc = _norm(c)
        if not nv or not nc:
            continue
        if nv in amb:
            continue
        gold_map[nv] = nc

    pred_keys = set(pred_map)
    gold_keys = set(gold_map)
    tp = sum(
        1 for k in pred_keys & gold_keys
        if pred_map[k] == gold_map[k]
    )
    fp_extra = len(pred_keys - gold_keys)
    fp_wrongfix = sum(
        1 for k in pred_keys & gold_keys
        if pred_map[k] != gold_map[k]
    )
    fp = fp_extra + fp_wrongfix
    fn = len(gold_keys - pred_keys) + fp_wrongfix

    if tp == 0 and fp == 0 and fn == 0:
        return {
            "precision": 1.0, "recall": 1.0, "f1": 1.0,
            "tp": 0, "fp": 0, "fn": 0, "score": 5.0,
        }
    if tp == 0:
        return {
            "precision": 0.0, "recall": 0.0, "f1": 0.0,
            "tp": 0, "fp": fp, "fn": fn, "score": 0.0,
        }
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )
    return {
        "precision": precision, "recall": recall, "f1": f1,
        "tp": tp, "fp": fp, "fn": fn,
        "score": float(max(0.0, min(5.0, f1 * 5.0))),
    }
