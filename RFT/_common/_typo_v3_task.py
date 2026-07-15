"""Typo-detection RFT task assets (v3 -- per-column audit).

This module bundles everything specific to the v3 typo-detection
task. ``experiments.demo_rft``, ``experiments.synth_typo_llm_v3``,
``experiments.split_typo_benchmark_v3``, and
``experiments.eval_typo_finetune_v3`` all import from here.

v3 changes (relative to v2's per-cell binary task in
``experiments/_typo_task.py``):

* **Column-level granularity.** One API call per (csv, column)
  pair, not per cell. The trainee receives the column header and
  the column's distinct values (with frequency counts, capped at
  500 distinct values) and must emit the FULL list of typos it
  finds. Values absent from that list are implicitly ``clean``.
* **Set-valued output.** The response is a JSON object with a
  single ``typos`` array; each entry is ``{value, correction,
  flavour, rationale}``. Empty array == "this column has no typos".
* **Python grader, not score_model.** Set-F1 between predicted
  and gold typo sets, computed deterministically (no grader-LLM
  variance). Score is ``F1 * 5`` to keep the [0, 5] range used
  by v1/v2.
* **Ambiguous-credit rule preserved.** Gold rows whose v2 label
  was ``ambiguous`` are aggregated into a separate
  ``ambiguous_values`` set per column; predictions on those
  values count as neither true nor false positives (they are
  excluded from both the precision and recall denominators).

The rulebook itself (what counts as a typo) is identical to v2 --
substantive spelling or punctuation edits only; no whitespace /
casing / spacing trivia. v3 reuses the same
``is_trivial_typo_correction`` filter via ``experiments._typo_task``.

Why column-at-a-time?

* v2's per-cell call sends the column context (header + sibling
  sample) with every cell, so for an N-cell column the same context
  is re-tokenised N times. The Bing21 eval corpus has ~300 (csv,
  column) pairs spread across ~1800 cells, so v3 makes ~6x fewer
  API calls at inference -- and many more after distinct-value
  deduplication, since most columns repeat values heavily.
* The natural signal for typos is "this value looks misspelled
  relative to the column's OTHER values". v3 puts that signal in
  the prompt directly (with frequencies, so rare typos stand out
  against common canonical forms).
"""

from __future__ import annotations

import json
from typing import Any, Iterable, Mapping, Sequence

# Re-export the trivial-typo filter from v2 -- v3 uses the SAME
# rulebook for what counts as a substantive typo, so we don't
# duplicate the implementation.
from RFT._common._typo_task import (
    TYPO_FLAVOURS,
    is_trivial_typo_correction,
)


# Filenames used by both ``prepare`` (which stages train/valid JSONL
# into the demo state dir) and ``submit`` (which uploads them). Kept
# distinct from v2's ``typo_train.jsonl`` so a v2 and v3 prepare
# can live in the same output dir without clobbering each other.
TYPO_V3_TRAIN_FILE = "typo_v3_train.jsonl"
TYPO_V3_VALID_FILE = "typo_v3_valid.jsonl"

# Trainee output flavours, mirroring v2.
TYPO_V3_FLAVOURS = TYPO_FLAVOURS

# Cap on distinct values shown to the trainee. Matches the existing
# convention in ``src/labeling/correction_suggester.py``: Bing21 columns
# top out at ~100 distinct values, so 500 means "show the entire column"
# in practice. The cap is a safety net for unusually-wide columns.
TYPO_V3_MAX_DISTINCT = 500


# ---------------------------------------------------------------------------
# Rulebook -- identical content to v2, framed for set-valued output
# ---------------------------------------------------------------------------

TYPO_V3_RULEBOOK = (
    "You are auditing ONE column of a CSV file for typos. The column "
    "header and EVERY distinct value in the column (up to "
    f"{TYPO_V3_MAX_DISTINCT}) are shown with their row counts. Your "
    "job is to identify every value that is a TYPO of one of the "
    "OTHER values in the same column, and to propose the intended "
    "correction.\n\n"
    "A value is a TYPO only when, judging from that value and the "
    "column's other values, BOTH of the following hold:\n"
    "  (a) you can name the specific defect, AND\n"
    "  (b) the intended correction is unambiguous and requires a real "
    "      character-level edit (not merely a layout fix).\n"
    "The two flavours that qualify are SPELLING and PUNCTUATION:\n"
    "  - spelling: a one- or two-character distortion of a word that "
    "    plausibly belongs in this column (e.g., 'Govenor' (1 row) "
    "    next to 'Governor' (47 rows) -> spelling typo). Adjacent "
    "    DUPLICATED WORDS also qualify as spelling typos when the "
    "    de-duplicated form is the plausible intended value (e.g., "
    "    'BUCKEYE LOCAL LOCAL SD' next to 'BUCKEYE LOCAL SD' -> "
    "    spelling typo whose correction is 'BUCKEYE LOCAL SD'; "
    "    'Aircraft Aircraft' -> 'Aircraft').\n"
    "  - punctuation: a wrong or missing punctuation token that "
    "    changes how the value is read (e.g., 'Address,State' -> "
    "    'Address, State'; '(575-) 894-8350' -> '(575) 894-8350').\n"
    "Do NOT flag values whose only difference from the intended form "
    "is whitespace, casing, or trivial punctuation -- those are "
    "normalisation issues caught upstream, not typos. Examples that "
    "are CLEAN under this rulebook even if they look 'off':\n"
    "  - 'New  York' (double space)  -- whitespace normalisation.\n"
    "  - 'STRATFORD-UPON-AVON' vs 'Stratford-upon-Avon' -- casing.\n"
    "  - 'Du Page' vs 'DuPage' (pure spacing) -- not a typo.\n"
    "Treat everything else as CLEAN; in particular do NOT flag:\n"
    "  - Recognised proper nouns, brand names, person names.\n"
    "  - Fixed-length codes (zip, SKU, FIPS, ISBN, version strings).\n"
    "  - Pure numeric values -- never typos by our definition.\n"
    "  - Date-shaped strings -- never typos by our definition.\n"
    "  - Standard idioms with elided punctuation ('New Years Day').\n"
    "  - Truncated / contracted values where you cannot recover the "
    "    full form ('CLARION SAFETY SYSTE', 'Twp'). Omit these.\n"
    "PROPER NOUNS, NAMES, AND ENTITIES -- EXTRA CAUTION.\n"
    "Most false positives come from flagging legitimate names "
    "(people, places, organisations, products, teams, song titles, "
    "movie titles, etc.) as typos. The default for any value that "
    "looks like a named entity is CLEAN. Only flag a name as a typo "
    "when ALL of the following are true:\n"
    "  (n1) the column header itself signals that values are NOT "
    "       free-form names -- i.e. the column is a controlled "
    "       vocabulary such as US state, country, month, day-of-"
    "       week, sport, department, status, category, etc., OR\n"
    "       the column has a clearly canonical majority spelling "
    "       and the suspect value is the lone deviant (e.g. one "
    "       row 'Govenor' against 47 rows 'Governor');\n"
    "  (n2) there is a SPECIFIC anchor value already present in "
    "       this column that differs by exactly 1-2 character edits "
    "       AND is more frequent than the suspect value; and\n"
    "  (n3) the defect is a clear character-level slip (transposed "
    "       letters, dropped/duplicated letter, obviously wrong "
    "       letter) -- not a legitimate spelling variant, regional "
    "       form, foreign-language form, historical name, "
    "       transliteration, abbreviation, nickname, alias, or "
    "       stylised brand spelling.\n"
    "If you cannot satisfy ALL THREE, the name is CLEAN. In "
    "particular, do NOT flag a name as a typo merely because:\n"
    "  - it looks 'unusual' or you do not personally recognise it;\n"
    "  - it could plausibly be a misspelling of a famous name not "
    "    present in this column;\n"
    "  - its spelling differs from the form you would expect (people "
    "    and places legitimately have non-standard spellings);\n"
    "  - it contains non-English letters, accents, or diacritics;\n"
    "  - it appears only once in the column (rarity alone is not "
    "    evidence -- many real names appear once).\n"
    "ALPHANUMERIC CODES AND IDS -- EXTRA CAUTION.\n"
    "Values that mix letters and digits within a single token "
    "(e.g., 'ZUR3260', 'D5876', '2016StrawPoll', 'B12X4', "
    "'ICC-ES-ESR3375') are codes or IDs and the default for them "
    "is CLEAN. There is no canonical spacing or punctuation "
    "schema to assume for such codes -- they legitimately appear "
    "in many internal layouts. Only flag a code as a typo when "
    "ALL of the following are true:\n"
    "  (c1) there is a SPECIFIC anchor value already present in "
    "       this column that differs by exactly 1-2 character "
    "       edits AND is more frequent than the suspect value; "
    "       and\n"
    "  (c2) the defect is a clear character-level slip in the "
    "       alphabetic portion (e.g., dropped or duplicated "
    "       letter) -- not a difference in spacing, casing, "
    "       hyphenation, or other punctuation between "
    "       alphanumeric segments.\n"
    "In particular, do NOT flag a code merely because:\n"
    "  - it concatenates letters and digits without separators "
    "    ('2016StrawPoll', 'MASSAIR');\n"
    "  - it contains or omits internal whitespace between "
    "    segments ('ZUR3260 R2R M', 'PRESIDIO H S');\n"
    "  - its hyphenation or punctuation differs from what you "
    "    would expect ('ESR 3375' vs 'ESR-3375').\n"
    "Bias toward HIGH PRECISION: false positives cost as much as "
    "false negatives. When in doubt about a value, omit it from "
    "the typos list. The intended correction MUST be one of the "
    "OTHER distinct values shown in this column (or a near-variant "
    "of one with case/whitespace fixed); never invent a correction "
    "that has no anchor in the column."
)


# ---------------------------------------------------------------------------
# Trainee prompt assembly
# ---------------------------------------------------------------------------

TYPO_V3_SYSTEM_INSTRUCTION = (
    "You are a data-quality reviewer auditing ONE CSV column for "
    "typos. Apply the rulebook below to the column's distinct values "
    "and return the FULL list of typos you find as a single JSON "
    "object. Reply with the JSON object only -- no Markdown fences, "
    "no surrounding text.\n\n"
    f"{TYPO_V3_RULEBOOK}\n\n"
    "Output JSON shape (strict):\n"
    "  typos: list of objects, each with keys:\n"
    "    value      -- the typo-bearing distinct value, copied "
    "verbatim from the input.\n"
    "    correction -- the intended canonical form (string).\n"
    "    flavour    -- 'spelling' | 'punctuation'.\n"
    "    rationale  -- one short sentence naming the defect.\n"
    "Return an EMPTY list if you find no typos. Every value NOT in "
    "the list is treated as CLEAN."
)


def format_distinct_values(
    values_with_counts: Sequence[tuple[str, int]],
    *,
    max_distinct: int = TYPO_V3_MAX_DISTINCT,
) -> tuple[str, int, bool]:
    """Render the column's distinct values as a numbered listing.

    Returns ``(listing_text, n_shown, truncated)``. Values are shown
    in count-descending order (canonical / common values first, rare
    candidates -- where typos cluster -- last). ``max_distinct``
    caps the listing for wide columns.

    Each line: ``  <idx>.  <count>x  '<value>'``.
    """
    items = sorted(
        values_with_counts,
        key=lambda kv: (-int(kv[1]), str(kv[0])),
    )
    truncated = len(items) > max_distinct
    shown = items[:max_distinct]
    width = len(str(len(shown)))
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
    max_distinct: int = TYPO_V3_MAX_DISTINCT,
) -> str:
    """Format one column-audit prompt.

    ``values_with_counts`` is a sequence of ``(value, row_count)``
    pairs covering every distinct non-null value in the column.
    """
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
    """Coerce / validate one gold typo entry.

    Required keys: ``value``, ``correction``. Optional: ``flavour``.
    Trivial corrections (pure whitespace / casing / punctuation
    rewrites) are rejected -- they're noise under the v2 rulebook
    and must not enter the training set.
    """
    value = entry.get("value")
    correction = entry.get("correction")
    if not isinstance(value, str) or not value:
        raise ValueError(f"gold typo missing/empty 'value': {entry!r}")
    if not isinstance(correction, str) or not correction:
        raise ValueError(
            f"gold typo missing/empty 'correction': {entry!r}"
        )
    if is_trivial_typo_correction(value, correction):
        raise ValueError(
            f"gold typo is trivial (whitespace/casing/punct only): "
            f"value={value!r}, correction={correction!r}"
        )
    flavour = entry.get("flavour") or "spelling"
    if flavour not in TYPO_V3_FLAVOURS:
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
      ambiguous_values      -- List of column values whose v2 gold was
                               ``ambiguous`` (predictions on these are
                               credited neither way -- treated as
                               neither TP nor FP by the grader).
      max_distinct          -- Override TYPO_V3_MAX_DISTINCT.

    The Python grader reads ``gold_typos`` and ``ambiguous_values``
    as JSON-encoded strings from the item.* namespace.
    """
    column = item["column"]
    source = item.get("source", "")
    values_with_counts = list(item["values_with_counts"])
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
    max_distinct = int(item.get("max_distinct", TYPO_V3_MAX_DISTINCT))

    distinct_set = {str(v) for v, _ in values_with_counts}
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
    return {
        "messages": [
            {"role": "developer", "content": TYPO_V3_SYSTEM_INSTRUCTION},
            {"role": "user", "content": user_msg},
        ],
        # Grader-visible (item.*). The Python grader receives the raw
        # ``item`` dict, so unlike v2 we do NOT JSON-encode strings
        # for textual template substitution -- but we DO JSON-encode
        # the typo list so the grader can decode it deterministically
        # with a single ``json.loads`` call.
        "column": column,
        "source": source,
        "gold_typos": json.dumps(gold_typos, ensure_ascii=False),
        "ambiguous_values": json.dumps(
            ambiguous_values, ensure_ascii=False
        ),
    }


# ---------------------------------------------------------------------------
# Response format (strict JSON schema)
# ---------------------------------------------------------------------------


def build_response_format() -> dict[str, Any]:
    """Strict JSON schema for the trainee's reply.

    The response is a single object with a ``typos`` array. Each
    entry in the array has ``value``, ``correction``, ``flavour``,
    and ``rationale`` (all required by OpenAI's strict-mode rules).
    """
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
                "enum": list(TYPO_V3_FLAVOURS),
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
            "name": "typo_column_audit",
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
# Python grader -- deterministic set-F1 between pred and gold typo sets
# ---------------------------------------------------------------------------
#
# The grader code below is what Azure RFT will execute inside the
# constrained Python sandbox (see
# https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/reinforcement-fine-tuning
# -- 256KB code, 2GB memory, no network). It must:
#   * never raise (errors -> training job fails);
#   * always return a numeric value in [0.0, 5.0];
#   * decode item.gold_typos / item.ambiguous_values from JSON
#     strings;
#   * decode the trainee output from sample.output_text.
# The local helper ``compute_column_f1`` below mirrors this logic so
# eval scripts can score offline without spinning up RFT.

GRADER_SOURCE = '''
"""Python grader for the v3 typo column-audit task.

Computes set-F1 between the trainee's predicted typo set and the gold
typo set per column. Returns ``F1 * 5`` so the score stays in the
[0.0, 5.0] range used by the v1/v2 score-model graders.

Match key: (normalised value, normalised correction). Normalisation
is case-insensitive with collapsed whitespace -- matches the v2
``correction equals gold up to case and collapsed whitespace`` rule.

Ambiguous values (gold-labelled ``ambiguous`` in the v2 benchmark)
are removed from BOTH the predicted and gold sets before scoring --
predictions on them are credited neither true positive nor false
positive, matching the v2 ambiguous-either-correct rule.
"""

import json
import re


def _norm(text):
    """Case-fold and collapse whitespace; safe on non-strings."""
    if not isinstance(text, str):
        return ""
    return re.sub(r"\\s+", " ", text).strip().casefold()


def _load_json_string(value, default):
    """item.gold_typos / item.ambiguous_values come in as JSON strings."""
    if isinstance(value, (list, dict)):
        return value
    if not isinstance(value, str) or not value:
        return default
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return default


def _extract_pred_typos(sample):
    """Pull the trainee's typos list out of sample.output_text safely."""
    text = ""
    if isinstance(sample, dict):
        text = sample.get("output_text") or ""
        if not text:
            # Fallback for output_json (when strict response_format
            # parsed correctly): the sandbox may surface either field.
            parsed = sample.get("output_json")
            if isinstance(parsed, dict) and "typos" in parsed:
                tl = parsed.get("typos")
                return tl if isinstance(tl, list) else []
    if not isinstance(text, str) or not text:
        return []
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(parsed, dict):
        return []
    typos = parsed.get("typos")
    return typos if isinstance(typos, list) else []


def grade(sample, item):
    try:
        pred_list = _extract_pred_typos(sample)
        gold_list = _load_json_string(item.get("gold_typos"), [])
        ambiguous = _load_json_string(item.get("ambiguous_values"), [])

        ambiguous_keys = {_norm(a) for a in ambiguous if a}

        # Build maps from normalised value -> normalised correction.
        # Skip schema-violating entries silently (don't penalise the
        # whole column for one malformed list item).
        pred_map = {}
        for entry in pred_list:
            if not isinstance(entry, dict):
                continue
            v = _norm(entry.get("value"))
            c = _norm(entry.get("correction"))
            if not v or not c:
                continue
            if v in ambiguous_keys:
                continue
            pred_map[v] = c

        gold_map = {}
        for entry in gold_list:
            if not isinstance(entry, dict):
                continue
            v = _norm(entry.get("value"))
            c = _norm(entry.get("correction"))
            if not v or not c:
                continue
            if v in ambiguous_keys:
                continue
            gold_map[v] = c

        # Per-value TP: predicted value is in gold AND correction
        # matches. FP: predicted value not in gold OR correction
        # disagrees. FN: gold value not predicted.
        pred_keys = set(pred_map)
        gold_keys = set(gold_map)
        tp = sum(
            1 for k in pred_keys & gold_keys
            if pred_map[k] == gold_map[k]
        )
        fp = len(pred_keys - gold_keys) + sum(
            1 for k in pred_keys & gold_keys
            if pred_map[k] != gold_map[k]
        )
        fn = len(gold_keys - pred_keys) + sum(
            1 for k in pred_keys & gold_keys
            if pred_map[k] != gold_map[k]
        )
        # Both empty -- column is genuinely clean, full credit.
        if tp == 0 and fp == 0 and fn == 0:
            return 5.0
        if tp == 0:
            return 0.0
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        if precision + recall == 0:
            return 0.0
        f1 = 2.0 * precision * recall / (precision + recall)
        return float(max(0.0, min(5.0, f1 * 5.0)))
    except Exception:
        # The Azure docs say grader exceptions can fail the whole
        # training job; never raise.
        return 0.0
'''


def build_grader(grader_model: str | None = None) -> dict[str, Any]:
    """Return a score-model (LLM) grader spec for v3.

    History: v3 originally shipped a Python grader (set-F1 in the
    Azure sandbox; see ``GRADER_SOURCE`` below). That grader caused
    repeated "Job failed" system errors during postprocessing on
    Azure RFT despite running cleanly locally. Until that is rooted
    out we use a score_model grader -- same pattern as v1 (``typo``)
    and v2 -- with a rubric that mirrors the set-F1 logic so the
    learning signal stays comparable.

    The grader is given the column header, source id, gold typo set
    (as a JSON array), the ambiguous-values list (as a JSON array),
    and the trainee's raw reply. It returns Score 0-5 per the rubric
    embedded in the developer prompt.
    """
    developer = (
        "You are evaluating a data-quality model's JSON output for "
        "a column-audit task. The model is given one CSV column "
        "(header + distinct values with counts) and must list every "
        "TYPO it finds, with the intended correction.\n\n"
        f"{TYPO_V3_RULEBOOK}\n\n"
        "You will receive (labelled plain-text fields, not a single "
        "JSON wrapper):\n"
        "  - column:           CSV column header.\n"
        "  - source:           source table id.\n"
        "  - gold_typos:       JSON array of {value, correction, "
        "flavour} objects -- the gold-labelled typos.\n"
        "  - ambiguous_values: JSON array of column values whose "
        "gold label is 'ambiguous'. A prediction on any of these "
        "values is neither rewarded nor penalised -- drop them "
        "from both sides before counting.\n"
        "  - model_output:     the model's raw reply. Expected "
        "shape: a JSON object with a 'typos' array; each entry has "
        "value, correction, flavour, rationale.\n\n"
        "Scoring procedure:\n"
        "  Step 1 -- Parse model_output as JSON. If parsing fails, "
        "or 'typos' is missing or not an array, score 0.\n"
        "  Step 2 -- Drop every model_output.typos entry and every "
        "gold_typos entry whose value (case-folded, whitespace-"
        "collapsed) appears in ambiguous_values.\n"
        "  Step 3 -- Match: a model entry matches a gold entry when "
        "BOTH .value AND .correction agree up to case and collapsed "
        "whitespace. Flavour mismatch does not break the match.\n"
        "  Step 4 -- Count TP (matched pairs), FP (unmatched model "
        "entries), FN (unmatched gold entries).\n\n"
        "Rubric (integer 0-5):\n"
        "  5: TP == |gold_typos_after_filter| AND FP == 0 (perfect "
        "set match); OR both sets empty (column is genuinely clean "
        "and model returned an empty typos array).\n"
        "  4: TP == |gold_typos_after_filter| AND FP <= 1 "
        "(perfect recall, one spurious flag); OR all gold values "
        "predicted but one correction is off by a single character.\n"
        "  3: at least half the gold values are caught and the "
        "spurious-flag count is at most equal to TP.\n"
        "  2: at least one TP but precision OR recall is poor; OR "
        "gold is empty and the model added 2-4 spurious flags.\n"
        "  1: model_output parses as JSON but TP == 0 (full miss "
        "when gold is non-empty), OR gold is empty and the model "
        "added 5+ spurious flags.\n"
        "  0: empty output, not valid JSON, or schema violation.\n\n"
        "Respond ONLY with two lines:\n"
        "Score: <0-5>\n"
        "Reasoning: <one short sentence>"
    )
    # NOTE: no whitespace inside {{ ... }} braces. Azure's grader
    # template parser does not strip them, and treats
    # ``{{ item.x }}`` as a variable literally named " item.x " --
    # which it then rejects with HTTP 400.
    user = (
        "Column: {{item.column}}\n"
        "Source: {{item.source}}\n"
        "Gold typos (JSON): {{item.gold_typos}}\n"
        "Ambiguous values (JSON): {{item.ambiguous_values}}\n"
        "\n"
        "Model output:\n"
        "{{sample.output_text}}"
    )
    return {
        "name": "typo_v3_grader",
        "type": "score_model",
        "model": grader_model,
        "input": [
            {"role": "developer", "content": developer},
            {"role": "user", "content": user},
        ],
        # Pass at >=4 so the trainee only gets full reward for high
        # recall + at most a single spurious flag.
        "pass_threshold": 4,
        "range": [0.0, 5.0],
    }


# ---------------------------------------------------------------------------
# Offline scoring helper (mirrors the python grader for eval scripts)
# ---------------------------------------------------------------------------


def compute_column_f1(
    pred_typos: Iterable[Mapping[str, Any]],
    gold_typos: Iterable[Mapping[str, Any]],
    ambiguous_values: Iterable[str] = (),
) -> dict[str, float | int]:
    """Compute set-F1 + TP/FP/FN locally.

    Match key: (normalised value, normalised correction), case-folded
    with collapsed whitespace. Predictions on ambiguous values are
    removed from both sides before scoring. Returns a dict with
    ``precision``, ``recall``, ``f1``, ``tp``, ``fp``, ``fn``,
    ``score`` (= ``f1 * 5``; ``5`` when both sets are empty).
    """
    import re as _re

    def _norm(text: Any) -> str:
        if not isinstance(text, str):
            return ""
        return _re.sub(r"\s+", " ", text).strip().casefold()

    amb = {_norm(a) for a in ambiguous_values if a}

    def _to_map(entries: Iterable[Mapping[str, Any]]) -> dict[str, str]:
        out: dict[str, str] = {}
        for e in entries:
            if not isinstance(e, Mapping):
                continue
            v = _norm(e.get("value"))
            c = _norm(e.get("correction"))
            if v and c and v not in amb:
                out[v] = c
        return out

    pred_map = _to_map(pred_typos)
    gold_map = _to_map(gold_typos)

    pred_keys = set(pred_map)
    gold_keys = set(gold_map)
    tp = sum(
        1 for k in pred_keys & gold_keys
        if pred_map[k] == gold_map[k]
    )
    correction_mismatches = sum(
        1 for k in pred_keys & gold_keys
        if pred_map[k] != gold_map[k]
    )
    fp = len(pred_keys - gold_keys) + correction_mismatches
    fn = len(gold_keys - pred_keys) + correction_mismatches

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
        if (precision + recall) else 0.0
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "score": f1 * 5.0,
    }


# ---------------------------------------------------------------------------
# English-only filter
# ---------------------------------------------------------------------------
#
# v3 is scoped to English data only (the trainee + grader rulebook
# is written for English defects; multilingual columns add a "is
# this a real typo or a legitimate non-English glyph?" axis the
# model has no signal on). Both ``split_typo_benchmark_v3`` and
# ``synth_typo_llm_v3`` apply this filter so train and eval stay
# aligned. The filter is intentionally heuristic, not language-id-
# library-based, so it has zero new dependencies.

# Default threshold: at least 98% of alphabetic characters across
# all distinct values must be ASCII. This passes columns that
# happen to contain a few foreign names (e.g. "Sao Paulo" in a
# US-state column) and rejects columns whose dominant content is
# Vietnamese, Italian-with-heavy-diacritics, Cyrillic, CJK, etc.
TYPO_V3_ENGLISH_VALUE_THRESHOLD = 0.98


def _ascii_alpha_fraction(text: str) -> float:
    """Return fraction of alphabetic chars in ``text`` that are ASCII.

    Returns ``1.0`` if ``text`` has no alphabetic characters at all
    (numeric / punctuation-only values count as "trivially English").
    """
    if not isinstance(text, str) or not text:
        return 1.0
    alpha_total = 0
    ascii_total = 0
    for ch in text:
        if ch.isalpha():
            alpha_total += 1
            if ord(ch) < 128:
                ascii_total += 1
    if alpha_total == 0:
        return 1.0
    return ascii_total / alpha_total


def is_english_column(
    column_header: str,
    values: Iterable[str],
    *,
    value_threshold: float = TYPO_V3_ENGLISH_VALUE_THRESHOLD,
) -> bool:
    """Heuristic English-only filter for a (header, values) pair.

    Rules:
      * Column header: every alphabetic character must be ASCII.
        A header like ``"Ocupación"`` or ``"Điểm giao dịch"`` is
        immediately disqualifying.
      * Values: at least ``value_threshold`` of all alphabetic
        characters concatenated across distinct values must be
        ASCII.

    This does NOT use a language-id library so it has zero new
    dependencies. It rejects Vietnamese (Latin-with-heavy-diacritics)
    and Cyrillic / CJK / Arabic outright, and tolerates English
    columns that happen to carry a few foreign names.
    """
    if column_header and _ascii_alpha_fraction(column_header) < 1.0:
        return False
    joined = "".join(v for v in values if isinstance(v, str))
    return _ascii_alpha_fraction(joined) >= value_threshold


__all__ = [
    "TYPO_V3_TRAIN_FILE",
    "TYPO_V3_VALID_FILE",
    "TYPO_V3_FLAVOURS",
    "TYPO_V3_MAX_DISTINCT",
    "TYPO_V3_RULEBOOK",
    "TYPO_V3_SYSTEM_INSTRUCTION",
    "TYPO_V3_ENGLISH_VALUE_THRESHOLD",
    "format_distinct_values",
    "build_user_message",
    "build_record",
    "build_response_format",
    "build_grader",
    "GRADER_SOURCE",
    "compute_column_f1",
    "is_english_column",
    "is_trivial_typo_correction",
]
