"""Shared, version-agnostic typo-task primitives.

Every typo RFT version imports these from here -- **never from a peer version**.
These are the rulebook-free, deterministic pieces shared across all typo
versions: the trainee output schema (:func:`build_response_format`) and the
three python sub-graders (trivial-FP gate, column-cap gate, gold-recall leg),
each a self-contained source string Azure runs in its sandbox.

The version-SPECIFIC pieces -- the score_model precision judge (which embeds a
per-version rulebook) and the reward formula -- stay in each version's own
``_grader*.py``; assemble a multigrader from those plus these primitives.

Source of truth: these bodies were lifted verbatim from the original
``typo/v5/_task.py`` (which keeps its own copy for the frozen v5-v8 snapshots).
"""

from __future__ import annotations

from typing import Any

from RFT._common.graders import DEFAULT_GRADER_MODEL, RFT_SCORE_MODEL_WHITELIST

# Neutral aliases + back-compat names so any consumer can import either.
GRADER_MODEL_WHITELIST = RFT_SCORE_MODEL_WHITELIST
TYPO_GRADER_MODEL_WHITELIST = RFT_SCORE_MODEL_WHITELIST
TYPO_DEFAULT_GRADER_MODEL = DEFAULT_GRADER_MODEL
TYPO_V5_GRADER_MODEL_WHITELIST = RFT_SCORE_MODEL_WHITELIST
TYPO_V5_DEFAULT_GRADER_MODEL = DEFAULT_GRADER_MODEL
TYPO_FLAVOURS = ("spelling", "punctuation", "spacing")


TYPO_V5_FLAVOURS: tuple[str, ...] = ("spelling", "punctuation", "spacing")

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
                "enum": list(TYPO_V5_FLAVOURS),
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

TYPO_V5_PYTHON_GRADER_SOURCE = '''
"""Python grader for the v5 typo column-audit task.

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

Predictions whose (value -> correction) edit is v5-trivial (pure
padding-whitespace or pure casing) are dropped silently -- those
are CLEAN per v5 sec.2 and must not count as predictions.

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


def _is_v5_trivial(original, correction):
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
            if _is_v5_trivial(v_raw, c_raw):
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

def build_python_grader() -> dict[str, Any]:
    """Python sub-grader: deterministic gold-set bookkeeping."""
    return {
        "type": "python",
        "name": "typo_v5_python_gold",
        "source": TYPO_V5_PYTHON_GRADER_SOURCE,
    }

TYPO_V5_PYTHON_TRIVIAL_GRADER_SOURCE = r'''
"""Python sub-grader: mechanical trivial-FP detector for the v4 typo task.

For each predicted typo `{value, correction}`, returns 0.0 if the edit
matches one of the deterministic CLEAN edit shapes (no-op, casing-only,
padding-whitespace, apostrophe-elision, Unicode-dash variants,
hyphen-vs-space swap, fixed-width prefix-truncation), else 1.0
(innocent -- leaves judgment to the score-model leg).

Final grader score = mean across predictions (1.0 if no predictions).
Combined with the score-model leg multiplicatively in calculate_output,
so any detected trivial-FP zeroes the prediction's contribution.

Defensive contract:
  * Never raises -- any exception returns 0.0.
  * Returns a plain ``float`` in [0.0, 1.0].
  * Uses only ``json`` and ``re`` from the standard library.
"""

import json
import re


# Unicode dash/hyphen variants that are typographically equivalent to
# ASCII '-' for trivial-FP detection. Detected via numeric ord()
# comparison rather than literal characters or chr() construction.


def _norm_dashes(s):
    out = []
    for ch in s:
        o = ord(ch)
        if 0x2010 <= o <= 0x2015 or o == 0x2212:
            out.append("-")
        else:
            out.append(ch)
    return "".join(out)


def _norm_ws(s):
    return re.sub(r"\s+", " ", s).strip()


def _is_trivial_fp(value, correction):
    if not isinstance(value, str) or not isinstance(correction, str):
        return False
    if not value or not correction:
        return value == correction  # no-op
    # Normalize Unicode dash variants to ASCII '-' before any rule;
    # purely typographic, no semantic loss.
    value = _norm_dashes(value)
    correction = _norm_dashes(correction)
    if value == correction:
        return True
    v_ws = _norm_ws(value)
    c_ws = _norm_ws(correction)
    # 1. casing-only (covers internal mixed-case too)
    if value.casefold() == correction.casefold():
        return True
    # 2. padding-whitespace only
    if v_ws == c_ws:
        return True
    # 3. casing + padding-whitespace combined
    if v_ws.casefold() == c_ws.casefold():
        return True
    # 4. apostrophe-elision: strip ASCII ' and curly U+2019 via ord().
    v_ap = "".join(ch for ch in value if ch != "'" and ord(ch) != 0x2019)
    c_ap = "".join(ch for ch in correction if ch != "'" and ord(ch) != 0x2019)
    if v_ap == c_ap or v_ap.casefold() == c_ap.casefold():
        return True
    # 5. hyphen-vs-space stylistic swap between adjacent tokens
    v_hs = value.replace("-", " ").split()
    c_hs = correction.replace("-", " ").split()
    if v_hs == c_hs:
        return True
    # 6. fixed-width prefix-truncation -- only when VALUE is the truncated
    #    (shorter) form and CORRECTION appends characters that complete a
    #    partial trailing word. Direction matters: a longer value being
    #    "corrected" to a shorter form is a spelling fix, not truncation.
    v_s, c_s = value.strip(), correction.strip()
    if (
        len(v_s) >= 12
        and v_s != c_s
        and c_s.startswith(v_s)
        and v_s[-1].isalnum()
    ):
        suffix = c_s[len(v_s):]
        if suffix and suffix[0].isalnum():
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
        preds = _extract_pred_typos(sample)
        if not preds:
            return 1.0
        n = 0
        innocent = 0
        for entry in preds:
            if not isinstance(entry, dict):
                continue
            v = entry.get("value")
            c = entry.get("correction")
            n += 1
            if not _is_trivial_fp(v, c):
                innocent += 1
        if n == 0:
            return 1.0
        return float(innocent) / float(n)
    except Exception:
        return 0.0
'''

def build_python_trivial_grader() -> dict[str, Any]:
    """Python sub-grader: mechanical trivial-FP gate.

    Returns 0.0 if any predicted (value -> correction) edit matches a
    deterministic CLEAN shape (no-op, casing, padding-ws, apostrophe,
    Unicode-dash variant, hyphen-vs-space swap, fixed-width prefix
    truncation). Multiplied into the multigrader output so trivial-
    FP predictions zero the reward.
    """
    return {
        "type": "python",
        "name": "typo_v5_python_trivial",
        "source": TYPO_V5_PYTHON_TRIVIAL_GRADER_SOURCE,
    }

TYPO_V5_COLUMN_CAP_ABSOLUTE = 5

TYPO_V5_COLUMN_CAP_FRACTION = 0.30

TYPO_V5_PYTHON_COLUMN_CAP_GRADER_SOURCE = r'''
"""Python sub-grader: column-level prediction-count cap.

Returns 1.0 if the model's predicted typo count for this column is
within the cap, 0.0 if it is over (the model is spamming).

Cap = max(K_ABS, ceil(FRAC * num_distinct)) where num_distinct is
derived from item.column_values_json (a JSON array of [value, count]
pairs). If num_distinct cannot be determined, the absolute floor is
used as the cap.

Multiplied into the multigrader output so any column where the model
exceeds the cap contributes 0.0, regardless of how it scores
elsewhere.

Defensive contract:
  * Never raises -- any exception returns 1.0 (innocent).
  * Returns a plain ``float`` in {0.0, 1.0}.
  * Uses only ``json`` and ``math`` from the standard library.
"""

import json
import math


K_ABS = 5
FRAC = 0.30


def _extract_pred_count(sample):
    if not isinstance(sample, dict):
        return 0
    oj = sample.get("output_json")
    if isinstance(oj, dict):
        tl = oj.get("typos")
        if isinstance(tl, list):
            return len(tl)
    ot = sample.get("output_text") or ""
    if not isinstance(ot, str) or not ot.strip():
        return 0
    try:
        parsed = json.loads(ot)
    except Exception:
        return 0
    if isinstance(parsed, dict):
        tl = parsed.get("typos")
        if isinstance(tl, list):
            return len(tl)
    return 0


def _extract_num_distinct(item):
    if not isinstance(item, dict):
        return None
    raw = item.get("column_values_json")
    if isinstance(raw, list):
        return len(raw)
    if not isinstance(raw, str) or not raw:
        return None
    try:
        parsed = json.loads(raw)
    except Exception:
        return None
    if isinstance(parsed, list):
        return len(parsed)
    return None


def grade(sample, item):
    try:
        n_pred = _extract_pred_count(sample)
        num_distinct = _extract_num_distinct(item)
        if num_distinct is None or num_distinct <= 0:
            cap = K_ABS
        else:
            cap = max(K_ABS, int(math.ceil(FRAC * float(num_distinct))))
        if n_pred > cap:
            return 0.0
        return 1.0
    except Exception:
        return 1.0
'''

def build_python_column_cap_grader() -> dict[str, Any]:
    """Python sub-grader: column-level prediction-count cap.

    Returns 0.0 if the model proposed too many typos for a single
    column (likely spam), else 1.0. Multiplied into ``calculate_output``
    so over-cap columns contribute 0.0 regardless of other sub-graders.

    Cap = ``max(K_ABS, ceil(FRAC * num_distinct))`` with
    ``K_ABS=5`` and ``FRAC=0.30`` (kept in sync with
    ``TYPO_V5_COLUMN_CAP_ABSOLUTE`` / ``TYPO_V5_COLUMN_CAP_FRACTION``).
    """
    return {
        "type": "python",
        "name": "typo_v5_python_column_cap",
        "source": TYPO_V5_PYTHON_COLUMN_CAP_GRADER_SOURCE,
    }


def build_python_gates_grader() -> dict[str, Any]:
    """Single python sub-grader = trivial-FP gate x column-cap gate, FUSED.

    Returns the PRODUCT of :func:`build_python_trivial_grader` and
    :func:`build_python_column_cap_grader`, computed in ONE python subprocess
    instead of two. The reward is byte-identical to multiplying the two
    separate sub-graders in ``calculate_output`` -- multiplication is
    associative and each leg keeps its OWN defensive default (trivial -> 0.0,
    cap -> 1.0 on a catastrophic error).

    The point is operational, not behavioural: each ``python`` sub-grader Azure
    runs gets its own per-invocation ``/mnt/data/<uuid>_data.json`` staged for
    it; collapsing two legs into one halves that staging exposure, which is the
    suspected trigger of the non-retriable "Runtime exceptions in Python grader
    ... FileNotFoundError" limit that killed every two-python-grader v11 run
    (see ``RFT/typo/v11/_grader.py``). v10 ``nogold``/``nogoldrec`` (which
    trained + deployed) used exactly ONE python grader.

    Composed from the two VERBATIM leg sources (their ``grade`` renamed to
    ``_grade_trivial`` / ``_grade_cap``) so the fused grader can never drift
    from the legs it fuses.
    """
    anchor = "def grade(sample, item):"
    legs: list[str] = []
    for tag, src in (
        ("trivial", TYPO_V5_PYTHON_TRIVIAL_GRADER_SOURCE),
        ("cap", TYPO_V5_PYTHON_COLUMN_CAP_GRADER_SOURCE),
    ):
        if src.count(anchor) != 1:
            raise RuntimeError(
                f"{tag} python grader source must define {anchor!r} exactly "
                f"once to be fused; found {src.count(anchor)}."
            )
        legs.append(src.replace(anchor, f"def _grade_{tag}(sample, item):"))

    combined = (
        '"""Fused trivial-FP x column-cap python gate (one subprocess).\n'
        "\n"
        "grade() returns _grade_trivial(...) * _grade_cap(...): identical reward\n"
        "to the two former python sub-graders, with a single /mnt/data staging\n"
        "op per rollout instead of two.\n"
        '"""\n'
        + legs[0]
        + legs[1]
        + "\n\ndef grade(sample, item):\n"
        "    return _grade_trivial(sample, item) * _grade_cap(sample, item)\n"
    )
    return {
        "type": "python",
        "name": "typo_v5_python_gates",
        "source": combined,
    }


# ---------------------------------------------------------------------------
# Shared eval/data helpers (used by recall benches + scorers across versions)
# ---------------------------------------------------------------------------

import json  # noqa: E402
import re  # noqa: E402
from typing import Iterable, Mapping, Sequence  # noqa: E402

TYPO_V5_MAX_DISTINCT = 500

def is_v5_trivial_correction(original: Any, correction: Any) -> bool:
    """Return True iff (original -> correction) is a v5-trivial edit.

    A correction is v5-trivial when it differs from the original
    ONLY by:

    * padding whitespace (leading/trailing or repeated runs collapsed
      to a single space), OR
    * casing.

    Both classes are explicitly CLEAN under v5 sec.2 -- caught by
    deterministic regexes upstream of the LLM.

    Critically, **spacing-tokenisation edits are NOT v5-trivial**.
    A change like ``DuPage`` -> ``Du Page`` or ``data base`` ->
    ``database`` alters how the value tokenises into words and is
    therefore a substantive ``spacing`` typo per v5 sec.1. v3's
    ``is_trivial_typo_correction`` would have dropped these; the
    v5 filter retains them.

    Punctuation-only edits are likewise NOT v5-trivial -- they fall
    under the v5 ``punctuation`` flavour.
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

def format_distinct_values(
    values_with_counts: Sequence[tuple[str, int]],
    *,
    max_distinct: int = TYPO_V5_MAX_DISTINCT,
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
    max_distinct: int = TYPO_V5_MAX_DISTINCT,
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

def compute_column_f1(
    pred_typos: Iterable[Mapping[str, Any]],
    gold_typos: Iterable[Mapping[str, Any]],
    ambiguous_values: Iterable[str] = (),
) -> dict[str, float | int]:
    """Set-F1 + TP/FP/FN with v5 ambiguous-value handling.

    Match key: (normalised value, normalised correction), case-folded
    with collapsed whitespace. Predictions whose (value -> correction)
    edit is v5-trivial are dropped silently (they cost nothing).
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
        if is_v5_trivial_correction(v, c):
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

# back-compat alias
is_trivial_correction = is_v5_trivial_correction
