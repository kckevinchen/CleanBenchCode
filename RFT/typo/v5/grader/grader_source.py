# Python sub-grader source for v5 typo column-audit task.
# This is a verbatim copy of ``_typo_v5_task.TYPO_V5_PYTHON_GRADER_SOURCE``
# for code review. Azure RFT executes the version embedded in grader.json
# inside its sandbox (256 KB code, 2 GB RAM, 1 CPU, 2 min, no network).

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
    return re.sub(r"\s+", " ", text).strip().casefold()


def _is_v5_trivial(original, correction):
    if not isinstance(original, str) or not isinstance(correction, str):
        return False
    if original == correction:
        return False
    o_norm = re.sub(r"\s+", " ", original).strip()
    c_norm = re.sub(r"\s+", " ", correction).strip()
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
