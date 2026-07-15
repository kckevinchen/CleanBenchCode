# Python grader source for v3 typo column-audit task.
# This file is a snapshot of ``experiments._typo_v3_task.GRADER_SOURCE`` for code review.
# Azure RFT executes this code in a sandboxed Python
# environment (256 KB code limit, 2 GB memory, no network).
# The grader must never raise; it always returns a number
# in [0.0, 5.0]. See scoring_rubric.md for the rubric.

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
    return re.sub(r"\s+", " ", text).strip().casefold()


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
