# Python column-cap gate sub-grader source for v5 typo task.
# Verbatim copy of ``_typo_v5_task.TYPO_V5_PYTHON_COLUMN_CAP_GRADER_SOURCE``
# for code review. Azure RFT runs the in-grader.json copy.

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
