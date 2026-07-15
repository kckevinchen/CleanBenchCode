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
# ASCII '-' for the purposes of trivial-FP detection. Same family of
# rule as the apostrophe variants below: collapses purely typographic
# differences before any structural comparison.
_DASH_CHARS = "\u2010\u2011\u2012\u2013\u2014\u2015\u2212"
_DASH_RE = re.compile(f"[{_DASH_CHARS}]")


def _norm_dashes(s):
    return _DASH_RE.sub("-", s)


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
    # 4. apostrophe-elision
    v_ap = value.replace("'", "").replace("\u2019", "")
    c_ap = correction.replace("'", "").replace("\u2019", "")
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
