"""The mechanical information barrier.

The reviewer sees dev data and is *instructed* not to quote it; this module is
the **backstop that enforces it**. It builds a blocklist of the concrete dev
values the reviewer could have seen (gold errors, their corrections, ambiguous
values, and the specific values surfaced as FP/FN evidence) and redacts any of
them that appear verbatim in the reviewer's suggestions — before those
suggestions reach the prompt editor. A suggestion whose actionable field is
gutted by redaction is dropped entirely.

Matching is whitespace-flexible and case-insensitive, mirroring the grader's
:func:`baseline.inconsistency._task.norm`. Very short or generic values are not
blocked (they would scrub legitimate rule language); the threshold is tunable.
"""

from __future__ import annotations

import re
from typing import Any

from baseline.inconsistency._task import norm

REDACTION = "\u00abREDACTED\u00bb"

# Generic tokens that may coincide with data values but must never be scrubbed
# out of rule language.
_STOPWORDS = {
    "true",
    "false",
    "none",
    "null",
    "name",
    "names",
    "date",
    "dates",
    "total",
    "value",
    "values",
    "yes",
    "no",
    "type",
    "types",
    "number",
    "count",
    "other",
    "column",
    "row",
    "text",
    "casing",
    "pattern",
    "punctuation",
}


def _candidate(value: str, min_len: int) -> str | None:
    """Return the normalised value if it is worth blocking, else ``None``."""
    nv = norm(value)
    low = nv.lower()
    if len(low) < min_len:
        return None
    if low in _STOPWORDS:
        return None
    if low.replace(".", "").replace(",", "").replace("-", "").isdigit() and len(low) < 5:
        return None  # short bare numbers over-match
    return nv


def _distinctive_header(header: str, min_len: int) -> str | None:
    """Return a column header only if it is *distinctive* (dev-specific) enough to block.

    Generic single-word headers (``category``, ``defect``, ``status``) double as
    ordinary rule vocabulary and appear in held-out data too — blocking them would
    corrupt rule language without preventing overfitting. Compound headers
    (``Defect Category``, ``Customer Address``) are far more likely to be a
    dev-specific identifier, so only those are blocked.
    """
    nv = norm(header)
    if len(nv.split()) < 2:
        return None
    return _candidate(nv, min_len)


def build_blocklist(
    *,
    dev_cols: dict[tuple[str, str], dict[str, Any]] | None = None,
    evidence: list[dict[str, Any]] | None = None,
    column_headers: set[str] | None = None,
    min_len: int = 4,
) -> set[str]:
    """Collect the dev values/identifiers to forbid in reviewer output.

    ``dev_cols`` is the gold restricted to dev sources (error originals +
    acceptable corrections + ambiguous values — ALL of these are blocked).
    ``evidence`` is the list of FP/FN records actually shown to the reviewer
    (their ``value`` / ``corrected_value`` / ``pred_correction`` fields).
    ``column_headers`` are dev column names; only *distinctive* (compound) ones
    are blocked (see :func:`_distinctive_header`). Any subset may be given.
    """
    raw: set[str] = set()
    for g in (dev_cols or {}).values():
        for nv, acc in g["error"].items():
            raw.add(nv)
            raw.update(acc)
        raw.update(g["amb"])
    for ev in evidence or []:
        for field in ("value", "corrected_value", "pred_correction", "gold_correction"):
            v = ev.get(field)
            items = v if isinstance(v, list) else [v]
            for x in items:
                if isinstance(x, str):
                    raw.add(x)
    block = {c for v in raw if (c := _candidate(v, min_len))}
    for h in column_headers or set():
        d = _distinctive_header(h, min_len)
        if d:
            block.add(d)
    return block


def _pattern_for(value: str) -> re.Pattern[str]:
    """Whitespace-flexible, case-insensitive matcher for one blocked value."""
    tokens = norm(value).split()
    body = r"\s+".join(re.escape(t) for t in tokens)
    return re.compile(body, re.IGNORECASE)


def _redact(text: str, patterns: list[re.Pattern[str]]) -> tuple[str, int]:
    n = 0
    for pat in patterns:
        text, k = pat.subn(REDACTION, text)
        n += k
    return text, n


def scrub_text(
    text: str, blocklist: set[str], *, placeholder: str = "an example"
) -> tuple[str, int]:
    """Redact any blocked value from free text with a readable placeholder.

    Used on the EDITOR's OUTPUT (the evolved rulebook) so a dev value can never
    sit in the prompt — whether it arrived via a leaked suggestion (the filter
    already strips those) OR was independently hallucinated by the editor as an
    example that happens to coincide with a dev value/header. Keeps the rulebook
    coherent by substituting a neutral phrase rather than a redaction marker.
    """
    n = 0
    for v in blocklist:
        pat = _pattern_for(v)
        text, k = pat.subn(placeholder, text)
        n += k
    return text, n


# The actionable field the editor relies on; a hit here means the whole
# suggestion is value-dependent and is dropped.
_ACTIONABLE_FIELD = "rule_change"


def scrub_critique(
    critique: dict[str, Any],
    blocklist: set[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Redact blocked values from a reviewer critique; drop gutted suggestions.

    EVERY string field of each suggestion is scanned (not just the named ones),
    so an unexpected field cannot smuggle a value to the editor. A suggestion is
    dropped when its ``rule_change`` — the only field the editor acts on — was
    redacted, since a leaked-value-dependent instruction does not generalise.
    Returns ``(filtered_critique, report)``.
    """
    patterns = [_pattern_for(v) for v in blocklist]
    out = dict(critique)
    redactions = 0
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []

    summary = critique.get("summary")
    if isinstance(summary, str):
        out["summary"], k = _redact(summary, patterns)
        redactions += k

    for sug in critique.get("suggestions", []) or []:
        if not isinstance(sug, dict):
            continue
        new = dict(sug)
        rule_hit = 0
        for f, v in sug.items():
            if isinstance(v, str):
                new[f], k = _redact(v, patterns)
                redactions += k
                if f == _ACTIONABLE_FIELD:
                    rule_hit = k
        if rule_hit:
            dropped.append({"suggestion": sug, "reason": "rule_change contained dev value(s)"})
        else:
            kept.append(new)

    out["suggestions"] = kept
    report = {
        "blocklist_size": len(blocklist),
        "redactions": redactions,
        "n_dropped": len(dropped),
        "dropped": dropped,
        "n_kept": len(kept),
    }
    return out, report
