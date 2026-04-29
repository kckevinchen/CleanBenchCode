"""Shared column-name canonicalization for LLM-returned issue payloads.

The LLM occasionally returns shortened versions of multi-line headers
(e.g. ``"District"`` for the actual header ``"District\\nCode"``).
Both the report builder (which feeds the clustering aggregator) and
the labeling server (which serves clusters back to the UI) need to
resolve those mangled names against the authoritative frame headers,
so clustering merges them and the UI highlights the right cells.

This module is the single source of truth for that resolution.
Centralising it here means a fix in the matching tiers automatically
applies to both the offline pipeline and the labeler API.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Sequence


def fold_header(s: object) -> str:
    """Normalize a header for tolerant comparison.

    Lowercases, folds all runs of whitespace (including embedded
    ``\\n`` from multi-line headers) into a single space, and strips.
    Nothing in the labeling pipeline depends on this fold being
    bijective; it's only used for tolerant equality / prefix matches
    against frame headers.
    """
    return " ".join(str(s or "").lower().split())


def canonicalize_columns(
    columns: Iterable[Any],
    frame_headers: Sequence[str],
) -> List[str]:
    """Resolve ``columns`` against ``frame_headers``.

    Resolution proceeds in three deterministic tiers:

    1. **exact membership** (cheap; the common case for clean runs).
    2. **case+whitespace-folded equality** (handles only-newline
       mangling like ``"District Code"`` vs ``"District\\nCode"``).
    3. **case+whitespace-folded prefix match**, with ambiguity broken
       by picking the shortest header (most specific column).
    4. **word-boundary suffix/contains match** (catches LLM tokens
       like ``"Code"`` for ``"County Code"`` / ``"District Code"``).
       Multiple matches are kept so downstream consumers can decide
       (the labeling UI shows them all rather than silently dropping
       one).

    A column that doesn't resolve to any frame header is **dropped**
    rather than carried through unmatched - keeping bad strings in
    the output causes the labeling UI to fail silently when looking
    up cell highlights.

    Returns a list of frame-header strings in stable order with no
    duplicates.
    """
    cols = list(columns or [])
    if not cols:
        return []
    header_set = set(frame_headers)
    folded = [(h, fold_header(h)) for h in frame_headers]
    resolved: List[str] = []
    for col in cols:
        if col in header_set:
            if col not in resolved:
                resolved.append(col)
            continue
        f = fold_header(col)
        if not f:
            continue
        hit = next((h for h, fh in folded if fh == f), None)
        cands: List[str] = []
        if hit is None:
            cands = [h for h, fh in folded if fh.startswith(f) or f.startswith(fh)]
        if hit is None and not cands:
            tokens = f.split()
            cands = [
                h for h, fh in folded
                if fh.endswith(" " + f)
                or fh.endswith("\n" + f)
                or all(tok in fh.split() for tok in tokens)
            ]
        if hit is None and cands:
            cands.sort(key=len)
            for h in cands:
                if h not in resolved:
                    resolved.append(h)
            continue
        if hit is not None and hit not in resolved:
            resolved.append(hit)
    return resolved


def canonicalize_cluster_columns(
    cluster: Mapping[str, Any],
    frame_headers: Sequence[str],
) -> Dict[str, Any]:
    """Return a copy of ``cluster`` with ``columns`` resolved.

    Convenience wrapper around :func:`canonicalize_columns` that
    preserves all other fields of the cluster mapping.
    """
    out = dict(cluster)
    out["columns"] = canonicalize_columns(cluster.get("columns") or [], frame_headers)
    return out
