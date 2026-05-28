"""Classify each LLM decision into a per-channel sub-label, with caching.

For every benchmark-label JSONL we audit, this script collapses cells to
the LLM's decision unit (matching ``notebooks/llm_label_error_audit.ipynb``)
and assigns a *sub-label* drawn from the prompt taxonomy
(``prompts/error_types.yaml``):

* **typo**          - spelling / whitespace / punctuation / casing
                      (heuristic; derived from original_value vs corrected_value).
* **inconsistency** - text / casing / pattern / punctuation / number_format
                      (taken directly from the existing ``cluster_kinds`` field;
                      the labeling pipeline already records this).
* **semantic_incompatible** - type_mismatch / domain_departure /
                              hedge_advisory / peer_convention_misread /
                              free_text_in_constrained
                              (LLM-classified; cached).
* **fd_violation** - real_constraint / spurious_inferred /
                     named_entity_key / date_key / advisory_only
                     (LLM-classified; cached).

Output: ``label_result/decision_subtypes_cache.jsonl`` -- one row per
decision keyed by ``(channel, source, column, decision_id)`` so the
notebook can join in O(1).

Re-running is idempotent: existing cache entries are kept; only the
new decisions hit the LLM. To force re-classification of one channel
pass ``--force-channel semantic``.

Usage
-----
    python -m experiments.classify_decision_subtypes
    python -m experiments.classify_decision_subtypes --force-channel semantic
    python -m experiments.classify_decision_subtypes --limit 25  # smoke test
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.core.llm_client import LLMClient  # noqa: E402

logger = logging.getLogger("subtypes")

LABEL_DIR = REPO / "label_result"
CACHE_PATH = LABEL_DIR / "decision_subtypes_cache.jsonl"

# Same files the audit notebook consumes.
FILES = {
    "typo":          LABEL_DIR / "benchmark_typo_20260507T215150Z_gpt5-family.canonical.jsonl",
    "inconsistency": LABEL_DIR / "benchmark_inconsistency_20260515T010112Z_gpt5-family.jsonl",
    "semantic":      LABEL_DIR / "benchmark_semantic_incompatible_20260508T164531Z_gpt5-family.jsonl",
    "fd_violation":  LABEL_DIR / "benchmark_fd_violation_20260514T063141Z_gpt5-family.jsonl",
}

# Sub-label taxonomies (used to coerce LLM output into a fixed set).
SUBTYPES = {
    "typo": {"spelling", "whitespace", "punctuation", "casing", "unknown"},
    "inconsistency": {"text", "casing", "pattern", "punctuation", "number_format", "unknown"},
    "semantic": {
        "type_mismatch",            # phone in a Name column
        "domain_departure",         # status not in {Open, Closed}
        "hedge_advisory",           # "If ...", "Verify ...", "Confirm ..."
        "peer_convention_misread",  # header says percent but every row is "1/3"
        "free_text_in_constrained", # model over-flagged a free-text/notes column
        "unknown",
    },
    "fd_violation": {
        "real_constraint",   # zip->state, country->capital
        "spurious_inferred", # LLM proposed an FD that isn't a real constraint
        "named_entity_key",  # determinant is a name (weak FD)
        "date_key",          # date treated as determinant
        "advisory_only",     # suggestion is "verify" not a literal fix
        "unknown",
    },
}


# ---------------------------------------------------------------------------
# 1. Decision-unit aggregation (mirrors the notebook)
# ---------------------------------------------------------------------------


def _stable_label(labels: Iterable[str]) -> str:
    s = set(labels)
    if "error" in s:
        return "error"
    if "clean" in s:
        return "clean"
    return "ambiguous"


def _decision_id(channel: str, row: Dict[str, Any]) -> Tuple:
    """Build the same decision key the notebook uses, per channel."""
    src = row["source"]
    col = row.get("column")
    if channel == "typo":
        return (src, col, row.get("original_value"))
    if channel == "semantic":
        sugg = row.get("llm_suggested_value")
        return (src, col, str(sugg)[:200] if sugg is not None else None)
    if channel == "inconsistency":
        cids = row.get("cluster_ids") or []
        return (src, col, tuple(cids) if isinstance(cids, list) else (cids,))
    if channel == "fd_violation":
        dids = row.get("detection_ids") or []
        first = dids[0] if isinstance(dids, list) and dids else None
        return (src, first)
    raise ValueError(f"unknown channel {channel}")


def load_decisions(channel: str, path: Path) -> List[Dict[str, Any]]:
    """Collapse cells to one row per decision; mirror DECISIONS in the notebook."""
    by_key: Dict[Tuple, Dict[str, Any]] = {}
    for line in path.open():
        r = json.loads(line)
        k = _decision_id(channel, r)
        slot = by_key.setdefault(
            k,
            {
                "channel": channel,
                "source": r["source"],
                "column": r.get("column"),
                "decision_key": list(k[1:]) if isinstance(k, tuple) else [k],
                "labels": [],
                "rep": r,
            },
        )
        slot["labels"].append(r["label"])
    out: List[Dict[str, Any]] = []
    for k, slot in by_key.items():
        rep = slot["rep"]
        out.append(
            {
                "channel": channel,
                "source": slot["source"],
                "column": slot["column"],
                "decision_key": slot["decision_key"],
                "n_cells": len(slot["labels"]),
                "verdict": _stable_label(slot["labels"]),
                "original_value": rep.get("original_value"),
                "llm_suggested_value": rep.get("llm_suggested_value"),
                "corrected_value": rep.get("corrected_value")
                or rep.get("corrected_value_canonicalized"),
                "cluster_kinds": rep.get("cluster_kinds") or [],
                "note": rep.get("note") or "",
            }
        )
    return out


# ---------------------------------------------------------------------------
# 2. Heuristic sub-labels (typo + inconsistency)
# ---------------------------------------------------------------------------


_PUNCT = re.compile(r"[^\w\s]")
_WS_INTERNAL = re.compile(r"\s+")
_DIGITS = re.compile(r"\d")


def classify_typo(orig: Any, corrected: Any) -> Tuple[str, str]:
    """Return (sub_label, reason).

    Priority order (a typo that crosses several axes is reported by its
    *primary* axis so the breakdown stays interpretable):
      1. ``spelling``    -- alphabetic letters differ (case-folded)
      2. ``whitespace``  -- internal whitespace structure differs
      3. ``punctuation`` -- only non-alphanumeric, non-space chars differ
      4. ``casing``      -- pure case-only flip
    ``unknown`` covers: missing correction, no-op correction, empty input.
    """
    if not isinstance(orig, str) or not orig:
        return "unknown", "empty_original"
    c = corrected if isinstance(corrected, str) and corrected else None
    if not c:
        return "unknown", "no_correction"
    if orig == c:
        return "unknown", "noop_correction"

    # 1) Spelling -- alpha letters differ (case-insensitive).
    o_letters = re.sub(r"[^A-Za-z]", "", orig).lower()
    c_letters = re.sub(r"[^A-Za-z]", "", c).lower()
    if o_letters != c_letters:
        return "spelling", "letters_differ"

    # 2) Whitespace -- collapse multi-space + trim; if those differ it's a
    #    whitespace-structure issue.
    o_ws = _WS_INTERNAL.sub(" ", orig).strip()
    c_ws = _WS_INTERNAL.sub(" ", c).strip()
    if o_ws != c_ws or (orig.strip() != orig) or (c.strip() != c) or (
        _WS_INTERNAL.sub(" ", orig) != orig
        or _WS_INTERNAL.sub(" ", c) != c
    ):
        return "whitespace", "whitespace_structure_differs"

    # 3) Punctuation -- letters & whitespace identical; only punct differs.
    o_punct = "".join(ch for ch in orig if not ch.isalnum() and not ch.isspace())
    c_punct = "".join(ch for ch in c if not ch.isalnum() and not ch.isspace())
    if o_punct != c_punct:
        return "punctuation", "punct_differs"

    # 4) Casing -- everything else matches; pure case flip.
    if orig != c:
        return "casing", "case_only"

    return "unknown", "no_difference_found"


def classify_inconsistency(cluster_kinds: List[str]) -> Tuple[str, str]:
    if not cluster_kinds:
        return "unknown", "empty_cluster_kinds"
    # Take the dominant kind.
    c = Counter(cluster_kinds).most_common()
    top, n = c[0]
    if top not in SUBTYPES["inconsistency"]:
        return "unknown", f"unmapped:{top}"
    return top, f"dominant:{top}({n}/{sum(x for _, x in c)})"


# ---------------------------------------------------------------------------
# 3. LLM-driven sub-labels (semantic + fd_violation)
# ---------------------------------------------------------------------------


SEM_SYSTEM = (
    "You are classifying a single LLM 'semantic_incompatible' detection into "
    "ONE of a fixed set of sub-labels so we can summarise failure modes. "
    "Respond with ONLY a JSON object."
)

SEM_PROMPT = """We previously asked an LLM to flag cells whose semantic
type/domain doesn't fit their column. For ONE such detection below,
choose the BEST sub-label.

Sub-labels (pick exactly one):
  - type_mismatch          : value type doesn't match the column (e.g. a phone number in a Name column)
  - domain_departure       : value's a sensible type but outside the column's observed set (e.g. "Pending review" when peers are {{Open, Closed}})
  - hedge_advisory         : the LLM's suggestion is advisory ("If...", "Verify...", "Confirm..."), not a concrete replacement -- favour this for clearly non-actionable suggestions
  - peer_convention_misread: header looks one way but the column's peers follow a different convention; the LLM didn't notice
  - free_text_in_constrained: the column actually allows free text and the LLM over-flagged
  - unknown                : none of the above

DETECTION
---------
source:        {source}
column:        {column}
original_value: {orig}
llm_suggested_value: {sugg}

Respond with EXACTLY one JSON object:
  {{"sub_label": "<one of the labels above>", "reason": "<<=30 words>"}}"""


FD_SYSTEM = (
    "You are classifying a single LLM 'fd_violation' detection into "
    "ONE of a fixed set of sub-labels so we can summarise failure modes. "
    "Respond with ONLY a JSON object."
)

FD_PROMPT = """We previously asked an LLM to flag rows where a functional
dependency was violated. For ONE such detection below, choose the BEST
sub-label characterising what KIND of FD the LLM proposed.

Sub-labels (pick exactly one):
  - real_constraint   : a well-known real-world FD (zip->state, country->capital, productCode->category)
  - spurious_inferred : the LLM *guessed* an FD that isn't a real constraint
  - named_entity_key  : the determinant is a person/place name -- a notoriously weak determinant
  - date_key          : the determinant is a date (also a weak determinant)
  - advisory_only     : the suggestion text is "verify"/"reconcile" rather than a concrete fix
  - unknown           : none of the above

DETECTION
---------
source:        {source}
column:        {column}
original_value: {orig}
llm_suggested_value: {sugg}

Respond with EXACTLY one JSON object:
  {{"sub_label": "<one of the labels above>", "reason": "<<=30 words>"}}"""


def _truncate(v: Any, n: int = 200) -> str:
    if v is None:
        return "<null>"
    s = str(v)
    return s if len(s) <= n else s[:n] + "..."


def _parse_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Extract the first JSON object from a model reply."""
    if not text:
        return None
    # Strip code fences if present.
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.S)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    # Fallback: regex-extract the first {...} block.
    m = re.search(r"\{.*\}", t, flags=re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def call_llm_subtype(
    client: LLMClient,
    channel: str,
    decision: Dict[str, Any],
    model: str,
) -> Tuple[str, str]:
    if channel == "semantic":
        system = SEM_SYSTEM
        prompt = SEM_PROMPT.format(
            source=decision["source"],
            column=decision.get("column") or "<null>",
            orig=_truncate(decision.get("original_value")),
            sugg=_truncate(decision.get("llm_suggested_value")),
        )
    elif channel == "fd_violation":
        system = FD_SYSTEM
        prompt = FD_PROMPT.format(
            source=decision["source"],
            column=decision.get("column") or "<null>",
            orig=_truncate(decision.get("original_value")),
            sugg=_truncate(decision.get("llm_suggested_value")),
        )
    else:
        return "unknown", "unsupported_channel"

    resp = client.query(model, prompt, reasoning_effort="low", system_prompt=system)
    if not resp.ok:
        return "unknown", f"llm_error:{(resp.error or '')[:80]}"
    obj = _parse_json_object(resp.text)
    if not obj or "sub_label" not in obj:
        return "unknown", f"unparseable:{resp.text[:80]}"
    sub = str(obj.get("sub_label", "")).strip()
    reason = str(obj.get("reason", ""))[:200]
    if sub not in SUBTYPES[channel]:
        return "unknown", f"unmapped:{sub}"
    return sub, reason


# ---------------------------------------------------------------------------
# 4. Cache I/O
# ---------------------------------------------------------------------------


def cache_key(channel: str, decision: Dict[str, Any]) -> str:
    """Stable per-decision cache key.

    Format mirrors the notebook's join key exactly:
        ``<channel>||<source>||<column>||<channel-specific tail>``

    The tail is the LAST element of ``decision_key``. ``load_decisions``
    records the full per-channel key tuple (minus ``source``) and the
    column appears at position 0 for every multi-part-key channel
    except ``fd_violation`` (whose decision_key is a single-element list).

    JSON-roundtrip subtlety: ``inconsistency``'s tail is a tuple at
    construction time but becomes a list once it has been written to
    and re-read from the cache file. Normalize back to a tuple before
    stringifying so the key is identical in both directions.
    """
    parts = [channel, decision["source"], str(decision.get("column"))]
    tail = decision["decision_key"][-1]
    if channel == "inconsistency" and isinstance(tail, list):
        tail = tuple(tail)
    parts.append(str(tail))
    return "||".join(parts)


def load_cache(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for line in path.open():
        r = json.loads(line)
        out[r["cache_key"]] = r
    return out


def write_cache(path: Path, cache: Dict[str, Dict[str, Any]]) -> None:
    # Stable order to keep diffs sensible.
    keys = sorted(cache.keys())
    with path.open("w") as fh:
        for k in keys:
            fh.write(json.dumps(cache[k], ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# 5. Main driver
# ---------------------------------------------------------------------------


def classify_channel(
    channel: str,
    decisions: List[Dict[str, Any]],
    cache: Dict[str, Dict[str, Any]],
    force: bool,
    *,
    client: Optional[LLMClient],
    model: str,
    workers: int,
    limit: Optional[int] = None,
) -> int:
    """Classify every decision in ``decisions`` for ``channel``.

    Returns the number of NEW cache entries written.
    """
    todo: List[Dict[str, Any]] = []
    for d in decisions:
        key = cache_key(channel, d)
        if not force and key in cache:
            continue
        todo.append(d)
    if limit is not None:
        todo = todo[:limit]

    new_n = 0
    if channel in ("typo", "inconsistency"):
        for d in todo:
            if channel == "typo":
                sub, reason = classify_typo(d.get("original_value"), d.get("corrected_value"))
            else:
                sub, reason = classify_inconsistency(d.get("cluster_kinds") or [])
            cache[cache_key(channel, d)] = {
                "cache_key": cache_key(channel, d),
                "channel": channel,
                "source": d["source"],
                "column": d.get("column"),
                "decision_key": d["decision_key"],
                "verdict": d["verdict"],
                "n_cells": d["n_cells"],
                "sub_label": sub,
                "reason": reason,
            }
            new_n += 1
        return new_n

    # LLM-driven channels.
    assert client is not None, "LLM client required for semantic / fd_violation"

    def _run(d: Dict[str, Any]) -> Tuple[Dict[str, Any], str, str]:
        sub, reason = call_llm_subtype(client, channel, d, model)
        return d, sub, reason

    if not todo:
        return 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_run, d) for d in todo]
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                d, sub, reason = fut.result()
            except Exception as exc:  # noqa: BLE001
                logger.warning("classify failure: %s", exc)
                continue
            cache[cache_key(channel, d)] = {
                "cache_key": cache_key(channel, d),
                "channel": channel,
                "source": d["source"],
                "column": d.get("column"),
                "decision_key": d["decision_key"],
                "verdict": d["verdict"],
                "n_cells": d["n_cells"],
                "sub_label": sub,
                "reason": reason,
            }
            new_n += 1
            if i % 25 == 0:
                logger.info("  [%s] classified %d / %d", channel, i, len(todo))
    return new_n


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force-channel", action="append", default=[], help="Re-classify this channel (can repeat).")
    ap.add_argument("--channels", nargs="+", default=list(FILES.keys()))
    ap.add_argument("--model", default="gpt-5.5,gpt-5.4,gpt-5.2")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="Cap per-channel decisions (smoke testing).")
    ap.add_argument("--cache", type=Path, default=CACHE_PATH)
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    cache = load_cache(args.cache)
    logger.info("cache start size: %d", len(cache))

    need_llm = any(ch in ("semantic", "fd_violation") for ch in args.channels)
    client = LLMClient() if need_llm else None

    for channel in args.channels:
        if channel not in FILES:
            logger.warning("unknown channel %r -- skipping", channel)
            continue
        force = channel in args.force_channel
        path = FILES[channel]
        decisions = load_decisions(channel, path)
        logger.info("[%s] %d decisions (force=%s)", channel, len(decisions), force)
        new_n = classify_channel(
            channel,
            decisions,
            cache,
            force,
            client=client,
            model=args.model,
            workers=args.workers,
            limit=args.limit,
        )
        logger.info("[%s] +%d new cache entries", channel, new_n)
        # Persist after every channel so a crash mid-LLM-sweep doesn't lose work.
        write_cache(args.cache, cache)

    logger.info("cache final size: %d -> %s", len(cache), args.cache)


if __name__ == "__main__":
    main()
