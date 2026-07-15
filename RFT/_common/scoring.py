"""Generic eval helpers for the ``run`` verb.

The deployed-model eval (recall_bench / compare_ft_vs_base) was previously
re-implemented per version with a hard-coded ``DEPLOYMENT`` name. These two
factory functions cover what those scripts actually varied -- the JSON array
key the trainee emits and the gold/ambiguous field names -- so a spec wires
them up in two lines (see :mod:`RFT._common.spec`).

The scorer implements the canonical cell-level micro-F1 from ``RFT.md`` §3:
pool predicted values across all (source, column) rows, compare to gold;
``ambiguous`` values are FP-immune (a prediction landing on one is neither a
true nor a false positive) and earn no recall credit.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

_FENCE_OPEN = re.compile(r"^```(?:json)?\s*")
_FENCE_CLOSE = re.compile(r"\s*```\s*$")


def _norm(value: Any) -> str:
    """Whitespace-collapsed string key used for matching."""
    return " ".join(str(value).split())


def _loads_maybe(raw: str | None) -> Any:
    """``json.loads`` tolerating ```` ```json ```` fences; ``None`` on failure."""
    if not raw:
        return None
    s = raw.strip()
    if s.startswith("```"):
        s = _FENCE_CLOSE.sub("", _FENCE_OPEN.sub("", s))
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return None


def _as_list(value: Any) -> list[Any]:
    """Coerce a field that may be a JSON-encoded string or already a list."""
    if isinstance(value, str):
        parsed = _loads_maybe(value)
        return parsed if isinstance(parsed, list) else []
    return value if isinstance(value, list) else []


def make_json_array_parser(
    array_key: str, value_key: str = "value"
) -> Callable[[str | None], set[str]]:
    """Parser that pulls ``obj[array_key][*][value_key]`` into a normalised set.

    e.g. typo emits ``{"typos": [{"value": ...}]}`` -> ``array_key="typos"``;
    inconsistency emits ``{"findings": [{"value": ...}]}``.
    """

    def parse(raw: str | None) -> set[str]:
        obj = _loads_maybe(raw)
        if not isinstance(obj, dict):
            return set()
        out: set[str] = set()
        for item in obj.get(array_key) or []:
            if isinstance(item, dict) and isinstance(item.get(value_key), str):
                out.add(_norm(item[value_key]))
        return out

    return parse


def make_cell_micro_f1_scorer(
    *, gold_field: str, ambiguous_field: str | None = None
) -> Callable[[list[dict[str, Any]], list[set[str]]], dict[str, Any]]:
    """Cell-level micro-F1 scorer (RFT.md §3).

    ``rows`` are eval.jsonl records; ``preds[i]`` is the parsed prediction set
    for ``rows[i]``. ``gold_field`` / ``ambiguous_field`` may be JSON-encoded
    strings (as the splits store them) or plain lists.
    """

    def score(rows: list[dict[str, Any]], preds: list[set[str]]) -> dict[str, Any]:
        tp = fp = fn = 0
        for row, pred in zip(rows, preds):
            gold = {_norm(v) for v in _as_list(row.get(gold_field))}
            ambig = (
                {_norm(v) for v in _as_list(row.get(ambiguous_field))}
                if ambiguous_field
                else set()
            )
            tp += len(pred & gold)
            fp += len(pred - gold - ambig)  # ambiguous predictions are immune
            fn += len(gold - pred)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        return {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "micro_f1": f1,
            "n_rows": len(rows),
        }

    return score
