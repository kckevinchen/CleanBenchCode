"""Decide which clusters are worth a web round-trip.

We only verify clusters that look like *named-entity* claims (org names,
addresses, etc.) where the web has a chance of carrying ground truth.
Free-text fields, numeric outliers, formatting nits, etc. are filtered
out cheaply so we don't burn API calls on them.

Heuristics, in order of evidence strength:

1. ``error_type`` is one of a short whitelist (typo, naming, address).
2. The flagged column header matches a known entity-column regex
   (``name``, ``address``, ``street``, ``city``, ``zip``, ``vendor``,
   ``organization``, ...).
3. The flagged cell value is short (< 200 chars), non-empty, contains
   at least one letter, and is not pure numeric / pure punctuation.

A cluster passes the gate when the column or error_type matches AND
the cell value passes the value gate. Each branch can be tightened or
loosened by the caller via :class:`GateConfig`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, List, Mapping, Optional, Sequence


class EntityKind(str, Enum):
    """Which kind of entity a cluster's column appears to describe.

    Used to bias the web query (e.g. for ``ADDRESS`` we add a state /
    country hint, for ``ORG`` we add ``official site``).
    """

    ORG = "org"           # company, agency, nonprofit, school, ...
    PERSON = "person"     # individual's name
    ADDRESS = "address"   # street / city / state / postal-code field
    PHONE = "phone"
    URL = "url"
    OTHER = "other"       # gate-eligible but kind unknown
    NONE = "none"         # not gate-eligible


# Column-name patterns. Compiled once. Word-boundary matching so
# ``zip_code`` matches ``zip`` but ``unzipper`` does not.
_COLUMN_PATTERNS: dict[EntityKind, re.Pattern[str]] = {
    EntityKind.ORG: re.compile(
        r"\b(?:"
        r"company|companies|organi[sz]ation|agency|agencies|firm|"
        r"vendor|supplier|contractor|partnership|corp(?:oration)?|"
        r"nonprofit|institution|institute|school|district|department|"
        r"office|bureau|business|entity|landlord|tenant|grantee|"
        r"recipient|provider|payee"
        r")\b",
        re.IGNORECASE,
    ),
    EntityKind.ADDRESS: re.compile(
        r"\b(?:"
        r"address|addr|street|st|road|rd|avenue|ave|"
        r"city|state|country|county|zip(?:[\s_-]?code)?|postal[\s_-]?code|"
        r"location|venue"
        r")\b",
        re.IGNORECASE,
    ),
    EntityKind.PHONE: re.compile(r"\bphone|fax|telephone|mobile|cell\b", re.IGNORECASE),
    EntityKind.URL: re.compile(r"\b(?:url|website|web[\s_-]?site|homepage|link)\b", re.IGNORECASE),
    EntityKind.PERSON: re.compile(
        # Conservative: only obvious person-name columns. We do not
        # verify person names by default (privacy) - a separate gate
        # config flag toggles this.
        r"\b(?:first[\s_-]?name|last[\s_-]?name|surname|"
        r"contact[\s_-]?name|representative|principal|owner[\s_-]?name)\b",
        re.IGNORECASE,
    ),
}


# Default error_type whitelist (keys from prompts/error_types.yaml).
# ``typo`` and ``text_inconsistency`` are the prime targets; we keep a
# small set so we don't flood the verifier with formatting nits.
DEFAULT_ERROR_TYPES: frozenset[str] = frozenset({
    "typo",
    "text_inconsistency",
    "casing_inconsistency",
    "pattern_inconsistency",
    "cross_record_inconsistency",
    "cross_source_naming_conflict",
})


@dataclass
class GateConfig:
    """Tunable gate parameters."""

    error_types: frozenset[str] = field(default_factory=lambda: DEFAULT_ERROR_TYPES)
    max_value_chars: int = 200
    min_value_chars: int = 2
    # Person verification is OFF by default to avoid PII fishing.
    verify_person: bool = False


def _classify_column(header: str) -> EntityKind:
    """Match a column header against entity-column regexes.

    Returns :attr:`EntityKind.NONE` if no pattern matches. Multi-line
    headers (LLM-mangled) are flattened first since the regexes use
    word boundaries that newlines don't satisfy.
    """
    if not header:
        return EntityKind.NONE
    flat = re.sub(r"\s+", " ", str(header).strip())
    if not flat:
        return EntityKind.NONE
    for kind, pat in _COLUMN_PATTERNS.items():
        if pat.search(flat):
            return kind
    return EntityKind.NONE


_NUMERIC_RE = re.compile(r"^[\s\d.,+\-/$%()]*$")
_LETTER_RE = re.compile(r"[A-Za-z\u00C0-\u024F]")


def _value_eligible(value: str, cfg: GateConfig) -> bool:
    if not value:
        return False
    s = str(value).strip()
    if len(s) < cfg.min_value_chars or len(s) > cfg.max_value_chars:
        return False
    if _NUMERIC_RE.match(s):
        # All-numeric (with currency / punctuation) is not an entity.
        # Phones / ZIPs would technically pass this but they're caught
        # earlier by column kind.
        return False
    if not _LETTER_RE.search(s):
        return False
    return True


def classify_cluster(
    cluster: Mapping[str, object],
    *,
    cfg: Optional[GateConfig] = None,
) -> EntityKind:
    """Best-guess entity kind for ``cluster``. ``NONE`` if not gate-eligible.

    The cluster's first column (post-canonicalisation) decides the
    kind. We fall back to error_type-only matching for clusters whose
    column header is empty or unrecognised - that keeps the verifier
    useful for rare runs where the LLM forgot to fill in ``columns``.
    """
    cfg = cfg or GateConfig()
    columns = cluster.get("columns") or []
    header = ""
    if isinstance(columns, list) and columns:
        header = str(columns[0])
    kind = _classify_column(header)
    if kind == EntityKind.PERSON and not cfg.verify_person:
        return EntityKind.NONE
    if kind != EntityKind.NONE:
        return kind
    # No column match: if the error_type is in the whitelist AND we
    # have a value that looks textual, treat as OTHER.
    et_list = cluster.get("error_types") or []
    if isinstance(et_list, list) and any(t in cfg.error_types for t in et_list):
        return EntityKind.OTHER
    return EntityKind.NONE


def should_verify(
    cluster: Mapping[str, object],
    *,
    cell_value: Optional[str] = None,
    cfg: Optional[GateConfig] = None,
) -> bool:
    """Return ``True`` if this cluster is worth a web verification call.

    ``cell_value`` is the actual value from the row at the flagged
    column. The caller must supply it (we don't have access to the
    frame here). When ``cell_value`` is None the value gate is skipped
    so the function can be used as a cheap pre-filter.
    """
    cfg = cfg or GateConfig()

    # Hard PII reject: even if the error_type is whitelisted, a column
    # whose header reads as a person-name field must NOT go to the web
    # unless the caller explicitly opts in. This has to happen before
    # the error_type branch below, otherwise PERSON columns with a
    # whitelisted error_type (typo, text_inconsistency, ...) would
    # still leak through as ``OTHER`` via :func:`classify_cluster`.
    columns = cluster.get("columns") or []
    raw_header = ""
    if isinstance(columns, list) and columns:
        raw_header = str(columns[0])
    raw_kind = _classify_column(raw_header)
    if raw_kind == EntityKind.PERSON and not cfg.verify_person:
        return False

    et_list = cluster.get("error_types") or []
    if not isinstance(et_list, list):
        et_list = [et_list]
    et_match = any(t in cfg.error_types for t in et_list)

    kind = classify_cluster(cluster, cfg=cfg)
    column_match = kind not in (EntityKind.NONE,)

    if not (et_match or column_match):
        return False

    if cell_value is not None and not _value_eligible(cell_value, cfg):
        return False

    return True


def filter_eligible(
    clusters: Sequence[Mapping[str, object]],
    *,
    value_lookup,  # callable: (cluster) -> Optional[str]
    cfg: Optional[GateConfig] = None,
) -> List[Mapping[str, object]]:
    """Return only the clusters that pass :func:`should_verify`.

    ``value_lookup`` is a callable that resolves the cell value at the
    cluster's first (row, column). Pulled out as a callback so this
    module stays free of pandas / IO dependencies.
    """
    cfg = cfg or GateConfig()
    out: List[Mapping[str, object]] = []
    for c in clusters:
        try:
            val = value_lookup(c)
        except Exception:
            val = None
        if should_verify(c, cell_value=val, cfg=cfg):
            out.append(c)
    return out


__all__ = [
    "EntityKind",
    "GateConfig",
    "DEFAULT_ERROR_TYPES",
    "classify_cluster",
    "should_verify",
    "filter_eligible",
]
