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
    EntityKind.PHONE: re.compile(
        r"\b(?:phone|fax|telephone|mobile|cell)\b", re.IGNORECASE
    ),
    EntityKind.URL: re.compile(r"\b(?:url|website|web[\s_-]?site|homepage|link)\b", re.IGNORECASE),
    EntityKind.PERSON: re.compile(
        # Conservative: only obvious person-name columns. We do not
        # verify person names by default (privacy) - a separate gate
        # config flag toggles this.
        # Patterns include the explicit name parts (first / last /
        # surname), name-bearing role columns ("Sales Person",
        # "Person Name", "Contact Name"), and entity-of-record roles
        # that are typically populated with a person's name in real
        # tables (representative, principal, owner_name, applicant,
        # signer, signatory, author). We deliberately keep these
        # unambiguous - "name" alone could be a product / project /
        # account name and stays untouched.
        r"\b(?:"
        r"first[\s_-]?name|last[\s_-]?name|sur[\s_-]?name|middle[\s_-]?name|"
        r"full[\s_-]?name|person[\s_-]?name|sales[\s_-]?person|sales[\s_-]?rep(?:resentative)?|"
        r"contact[\s_-]?name|representative|principal|owner[\s_-]?name|"
        r"applicant|signer|signatory|author|attendee|guest"
        r")\b",
        re.IGNORECASE,
    ),
}


# Default error_type whitelist (keys from prompts/error_types.yaml).
# Web verification is useful for any error_type whose ground truth
# can plausibly be looked up online (entity names, addresses, FD
# relationships, cross-table consistency). We deliberately exclude
# pure-formatting / pure-arithmetic types - the web cannot adjudicate
# whether a column should use ``%`` vs decimal, or whether a row's
# Total = Item1 + Item2 holds.
DEFAULT_ERROR_TYPES: frozenset[str] = frozenset({
    # Single-cell: classic entity-name / address corrections. The
    # merged ``inconsistency`` umbrella covers what used to be
    # text/casing/pattern/punctuation/number_format. Punctuation and
    # number-format clusters are weeded out cheaply by the column /
    # value gates below before any web call is attempted.
    "typo",
    "inconsistency",
    "semantic_incompatible",
    # Multi-row: web can confirm whether two records are really the
    # same real-world entity (or distinct entities that happen to share
    # a key field).
    "row_duplication",
    # Multi-column: web can verify functional-dependency relationships
    # (license -> business name -> address, etc.) for a single row.
    "fd_violation",
    # Multi-table: web confirms which source carries the correct
    # value when two sources disagree.
    "inter_table_inconsistency",
})


# Error types that are NEVER worth a web round-trip - their ground
# truth is computable from the row/column itself, not from external
# evidence. The gate hard-rejects clusters whose ONLY error_types
# come from this set, regardless of the operator's whitelist.
NON_WEB_VERIFIABLE_TYPES: frozenset[str] = frozenset({
    "row_calculation_inconsistency",
    "string_calculation_inconsistency",
    "numeric_outlier",
})


@dataclass
class GateConfig:
    """Tunable gate parameters."""

    error_types: frozenset[str] = field(default_factory=lambda: DEFAULT_ERROR_TYPES)
    max_value_chars: int = 200
    min_value_chars: int = 2
    # Person verification is ON by default. The verifier uses a
    # lenient criterion for person-name typos -- a name is valid as
    # long as the EXACT suspect spelling exists somewhere on the web
    # as a real person's name (anywhere in the world). It does NOT
    # need to match the specific person referenced by the row. This
    # avoids the systematic false-positive problem where rare-but-
    # real spellings (e.g. "Murry", "Catherin") get flagged as typos.
    # Flip to ``False`` if you need to suppress all person-name web
    # calls (e.g. for a PII-sensitive audit).
    verify_person: bool = True
    # Multi-row caps: skip clusters with fewer than this many distinct
    # rows (a 1-row "multi-row" cluster is just a single_cell), and
    # cap the number of records we send to the agent for a given
    # multi-row check.
    multi_row_min: int = 2
    multi_row_cap: int = 6
    # Multi-column caps: skip clusters with fewer than this many
    # columns under review, and cap how many we render in the prompt.
    multi_col_min: int = 2
    multi_col_cap: int = 8


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


def cluster_shape(cluster: Mapping[str, object]) -> dict:
    """Classify a cluster's spatial layout for verification routing.

    Returns a dict with three keys:

    * ``pattern`` - one of ``"single_cell"`` | ``"multi_row"`` |
      ``"multi_column"`` | ``"composite"``.
    * ``rows`` - sorted unique list of row indices the cluster spans.
      Built from ``cluster.row`` plus the union of
      ``detections[*].affected_rows``.
    * ``columns`` - the cluster's ``columns`` list, unmodified.

    The function is read-only and never raises; malformed inputs
    degrade to ``single_cell`` with whatever the caller passed in.
    """
    columns: List[str] = []
    raw_cols = cluster.get("columns") or []
    if isinstance(raw_cols, list):
        columns = [str(c) for c in raw_cols if c is not None]

    rows: List[int] = []
    seen: set[int] = set()
    primary = cluster.get("row")
    if isinstance(primary, int):
        rows.append(primary)
        seen.add(primary)
    elif isinstance(primary, float) and not (primary != primary):  # not NaN
        ri = int(primary)
        rows.append(ri)
        seen.add(ri)
    for d in cluster.get("detections") or []:
        if not isinstance(d, Mapping):
            continue
        for r in d.get("affected_rows") or []:
            try:
                ri = int(r)
            except (TypeError, ValueError):
                continue
            if ri in seen:
                continue
            seen.add(ri)
            rows.append(ri)
    rows.sort()

    multi_row = len(rows) >= 2
    multi_col = len(columns) >= 2
    if multi_row and multi_col:
        pattern = "composite"
    elif multi_row:
        pattern = "multi_row"
    elif multi_col:
        pattern = "multi_column"
    else:
        pattern = "single_cell"
    return {"pattern": pattern, "rows": rows, "columns": columns}


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
    # whitelisted error_type (typo, inconsistency, ...) would
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

    # Hard reject error_types that are not adjudicable on the web.
    # If a cluster has at least one web-verifiable type alongside a
    # non-web one, we still proceed (the agent will focus on the web
    # angle); the rejection only triggers when ALL listed types are
    # in the non-web set.
    if et_list and all(t in NON_WEB_VERIFIABLE_TYPES for t in et_list):
        return False

    et_match = any(t in cfg.error_types for t in et_list)
    shape = cluster_shape(cluster)
    pattern = shape["pattern"]

    if pattern == "multi_row":
        # Multi-row: at least N rows + an error_type that can be
        # adjudicated by checking whether the records refer to the
        # same real-world entity.
        if len(shape["rows"]) < cfg.multi_row_min:
            return False
        return et_match or "row_duplication" in et_list

    if pattern in ("multi_column", "composite"):
        # Multi-column: need >=2 columns and an error_type that
        # speaks to inter-column consistency.
        if len(shape["columns"]) < cfg.multi_col_min:
            return False
        multicol_ok = {
            "fd_violation",
            "inter_table_inconsistency",
            "inconsistency",
            "semantic_incompatible",
        }
        return et_match or any(t in multicol_ok for t in et_list)

    # ---- single_cell (existing semantics) -------------------------
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
    "NON_WEB_VERIFIABLE_TYPES",
    "classify_cluster",
    "cluster_shape",
    "should_verify",
    "filter_eligible",
]
