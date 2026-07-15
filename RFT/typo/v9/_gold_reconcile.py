"""Gold-set reconciliation for v9 (proper-noun char-level slips).

Why this exists
---------------
v9's precision grader puts proper nouns OUT OF SCOPE for character-level
typo corrections: a prediction whose ``value`` is a name / place / brand /
address / scientific binomial and whose ``correction`` is a character-level
edit scores as a hard false positive (reward 0.0). See
``RFT/typo/v9/prompts/rulebook.md`` sec.3 and ``_grader_variants.py``.

But the reward formula also multiplies/adds in ``typo_v5_python_gold`` --
recall against the gold set (``_grader_variants.py``). The inherited gold
set (v6/v7/v8 splits) still contains many proper-noun-token char slips
(``Brenalillo -> Bernalillo``, ``Flecther -> Fletcher``, ...). Those cells
create a lose-lose gradient: the recall leg penalises NOT flagging them
(counted as missed) while the precision leg penalises flagging them (hard
FP). Under r2 (harmonic) such a column collapses to ~0 reward either way.

This module classifies each gold typo cell so a builder can MOVE the
proper-noun-token char slips from ``gold_typos`` into ``ambiguous_values``
(FP-immune, no recall credit) -- removing the recall-side conflict while
leaving the precision rule to govern. Cells that v9 keeps in scope --
structural defects on names, and common-word-token fixes inside a name --
stay in ``gold_typos``.

Design constraints
------------------
* **Deterministic and offline.** No network, no LLM judge, no third-party
  deps. RFT.md sec.1 forbids LLM-judged data in the eval set, so the
  classifier is a pure function over (column, value, correction). Its output
  is meant to be reviewed by a human (see ``classify_cell`` reasons and the
  builder's report) -- the heuristic generates the candidate list; a human
  confirms it.
* **Column-header driven.** With no dictionary available we cannot tell an
  ordinary-word slip (``Recieved -> Received``, keep) from a proper-noun
  slip (``Brenalillo -> Bernalillo``, reconcile) by the word alone. The
  column header is the strong signal: ``City`` / ``Agency Name`` /
  ``Supplier`` hold proper nouns; ``Desc`` / ``Industry`` hold ordinary
  text. Cells in non-entity columns are always kept.
* **Conservative + auditable.** When the column is ambiguous, the cell is
  flagged ``review`` (kept in gold by default) rather than silently moved.
  Every decision carries a one-line reason for the report.
"""

from __future__ import annotations

import re
from typing import Literal

Decision = Literal["keep", "reconcile", "review"]


# ---------------------------------------------------------------------------
# Column-header classification
# ---------------------------------------------------------------------------

# Header tokens that mark a column as holding PROPER NOUNS (entity column).
# Char-level slips on the name token of such columns are reconcile candidates.
_ENTITY_HEADER_TOKENS: frozenset[str] = frozenset({
    "name", "names",
    "city", "town", "village", "borough", "township", "county", "state",
    "province", "country", "region", "district", "jurisdiction", "ward",
    "constituency", "location", "place", "address", "street", "suburb",
    "agency", "agencies", "company", "companies", "organisation",
    "organization", "org", "supplier", "vendor", "payee", "applicant",
    "sponsor", "provider", "institution", "employer", "manufacturer",
    "brand", "site", "facility", "store", "branch", "customer", "client",
    "contact", "author", "owner", "landlord", "tenant", "party", "entity",
    "business", "firm", "bank", "school", "university", "college",
    "hospital", "department", "authority", "council", "board", "trust",
    "person", "people", "individual", "staff", "officer", "manager",
    "director", "chairman", "ceo", "founder", "publisher", "developer",
})

# Header tokens that mark a column as ordinary descriptive TEXT. Slips here
# are ordinary-word typos and stay in gold. Used to override the entity set
# when both appear (text wins -- safer to keep in gold).
_TEXT_HEADER_TOKENS: frozenset[str] = frozenset({
    "desc", "description", "comment", "comments", "note", "notes", "remark",
    "remarks", "summary", "detail", "details", "industry", "sector",
    "type", "category", "class", "status", "reason", "subject", "food",
    "foods", "ingredient", "ingredients", "material", "materials", "fuel",
    "occupation", "role", "keyword", "keywords", "tag", "tags", "message",
})


def _header_tokens(header: str) -> list[str]:
    return [t for t in re.split(r"[^A-Za-z]+", header.lower()) if t]


def classify_column(header: str) -> Literal["entity", "text", "unknown"]:
    """Classify a column header as entity (proper nouns), text, or unknown."""
    toks = set(_header_tokens(header))
    if toks & _TEXT_HEADER_TOKENS:
        return "text"
    if toks & _ENTITY_HEADER_TOKENS:
        return "entity"
    return "unknown"


# ---------------------------------------------------------------------------
# Common institutional/descriptor words that legitimately appear INSIDE
# proper-noun values. A char-level slip on one of these is an ordinary-word
# fix that v9 KEEPS in scope (rulebook sec.3 "judge the token"). Not a full
# dictionary -- just the descriptor vocabulary that co-occurs in entity names.
# ---------------------------------------------------------------------------

_COMMON_TOKENS: frozenset[str] = frozenset({
    "development", "district", "county", "center", "centre", "association",
    "hospital", "department", "departments", "service", "services",
    "accommodation", "partner", "partners", "council", "school", "schools",
    "university", "college", "market", "room", "board", "trust", "group",
    "holdings", "company", "agency", "authority", "division", "region",
    "regional", "national", "international", "community", "charter",
    "medical", "family", "health", "fire", "drug", "court", "juvenile",
    "liaison", "nursing", "rehabilitation", "foundation", "society",
    "committee", "commission", "office", "bureau", "institute", "academy",
    "clinic", "pharmacy", "ministry", "north", "south", "east", "west",
    "northern", "southern", "eastern", "western", "central", "upper",
    "lower", "saint", "mount", "lake", "river", "valley", "hill", "hills",
    "park", "grove", "street", "lane", "road", "avenue", "drive", "place",
    "square", "terrace", "boulevard", "highway", "route", "suite", "floor",
    "unit", "building", "tower", "plaza", "store", "shop", "bank", "credit",
    "union", "mutual", "insurance", "financial", "capital", "ventures",
    "enterprises", "industries", "manufacturing", "construction",
    "engineering", "consulting", "solutions", "systems", "technologies",
    "logistics", "transport", "transportation", "distribution", "supply",
    "supplies", "wholesale", "retail", "trading", "energy", "power",
    "electric", "media", "communications", "publishing", "productions",
    "studios", "records", "beverage", "restaurant", "hotel", "resort",
    "hospitality", "tourism", "travel", "airlines", "airways", "automotive",
    "motors", "equipment", "nonviolent", "violent", "unsweetened",
    "sweetened", "canned", "frozen", "fresh", "silky", "smooth", "salty",
    "general", "operations", "operating", "youth",
    # common descriptor adjectives/nouns that appear in org/vendor names and
    # are ordinary dictionary words (a slip on one is an in-scope common-word
    # fix, not a proper-noun slip)
    "independent", "environmental", "professional", "technical", "financial",
    "commercial", "residential", "industrial", "agricultural", "educational",
    "electrical", "mechanical", "dental", "surgical", "clinical", "federal",
    "municipal", "public", "private", "cooperative", "integrated", "advanced",
    "applied", "associated", "allied", "combined", "consolidated", "certified",
    "registered", "licensed", "qualified", "accredited", "sustainable",
    "renewable", "global", "regional", "mountain", "lakes", "valleys",
})


def _norm_token(tok: str) -> str:
    """Lowercase, strip surrounding punctuation (keep internal apostrophes)."""
    return tok.strip(".,&()[]{}'\"/:;*-").lower()


def _alnum_only(s: str) -> str:
    return re.sub(r"[^0-9a-z]", "", s.lower())


def _is_structural(value: str, correction: str) -> bool:
    """True if the defect is structural (in scope on names per v9 sec.3).

    Covers: word-count change (spacing / token removal), adjacent duplicated
    word, intra-value duplicate-with-separator (``X/X`` -> ``X``), and
    punctuation/spacing-only edits.
    """
    v_toks = value.split()
    c_toks = correction.split()
    # word-count change -> spacing / duplicated-word / token-removal defect
    if len(v_toks) != len(c_toks):
        return True
    # punctuation- or spacing-only difference
    if _alnum_only(value) == _alnum_only(correction):
        return True
    # adjacent duplicated word/segment collapsed. Tokenise on whitespace AND
    # separators (`/`, `-`) so intra-token duplication is caught too, e.g.
    # `North Bennington/Bennington` -> `North Bennington`.
    def _dedup_adjacent(text: str) -> list[str]:
        toks = [t for t in re.split(r"[\s/\-]+", text.lower()) if t]
        out: list[str] = []
        for t in toks:
            if not out or out[-1] != t:
                out.append(t)
        return out

    if _dedup_adjacent(value) == _dedup_adjacent(correction):
        return True
    return False


def _changed_tokens(value: str, correction: str) -> list[tuple[str, str]]:
    """Aligned (value_token, correction_token) pairs that differ.

    Only meaningful when token counts match (char-level edit case).
    """
    v_toks = value.split()
    c_toks = correction.split()
    if len(v_toks) != len(c_toks):
        return []
    out = []
    for a, b in zip(v_toks, c_toks):
        if _norm_token(a) != _norm_token(b):
            out.append((a, b))
    return out


def classify_cell(column: str, value: str, correction: str) -> tuple[Decision, str]:
    """Classify one gold typo cell for v9 reconciliation.

    Returns ``(decision, reason)`` where decision is:
      * ``keep``      -- stays in ``gold_typos`` (in scope under v9).
      * ``reconcile`` -- move ``value`` into ``ambiguous_values`` (proper-noun
                         token char slip; FP for the v9 precision grader).
      * ``review``    -- uncertain (ambiguous column); kept in gold by default
                         but surfaced for human review.
    """
    if not isinstance(value, str) or not isinstance(correction, str):
        return "keep", "non-string cell"
    if value == correction:
        return "keep", "no-op"

    if _is_structural(value, correction):
        return "keep", "structural defect (in scope on names)"

    changed = _changed_tokens(value, correction)
    if not changed:
        return "keep", "no aligned char-level change"

    col = classify_column(column)
    if col == "text":
        return "keep", "ordinary-text column"

    # All changed tokens are recognised descriptors -> common-word fix.
    if all(_norm_token(b) in _COMMON_TOKENS for _, b in changed):
        toks = ", ".join(f"{a}->{b}" for a, b in changed)
        return "keep", f"common-word-token fix ({toks})"

    name_like = ", ".join(f"{a}->{b}" for a, b in changed)
    if col == "entity":
        return "reconcile", f"proper-noun-token char slip in entity column ({name_like})"
    return "review", f"char slip in unknown-type column ({name_like})"
