"""Per-cluster web-grounded verifier.

Runs every gate-eligible cluster through a single Foundry Agents call:

1. Build the user prompt from the suspect cell, sibling values, and
   full row context.
2. Hand it to an :class:`AgentBackend` (production: the Web-Search-
   equipped Foundry agent; tests: :class:`StubAgentBackend`).
3. Parse the agent's JSON reply into a structured :class:`Verification`,
   filtering out citation URLs the runtime did not actually record as
   Web Search sources (anti-hallucination).

The agent owns query formulation and result fetching; we never see raw
snippet text. We DO see the URL citations the agent emitted via
``url_citation`` annotations, plus the Bing search query URLs from the
run steps - both of which feed into the labeling UI.

The output is a list of :class:`Verification` records ready to dump
to JSON. We never overwrite a verification record from a previous
run unless the operator explicitly asks for it (the CLI flag
``--force`` clears the cache).
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import yaml

from .agent_backend import AgentBackend, AgentVerdict, Citation
from .gate import EntityKind, GateConfig, classify_cluster, cluster_shape

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------


@dataclass
class EvidenceCitation:
    url: str
    title: str
    quote: str


@dataclass
class Verification:
    """One verifier output row, persisted as JSON."""

    cluster_id: str
    source: str
    row: Optional[int]
    column: str
    suspect_value: str
    entity_kind: str
    verdict: str = "ambiguous"
    canonical_value: Optional[str] = None
    confidence: float = 0.0
    evidence: List[EvidenceCitation] = field(default_factory=list)
    reasoning: str = ""
    # Bing search query URLs (api host swapped for www.bing.com per
    # display requirements) the agent fired during this run.
    queries: List[str] = field(default_factory=list)
    # Total Web Search tool calls the agent made for this cluster.
    n_results: int = 0
    # Unique citation URLs the agent recorded in url_citation annotations.
    n_unique_urls: int = 0
    error: Optional[str] = None
    parse_error: Optional[str] = None
    # Carried over from the LLM era: how many backend calls + which
    # endpoint produced this verdict. We keep both for backwards-
    # compat with on-disk records and the labeling UI.
    llm_attempts: int = 0
    llm_endpoint: str = ""
    # ---- Multi-row / multi-column extensions (additive) -----------
    # Spatial layout the verifier processed for this cluster.
    # ``pattern`` is one of ``"single_cell"`` | ``"multi_row"`` |
    # ``"multi_column"`` | ``"composite"``. ``rows`` / ``columns_all``
    # are the full row + column lists the cluster spans (``row`` and
    # ``column`` remain the primary anchors for the UI).
    pattern: str = "single_cell"
    rows: List[int] = field(default_factory=list)
    columns_all: List[str] = field(default_factory=list)
    # One projected snapshot per record actually shown to the agent
    # (column -> str value). For single_cell this is the single
    # suspect cell wrapped in a 1-element list; for multi_row it
    # holds one entry per affected row, projected to the cluster's
    # columns when set or to the full row otherwise.
    record_snapshots: List[Dict[str, str]] = field(default_factory=list)
    # Optional structured fix returned by the agent for multi-column
    # patterns: ``{column_name: corrected_value}``. ``canonical_value``
    # remains the human-readable summary; this field is consumed by
    # the labeling UI when applying a multi-column correction.
    canonical_record: Optional[Dict[str, str]] = None
    ts: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class VerifierConfig:
    """Tunables for the verifier loop."""

    # Carried for output-schema compatibility (``Verification.llm_endpoint``);
    # the actual model is configured on the Foundry agent in the portal.
    model: str = "gpt-5.2"
    reasoning_effort: str = "medium"
    reasoning_summary: str = "concise"
    concurrency: int = 4
    # Per-cluster cap on snippets / citations we surface to the LLM.
    # The agent decides on its own how many searches to run; this is
    # only a hard cap on what we copy back into the Verification record.
    max_snippets: int = 6
    # Hard cap on snippet length (chars) before we truncate.
    snippet_chars: int = 600
    # Top-N distinct sibling values to include for in-corpus context.
    sibling_samples: int = 8
    # Path to the YAML prompt file. Resolved against the project root
    # (``<repo>/prompts/verify_entity.yaml``) by default so the
    # verifier doesn't depend on the caller's CWD.
    prompt_path: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[2]
        / "prompts" / "verify_entity.yaml"
    )
    # Gate config (filters which clusters to run).
    gate: GateConfig = field(default_factory=GateConfig)
    # When True, even cached verifications get recomputed (caller
    # clears the on-disk cache; the verifier itself only honours this
    # flag for diagnostics).
    force: bool = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)
_OBJECT_RE = re.compile(r"\{[\s\S]*\}")


def _parse_json_object(text: str) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Extract a JSON object from possibly-fenced agent output."""
    if not text:
        return None, "empty response"
    candidates: List[str] = []
    for m in _FENCE_RE.finditer(text):
        candidates.append(m.group(1))
    candidates.append(text.strip())
    m = _OBJECT_RE.search(text)
    if m:
        candidates.append(m.group(0))
    for c in candidates:
        c = c.strip()
        if not c:
            continue
        try:
            payload = json.loads(c)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload, None
    return None, "failed to parse a JSON object"


def _format_sibling_values(values: Sequence[str]) -> str:
    if not values:
        return "  (none)"
    return "\n".join(f"  - {v}" for v in values)


# Error types whose entire premise is that the suspect is a *small edit*
# of one canonical string (typo / formatting / casing variants). For
# these, a verified ``confirmed_error`` correction MUST be a near-edit of
# the suspect value; if the agent's canonical is a different entity that
# merely shares some words, the web found a look-alike rather than the
# corrected form and the flag is a false positive.
_EDIT_SCALE_ERROR_TYPES = frozenset({
    "typo",
    "casing_inconsistency",
    "pattern_inconsistency",
    "punctuation_inconsistency",
    "number_format_inconsistency",
    "text_inconsistency",
    "inconsistency",
    "spelling",
})

# Error types where a large suspect->canonical rewrite is legitimate (the
# correct value can be wholly unrelated to the suspect), so the distinct-
# entity guard must NOT fire for them.
_EXPANSIVE_ERROR_TYPES = frozenset({
    "semantic_incompatible",
    "numeric_outlier",
})

# Short function words ignored when counting "brand new" content words
# between a suspect and its proposed canonical correction.
_SIMILARITY_STOPWORDS = frozenset({
    "a", "an", "the", "of", "and", "or", "in", "on", "at", "to", "for",
    "by", "with", "is", "are", "as", "from", "into",
})


def _normalize_for_similarity(value: str) -> str:
    """Lower-case + NFKC + collapse whitespace for similarity scoring."""
    import unicodedata

    norm = unicodedata.normalize("NFKC", value or "")
    return " ".join(norm.lower().split())


def _similarity_tokens(value: str) -> List[str]:
    """Split ``value`` into alphanumeric tokens (hyphens are separators)."""
    return [t for t in re.split(r"[^0-9a-z]+", value) if t]


def _looks_like_distinct_entity(suspect: str, canonical: str) -> bool:
    """True when ``canonical`` is too different from ``suspect`` to be a
    typo-scale correction - i.e. the web most likely surfaced a *similar
    but distinct* entity rather than the suspect's corrected spelling.

    A genuine typo / formatting fix preserves almost the whole string and
    changes at most a token or two (e.g. ``"Calavares COE"`` ->
    ``"Calaveras COE"``). A look-alike entity introduces multiple brand-
    new content words (e.g. ``"The Dougla Body: ... Blackness"`` ->
    ``"Dougla Poetics: ... Mixedness in Trinidad"``). We combine a char-
    level similarity ratio with a count of content words in the canonical
    that have no near-match token in the suspect; both must indicate a
    large divergence before we declare a distinct entity, so genuine
    multi-word typo fixes are not downgraded.
    """
    sn = _normalize_for_similarity(suspect)
    cn = _normalize_for_similarity(canonical)
    if not sn or not cn or sn == cn:
        return False

    ratio = SequenceMatcher(None, sn, cn).ratio()

    suspect_tokens = _similarity_tokens(sn)
    new_content_words = 0
    for tok in _similarity_tokens(cn):
        if tok in _SIMILARITY_STOPWORDS or len(tok) <= 2:
            continue
        if any(
            SequenceMatcher(None, tok, s).ratio() >= 0.70
            for s in suspect_tokens
        ):
            continue
        new_content_words += 1

    # Wholesale rewrite (little char overlap) -> distinct entity.
    if ratio < 0.55:
        return True
    # Otherwise require BOTH a sub-threshold char ratio AND >=2 brand-new
    # content words so single-token typo fixes (which keep a high ratio
    # and add at most one "new" token) are never downgraded.
    return ratio < 0.90 and new_content_words >= 2


def _content_tokens_cased(value: str) -> List[str]:
    """Alphanumeric tokens with case PRESERVED (NFKC, no lower-casing).

    The vacuous-correction guard must distinguish a real spelling/casing
    change (``"iphone"`` -> ``"iPhone"``) from a pure reorder/reformat, so
    unlike :func:`_similarity_tokens` it does NOT fold case.
    """
    import unicodedata

    norm = unicodedata.normalize("NFKC", value or "")
    return [t for t in re.split(r"[^0-9A-Za-z]+", norm) if t]


def _is_vacuous_correction(suspect: str, canonical: str) -> bool:
    """True when a ``confirmed_error`` canonical does not actually correct
    any token's spelling - it only reorders, drops, or reformats tokens
    the suspect already contains (or echoes the suspect verbatim).

    For an edit-scale / typo verdict this is self-contradictory: the agent
    says "this value is misspelled" yet its canonical keeps every flagged
    word unchanged. Real example::

        suspect   = "STOCKROOM - TOCKWOTTEN ST 271"
        canonical = "271 TOCKWOTTEN ST"   # drop + reorder; TOCKWOTTEN kept

    The canonical drops "STOCKROOM" and moves "271", but every word it
    keeps ("271", "TOCKWOTTEN", "ST") is byte-identical to the suspect, so
    NO spelling was corrected - the confirmed_error verdict contradicts
    its own canonical.

    Does NOT trigger when a token's spelling actually changed
    ("Calavares" -> "Calaveras"), nor for a casing fix ("iphone" ->
    "iPhone", which differs case-sensitively), nor for an order-preserving
    whitespace/format normalisation (same tokens in the same order).
    """
    import unicodedata

    # Verbatim echo: the canonical is literally the suspect (NFKC only,
    # case + whitespace preserved so a casing / whitespace fix is NOT
    # mistaken for an echo).
    s_keep = unicodedata.normalize("NFKC", suspect or "")
    c_keep = unicodedata.normalize("NFKC", canonical or "")
    if not s_keep or not c_keep:
        return False
    if s_keep == c_keep:
        return True

    s_tokens = _content_tokens_cased(suspect)
    c_tokens = _content_tokens_cased(canonical)
    if not s_tokens or not c_tokens:
        return False
    # Same tokens in the same order => only whitespace/punctuation
    # changed: a legitimate reformat, not a contradiction.
    if c_tokens == s_tokens:
        return False
    # Every canonical token already appears VERBATIM (case-sensitive) in
    # the suspect => the canonical introduced no corrected spelling; it
    # merely reordered / dropped existing tokens.
    s_set = set(s_tokens)
    return all(t in s_set for t in c_tokens)


def _cache_key(cluster_id: str, prompt: str) -> str:
    """Stable per-cluster cache key. Includes a hash of the prompt so
    a prompt-template change invalidates the cache automatically.
    """
    h = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
    return f"{cluster_id}::{h}"


def _format_records_under_review(
    records: Sequence[Mapping[str, str]],
    rows: Sequence[int],
    cluster_columns: Sequence[str],
    *,
    suspect_column: str = "",
    cap_value_chars: int = 120,
    max_columns_per_record: int = 24,
) -> str:
    """Render the ``RECORDS UNDER REVIEW`` block of the prompt.

    Each record is one row's projected snapshot. Columns that are
    part of the cluster (i.e. flagged for review) are marked with
    ``" *"`` so the agent knows which fields the detector
    suspects. For single-cell clusters ``cluster_columns`` is empty
    and ``suspect_column`` carries the one flagged column.
    """
    if not records:
        return "  (no records available)"

    flagged: set[str] = {c for c in cluster_columns if c}
    if suspect_column:
        flagged.add(suspect_column)

    def _fmt_value(val: Any) -> str:
        s = "" if val is None else str(val)
        s = s.replace("\n", " ").replace("\r", " ").strip()
        if len(s) > cap_value_chars:
            s = s[: cap_value_chars].rstrip() + "\u2026"
        return s

    lines: List[str] = []
    for idx, rec in enumerate(records):
        row_idx = rows[idx] if idx < len(rows) else None
        header = (
            f"  Record {idx + 1} (row {row_idx}):"
            if isinstance(row_idx, int)
            else f"  Record {idx + 1}:"
        )
        lines.append(header)
        # Order: flagged columns first (preserving cluster_columns
        # order), then other columns in insertion order. This lets
        # the agent see the columns under review at the top of each
        # record block.
        items = list(rec.items())
        head = [(k, v) for k, v in items if k in flagged]
        # Preserve cluster_columns order for the head section.
        if cluster_columns:
            order = {c: i for i, c in enumerate(cluster_columns)}
            head.sort(key=lambda kv: order.get(kv[0], 1_000_000))
        tail = [(k, v) for k, v in items if k not in flagged]
        ordered = head + tail
        rendered = 0
        for col, val in ordered:
            mark = " *" if col in flagged else ""
            lines.append(f"    - {col}{mark}: {_fmt_value(val)}")
            rendered += 1
            if rendered >= max_columns_per_record:
                remaining = len(ordered) - rendered
                if remaining > 0:
                    lines.append(
                        f"    - ... and {remaining} more columns"
                    )
                break
    return "\n".join(lines)


def _audit_error_type_focus(error_types: Sequence[str]) -> str:
    """Build the column-audit "FOCUS" preamble for one or more
    detector error types.

    The labelling-effort artifact records every column whose
    Good-Turing arm hit ``status="exhausted"``; that arm IS the
    error type we want the agent to look hardest for. We surface a
    short, taxonomy-aware blurb so the agent's queries (and its
    confirmed_error bar) match the defect class the detector
    couldn't fully exhaust.

    Returns an empty string when the cluster carries no error_types
    (legacy callers); otherwise returns a paragraph that ENDS with
    a blank line so the caller can splice it in front of the verdict
    rubric without extra glue. """
    et = [str(t).strip().lower() for t in error_types if str(t).strip()]
    if not et:
        return ""
    # Per-defect blurbs. Keys are the canonical error_type strings
    # written by the detector / per-arm artifact (see data/error_type.csv
    # and the run_good_turing pipeline). Each blurb tells the agent
    # WHAT to look for and WHAT counts as confirmed_error / false_positive
    # for that defect class.
    BLURBS: Dict[str, str] = {
        "typo": (
            "  - typo: a misspelling of a real-world entity name (org, "
            "place, product, person). Confirmed_error requires the web "
            "to surface a single dominant canonical spelling that "
            "differs from the suspect_value by a small edit distance "
            "(swapped / missing / extra letters)."
        ),
        "casing_inconsistency": (
            "  - casing_inconsistency: the value uses a casing pattern "
            "that disagrees with the dominant casing convention shown "
            "in OTHER VALUES IN THIS COLUMN. Confirmed_error requires "
            "BOTH that the dominant convention is clear (e.g. >80% of "
            "siblings agree) AND that the web confirms the canonical "
            "casing for this specific entity (e.g. \"iPhone\", "
            "\"USA\", \"McDonald's\")."
        ),
        "pattern_inconsistency": (
            "  - pattern_inconsistency: the value's structural shape "
            "(separators, digit groups) disagrees with the dominant "
            "pattern in OTHER VALUES IN THIS COLUMN (e.g. "
            "\"800-123-4567\" vs. \"8001234567\"). Confirmed_error "
            "requires the web (or a clear majority of siblings) to "
            "back the canonical pattern. When the value's CONTENT is "
            "correct and only its formatting differs, prefer "
            "false_positive unless the column header strongly implies "
            "a single required format."
        ),
        "punctuation_inconsistency": (
            "  - punctuation_inconsistency: the value uses different "
            "punctuation / whitespace from the dominant convention "
            "(e.g. en-dash vs. hyphen, double space vs. single)."
        ),
        "number_format_inconsistency": (
            "  - number_format_inconsistency: the value mixes numeric "
            "encodings (e.g. \"15%\" alongside \"0.15\", or \"1,000\" "
            "alongside \"1000\"). Use the column header + sibling "
            "values - not the web - to decide which encoding is "
            "canonical for this column."
        ),
        "text_inconsistency": (
            "  - text_inconsistency: the value is a less-common text "
            "variant of the same entity that appears with a more "
            "canonical name elsewhere in the column (e.g. \"USA\" vs. "
            "\"United States\"). Confirmed_error requires the web to "
            "confirm the canonical name for this specific entity."
        ),
        "inconsistency": (
            "  - inconsistency: the value disagrees with the dominant "
            "convention in OTHER VALUES IN THIS COLUMN (umbrella "
            "covering text / casing / pattern / punctuation / number-"
            "format variants). Name the sub-pattern in your reasoning."
        ),
        "semantic_incompatible": (
            "  - semantic_incompatible: the value is out-of-domain for "
            "this column (e.g. a phone number in a NAME column, a "
            "date in a SALARY column). Confirmed_error requires the "
            "web (or the column header semantics) to make the "
            "incompatibility unambiguous."
        ),
        "numeric_outlier": (
            "  - numeric_outlier: the numeric value deviates strongly "
            "from the bulk distribution. Confirmed_error requires "
            "external evidence that the value is wrong (e.g. the web "
            "shows a salary band, a price range, a known cap). Do "
            "NOT flag a value just because it's rare - only flag it "
            "when authoritative evidence contradicts it."
        ),
    }
    DEFAULT_BLURB = (
        "  - {et}: focus your check on this defect class as defined "
        "in the project's error-type taxonomy."
    )
    blurbs = []
    seen: set = set()
    for t in et:
        if t in seen:
            continue
        seen.add(t)
        blurbs.append(BLURBS.get(t, DEFAULT_BLURB.format(et=t)))
    label = (
        "FOCUS - this column was budget-capped on the following "
        "detector error type"
        + ("s" if len(blurbs) > 1 else "")
        + ", so its run-budget was exhausted while still finding new "
        "defects of this kind. Search and reason WITH THIS DEFECT "
        "CLASS IN MIND - prefer queries and evidence that adjudicate "
        "it, and apply the verdict criteria below in light of it:"
    )
    return label + "\n" + "\n".join(blurbs) + "\n\n"


def _format_task_block(
    pattern: str,
    columns: Sequence[str],
    n_rows: int,
    *,
    entity_kind: str = "other",
    error_types: Sequence[str] = (),
) -> str:
    """Pattern-specific TASK instructions slotted into the prompt.

    The verifier composes this in code (rather than in the YAML)
    because each pattern asks the agent a fundamentally different
    question about the records under review. ``entity_kind`` is
    used to specialise the single-cell block - notably, person-name
    typos use a flipped-default policy (any web hit on the suspect
    spelling is strong evidence of a false positive).

    ``error_types`` is the cluster's detector error_type list. It
    currently only specialises the ``column_audit`` block (so the
    audit prompt focuses on the SAME defects the column was budget-
    capped on), but it is plumbed in from the call site so other
    pattern blocks can adopt the same convention if needed.
    """
    cols = [c for c in columns if c]
    cols_str = ", ".join(cols) if cols else ""
    et_list = [str(t).strip() for t in error_types if str(t).strip()]

    if pattern == "single_cell" and entity_kind == "person":
        # Person-name typos are systematically over-flagged: a name
        # that *looks* misspelled (e.g. "Murry" vs "Murray", "Catherin"
        # vs "Catherine") is often a real, less-common spelling that
        # SOMEONE - somewhere in the world - actually uses. We do NOT
        # need to confirm the row's specific person; we only need to
        # confirm the spelling is in real-world use as a person name.
        # The web is the authoritative arbiter: any plausible match
        # for the EXACT suspect spelling on a person profile,
        # directory, news article, publication, or social-media bio
        # is enough to mark this a false_positive.
        return (
            "Person-name typo verification.\n"
            "\n"
            "The detector flagged the suspect_value as a possible "
            "misspelling of a more common name (suggested_fix). "
            "Names are unusual: rare and unconventional spellings "
            "exist for almost every name. We therefore use a LENIENT "
            "criterion - the suspect spelling is valid as long as "
            "AT LEAST ONE real human, ANYWHERE in the world, uses "
            "the EXACT same spelling. We do NOT need to verify the "
            "specific person referenced by the row.\n"
            "\n"
            "Use the Web Search tool. Run AT LEAST one query for "
            "the suspect_value as written (in quotes). You may add "
            "context terms like ``name``, ``person``, or ``profile`` "
            "if the bare query is too noisy, but DO NOT narrow with "
            "row-specific fields (employer, address, etc.). The "
            "question is \"does this spelling exist as a real human "
            "name?\" - not \"is this the right person for this row?\"\n"
            "\n"
            "Decide:\n"
            "  * \"false_positive\" - your search returned ONE OR "
            "MORE plausible matches for the EXACT suspect spelling "
            "as a real person's name (any human, anywhere). Cite "
            "the URL(s) used as evidence in the ``evidence`` array "
            "(person profile, directory, news article, publication, "
            "social-media bio, obituary, faculty page, etc.). Set "
            "``canonical_value`` to null. The bar is LOW: a single "
            "credible match is enough. Genealogy / surname-history "
            "pages and name-meaning sites also count if they show "
            "the spelling is in actual use.\n"
            "  * \"confirmed_error\" - the EXACT suspect spelling "
            "has NO web presence as a real person's name. This bar "
            "is HIGH: you must have run at least one quoted search "
            "and seen zero plausible person-name results. Cite the "
            "supporting URL for the suggested_fix as the canonical "
            "alternative.\n"
            "  * \"ambiguous\" - the search returned no useful "
            "evidence either way (the agent's tool returned 0 "
            "results, or only off-topic hits). Set "
            "``canonical_value`` to null and explain in "
            "``reasoning``.\n"
            "\n"
            "When in doubt between false_positive and confirmed_error, "
            "prefer false_positive. A name that produces ANY "
            "plausible web hit is more likely a real rare-spelling "
            "name than a true typo. Do NOT return confirmed_error "
            "unless the suspect spelling truly has no web presence."
        )

    if pattern == "column_audit":
        # Column-wide audit. The detector either never flagged this
        # cell or flagged it without enough budget to fully explore
        # the column - we are auditing the WHOLE column for missed
        # errors, so the agent has NO prior suspicion to anchor on.
        # The bar for confirmed_error is therefore HIGHER than the
        # standard single_cell case: we do not want to manufacture
        # false positives just because the operator asked us to scan.
        #
        # ``error_types`` (when present) tells the agent which defect
        # categories the column was budget-capped on. We surface them
        # explicitly so the agent's queries + verdict criteria match
        # the run-budget-capped detector. When no error types are
        # supplied (legacy callers) the audit reverts to a generic
        # "is the value correct?" framing.
        et_focus = _audit_error_type_focus(et_list)
        return (
            "Column-wide audit. The detector flagged this column "
            "as under-explored (its run-budget cap was hit while it "
            "was still finding new defects), so we are auditing every "
            "distinct value in the column for missed errors. The "
            "suspect_value below was NOT necessarily flagged by the "
            "detector - your job is to decide whether the value is "
            "correct as written, using the web as the authoritative "
            "arbiter.\n"
            "\n"
            f"{et_focus}"
            "Use the Web Search tool. Run AT LEAST one query for the "
            "suspect_value (in quotes), optionally combined with the "
            "column header and any high-signal sibling columns from "
            "the RECORDS UNDER REVIEW block. Then:\n"
            "  * \"confirmed_error\" - the web returned authoritative "
            "evidence (official directory, government registry, "
            "Wikipedia, established news source, vendor's own site) "
            "that the suspect spelling / value is WRONG, AND a single "
            "canonical alternative is supported by at least one "
            "quoted snippet. Provide it as ``canonical_value``. Be "
            "conservative: when the web is silent or returns only "
            "weak / off-topic hits, do NOT flag the cell - prefer "
            "false_positive or ambiguous.\n"
            "  * \"false_positive\" - the value is consistent with "
            "the web (i.e. the search returned at least one credible "
            "match for the EXACT suspect spelling). Set "
            "``canonical_value`` to null. The bar is LOW: a single "
            "credible match is enough.\n"
            "  * \"ambiguous\" - the web does not adjudicate the "
            "value either way (zero relevant hits, or only off-topic "
            "results). Set ``canonical_value`` to null and explain "
            "in ``reasoning``.\n"
            "\n"
            "When in doubt between confirmed_error and any other "
            "verdict, pick the other verdict. This is an audit, not "
            "a screen - we surface confirmed_error only when the "
            "web evidence is unambiguous."
        )

    if pattern == "single_cell":
        return (
            "Use the Web Search tool to find authoritative evidence about "
            "the suspect cell value (and, if useful, the suggested fix). "
            "Decide which verdict best fits using ONLY the evidence the "
            "tool surfaced.\n"
            "  * \"confirmed_error\" - the suspect value is wrong and a "
            "single canonical alternative is supported by at least one "
            "quoted snippet. Provide it as ``canonical_value``.\n"
            "  * \"false_positive\" - the suspect value is in fact "
            "correct. Set ``canonical_value`` to null.\n"
            "  * \"ambiguous\" - evidence is missing or contradictory. "
            "Set ``canonical_value`` to null.\n"
            "Notes for person-name columns: the suspect_value may be a "
            "private individual rather than a public figure. If the "
            "Web Search tool surfaces no authoritative public record "
            "for the person, return verdict=\"ambiguous\" with a brief "
            "``reasoning`` that says so. Use the OTHER row fields (e.g. "
            "company, address, role) shown in RECORDS UNDER REVIEW to "
            "scope the search before falling back to ambiguous."
        )
    if pattern == "multi_row":
        return (
            f"Use the Web Search tool to determine whether the {n_rows} "
            "records under review describe the SAME real-world entity. "
            "The detector suspects they may be duplicate records of one "
            "entity that has been written down in different ways.\n"
            "  * \"confirmed_error\" - the records ARE the same entity "
            "and at least one record has a wrong / non-canonical "
            "representation. Set ``canonical_value`` to the canonical "
            "name (or a short description like \"merge as <Name>\"). "
            "Cite at least one URL whose page text contains the "
            "canonical name verbatim.\n"
            "  * \"false_positive\" - the records refer to DISTINCT "
            "real-world entities (i.e. they are NOT duplicates). Set "
            "``canonical_value`` to null and explain in ``reasoning`` "
            "which differentiating fact the web confirmed.\n"
            "  * \"ambiguous\" - the web does not have enough evidence "
            "to decide."
        )
    if pattern == "multi_column":
        return (
            "Use the Web Search tool to verify whether the columns "
            f"under review ({cols_str}) hold values that are mutually "
            "consistent for THIS record - i.e. they all describe the "
            "same real-world entity according to authoritative sources.\n"
            "  * \"confirmed_error\" - at least one column's value is "
            "inconsistent with the others according to the web. Set "
            "``canonical_value`` to a short human-readable summary of "
            "the corrected record AND populate ``canonical_record`` "
            "with ``{col: corrected_value}`` for every column you are "
            "correcting (omit columns whose value is already correct).\n"
            "  * \"false_positive\" - the columns are mutually "
            "consistent. Set ``canonical_value`` and ``canonical_record`` "
            "to null.\n"
            "  * \"ambiguous\" - the web does not adjudicate the "
            "relationship clearly."
        )
    # composite (multi-row + multi-column)
    return (
        f"Use the Web Search tool to verify whether the {n_rows} "
        f"records share consistent values across the columns "
        f"({cols_str}). The detector suspects either a functional-"
        "dependency violation (different rows disagree on what the "
        "value of a column should be for a shared key) or a duplicate "
        "group with inconsistent attributes.\n"
        "  * \"confirmed_error\" - the records refer to the same "
        "entity / key but disagree on at least one column's value, "
        "and the web supports a single canonical value. Use "
        "``canonical_record`` to map ``{col: corrected_value}``.\n"
        "  * \"false_positive\" - the records are mutually consistent "
        "or refer to distinct entities. Set ``canonical_value`` to null.\n"
        "  * \"ambiguous\" - insufficient web evidence."
    )


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------


@dataclass
class Verifier:
    """Run :class:`Verification` for each gate-eligible cluster.

    The verifier is intentionally narrow: it never opens DataFrames
    itself - the caller passes a ``cell_lookup`` callback that
    resolves ``(source, row, column) -> str`` and a
    ``column_samples`` callback that returns up to N distinct sibling
    values for a column. ``row_lookup`` is optional and provides the
    full row dictionary for cross-column context.
    """

    agent: AgentBackend
    cell_lookup: Callable[[str, int, str], Optional[str]]
    column_samples: Callable[[str, str, int], List[str]]
    row_lookup: Optional[Callable[[str, int], Mapping[str, Any]]] = None
    cfg: VerifierConfig = field(default_factory=VerifierConfig)
    _prompt: Mapping[str, str] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        self._prompt = self._load_prompt(self.cfg.prompt_path)
        # Plumb the system prompt onto the backend so it doesn't have
        # to be re-passed per call. The backend uses it as the run's
        # ``instructions`` override.
        if hasattr(self.agent, "system_prompt"):
            try:
                setattr(self.agent, "system_prompt", self._prompt["system"])
            except Exception:  # pragma: no cover (immutable backend)
                pass

    # ---- prompt loading ------------------------------------------

    @staticmethod
    def _load_prompt(path: Path) -> Mapping[str, str]:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"verify prompt not found: {path}")
        with path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        if not isinstance(raw, Mapping):
            raise ValueError(f"{path}: top level must be a mapping")
        system = str(raw.get("system", "")).strip()
        template = str(raw.get("user_template", ""))
        if not system or not template:
            raise ValueError(f"{path}: 'system' and 'user_template' required")
        return {"system": system, "user_template": template}

    # ---- single-cluster pipeline ---------------------------------

    def _verify_one(self, cluster: Mapping[str, Any]) -> Verification:
        cid = str(cluster.get("cluster_id") or "")
        source = str(cluster.get("source") or "")

        shape = cluster_shape(cluster)
        shape_pattern = shape["pattern"]
        all_rows: List[int] = list(shape["rows"])
        all_cols: List[str] = list(shape["columns"])

        # Pattern-routing intent. ``cluster_shape`` is purely
        # structural: any cluster whose ``detections[*].affected_rows``
        # list >=2 entries gets shaped ``multi_row``, even when the
        # error is a per-cell typo / pattern_inconsistency and the
        # extra rows are sibling evidence rather than candidate
        # duplicate records. Routing the prompt purely on shape would
        # ask the agent the wrong question ("are these rows the same
        # real-world entity?") for typo / inconsistency clusters. The
        # multi-row / composite prompts are only appropriate when the
        # detector itself claims the rows are duplicates - i.e.
        # ``row_duplication`` appears in the cluster's error_types.
        et_list = [str(t) for t in (cluster.get("error_types") or []) if t]
        has_duplication = "row_duplication" in et_list
        pattern = shape_pattern
        if pattern == "multi_row" and not has_duplication:
            pattern = "single_cell"
        elif pattern == "composite" and not has_duplication:
            pattern = "multi_column" if len(all_cols) >= 2 else "single_cell"

        # Apply caps from the gate config so we never blast a
        # 200-row dup group into one prompt.
        capped_rows = all_rows[: max(1, self.cfg.gate.multi_row_cap)] if all_rows else []
        capped_cols = all_cols[: max(1, self.cfg.gate.multi_col_cap)] if all_cols else []

        # Anchor on ``cluster.row`` (the LLM's primary suspect) when
        # available - ``cluster_shape`` sorts rows, so
        # ``capped_rows[0]`` is the smallest index, not necessarily
        # the row the detector flagged. This matters for multi_row
        # clusters demoted to single_cell above (typo with sibling
        # rows): the primary cell to verify is the one the detector
        # pointed at.
        explicit_row = cluster.get("row")
        if not isinstance(explicit_row, int):
            explicit_row = None
        primary_row = explicit_row if explicit_row is not None else (
            capped_rows[0] if capped_rows else None
        )
        primary_col = capped_cols[0] if capped_cols else ""

        # For demoted single_cell clusters the prompt should only show
        # the primary row (the suspect cell), not the full sibling
        # list - we already decided this is a per-cell defect and
        # showing 10 sibling rows would re-introduce the duplication
        # framing we just demoted away from.
        prompt_rows: List[int]
        if pattern == "single_cell":
            prompt_rows = [primary_row] if isinstance(primary_row, int) else []
        elif pattern == "multi_column":
            # multi_column verifies one record's column relationships,
            # not multiple rows. Even when ``shape_pattern`` was
            # ``composite`` and we demoted (no ``row_duplication``
            # error_type), we anchor on the LLM's primary row.
            prompt_rows = (
                [primary_row] if isinstance(primary_row, int)
                else capped_rows[:1]
            )
        else:  # multi_row / true composite (row_duplication intent)
            prompt_rows = list(capped_rows)

        kind = classify_cluster(cluster, cfg=self.cfg.gate)

        # ---- build per-record snapshots --------------------------
        records: List[Dict[str, str]] = []
        if pattern == "single_cell":
            # Existing behavior: a single cell, with the full row
            # populating the snapshot for cross-column hints. The
            # agent sees the row marked with ``*`` on the suspect
            # column.
            cell_val = ""
            try:
                if isinstance(primary_row, int) and primary_col:
                    cell_val = self.cell_lookup(source, primary_row, primary_col) or ""
            except Exception as exc:  # noqa: BLE001
                logger.warning("cell_lookup failed for %s: %s", cid, exc)
            row_dict: Mapping[str, Any] = {}
            if (
                self.row_lookup is not None
                and isinstance(primary_row, int)
            ):
                try:
                    row_dict = self.row_lookup(source, primary_row) or {}
                except Exception as exc:  # noqa: BLE001
                    logger.warning("row_lookup failed for %s: %s", cid, exc)
                    row_dict = {}
            if row_dict:
                snap = {str(k): ("" if v is None else str(v)) for k, v in row_dict.items()}
                # If primary_col is missing from the row, splice in
                # the explicitly-fetched cell so the prompt is
                # never silently empty.
                if primary_col and primary_col not in snap:
                    snap[primary_col] = cell_val
            else:
                snap = {primary_col: cell_val} if primary_col else {}
            records.append(snap)
            suspect_value_for_record = cell_val
        else:
            # Multi-row, multi-column, or composite: one snapshot
            # per affected row. ``prompt_rows`` already encodes the
            # routing decision - for ``multi_column`` we iterate
            # exactly one row (the LLM's anchor) instead of every
            # sibling row carried by the structural shape.
            for r in prompt_rows:
                row_dict = {}
                if self.row_lookup is not None:
                    try:
                        row_dict = self.row_lookup(source, r) or {}
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "row_lookup failed for %s row=%s: %s", cid, r, exc,
                        )
                        row_dict = {}
                if capped_cols:
                    # Project to the cluster's columns first; for
                    # multi-row clusters with no cluster.columns the
                    # projection is the full row.
                    snap = {
                        c: ("" if row_dict.get(c) is None else str(row_dict.get(c, "")))
                        for c in capped_cols
                    }
                    # Also keep a few high-signal identifier columns
                    # (e.g. license #, ID, name) so the agent can
                    # ground its searches.
                    for c, v in row_dict.items():
                        if c in snap:
                            continue
                        snap[str(c)] = "" if v is None else str(v)
                else:
                    snap = {
                        str(c): ("" if v is None else str(v))
                        for c, v in row_dict.items()
                    }
                records.append(snap)
            suspect_value_for_record = ""

        # ---- build the Verification skeleton ---------------------
        # ``suspect_value`` is the textual value of the primary
        # suspect cell for single_cell, or a short JSON summary for
        # multi-record patterns (kept short so it fits cleanly in
        # the labeling UI).
        if pattern == "single_cell":
            display_suspect = suspect_value_for_record
        else:
            try:
                display_suspect = json.dumps(records, ensure_ascii=False)[:1000]
            except (TypeError, ValueError):
                display_suspect = ""

        ver = Verification(
            cluster_id=cid,
            source=source,
            row=primary_row if isinstance(primary_row, int) else None,
            column=primary_col,
            suspect_value=str(display_suspect),
            entity_kind=kind.value,
            pattern=pattern,
            rows=list(all_rows),
            columns_all=list(all_cols),
            record_snapshots=[dict(r) for r in records],
        )

        # ---- early-exit: missing data ----------------------------
        if pattern == "single_cell":
            if not suspect_value_for_record:
                ver.error = "suspect cell value unavailable"
                return ver
        else:
            if not records or not any(records):
                ver.error = "row context unavailable for multi-record cluster"
                return ver

        # ---- suggested fix + sibling values ----------------------
        syn = cluster.get("synthesis") or {}
        if not isinstance(syn, Mapping):
            syn = {}
        suggested_fix = (
            (syn.get("canonical_fix") or "")
            or (cluster.get("suggested_fix") or "")
        )

        siblings: List[str] = []
        if pattern == "single_cell" and primary_col:
            try:
                raw_sib = self.column_samples(
                    source, primary_col, self.cfg.sibling_samples,
                ) or []
            except Exception as exc:  # noqa: BLE001
                logger.warning("column_samples failed for %s: %s", cid, exc)
                raw_sib = []
            cleaned = [
                s for s in (str(s).strip() for s in raw_sib)
                if s and s != suspect_value_for_record.strip()
            ]
            seen_sib: set[str] = set()
            siblings = [
                s for s in cleaned
                if not (s in seen_sib or seen_sib.add(s))  # type: ignore[func-returns-value]
            ][: self.cfg.sibling_samples]

        # ---- prompt construction ---------------------------------
        records_block = _format_records_under_review(
            records=records,
            rows=prompt_rows,
            cluster_columns=capped_cols,
            suspect_column=primary_col if pattern == "single_cell" else "",
        )
        # ``verify_mode`` is an opt-in cluster marker that overrides
        # the task-block framing without touching the rest of the
        # prompt. The column-wide audit script (run_web_verify_column)
        # sets ``verify_mode="column_audit"`` on synthetic single-
        # cell clusters so the agent gets neutral "is this value
        # correct?" framing instead of the "the detector flagged
        # this cell - confirm or refute" framing.
        verify_mode = str(cluster.get("verify_mode") or "").strip()
        task_pattern = (
            "column_audit"
            if verify_mode == "column_audit" and pattern == "single_cell"
            else pattern
        )
        task_block = _format_task_block(
            pattern=task_pattern,
            columns=capped_cols,
            n_rows=max(len(prompt_rows), 1),
            entity_kind=kind.value,
            error_types=[
                str(t) for t in (cluster.get("error_types") or []) if t
            ],
        )

        prompt = self._prompt["user_template"].format(
            source=source,
            pattern=pattern,
            column=primary_col or "(unknown)",
            columns=", ".join(capped_cols) if capped_cols else "(none)",
            row=primary_row if isinstance(primary_row, int) else "n/a",
            rows=", ".join(str(r) for r in prompt_rows) if prompt_rows else "n/a",
            n_records=len(records),
            entity_kind=kind.value,
            error_types=", ".join(
                str(t) for t in (cluster.get("error_types") or [])
            ) or "(none)",
            suspect_value=(
                suspect_value_for_record
                if pattern == "single_cell"
                else "(see RECORDS UNDER REVIEW)"
            ),
            suggested_fix=suggested_fix or "(none)",
            sibling_values=_format_sibling_values(siblings),
            records_block=records_block,
            task_block=task_block,
        )

        verdict: AgentVerdict = self.agent.verify(
            prompt, cache_key=_cache_key(cid, prompt),
        )
        ver.llm_attempts = 1
        ver.llm_endpoint = getattr(self.agent, "endpoint", "") or self.agent.name
        ver.queries = list(verdict.bing_query_urls)
        ver.n_results = int(verdict.n_search_calls)
        ver.n_unique_urls = len(verdict.citations)

        if verdict.error:
            ver.error = verdict.error
            return ver
        if not verdict.text:
            ver.error = "agent returned empty response"
            return ver

        payload, parse_err = _parse_json_object(verdict.text)
        ver.parse_error = parse_err
        if not payload:
            return ver

        verdict_str = str(payload.get("verdict") or "").strip().lower()
        if verdict_str not in ("confirmed_error", "false_positive", "ambiguous"):
            ver.parse_error = f"unknown verdict: {verdict_str!r}"
            return ver
        ver.verdict = verdict_str

        canon = payload.get("canonical_value")
        if canon in (None, "", "null", "None"):
            ver.canonical_value = None
        else:
            ver.canonical_value = str(canon).strip() or None

        # Optional structured per-column fix for multi-column /
        # composite patterns. We accept any mapping the agent emits
        # and filter to columns we asked about (ignoring keys for
        # columns the agent invented).
        canon_record_raw = payload.get("canonical_record")
        if isinstance(canon_record_raw, Mapping) and capped_cols:
            allowed_cols = {c for c in capped_cols if c}
            cleaned_record: Dict[str, str] = {}
            for k, v in canon_record_raw.items():
                key = str(k).strip()
                if not key or key not in allowed_cols:
                    continue
                if v in (None, "", "null", "None"):
                    continue
                cleaned_record[key] = str(v).strip()
            ver.canonical_record = cleaned_record or None

        try:
            ver.confidence = float(payload.get("confidence") or 0.0)
        except (TypeError, ValueError):
            ver.confidence = 0.0
        ver.confidence = max(0.0, min(1.0, ver.confidence))
        ver.reasoning = str(payload.get("reasoning") or "").strip()

        # Anti-hallucination filter: only keep cites whose URL the
        # agent runtime actually recorded as a Web Search source
        # (i.e. it appeared as a ``url_citation`` annotation).
        allowed_urls = {c.url for c in verdict.citations if c.url}
        url_to_title = {c.url: c.title for c in verdict.citations}
        for ev in payload.get("evidence") or []:
            if not isinstance(ev, Mapping):
                continue
            url = str(ev.get("url") or "").strip()
            if not url or url not in allowed_urls:
                continue
            ver.evidence.append(EvidenceCitation(
                url=url,
                title=str(ev.get("title") or url_to_title.get(url, "")).strip(),
                quote=str(ev.get("quote") or "").strip()[: self.cfg.snippet_chars],
            ))
            if len(ver.evidence) >= self.cfg.max_snippets:
                break

        # Fallback: when the agent omitted ``payload.evidence`` (or
        # every URL it cited was filtered as non-grounded) but the
        # runtime DID record ``url_citation`` annotations, surface
        # those SDK-attributed citations as evidence so we don't
        # silently drop trustworthy URLs. This is common with the
        # Bing Grounding tool: the agent emits a JSON-only response
        # and forgets to copy the URLs from its own annotations
        # back into the ``evidence`` array. The annotations are by
        # construction grounded (the runtime wrote them, not the
        # model), so this preserves the anti-hallucination guarantee.
        # The agent's ``reasoning`` is reused as the quote since we
        # never see the underlying snippet text from Bing.
        if not ver.evidence and verdict.citations:
            quote_text = (ver.reasoning or "").strip()[: self.cfg.snippet_chars]
            seen: set[str] = set()
            for c in verdict.citations:
                url = (c.url or "").strip()
                if not url or url in seen:
                    continue
                seen.add(url)
                ver.evidence.append(EvidenceCitation(
                    url=url,
                    title=(c.title or "").strip(),
                    quote=quote_text,
                ))
                if len(ver.evidence) >= self.cfg.max_snippets:
                    break

        # Tighten the verdict if it claims confirmed_error without
        # surviving evidence: downgrade to ambiguous.
        if ver.verdict == "confirmed_error" and not ver.evidence:
            ver.verdict = "ambiguous"
            ver.canonical_value = None
            ver.canonical_record = None
            ver.parse_error = (ver.parse_error or "") + (
                "; downgraded to ambiguous: no evidence URLs survived"
            ).lstrip("; ")

        # Distinct-entity guard for edit-scale defects (typo / formatting
        # / casing). When the agent "confirms" an error but its canonical
        # value is a different entity that merely resembles the suspect
        # (e.g. suspect "The Dougla Body: ... Blackness" -> canonical
        # "Dougla Poetics: ... Mixedness in Trinidad"), the web surfaced a
        # look-alike rather than the corrected spelling. A typo correction
        # must be a near-edit of the suspect; a distinct entity means the
        # original value is most likely a legitimate, different value, so
        # the detector flag is a false positive. We only apply this to
        # single-cell edit-scale error types and never to defect classes
        # where a large rewrite is expected (semantic_incompatible /
        # numeric_outlier).
        if (
            pattern == "single_cell"
            and ver.verdict == "confirmed_error"
            and ver.canonical_value
            and suspect_value_for_record
        ):
            et_lower = {str(t).strip().lower() for t in et_list}
            scoped = bool(et_lower & _EDIT_SCALE_ERROR_TYPES) and not (
                et_lower & _EXPANSIVE_ERROR_TYPES
            )
            if scoped and _looks_like_distinct_entity(
                suspect_value_for_record, ver.canonical_value
            ):
                rejected_canonical = ver.canonical_value
                ver.verdict = "false_positive"
                ver.canonical_value = None
                ver.canonical_record = None
                # FP confidence = certainty the suspect is correct. This is
                # a heuristic override, not positive web confirmation, so
                # cap it at a moderate level rather than carrying over the
                # agent's (now-inapplicable) confirmed_error confidence.
                ver.confidence = min(ver.confidence, 0.5)
                ver.reasoning = (
                    (ver.reasoning or "").strip()
                    + (
                        " [Automated review: the proposed correction "
                        f"{rejected_canonical!r} is a different entity than "
                        "the flagged value (not a small spelling/format "
                        "edit), so the value is treated as correct as "
                        "written rather than a typo.]"
                    )
                ).strip()
                ver.parse_error = (ver.parse_error or "") + (
                    "; downgraded to false_positive: canonical "
                    f"{rejected_canonical!r} is a distinct entity from "
                    f"suspect {suspect_value_for_record!r}, not a "
                    "typo-scale correction"
                ).lstrip("; ")
            elif scoped and _is_vacuous_correction(
                suspect_value_for_record, ver.canonical_value
            ):
                # Self-contradictory verdict: confirmed_error, but the
                # canonical does not change any token's spelling - it only
                # reorders / drops / reformats words the suspect already
                # contains (e.g. suspect "STOCKROOM - TOCKWOTTEN ST 271"
                # -> canonical "271 TOCKWOTTEN ST", which keeps the flagged
                # "TOCKWOTTEN" verbatim). The agent confirmed an error its
                # own canonical fails to correct, so we cannot trust either
                # direction -> ambiguous (needs a human), canonical
                # cleared. We do NOT assert false_positive here: the
                # flagged spelling might still be a real typo (the web
                # evidence may even support it), the agent just produced an
                # inconsistent canonical.
                rejected_canonical = ver.canonical_value
                ver.verdict = "ambiguous"
                ver.canonical_value = None
                ver.canonical_record = None
                # Ambiguous confidence must stay in [0, 0.49] per the
                # prompt's calibration contract.
                ver.confidence = min(ver.confidence, 0.49)
                ver.reasoning = (
                    (ver.reasoning or "").strip()
                    + (
                        " [Automated review: the proposed correction "
                        f"{rejected_canonical!r} does not change the "
                        "spelling of the flagged value (it only reorders / "
                        "drops / reformats words it already contains), so "
                        "the confirmed_error verdict is inconclusive.]"
                    )
                ).strip()
                ver.parse_error = (ver.parse_error or "") + (
                    "; downgraded to ambiguous: canonical "
                    f"{rejected_canonical!r} does not correct any token of "
                    f"suspect {suspect_value_for_record!r} (vacuous "
                    "reorder/reformat, not a spelling fix)"
                ).lstrip("; ")

        return ver

    # ---- batch -----------------------------------------------------

    def run(
        self,
        clusters: Sequence[Mapping[str, Any]],
        *,
        progress: Optional[Callable[[int, int], None]] = None,
    ) -> List[Verification]:
        """Run the verifier over ``clusters``. Eligibility filtering
        is the caller's job (use :func:`src.verifier.gate.filter_eligible`).
        """
        clusters = list(clusters)
        total = len(clusters)
        out: List[Verification] = []
        if total == 0:
            return out

        lock = threading.Lock()
        done = {"n": 0}

        def _job(c: Mapping[str, Any]) -> Verification:
            v = self._verify_one(c)
            with lock:
                done["n"] += 1
                if progress is not None:
                    progress(done["n"], total)
            return v

        if self.cfg.concurrency <= 1:
            for c in clusters:
                out.append(_job(c))
        else:
            with ThreadPoolExecutor(max_workers=self.cfg.concurrency) as pool:
                futures = [pool.submit(_job, c) for c in clusters]
                for fut in as_completed(futures):
                    out.append(fut.result())

        out.sort(key=lambda v: v.cluster_id)
        return out


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def write_verifications(path: Path, records: Sequence[Verification]) -> None:
    """Write ``records`` as a single JSON array under ``path``.

    Uses an atomic ``os.replace`` so partial files never appear if the
    process is killed mid-write.
    """
    import os
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [v.to_dict() for v in records]
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def load_verifications(path: Path) -> List[Verification]:
    """Read a previous run's verifications, or ``[]`` if missing."""
    path = Path(path)
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(raw, list):
        return []
    out: List[Verification] = []
    for r in raw:
        if not isinstance(r, dict):
            continue
        evidence = []
        for ev in r.get("evidence") or []:
            if isinstance(ev, dict):
                evidence.append(EvidenceCitation(
                    url=str(ev.get("url") or ""),
                    title=str(ev.get("title") or ""),
                    quote=str(ev.get("quote") or ""),
                ))
        # Multi-row / multi-column extension fields. Older records
        # written before the schema bump simply default these to
        # empty / None, which preserves the single_cell semantics.
        rows_raw = r.get("rows") or []
        rows: List[int] = []
        if isinstance(rows_raw, list):
            for x in rows_raw:
                try:
                    rows.append(int(x))
                except (TypeError, ValueError):
                    continue
        columns_all_raw = r.get("columns_all") or []
        columns_all = [str(c) for c in columns_all_raw] if isinstance(columns_all_raw, list) else []
        snapshots_raw = r.get("record_snapshots") or []
        record_snapshots: List[Dict[str, str]] = []
        if isinstance(snapshots_raw, list):
            for s in snapshots_raw:
                if isinstance(s, dict):
                    record_snapshots.append({str(k): "" if v is None else str(v) for k, v in s.items()})
        canon_rec_raw = r.get("canonical_record")
        canon_rec: Optional[Dict[str, str]] = None
        if isinstance(canon_rec_raw, dict) and canon_rec_raw:
            canon_rec = {str(k): "" if v is None else str(v) for k, v in canon_rec_raw.items()}

        out.append(Verification(
            cluster_id=str(r.get("cluster_id") or ""),
            source=str(r.get("source") or ""),
            row=r.get("row") if isinstance(r.get("row"), int) else None,
            column=str(r.get("column") or ""),
            suspect_value=str(r.get("suspect_value") or ""),
            entity_kind=str(r.get("entity_kind") or "other"),
            verdict=str(r.get("verdict") or "ambiguous"),
            canonical_value=(
                str(r["canonical_value"])
                if r.get("canonical_value") not in (None, "", "null") else None
            ),
            confidence=float(r.get("confidence") or 0.0),
            evidence=evidence,
            reasoning=str(r.get("reasoning") or ""),
            queries=[str(q) for q in (r.get("queries") or [])],
            n_results=int(r.get("n_results") or 0),
            n_unique_urls=int(r.get("n_unique_urls") or 0),
            error=r.get("error") if r.get("error") else None,
            parse_error=r.get("parse_error") if r.get("parse_error") else None,
            llm_attempts=int(r.get("llm_attempts") or 0),
            llm_endpoint=str(r.get("llm_endpoint") or ""),
            pattern=str(r.get("pattern") or "single_cell"),
            rows=rows,
            columns_all=columns_all,
            record_snapshots=record_snapshots,
            canonical_record=canon_rec,
            ts=str(r.get("ts") or ""),
        ))
    return out
