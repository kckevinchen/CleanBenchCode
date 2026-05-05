"""Relation-discovery: enumerate multi-column relations per error type.

This is "Phase D", a pre-pass that runs ONCE per (source, error_type)
before per-row detection. The LLM sees only the compact whole-table
digest plus a small stratified row sample, and is asked to enumerate
every relation / dependency / constraint that should hold over the
table for a single error type. The resulting relations are persisted
and later fed into per-type detection prompts as a closed
"RELATIONS TO CHECK" list, so detection no longer has to re-discover
the rules from scratch on every row chunk.

Cost shape: one LLM call per (source, error_type) per pass. A small
constant — the multi-column / multi-row error types are a handful of
keys, not the full taxonomy — so the discovery cost is independent
of table height.

Module structure note
---------------------
``SchemaDiscoverer._discover_single`` and ``_discover_table`` import
``multi_row_priors`` / ``multi_table_priors`` / ``table_digest``
**inside** the function body (``# noqa: PLC0415``) on purpose:

* ``multi_row_priors`` and ``multi_table_priors`` import :class:`Relation`
  and helpers from this module, creating a bidirectional dependency.
  Top-level imports would cause an ``ImportError`` at module load.
* ``table_digest._to_float`` is only needed inside the numeric-outlier
  helper; deferring keeps the import graph cleaner for callers that
  never trigger that branch.

Any future refactor that breaks the cycle (e.g. extracting a small
``relations_types`` module exposing :class:`Relation` plus the
``species_key`` helper) can promote these to top-level imports.

Canonicalization (see :func:`Relation.species_key`) folds two
phrasings of the same rule into the same species key. That key is
the unit consumed by the adaptive Good-Turing driver when we
graduate this from a single-pass tool to a multi-pass coverage
estimator (planned Phase D follow-up).
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import re
import threading
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

import pandas as pd
import yaml

from .columns import canonicalize_columns
from .error_taxonomy import ErrorTaxonomy, ErrorType
from .llm_client import LLMClient, LLMResponse
from .table_digest import TableDigest

logger = logging.getLogger(__name__)


from .jsonl_io import JSON_FENCE_RE as _JSON_BLOCK_RE, extract_json_payloads
_WS_RE = re.compile(r"\s+")
# Word characters (variable / column names, numbers).
_WORD_RE = re.compile(r"\w+")
# Operator tokens whose presence materially changes a predicate's
# meaning ("a <= b" vs "a < b"). Multi-char forms first so we don't
# greedy-match them as two single chars.
_OPERATOR_RE = re.compile(r"(<=|>=|==|!=|[+\-*/=<>!])")
# Combined tokenizer: a token is either a word run or an operator run.
_TOKEN_RE = re.compile(r"\w+|<=|>=|==|!=|[+\-*/=<>!]")


# ---------------------------------------------------------------------------
# Relation dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Relation:
    """One multi-column rule discovered for a (source, error_type).

    The free-form ``predicate`` is the canonical natural-language form
    the LLM emits; ``executable`` is an optional machine-evaluable
    expression for kinds that admit it (calculations, aggregations,
    ordered comparisons). ``species_key`` is the stable hash used by
    the adaptive coverage estimator to deduplicate paraphrases of the
    same underlying rule across passes.
    """

    source: str
    error_type: str
    kind: str
    columns: Tuple[str, ...]
    predicate: str
    rationale: str = ""
    confidence: str = "medium"
    lhs: Tuple[str, ...] = ()
    rhs: Tuple[str, ...] = ()
    executable: Optional[str] = None
    raw: Mapping[str, Any] = field(default_factory=dict)
    # Multi-table fields. Empty for single-source row/group relations.
    # ``tables`` lists the sheet/source names this relation spans (a
    # workbook may contain N sheets and the rule may touch a subset).
    # ``column_refs`` carries the qualified ``(sheet, column)`` pairs
    # so the validator can resolve which frame each column lives in.
    tables: Tuple[str, ...] = ()
    column_refs: Tuple[Tuple[str, str], ...] = ()

    def as_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Tuples are not JSON-serialisable as themselves but ``asdict``
        # already converted them - just normalise to lists for stable
        # JSON output.
        d["columns"] = list(self.columns)
        d["lhs"] = list(self.lhs)
        d["rhs"] = list(self.rhs)
        d["tables"] = list(self.tables)
        d["column_refs"] = [list(cr) for cr in self.column_refs]
        d["raw"] = dict(self.raw)
        d["species_key"] = self.species_key()
        return d

    def species_key(self) -> str:
        """Stable canonical hash for adaptive Good-Turing accounting.

        Two relations collapse to the same species when:
          * the error type matches, AND
          * the kind matches, AND
          * the column set matches (sorted; lhs/rhs preserved separately
            for directional kinds), AND
          * a fold of the predicate text matches (lower, whitespace
            collapsed, punctuation stripped beyond a small allowed set).

        Bijection collapse: when the kind is ``fd`` and BOTH ``lhs``
        and ``rhs`` are single-column, ``A -> B`` and ``B -> A`` fold
        to the same species. The two directions encode the same
        functional constraint (the columns are bijective in this
        slice) and double-counting them inflates the species pool
        without giving the LLM more signal.

        Conservative dedup is fine: over-counting just makes the
        adaptive driver run an extra pass; under-counting (false
        merges) silently hides relations.
        """
        cols = ",".join(sorted(self.columns))
        # Tables (workbook scope) are part of the species key so two
        # cross-table FDs that happen to share predicates but live
        # in different workbooks do NOT collapse together.
        tbls = ",".join(sorted(self.tables))
        if (
            self.kind == "fd"
            and len(self.lhs) == 1
            and len(self.rhs) == 1
        ):
            # Symmetric pair: sort the two columns so direction is
            # erased. Predicate is also dropped from the key because
            # the two directions naturally phrase the rule
            # differently (e.g. "A determines B" vs "B determines A")
            # and we explicitly want them to collapse.
            pair = ",".join(sorted([self.lhs[0], self.rhs[0]]))
            payload = f"{self.error_type}|{self.kind}|{cols}|{tbls}|bijection|{pair}"
        else:
            lhs = ",".join(sorted(self.lhs))
            rhs = ",".join(sorted(self.rhs))
            pred = _fold_predicate(self.predicate)
            payload = f"{self.error_type}|{self.kind}|{cols}|{tbls}|{lhs}|{rhs}|{pred}"
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def _fold_predicate(text: str) -> str:
    # NFKC folds full-width / compatibility forms (e.g. ``＞`` -> ``>``,
    # full-width digits, no-break spaces) so two equivalent predicates
    # written in different unicode forms collapse to the same species.
    s = unicodedata.normalize("NFKC", text or "").lower()
    # Normalize comparison-operator variants. NFKC handles the unicode
    # math operators; this pass collapses the ASCII-spelt synonyms so
    # ``a != b`` / ``a <> b`` / ``a ≠ b`` (post-NFKC) all fold to the
    # same species key. Without this, two paraphrases of the same
    # rule end up in different Good-Turing buckets and the dashboard
    # renders both. Keep this list tight: only identity-preserving
    # rewrites belong here.
    s = (
        s.replace("≠", "!=")
         .replace("≤", "<=")
         .replace("≥", ">=")
         .replace("<>", "!=")
    )
    # Tokenise words and operators independently so ``a==b`` and
    # ``a == b`` and ``a  ==   b`` all fold to the same canonical
    # form. Without this split the previous regex kept "a==b" as one
    # token but "a == b" as three, splitting otherwise-identical
    # predicates into different species and inflating the GT pool.
    tokens = _TOKEN_RE.findall(s)
    return " ".join(tokens)


# ---------------------------------------------------------------------------
# Per-error-type prompt config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DiscoveryTypeSpec:
    """Per-error-type entry in ``prompts/discover_relations.yaml``."""

    key: str
    kinds: Tuple[str, ...]
    # ``row`` (default): single-row predicate - the rule is true or
    # false on each row independently. ``group``: multi-row /
    # inter-record predicate - the rule's truth depends on a set of
    # rows that share a key (FDs, duplicates, cross-record conflicts,
    # total-row aggregates). ``table``: multi-table / multi-source
    # predicate - the rule spans two or more sibling sheets that
    # belong to the same workbook (FK orphans, naming conflicts,
    # representation conflicts). The discovery runner uses this to
    # decide whether to compute and inject structural priors /
    # verbatim group blocks / workbook context alongside the digest +
    # stratified sample.
    scope: str = "row"


@dataclass(frozen=True)
class DiscoveryPromptConfig:
    """Loaded ``prompts/discover_relations.yaml`` (system + template + per-type)."""

    system_prompt: str
    user_template: str
    types: Mapping[str, DiscoveryTypeSpec]
    source_path: Optional[Path] = None

    def render(
        self,
        *,
        error_type: ErrorType,
        digest: TableDigest,
        row_sample_md: str,
        priors_section: str = "",
    ) -> str:
        spec = self.types.get(error_type.key)
        if spec is None:
            raise KeyError(
                f"discover_relations.yaml has no entry for "
                f"error_type '{error_type.key}'"
            )
        kinds_block = "\n".join(f"  - {k}" for k in spec.kinds)
        return self.user_template.format(
            type_key=error_type.key,
            type_name=error_type.name,
            type_category=error_type.category,
            type_description=error_type.description.strip(),
            type_example=error_type.example.strip()
            or "(no reference example provided)",
            type_notes=error_type.notes.strip() or "(no additional guidance)",
            table_overview=digest.to_prompt_block(),
            row_sample=row_sample_md,
            priors_section=priors_section
            or "(no structural priors for this error type)",
            error_type_kinds=kinds_block,
        )


def load_discovery_prompt(path: Path | str) -> DiscoveryPromptConfig:
    """Parse ``prompts/discover_relations.yaml`` into a config object."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Discovery prompt not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, Mapping):
        raise ValueError(f"{p}: top level must be a mapping")
    base = raw.get("base") or {}
    if not isinstance(base, Mapping):
        raise ValueError(f"{p}: 'base' must be a mapping")
    system_prompt = str(base.get("system", "")).strip()
    user_template = str(base.get("user_template", ""))
    if not system_prompt or not user_template:
        raise ValueError(
            f"{p}: 'base.system' and 'base.user_template' are required"
        )
    types_raw = raw.get("types") or {}
    if not isinstance(types_raw, Mapping):
        raise ValueError(f"{p}: 'types' must be a mapping")
    types: Dict[str, DiscoveryTypeSpec] = {}
    for key, body in types_raw.items():
        if not isinstance(body, Mapping):
            raise ValueError(f"{p}: types.{key} must be a mapping")
        kinds = body.get("kinds") or []
        if not isinstance(kinds, list) or not all(isinstance(k, str) for k in kinds):
            raise ValueError(f"{p}: types.{key}.kinds must be a list of strings")
        scope_raw = body.get("scope", "row")
        if scope_raw not in ("row", "group", "table"):
            raise ValueError(
                f"{p}: types.{key}.scope must be 'row', 'group', or "
                f"'table' (got {scope_raw!r})"
            )
        types[str(key)] = DiscoveryTypeSpec(
            key=str(key), kinds=tuple(kinds), scope=str(scope_raw),
        )
    return DiscoveryPromptConfig(
        system_prompt=system_prompt,
        user_template=user_template,
        types=types,
        source_path=p,
    )


# ---------------------------------------------------------------------------
# Stratified row sampling
# ---------------------------------------------------------------------------


def sample_rows_for_prompt(
    df: pd.DataFrame,
    *,
    n_samples: int = 5,
    total_row_indices: Sequence[int] = (),
    max_cell_chars: int = 80,
    seed: Optional[int] = None,
    outlier_indices: Sequence[int] = (),
) -> str:
    """Return a Markdown table with a stratified row sample.

    When ``seed`` is ``None`` (default), picks ``n_samples`` row
    indices spread evenly across the table (first, last, evenly-spaced
    interior rows) - deterministic, reproducible, no per-pass
    variation. This is the right choice for a one-shot run.

    When ``seed`` is an integer, draws ``n_samples`` indices
    pseudo-randomly without replacement using ``random.Random(seed)``.
    Use this in adaptive multi-pass mode to surface different rows
    each pass and let the LLM see different evidence; the seed is
    deterministic so a pass can be replayed exactly.

    In both cases rows the digest has flagged as total / summary rows
    are EXCLUDED - those distort grounding for non-aggregation rules.
    Falls back to "every row" when the table is shorter than the
    requested sample.

    When ``outlier_indices`` is non-empty, a second labeled markdown
    block is appended showing those rows. They are picked elsewhere
    (typically by :func:`pick_outlier_indices`) using robust z-scores
    on numeric columns and rare-value membership for text columns.
    Outlier rows are de-duplicated against the stratified picks so
    the same row is never rendered twice.
    """
    if df.empty:
        return "(empty table - no row sample available)"

    def _truncate(v: object) -> str:
        s = "" if pd.isna(v) else str(v)
        if len(s) > max_cell_chars:
            return s[: max_cell_chars - 1] + "\u2026"
        return s

    def _render(picks: Sequence[int]) -> str:
        sample = df.iloc[list(picks)].copy()
        sample = sample.astype(object).map(_truncate)
        sample.insert(0, "__row__", list(picks))
        return sample.to_markdown(index=False)

    n_rows = len(df)
    excluded = set(int(i) for i in total_row_indices)
    candidates = [i for i in range(n_rows) if i not in excluded]
    if not candidates:
        candidates = list(range(n_rows))

    if len(candidates) <= n_samples:
        picks = candidates
    elif seed is None:
        # Evenly-spaced indices into the candidate list, including
        # both endpoints.
        step = (len(candidates) - 1) / (n_samples - 1)
        picks = sorted({candidates[int(round(i * step))] for i in range(n_samples)})
    else:
        rng = random.Random(int(seed))
        picks = sorted(rng.sample(candidates, n_samples))

    parts = ["SAMPLE ROWS (stratified):", _render(picks)]
    extra = [int(i) for i in outlier_indices if 0 <= int(i) < n_rows and int(i) not in set(picks)]
    if extra:
        parts.append("")
        parts.append(
            "SAMPLE ROWS (outliers / rare values - flagged by robust "
            "z-score on numeric columns or rare-value membership on "
            "text columns; use these to test whether candidate rules "
            "still hold at the extremes):"
        )
        parts.append(_render(sorted(set(extra))))
    return "\n".join(parts)


def pick_outlier_indices(
    df: pd.DataFrame,
    digest: TableDigest,
    *,
    n: int = 3,
    z_threshold: float = 3.5,
    exclude: Sequence[int] = (),
) -> List[int]:
    """Pick up to ``n`` row indices that look like distributional
    outliers given the digest.

    Two signals:

    1. **Numeric robust z-score**. For each numeric column with a
       ``numeric_summary``, compute ``|x - median| / (1.4826 * MAD)``
       per row (median absolute deviation; the constant 1.4826 makes
       MAD a consistent estimator of sigma under normality). Rank
       rows by their MAX z-score across numeric columns. A row
       qualifies when this max exceeds ``z_threshold`` (or when the
       column is constant, when its value differs from the median).
    2. **Rare-value presence**. Any row whose cell holds a value
       that appears only once in its column (a "long-tail" value)
       is a candidate, even when no numeric signal flags it.

    Total / summary rows from the digest are excluded - they
    *always* look like numeric outliers (they're the sums) and
    would crowd out interesting evidence.

    Returns at most ``n`` indices, sorted ascending. Deterministic.
    """
    if df.empty or n <= 0:
        return []

    excluded = {int(i) for i in digest.total_row_indices}
    excluded.update(int(i) for i in exclude)

    n_rows = len(df)
    scores: Dict[int, float] = {}

    # 1. Numeric robust z-scores.
    for col_name, col_digest in digest.columns.items():
        if col_digest.dtype_kind != "numeric" or not col_digest.numeric_summary:
            continue
        if col_name not in df.columns:
            continue
        ns = col_digest.numeric_summary
        median = ns.get("median")
        if median is None:
            continue
        # Per-row coercion. Cells that don't parse contribute 0.
        from .table_digest import _to_float as _coerce  # local import; small helper

        col = df[col_name].where(df[col_name].notna(), "").astype(str).str.strip()
        floats = [_coerce(v) for v in col.tolist()]
        finite = [x for x in floats if x is not None]
        if not finite:
            continue
        # MAD (median absolute deviation), with the 1.4826 scaling.
        abs_dev = [abs(x - median) for x in finite]
        mad = float(pd.Series(abs_dev).median()) if abs_dev else 0.0
        scale = 1.4826 * mad if mad > 0 else 0.0
        if scale <= 0:
            # Constant (or near-constant) column - the previous code
            # treated any deviation as ``+inf`` z-score, which floods
            # ``scores`` with ungroundable picks whenever a single
            # row drifts by 1e-6. Skip the column entirely instead;
            # rare-value membership will still surface true outliers
            # via the second pass below.
            continue
        for idx, x in enumerate(floats):
            if x is None or idx in excluded:
                continue
            z = abs(x - median) / scale
            if z >= z_threshold:
                scores[idx] = max(scores.get(idx, 0.0), z)

    # 2. Rare-value presence in text columns.
    rare_bonus: Dict[int, int] = {}
    for col_name, col_digest in digest.columns.items():
        if col_digest.dtype_kind != "text" or not col_digest.rare_values:
            continue
        if col_name not in df.columns:
            continue
        rare_set = set(col_digest.rare_values)
        col = df[col_name].where(df[col_name].notna(), "").astype(str).str.strip()
        for idx, v in enumerate(col.tolist()):
            if idx in excluded or not v:
                continue
            if v in rare_set:
                rare_bonus[idx] = rare_bonus.get(idx, 0) + 1

    if not scores and not rare_bonus:
        return []

    # Combine: numeric outliers ranked by z, then rare-value rows ranked
    # by rare-column count. Numeric signal wins when both exist.
    ranked_numeric = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    ranked_rare = sorted(
        ((i, c) for i, c in rare_bonus.items() if i not in scores),
        key=lambda kv: (-kv[1], kv[0]),
    )

    picks: List[int] = []
    for idx, _ in ranked_numeric:
        if idx not in picks:
            picks.append(idx)
        if len(picks) >= n:
            break
    if len(picks) < n:
        for idx, _ in ranked_rare:
            if idx not in picks:
                picks.append(idx)
            if len(picks) >= n:
                break
    return sorted(picks[:n])


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------


def _coerce_relations_payload(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return [r for r in payload if isinstance(r, dict)]
    if isinstance(payload, dict):
        for key in ("relations", "rules", "constraints"):
            v = payload.get(key)
            if isinstance(v, list):
                return [r for r in v if isinstance(r, dict)]
        return [payload]
    return []


def _parse_relations_json(text: str) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    if not text:
        return [], "empty response"
    candidates: List[str] = []
    for m in _JSON_BLOCK_RE.finditer(text):
        candidates.append(m.group(1))
    candidates.append(text.strip())
    # Bracket-balanced fallback handles non-fenced prose+JSON.
    candidates.extend(extract_json_payloads(text))
    for cand in candidates:
        cand = cand.strip()
        if not cand:
            continue
        try:
            payload = json.loads(cand)
        except json.JSONDecodeError:
            continue
        return _coerce_relations_payload(payload), None
    return [], "failed to parse JSON from response"


def _normalize_columns(value: Any, frame_headers: Sequence[str]) -> Tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)):
        return ()
    resolved = canonicalize_columns(value, frame_headers)
    return tuple(resolved)


def _list_len(value: Any) -> int:
    """Number of column tokens the LLM declared. Used to spot
    hallucinated columns: if canonicalization drops any, the
    relation references something not in the schema."""
    if value is None:
        return 0
    if isinstance(value, str):
        return 1 if value.strip() else 0
    if isinstance(value, (list, tuple)):
        return sum(1 for v in value if isinstance(v, str) and v.strip())
    return 0


def _resolve_multi_table_refs(
    *,
    raw: Mapping[str, Any],
    default_tables: Tuple[str, ...],
    sheet_columns: Mapping[str, Sequence[str]],
) -> Tuple[Tuple[str, ...], Tuple[Tuple[str, str], ...]]:
    """Parse the optional ``tables`` and ``column_refs`` payload.

    For row/group scope (``default_tables == ()``), this is a no-op
    and returns ``((), ())``. For table scope it:
      * intersects any LLM-declared ``tables`` with the sibling
        sheets in scope, falling back to ``default_tables`` when
        the LLM omitted the field;
      * canonicalises ``column_refs`` per-sheet, dropping any
        ``[sheet, col]`` pair where ``sheet`` is unknown or
        ``col`` does not resolve in that sheet's headers.
    """
    if not default_tables:
        return (), ()

    tables_raw = raw.get("tables")
    if isinstance(tables_raw, list) and tables_raw:
        seen_t: set[str] = set()
        tables: List[str] = []
        for t in tables_raw:
            if not isinstance(t, str):
                continue
            t_norm = t.strip()
            if not t_norm or t_norm in seen_t:
                continue
            if t_norm in sheet_columns:
                tables.append(t_norm)
                seen_t.add(t_norm)
        if not tables:
            tables = list(default_tables)
    else:
        tables = list(default_tables)

    refs_raw = raw.get("column_refs")
    column_refs: List[Tuple[str, str]] = []
    if isinstance(refs_raw, list):
        for entry in refs_raw:
            if not isinstance(entry, (list, tuple)) or len(entry) != 2:
                logger.debug("dropping malformed column_ref %r", entry)
                continue
            sheet_part, col_part = entry
            if not isinstance(sheet_part, str) or not isinstance(col_part, str):
                logger.debug("dropping non-string column_ref %r", entry)
                continue
            sheet = sheet_part.strip()
            col = col_part.strip()
            if not sheet or not col:
                logger.debug("dropping empty column_ref %r", entry)
                continue
            if sheet not in sheet_columns:
                logger.debug(
                    "dropping column_ref %r: sheet not in scope (%s)",
                    entry, sorted(sheet_columns.keys()),
                )
                continue
            resolved = canonicalize_columns([col], sheet_columns[sheet])
            if not resolved:
                logger.debug(
                    "dropping column_ref %r: column not in sheet headers",
                    entry,
                )
                continue
            column_refs.append((sheet, resolved[0]))

    return tuple(tables), tuple(column_refs)


_VALUE_ANCHOR_KINDS = frozenset({
    # Kinds prone to the "if A == 'literal' then B == 'literal'"
    # pattern that pollutes downstream detectors. We deliberately do
    # NOT filter arithmetic / FD / total_row kinds whose executables
    # legitimately reference numeric constants (rates, bounds, 100, 0).
    "implication",
    "mandatory_when",
    "equality_within",
    "factual",
    "bounded",
    "cap",
    "equality",
})

_QUOTED_LITERAL_RE = re.compile(r"""['"]([^'"\n]{1,80})['"]""")


def _is_value_anchored(
    raw: Mapping[str, Any],
    digest: Optional[TableDigest],
    cols: Sequence[str],
    *,
    closed_enum_max_distinct: int = 4,
) -> Optional[str]:
    """Detect rules anchored on a specific cell value of a non-enum column.

    Returns a short reason string when the relation looks value-keyed,
    or ``None`` when it passes. The pattern we filter:

        kind:       implication / mandatory_when / equality_within / ...
        executable: ``... 'Exemption 10 MW' ...`` or ``"Yes" -> B == 100``

    A relation is rejected when any quoted literal in its
    ``executable`` (or ``predicate`` if no executable) matches a
    ``top_values`` entry of one of its referenced columns AND that
    column has more than ``closed_enum_max_distinct`` distinct
    values. The reasoning: if the conditioning column is a closed
    enum (status / flag with a handful of values) the literal
    anchor is legitimate; if the column is high-cardinality, the
    rule is sample-fitted to one specific cell and would explode
    into N siblings under aggregation.
    """
    kind = str(raw.get("kind") or "").strip()
    if kind not in _VALUE_ANCHOR_KINDS:
        return None
    if digest is None or not cols:
        return None

    executable = raw.get("executable")
    predicate = raw.get("predicate")
    text = str(executable) if isinstance(executable, str) and executable.strip() else ""
    if not text:
        text = str(predicate) if isinstance(predicate, str) else ""
    if not text:
        return None

    literals = _QUOTED_LITERAL_RE.findall(text)
    if not literals:
        return None

    # Build a lookup of {col_name: (set_of_known_values, n_distinct)}
    # for the columns referenced by this relation. We pool top_values,
    # sample_values, and rare_values because the digest only keeps a
    # short top_values list (default 5); a high-cardinality column
    # like Airline with 20 distinct values can have 'Virgin Atlantic'
    # in its sample_values without ever reaching top_values, and the
    # LLM still latches onto it. Pooling all surfaced literals catches
    # those leaks.
    refs: Dict[str, Tuple[set, int]] = {}
    for col in cols:
        cd = digest.columns.get(col) if digest.columns else None
        if cd is None:
            continue
        known: set = set()
        for v, _c in (cd.top_values or []):
            known.add(str(v))
        for v in (cd.sample_values or []):
            known.add(str(v))
        for v in (cd.rare_values or []):
            known.add(str(v))
        refs[col] = (known, int(cd.n_distinct))

    for lit in literals:
        for col, (known, n_distinct) in refs.items():
            if lit in known and n_distinct > closed_enum_max_distinct:
                return (
                    f"value-anchored: literal {lit!r} matches a known "
                    f"value of column {col!r} (n_distinct={n_distinct} "
                    f"> {closed_enum_max_distinct} closed-enum cap)"
                )
    return None


# ---------------------------------------------------------------------------
# Per-(source, error_type) discovery report
# ---------------------------------------------------------------------------


@dataclass
class DiscoveryReport:
    """Result of one (source, error_type) discovery LLM call."""

    source: str
    error_type: str
    error_type_name: str
    error_type_category: str
    model: str
    prompt: str
    relations: List[Relation] = field(default_factory=list)
    raw_text: str = ""
    reasoning_summary: str = ""
    endpoint: str = ""
    attempts: int = 0
    error: Optional[str] = None
    parse_error: Optional[str] = None

    def as_dict(self, *, include_raw: bool = True) -> Dict[str, Any]:
        d = {
            "source": self.source,
            "error_type": self.error_type,
            "error_type_name": self.error_type_name,
            "error_type_category": self.error_type_category,
            "model": self.model,
            "endpoint": self.endpoint,
            "attempts": self.attempts,
            "error": self.error,
            "parse_error": self.parse_error,
            "n_relations": len(self.relations),
            "relations": [r.as_dict() for r in self.relations],
        }
        if include_raw:
            d["prompt"] = self.prompt
            d["raw_text"] = self.raw_text
            d["reasoning_summary"] = self.reasoning_summary
        return d


# ---------------------------------------------------------------------------
# Discoverer
# ---------------------------------------------------------------------------


class SchemaDiscoverer:
    """Run relation-discovery LLM calls over (source, error_type) pairs.

    One call per (source, error_type) pair. Calls are dispatched
    through a thread pool so multiple sources / error types run
    concurrently across the LLM endpoint pool, mirroring
    :class:`PerTypeCleaningDetector`.

    The discoverer is intentionally STATELESS w.r.t. perturbation:
    callers wanting multi-pass adaptive discovery rebuild the
    discoverer per pass with whatever variant prompt-config / digest
    sample they want. This keeps the single-pass code path trivial
    while leaving room to graft the adaptive driver on top.
    """

    def __init__(
        self,
        llm: LLMClient,
        prompt_config: DiscoveryPromptConfig,
        taxonomy: ErrorTaxonomy,
        model: str,
        *,
        digests: Mapping[str, TableDigest],
        source_frames: Mapping[str, pd.DataFrame],
        error_types: Optional[Sequence[ErrorType]] = None,
        reasoning_effort: str = "medium",
        reasoning_summary: str = "detailed",
        concurrency: Optional[int] = None,
        n_row_samples: int = 5,
        workbooks: Optional[Mapping[str, Sequence[str]]] = None,
    ) -> None:
        self.llm = llm
        self.prompt_config = prompt_config
        self.taxonomy = taxonomy
        self.model = model
        self.digests = dict(digests)
        self.source_frames = dict(source_frames)
        # Workbook grouping: ``workbook_id -> [sheet_source_name, ...]``.
        # Only workbooks with two or more sheets are eligible for
        # ``scope=table`` discovery. ``None`` (the default) treats
        # every source as its own workbook -> table-scope arms are
        # automatically skipped, preserving single-source behaviour.
        wb_map: Dict[str, List[str]] = {}
        if workbooks is not None:
            for wb_id, sheets in workbooks.items():
                wb_map[str(wb_id)] = [
                    str(s) for s in sheets if str(s) in self.source_frames
                ]
        # Filter to multi-sheet workbooks; single-sheet ones contribute
        # nothing to cross-table discovery.
        self.workbooks: Dict[str, Tuple[str, ...]] = {
            wb_id: tuple(sheets)
            for wb_id, sheets in wb_map.items()
            if len(sheets) >= 2
        }
        # Default: every taxonomy type that has a discovery entry. The
        # discovery YAML deliberately omits the single-column types,
        # so this filter naturally restricts to multi-column / multi-
        # row / multi-table types.
        all_types = list(error_types) if error_types is not None else list(
            taxonomy.types
        )
        self.error_types = [
            et for et in all_types if et.key in self.prompt_config.types
        ]
        if not self.error_types:
            raise ValueError(
                "SchemaDiscoverer: no error types match discover_relations.yaml; "
                "check the taxonomy / discovery YAML pairing."
            )
        self.reasoning_effort = reasoning_effort
        self.reasoning_summary = reasoning_summary
        default_concurrency = min(8, max(1, len(self.llm.endpoints)))
        self.concurrency = concurrency if concurrency and concurrency > 0 else default_concurrency
        self.n_row_samples = max(1, int(n_row_samples))

        # Pre-render row samples once per source. Sampling is
        # deterministic, so this is cheap and avoids duplicating the
        # work across error types. We also pre-pick outlier rows
        # using the digest so every prompt sees both stratified
        # rows AND distributional extremes (numeric robust z-score
        # / rare-value membership) - this catches sample-fitted
        # rules where a stratified sample of 5 hides the outlier.
        self._row_samples: Dict[str, str] = {}
        self._outlier_indices: Dict[str, List[int]] = {}
        for src, frame in self.source_frames.items():
            digest = self.digests.get(src)
            totals = digest.total_row_indices if digest else ()
            outliers = (
                pick_outlier_indices(frame, digest, n=3)
                if digest is not None else []
            )
            self._outlier_indices[src] = outliers
            self._row_samples[src] = sample_rows_for_prompt(
                frame,
                n_samples=self.n_row_samples,
                total_row_indices=totals,
                outlier_indices=outliers,
            )

    # ------------------------------------------------------------------
    # Single-call execution
    # ------------------------------------------------------------------

    def _discover(
        self,
        source: str,
        error_type: ErrorType,
        *,
        seed: Optional[int] = None,
    ) -> DiscoveryReport:
        """Dispatch to the per-scope discovery path.

        For ``scope=row`` and ``scope=group`` types, ``source`` is a
        sheet/source key in ``self.source_frames``. For
        ``scope=table`` types, ``source`` is a workbook id in
        ``self.workbooks`` and the call enumerates relations across
        that workbook's sibling sheets.
        """
        spec = self.prompt_config.types.get(error_type.key)
        if spec is not None and spec.scope == "table":
            return self._discover_table(source, error_type, seed=seed)
        return self._discover_single(source, error_type, seed=seed)

    def _discover_single(
        self,
        source: str,
        error_type: ErrorType,
        *,
        seed: Optional[int] = None,
    ) -> DiscoveryReport:
        digest = self.digests.get(source)
        if digest is None:
            raise KeyError(f"No digest available for source {source!r}")
        if seed is None:
            row_sample_md = self._row_samples.get(source, "(no row sample)")
        else:
            frame = self.source_frames[source]
            totals = digest.total_row_indices if digest else ()
            row_sample_md = sample_rows_for_prompt(
                frame,
                n_samples=self.n_row_samples,
                total_row_indices=totals,
                seed=seed,
                outlier_indices=self._outlier_indices.get(source, ()),
            )

        # Multi-row priors. For ``scope=group`` error types, run the
        # cheap structural detectors and surface their output to the
        # LLM as a separate section. For ``scope=row`` types, the
        # section is omitted entirely so the row-level prompt does
        # not get cluttered with irrelevant context.
        spec = self.prompt_config.types.get(error_type.key)
        priors_section = ""
        if spec is not None and spec.scope == "group":
            try:
                # Local import: keeps ``relations`` lightweight when
                # only single-row types are exercised.
                from .multi_row_priors import (  # noqa: PLC0415
                    compute_priors,
                    render_group_blocks,
                    render_priors_block,
                )

                frame = self.source_frames[source]
                priors = compute_priors(
                    df=frame,
                    digest=digest,
                    error_type=error_type.key,
                    source=source,
                )
                priors_section = (
                    render_priors_block(frame, priors)
                    + "\n"
                    + render_group_blocks(frame, priors)
                )
            except Exception as exc:  # noqa: BLE001
                # Never let the priors pipeline break discovery -
                # fall back to no extra context and log the issue.
                logger.warning(
                    "multi_row_priors failed for source=%s type=%s: %s",
                    source, error_type.key, exc,
                )
                priors_section = (
                    "STRUCTURAL CANDIDATES\n"
                    "  (heuristic detector errored - see logs)\n"
                )

        prompt = self.prompt_config.render(
            error_type=error_type,
            digest=digest,
            row_sample_md=row_sample_md,
            priors_section=priors_section,
        )
        report = DiscoveryReport(
            source=source,
            error_type=error_type.key,
            error_type_name=error_type.name,
            error_type_category=error_type.category,
            model=self.model,
            prompt=prompt,
        )
        return self._send_and_parse(
            report, error_type, frame=self.source_frames[source],
            digest=digest, default_tables=(),
        )

    def _discover_table(
        self,
        workbook_id: str,
        error_type: ErrorType,
        *,
        seed: Optional[int] = None,
    ) -> DiscoveryReport:
        """Cross-table discovery path. ``workbook_id`` selects a
        multi-sheet workbook from ``self.workbooks``."""
        sheets = self.workbooks.get(workbook_id)
        if not sheets:
            # Workbook with <2 sheets shouldn't reach here, but be
            # defensive: emit an empty report so the runner stays
            # uniform.
            return DiscoveryReport(
                source=workbook_id,
                error_type=error_type.key,
                error_type_name=error_type.name,
                error_type_category=error_type.category,
                model=self.model,
                prompt="",
                error="workbook has <2 sheets; skipping cross-table discovery",
            )

        try:
            from .multi_table_priors import (  # noqa: PLC0415
                compute_table_priors,
                render_table_priors_block,
                render_workbook_context,
            )

            priors = compute_table_priors(
                sheets=sheets,
                frames=self.source_frames,
                digests=self.digests,
                error_type=error_type.key,
                workbook=workbook_id,
            )
            workbook_block = render_workbook_context(
                workbook=workbook_id,
                sheets=priors.sheets or sheets,
                frames=self.source_frames,
                digests=self.digests,
            )
            cross_block = render_table_priors_block(priors)
            priors_section = workbook_block + "\n" + cross_block
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "multi_table_priors failed for workbook=%s type=%s: %s",
                workbook_id, error_type.key, exc,
            )
            priors_section = (
                "WORKBOOK CONTEXT\n"
                "  (heuristic detector errored - see logs)\n"
            )

        # For the prompt's ``table_overview`` and ``row_sample`` we
        # render a per-sheet preview so the LLM has concrete grounding
        # without the full per-column digest of every sheet (kept
        # compact via WORKBOOK CONTEXT above).
        per_sheet_overview: List[str] = []
        per_sheet_samples: List[str] = []
        for s in sheets:
            df = self.source_frames.get(s)
            if df is None or df.empty:
                continue
            d = self.digests.get(s)
            per_sheet_overview.append(f"=== sheet {s!r} ===")
            if d is not None:
                per_sheet_overview.append(d.to_prompt_block())
            per_sheet_samples.append(f"=== sheet {s!r} ===")
            per_sheet_samples.append(
                sample_rows_for_prompt(
                    df,
                    n_samples=min(self.n_row_samples, 4),
                    total_row_indices=d.total_row_indices if d else (),
                    seed=seed,
                )
            )

        # Pick the first sheet as the "anchor" digest; ``render``
        # only uses the digest for its prompt block, which we
        # override below by passing a synthesised overview.
        anchor_sheet = sheets[0]
        anchor_digest = self.digests.get(anchor_sheet)
        if anchor_digest is None:
            return DiscoveryReport(
                source=workbook_id,
                error_type=error_type.key,
                error_type_name=error_type.name,
                error_type_category=error_type.category,
                model=self.model,
                prompt="",
                error=f"anchor sheet {anchor_sheet!r} has no digest",
            )

        # Synthesise the prompt manually so we can substitute a
        # multi-sheet table_overview without changing the public
        # ``DiscoveryPromptConfig.render`` signature.
        spec = self.prompt_config.types.get(error_type.key)
        kinds_block = "\n".join(f"  - {k}" for k in (spec.kinds if spec else ()))
        prompt = self.prompt_config.user_template.format(
            type_key=error_type.key,
            type_name=error_type.name,
            type_category=error_type.category,
            type_description=error_type.description.strip(),
            type_example=error_type.example.strip()
            or "(no reference example provided)",
            type_notes=error_type.notes.strip() or "(no additional guidance)",
            table_overview="\n".join(per_sheet_overview)
            or "(no per-sheet digest)",
            row_sample="\n".join(per_sheet_samples)
            or "(no row sample)",
            priors_section=priors_section
            or "(no structural priors for this error type)",
            error_type_kinds=kinds_block,
        )
        report = DiscoveryReport(
            source=workbook_id,
            error_type=error_type.key,
            error_type_name=error_type.name,
            error_type_category=error_type.category,
            model=self.model,
            prompt=prompt,
        )
        return self._send_and_parse(
            report, error_type, frame=None,
            digest=anchor_digest,
            default_tables=tuple(sheets),
        )

    def _send_and_parse(
        self,
        report: DiscoveryReport,
        error_type: ErrorType,
        *,
        frame: Optional[pd.DataFrame],
        digest: TableDigest,
        default_tables: Tuple[str, ...],
    ) -> DiscoveryReport:
        """Issue the LLM call and parse the response into ``Relation``
        objects.

        ``frame`` is the single-sheet DataFrame used for column
        canonicalisation in row/group scopes. For table scope it is
        ``None`` and the caller's ``default_tables`` lists every
        sibling sheet involved.
        """
        response: LLMResponse = self.llm.query(
            self.model,
            report.prompt,
            reasoning_effort=self.reasoning_effort,
            reasoning_summary=self.reasoning_summary,
            system_prompt=self.prompt_config.system_prompt,
        )
        report.endpoint = response.endpoint
        report.attempts = response.attempts
        report.raw_text = response.text
        report.reasoning_summary = response.reasoning_summary
        report.error = response.error

        if not response.ok:
            return report

        items, parse_err = _parse_relations_json(response.text)
        report.parse_error = parse_err

        # Build the headers union the LLM is allowed to reference. For
        # row/group scope this is just the source frame's columns; for
        # table scope we union every sibling sheet's columns and keep
        # a per-sheet column map so we can resolve qualified
        # ``[sheet, col]`` refs and detect cross-sheet hallucinations.
        if frame is not None:
            headers: List[str] = list(frame.columns)
            sheet_columns: Dict[str, List[str]] = {report.source: headers}
        else:
            headers = []
            sheet_columns = {}
            for s in default_tables:
                df_s = self.source_frames.get(s)
                if df_s is None:
                    continue
                sheet_columns[s] = list(df_s.columns)
                for h in df_s.columns:
                    if h not in headers:
                        headers.append(str(h))

        seen_keys: set[str] = set()
        relations: List[Relation] = []
        for raw in items:
            cols = _normalize_columns(raw.get("columns"), headers)
            cols_declared = _list_len(raw.get("columns"))
            if cols_declared and len(cols) != cols_declared:
                # The LLM named a column we cannot resolve - reject
                # the whole relation rather than silently truncating
                # it to a different (possibly meaningless) shape.
                logger.debug(
                    "Dropping relation with unresolved columns: %r", raw,
                )
                continue
            lhs = _normalize_columns(raw.get("lhs"), headers)
            rhs = _normalize_columns(raw.get("rhs"), headers)
            if (
                _list_len(raw.get("lhs")) != len(lhs)
                or _list_len(raw.get("rhs")) != len(rhs)
            ):
                logger.debug(
                    "Dropping relation with unresolved lhs/rhs: %r", raw,
                )
                continue
            if not cols:
                # Fall back to lhs ∪ rhs if `columns` was omitted.
                cols = tuple(dict.fromkeys(lhs + rhs))
                if not cols:
                    continue
            kind = str(raw.get("kind") or "").strip()
            if not kind:
                continue
            predicate = str(raw.get("predicate") or "").strip()
            if not predicate:
                continue
            anchor_reason = _is_value_anchored(raw, digest, cols)
            if anchor_reason is not None:
                # The LLM emitted a value-keyed rule despite the
                # PARAMETRIC ABSTRACTION clause. These are the
                # "if Description == 'Exemption 10 MW' then ..."
                # type relations the user flagged - they pollute
                # the downstream detector and explode under
                # aggregation. Drop with a debug log so it's
                # visible in the run trace if anyone wants to
                # audit the filter's recall.
                logger.debug(
                    "Dropping %s for %s: %s",
                    kind, error_type.key, anchor_reason,
                )
                continue
            executable_raw = raw.get("executable")
            executable = (
                str(executable_raw).strip()
                if isinstance(executable_raw, str) and executable_raw.strip()
                else None
            )

            # Multi-table fields. For table scope we resolve any
            # ``tables`` / ``column_refs`` declared by the LLM,
            # falling back to ``default_tables`` when omitted. For
            # row/group scope these are always empty tuples.
            tables_t, column_refs_t = _resolve_multi_table_refs(
                raw=raw,
                default_tables=default_tables,
                sheet_columns=sheet_columns,
            )

            relation = Relation(
                source=report.source,
                error_type=error_type.key,
                kind=kind,
                columns=cols,
                predicate=predicate,
                rationale=str(raw.get("rationale") or "").strip(),
                confidence=str(raw.get("confidence") or "medium").strip().lower(),
                lhs=lhs,
                rhs=rhs,
                executable=executable,
                raw=dict(raw),
                tables=tables_t,
                column_refs=column_refs_t,
            )
            key = relation.species_key()
            if key in seen_keys:
                continue
            seen_keys.add(key)
            relations.append(relation)
        report.relations = relations
        return report

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def run(
        self,
        *,
        seed: Optional[int] = None,
        skip: Optional[Any] = None,
        desc: str = "Discover",
    ) -> List[DiscoveryReport]:
        """Run one discovery pass.

        ``seed`` controls per-pass row sampling: ``None`` uses the
        deterministic evenly-spaced sample (single-pass mode);
        passing an integer reseeds the row sample so adaptive
        passes see different rows.

        ``skip`` is an optional callable ``(source, error_type_key)
        -> bool``; arms for which it returns truthy are skipped this
        pass. Used by the adaptive driver to retire saturated arms.

        ``desc`` is the tqdm progress-bar label; the adaptive driver
        overrides it with a per-pass title.
        """
        skip_fn = skip if callable(skip) else (lambda _s, _e: False)
        # Build the (job_key, error_type) pairs. ``job_key`` is a
        # source name for row/group-scope types and a workbook id for
        # table-scope types. Workbooks with <2 sheets contribute no
        # table-scope jobs (already filtered in ``__init__``).
        jobs: List[Tuple[str, ErrorType]] = []
        for et in self.error_types:
            spec = self.prompt_config.types.get(et.key)
            if spec is not None and spec.scope == "table":
                for wb_id in self.workbooks:
                    if skip_fn(wb_id, et.key):
                        continue
                    jobs.append((wb_id, et))
            else:
                for src in self.source_frames:
                    if src not in self.digests or skip_fn(src, et.key):
                        continue
                    jobs.append((src, et))
        total = len(jobs)
        # Demoted to debug: the tqdm bar already shows job count and
        # progress. The pass-summary log lines above the bar were the
        # main source of noise in the workbench job tail.
        logger.debug(
            "Relation discovery pass: %d job(s) (concurrency=%d, seed=%s)",
            total, self.concurrency, seed,
        )
        if total == 0:
            return []

        results: List[Optional[DiscoveryReport]] = [None] * total
        progress = tqdm(
            total=total,
            desc=desc,
            unit="call",
            dynamic_ncols=True,
            leave=True,
        )
        progress_lock = threading.Lock()
        counters = {"errors": 0, "parse": 0, "relations": 0}

        def _on_done(report: DiscoveryReport, src: str, et: ErrorType) -> None:
            with progress_lock:
                if report.error:
                    counters["errors"] += 1
                    logger.warning(
                        "discover failed: source=%s type=%s error=%s",
                        src, et.key, report.error,
                    )
                if report.parse_error:
                    counters["parse"] += 1
                counters["relations"] += len(report.relations)
                progress.set_postfix(
                    rels=counters["relations"],
                    err=counters["errors"],
                    parse=counters["parse"],
                    refresh=False,
                )
                progress.update(1)

        try:
            with logging_redirect_tqdm():
                with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
                    futures = {
                        pool.submit(self._discover, src, et, seed=seed): (idx, src, et)
                        for idx, (src, et) in enumerate(jobs)
                    }
                    for fut in as_completed(futures):
                        idx, src, et = futures[fut]
                        try:
                            report = fut.result()
                        except Exception as exc:  # pragma: no cover - defensive
                            logger.exception(
                                "Relation discovery failed for source=%s error_type=%s",
                                src, et.key,
                            )
                            report = DiscoveryReport(
                                source=src,
                                error_type=et.key,
                                error_type_name=et.name,
                                error_type_category=et.category,
                                model=self.model,
                                prompt="",
                                error=f"unhandled: {exc}",
                            )
                        results[idx] = report
                        _on_done(report, src, et)
        finally:
            progress.close()
        return [r for r in results if r is not None]


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------


def write_relations_jsonl(reports: Iterable[DiscoveryReport], path: Path) -> int:
    """Write one JSON line per (source, error_type, relation).

    Returns the total number of relation lines written. Reports that
    surface zero relations contribute nothing - the file's purpose is
    to be the closed list of relations downstream detection consumes.
    """
    n = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rep in reports:
            for rel in rep.relations:
                fh.write(json.dumps(rel.as_dict(), ensure_ascii=False) + "\n")
                n += 1
    return n


def write_discovery_reports_json(
    reports: Iterable[DiscoveryReport],
    path: Path,
    *,
    include_raw: bool = False,
) -> None:
    """Write the full per-call reports (slim by default; raw=True keeps
    the full LLM response text for debugging)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [rep.as_dict(include_raw=include_raw) for rep in reports]
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
