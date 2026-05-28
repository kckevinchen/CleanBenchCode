"""Step 4 of the multi-table inconsistency pipeline: per-violation
LLM verification.

The deterministic compiler in :mod:`src.core.multi_table.violations`
emits four kinds of records (``header_synonym``, ``dtype_drift``,
``duplicate_row``, ``attribute_conflict``, ``fk_orphan``). The first
two are purely structural and were approved at step 2;
``fk_orphan`` is mechanical (parent/child set containment).

``duplicate_row`` and ``attribute_conflict`` are the kinds where an
LLM can add genuine signal: a deterministic ``==`` on attribute
columns cannot tell whether a difference is a real data-quality
inconsistency or a benign variation (snapshot at different times,
unit/format drift declared elsewhere, NULL placeholder, legitimate
cross-sheet copy). We send these candidates to the LLM in a
batched per-(workbook, group) call asking for a per-violation
verdict + rationale.

Output shape (one line per violation_id; multi-row violations share
an id, so they collapse to one verification record):

.. code-block:: json

    {
      "violation_id": "...",
      "workbook": "...",
      "group_id": "u1",
      "kind": "attribute_conflict",
      "verdict": "confirmed_inconsistency"
               | "benign_variation"
               | "not_a_violation"
               | "uncertain",
      "confidence": "high" | "medium" | "low",
      "rationale": "...",
      "endpoint": "...",
      "attempts": 1,
      "error": null,
      "parse_error": null
    }

Failure modes are recorded but do not crash the pipeline: if the
LLM call errors or the response cannot be parsed, every violation
in the batch gets ``verdict="uncertain"`` with ``error`` populated,
so the downstream dashboard can still show the raw violation while
flagging the verifier failure.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
import yaml

from ..jsonl_io import JSON_FENCE_RE, JSON_OBJECT_RE
from ..llm_client import LLMClient, LLMResponse
from .resolver import ResolvedGroup, ResolvedWorkbook
from .violations import MultiTableViolation

logger = logging.getLogger(__name__)


VERDICTS = (
    "confirmed_inconsistency",
    "benign_variation",
    "not_a_violation",
    "uncertain",
)
CONFIDENCES = ("high", "medium", "low")

# Kinds that this verifier acts on. Other kinds get a default
# "uncertain" pass-through verdict so the downstream dashboard
# can render them uniformly without special-casing.
VERIFIABLE_KINDS = ("attribute_conflict", "duplicate_row")


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VerifyPromptConfig:
    """Loaded prompt config for ``verify_multi_table_violations.yaml``."""

    model: str
    reasoning_effort: str
    reasoning_summary: str
    system_prompt: str
    user_template: str


def load_verify_prompt(path: Path) -> VerifyPromptConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"verify prompt must be a mapping: {path}")
    return VerifyPromptConfig(
        model=str(raw.get("model") or "gpt-5.2"),
        reasoning_effort=str(raw.get("reasoning_effort") or "medium"),
        reasoning_summary=str(raw.get("reasoning_summary") or "detailed"),
        system_prompt=str(raw.get("system") or "").strip(),
        user_template=str(raw.get("user_template") or "").strip(),
    )


@dataclass(frozen=True)
class Verification:
    """One LLM verdict for one ``violation_id``."""

    violation_id: str
    workbook: str
    group_id: str
    kind: str
    verdict: str
    confidence: str
    rationale: str
    endpoint: str = ""
    attempts: int = 0
    error: Optional[str] = None
    parse_error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "violation_id": self.violation_id,
            "workbook": self.workbook,
            "group_id": self.group_id,
            "kind": self.kind,
            "verdict": self.verdict,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "endpoint": self.endpoint,
            "attempts": int(self.attempts),
            "error": self.error,
            "parse_error": self.parse_error,
        }


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------


def _render_column_mapping(group: ResolvedGroup) -> str:
    if not group.column_mapping:
        return "  (none)"
    parts: List[str] = []
    for canonical, entries in group.column_mapping.items():
        variants = ", ".join(
            f"{e.sheet}:{e.column!r}" for e in entries
        )
        parts.append(f"  - {canonical!r}: {variants}")
    return "\n".join(parts)


def _row_to_str(
    frames: Mapping[str, pd.DataFrame],
    sheet: str,
    row: int,
    columns: Sequence[str],
    *,
    max_cell_chars: int = 80,
) -> str:
    """Render one (sheet, row) tuple as ``col=value`` pairs.

    ``columns`` are canonical column names; we resolve them via the
    group's column_mapping at the call-site so this helper just
    formats. Falls back gracefully when the row is out of range or
    the column is absent on this sheet.
    """
    df = frames.get(sheet)
    if df is None or row < 0 or row >= len(df):
        return "(row not available)"
    pieces: List[str] = []
    for col in columns:
        if col not in df.columns:
            continue
        val = df.iat[row, df.columns.get_loc(col)]
        s = "" if val is None else str(val)
        if len(s) > max_cell_chars:
            s = s[:max_cell_chars] + "…"
        pieces.append(f"{col}={s!r}")
    return " | ".join(pieces) or "(no overlapping columns)"


def _render_violation_block(
    *,
    violations: Sequence[MultiTableViolation],
    group: ResolvedGroup,
    frames: Mapping[str, pd.DataFrame],
) -> str:
    """Render the violations to verify, with the underlying rows.

    For every violation we emit:

      [violation_id] kind   columns_in_dispute=[...]
        - sheet_a:row_x  col=value | col=value ...
        - sheet_b:row_y  col=value | col=value ...
        rationale: ...

    Sibling locations come from ``violation.linked`` so the LLM
    sees both sides in one block.
    """
    if not violations:
        return "  (none)"
    # Collect columns to show per row: union of PK + columns flagged on any
    # of the violations (so the LLM has enough context to judge a conflict).
    pk_cols: List[str] = list(group.primary_key or ())
    parts: List[str] = []
    for v in violations:
        cols_dispute = list(v.columns or ())
        # Try to render rows using the canonical column names.
        cols_to_render: List[str] = list(
            dict.fromkeys(pk_cols + cols_dispute)
        )
        # Resolve canonical -> raw column name per sheet via the
        # group's column mapping so we can look up cells in the
        # underlying DataFrames.
        canonical_to_raw_per_sheet: Dict[str, Dict[str, str]] = {}
        for canonical, entries in (group.column_mapping or {}).items():
            for e in entries:
                canonical_to_raw_per_sheet.setdefault(
                    e.sheet, {}
                )[canonical] = e.column

        head = (
            f"  [{v.violation_id}] {v.kind} "
            f"columns_in_dispute={cols_dispute}"
        )
        parts.append(head)
        # Render this row first.
        loc_pairs = [(v.sheet, int(v.row))]
        # Then linked rows.
        for s, r in v.linked or ():
            loc_pairs.append((s, int(r)))
        for sheet, row in loc_pairs:
            raw_cols = [
                canonical_to_raw_per_sheet.get(sheet, {}).get(c, c)
                for c in cols_to_render
            ]
            rendered = _row_to_str(frames, sheet, row, raw_cols)
            parts.append(f"    - {sheet}:row_{row}   {rendered}")
        if v.rationale:
            parts.append(f"    rationale_step3: {v.rationale}")
        parts.append("")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def _parse_response(
    text: str,
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """Pull ``verifications`` array out of the LLM's JSON response.

    Returns ``(records, parse_error)``. ``parse_error`` is ``None``
    on success.
    """
    if not text:
        return [], "empty response"
    body = text.strip()
    fence = JSON_FENCE_RE.search(body)
    if fence:
        body = fence.group(1)
    obj_match = JSON_OBJECT_RE.search(body)
    if not obj_match:
        return [], f"no JSON object in response: {text[:200]!r}"
    try:
        obj = json.loads(obj_match.group(0))
    except json.JSONDecodeError as exc:
        return [], f"JSON decode failed: {exc}"
    if not isinstance(obj, dict):
        return [], "top-level JSON is not an object"
    records = obj.get("verifications")
    if not isinstance(records, list):
        return [], "'verifications' missing or not a list"
    return records, None


def _normalize_verdict(raw: Any) -> str:
    s = str(raw or "").strip().lower()
    if s in VERDICTS:
        return s
    # Permissive synonyms.
    if s in {"confirmed", "true", "yes"}:
        return "confirmed_inconsistency"
    if s in {"benign", "ok", "expected", "false_positive_benign"}:
        return "benign_variation"
    if s in {"false", "no", "false_positive", "spurious"}:
        return "not_a_violation"
    return "uncertain"


def _normalize_confidence(raw: Any) -> str:
    s = str(raw or "").strip().lower()
    if s in CONFIDENCES:
        return s
    return "medium"


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def _select_verifiable(
    violations: Sequence[MultiTableViolation],
) -> List[MultiTableViolation]:
    """Filter to the kinds we ask the LLM about.

    Header / dtype / FK violations are mechanical: we pass them
    through without an LLM call (verdict=``not_verified``).
    """
    return [v for v in violations if v.kind in VERIFIABLE_KINDS]


def _passthrough(
    v: MultiTableViolation,
    *,
    verdict: str = "not_verified",
    confidence: str = "medium",
    rationale: str = "Mechanical kind; not sent to verifier.",
) -> Verification:
    return Verification(
        violation_id=v.violation_id,
        workbook=v.workbook,
        group_id=v.group_id,
        kind=v.kind,
        verdict=verdict,
        confidence=confidence,
        rationale=rationale,
    )


def verify_workbook_violations(
    *,
    resolved: ResolvedWorkbook,
    violations: Sequence[MultiTableViolation],
    frames: Mapping[str, pd.DataFrame],
    llm: LLMClient,
    model: str,
    prompt_cfg: VerifyPromptConfig,
    batch_size: int = 16,
) -> List[Verification]:
    """Run step 4 verification for one workbook.

    Iterates over the workbook's confirmed groups, batches its
    verifiable violations (``attribute_conflict`` /
    ``duplicate_row``), and fires one LLM call per batch. Other
    kinds get a pass-through ``not_verified`` verdict.

    Multi-row violations share a ``violation_id``; this function
    deduplicates on that id so the LLM is asked once per logical
    finding (the per-(sheet, row) records still appear in the
    output verbatim, all sharing the same verdict).
    """
    out: List[Verification] = []

    # Index: violation_id -> deduped record (with all linked rows).
    by_id: Dict[str, MultiTableViolation] = {}
    by_id_kind_count: Dict[str, int] = {}
    for v in violations:
        if v.workbook != resolved.workbook:
            continue
        by_id_kind_count[v.violation_id] = by_id_kind_count.get(v.violation_id, 0) + 1
        if v.violation_id in by_id:
            continue
        by_id[v.violation_id] = v

    # Pass-through for unverifiable kinds.
    for vid, v in list(by_id.items()):
        if v.kind not in VERIFIABLE_KINDS:
            out.append(_passthrough(v))
            del by_id[vid]

    # Group remaining by group_id so each LLM call sees a coherent
    # set of violations sharing one canonical column mapping / PK.
    by_group: Dict[str, List[MultiTableViolation]] = {}
    for v in by_id.values():
        by_group.setdefault(v.group_id, []).append(v)

    group_lookup: Dict[str, ResolvedGroup] = {
        g.group_id: g for g in resolved.groups if g.confirmed
    }

    for group_id, vios_in_group in by_group.items():
        group = group_lookup.get(group_id)
        if group is None:
            # Compiler emitted a violation against an unconfirmed
            # group? Should not happen, but be defensive.
            for v in vios_in_group:
                out.append(_passthrough(
                    v, verdict="uncertain",
                    rationale="Group not confirmed at step 2.",
                ))
            continue

        # Stable order keeps prompts deterministic across runs.
        vios_in_group.sort(key=lambda v: v.violation_id)
        for start in range(0, len(vios_in_group), batch_size):
            batch = vios_in_group[start:start + batch_size]
            out.extend(
                _verify_batch(
                    workbook=resolved.workbook,
                    group=group,
                    batch=batch,
                    frames=frames,
                    llm=llm,
                    model=model,
                    prompt_cfg=prompt_cfg,
                )
            )
    return out


def _verify_batch(
    *,
    workbook: str,
    group: ResolvedGroup,
    batch: Sequence[MultiTableViolation],
    frames: Mapping[str, pd.DataFrame],
    llm: LLMClient,
    model: str,
    prompt_cfg: VerifyPromptConfig,
) -> List[Verification]:
    """Issue one LLM call for ``batch`` and parse the response."""
    column_mapping_block = _render_column_mapping(group)
    violations_block = _render_violation_block(
        violations=batch, group=group, frames=frames,
    )
    prompt = prompt_cfg.user_template.format(
        workbook=workbook,
        group_id=group.group_id,
        group_kind=group.kind,
        sheets=list(group.sheets),
        primary_key=list(group.primary_key) or "(none)",
        column_mapping_block=column_mapping_block,
        violations_block=violations_block,
    )

    response: LLMResponse = llm.query(
        model,
        prompt,
        reasoning_effort=prompt_cfg.reasoning_effort,
        reasoning_summary=prompt_cfg.reasoning_summary,
        system_prompt=prompt_cfg.system_prompt,
    )

    if not response.ok:
        return [
            Verification(
                violation_id=v.violation_id,
                workbook=workbook,
                group_id=group.group_id,
                kind=v.kind,
                verdict="uncertain",
                confidence="low",
                rationale="Verifier LLM call failed.",
                endpoint=response.endpoint,
                attempts=response.attempts,
                error=response.error,
            )
            for v in batch
        ]

    raw_records, parse_err = _parse_response(response.text)
    if parse_err:
        logger.warning(
            "verifier parse error for workbook=%s group=%s: %s",
            workbook, group.group_id, parse_err,
        )
        return [
            Verification(
                violation_id=v.violation_id,
                workbook=workbook,
                group_id=group.group_id,
                kind=v.kind,
                verdict="uncertain",
                confidence="low",
                rationale="Verifier response could not be parsed.",
                endpoint=response.endpoint,
                attempts=response.attempts,
                parse_error=parse_err,
            )
            for v in batch
        ]

    # Index the LLM's records by violation_id, then drop hallucinated
    # ids and fill in missing ones with `uncertain`.
    by_id: Dict[str, Dict[str, Any]] = {}
    for r in raw_records:
        if not isinstance(r, dict):
            continue
        vid = str(r.get("violation_id") or "").strip()
        if vid:
            by_id[vid] = r

    out: List[Verification] = []
    for v in batch:
        rec = by_id.get(v.violation_id)
        if rec is None:
            out.append(Verification(
                violation_id=v.violation_id,
                workbook=workbook,
                group_id=group.group_id,
                kind=v.kind,
                verdict="uncertain",
                confidence="low",
                rationale="Verifier omitted this violation_id.",
                endpoint=response.endpoint,
                attempts=response.attempts,
            ))
            continue
        out.append(Verification(
            violation_id=v.violation_id,
            workbook=workbook,
            group_id=group.group_id,
            kind=v.kind,
            verdict=_normalize_verdict(rec.get("verdict")),
            confidence=_normalize_confidence(rec.get("confidence")),
            rationale=str(rec.get("rationale") or "").strip(),
            endpoint=response.endpoint,
            attempts=response.attempts,
        ))
    return out


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def write_verifications_jsonl(
    verifications: Iterable[Verification],
    path: Path,
) -> int:
    n = 0
    with Path(path).open("w", encoding="utf-8") as fh:
        for v in verifications:
            fh.write(json.dumps(v.to_dict(), ensure_ascii=False) + "\n")
            n += 1
    return n


def summarize(verifications: Sequence[Verification]) -> Dict[str, Any]:
    """Roll up verifications into ``summary.json`` shape."""
    by_verdict: Dict[str, int] = {}
    by_kind: Dict[str, int] = {}
    by_kind_verdict: Dict[str, Dict[str, int]] = {}
    n_errored = 0
    n_parse_errored = 0
    for v in verifications:
        by_verdict[v.verdict] = by_verdict.get(v.verdict, 0) + 1
        by_kind[v.kind] = by_kind.get(v.kind, 0) + 1
        by_kind_verdict.setdefault(v.kind, {})[v.verdict] = (
            by_kind_verdict.setdefault(v.kind, {}).get(v.verdict, 0) + 1
        )
        if v.error:
            n_errored += 1
        if v.parse_error:
            n_parse_errored += 1
    return {
        "n_verifications": len(verifications),
        "by_verdict": by_verdict,
        "by_kind": by_kind,
        "by_kind_verdict": by_kind_verdict,
        "n_errored": n_errored,
        "n_parse_errored": n_parse_errored,
    }
