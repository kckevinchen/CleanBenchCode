"""Cross-sheet UNIONABLE / JOINABLE group resolver.

This module replaces the LLM ``check()`` codegen path for the
``inter_table_inconsistency`` error type. The flow is:

    1. Compute structural priors (``multi_table_priors``).
    2. Call the LLM ONCE per workbook with the priors and ask it
       to (a) confirm each candidate group, (b) provide a
       canonical column mapping, (c) name the primary_key.
       The prompt lives at ``prompts/confirm_union_join.yaml``.
    3. Return a typed :class:`ResolvedWorkbook` carrying the
       confirmed groups + mappings. Downstream
       :mod:`src.core.multi_table.violations` consumes this
       structure to emit the four violation kinds deterministically.

Why this split? The LLM is asked exactly one question (does this
structural candidate represent a real union/join, and what is the
canonical name for each variant column?). It is NOT asked to
write Python, enumerate constraints, or judge individual rows.
Compared to the ``_discover_table`` path this gives:

    * Lower false-positive rate (the LLM cannot hallucinate row-level
      rules; it can only confirm/reject pre-computed structural
      candidates).
    * No sandboxed code execution.
    * Explainable output: every confirmed group ships with a
      canonical_name -> [(sheet, column), ...] mapping that the
      report and the frontend can render directly.

The module is stateless; one ``ResolvedWorkbook`` per workbook,
returned synchronously from :func:`resolve_workbook`. The runner
parallelises across workbooks at the call site.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
import yaml

from ..jsonl_io import JSON_FENCE_RE, JSON_OBJECT_RE
from ..llm_client import LLMClient, LLMResponse
from .priors import (
    FKCandidate,
    MultiTablePriors,
    UnionableGroupCandidate,
    compute_table_priors,
    render_workbook_context,
)
from ..table_digest import TableDigest

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ColumnMappingEntry:
    """One (sheet, column) pair under a canonical column name."""

    sheet: str
    column: str

    def to_dict(self) -> Dict[str, str]:
        return {"sheet": self.sheet, "column": self.column}


@dataclass(frozen=True)
class JoinSpec:
    """Parent/child binding for a confirmed JOIN group."""

    parent_sheet: str
    parent_columns: Tuple[str, ...]
    child_sheet: str
    child_columns: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "parent_sheet": self.parent_sheet,
            "parent_columns": list(self.parent_columns),
            "child_sheet": self.child_sheet,
            "child_columns": list(self.child_columns),
        }


@dataclass(frozen=True)
class ResolvedGroup:
    """One LLM-confirmed cross-sheet group (union or join).

    ``column_mapping`` maps a canonical column name to the list of
    (sheet, column) variants. Partial mappings are allowed: if a
    canonical only appears on a subset of ``sheets``, the
    union-time projection fills the missing sheets with NaN.
    """

    workbook: str
    group_id: str
    kind: str   # "union" | "join"
    confirmed: bool
    rationale: str
    sheets: Tuple[str, ...]
    column_mapping: Mapping[str, Tuple[ColumnMappingEntry, ...]]
    primary_key: Tuple[str, ...]   # canonical names; may be empty
    join: Optional[JoinSpec] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "workbook": self.workbook,
            "group_id": self.group_id,
            "kind": self.kind,
            "confirmed": bool(self.confirmed),
            "rationale": self.rationale,
            "sheets": list(self.sheets),
            "column_mapping": {
                canonical: [e.to_dict() for e in entries]
                for canonical, entries in self.column_mapping.items()
            },
            "primary_key": list(self.primary_key),
            "join": self.join.to_dict() if self.join else None,
        }


@dataclass
class ResolvedWorkbook:
    """LLM-confirmed groups for one workbook.

    ``priors`` is preserved for the downstream violation compiler
    and for diagnostics. ``raw_response`` and ``parse_error`` aid
    debugging when the LLM emits malformed JSON.
    """

    workbook: str
    sheets: Tuple[str, ...]
    groups: List[ResolvedGroup] = field(default_factory=list)
    priors: Optional[MultiTablePriors] = None
    prompt: str = ""
    raw_response: str = ""
    reasoning_summary: str = ""
    endpoint: str = ""
    attempts: int = 0
    error: Optional[str] = None
    parse_error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "workbook": self.workbook,
            "sheets": list(self.sheets),
            "n_groups": len(self.groups),
            "n_confirmed": sum(1 for g in self.groups if g.confirmed),
            "groups": [g.to_dict() for g in self.groups],
            "error": self.error,
            "parse_error": self.parse_error,
            "endpoint": self.endpoint,
            "attempts": self.attempts,
        }


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConfirmPromptConfig:
    """Loaded prompt config for ``confirm_union_join.yaml``."""

    model: str
    reasoning_effort: str
    reasoning_summary: str
    system_prompt: str
    user_template: str


def load_confirm_prompt(path: Path) -> ConfirmPromptConfig:
    """Load the cross-sheet confirmation prompt YAML.

    The YAML uses single-brace ``{name}`` placeholders for str.format
    (NOT Jinja). Literal braces in the JSON skeleton are escaped
    with ``{{`` / ``}}`` per str.format conventions.
    """
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"confirm prompt must be a mapping: {path}")
    return ConfirmPromptConfig(
        model=str(raw.get("model") or "gpt-5.2"),
        reasoning_effort=str(raw.get("reasoning_effort") or "medium"),
        reasoning_summary=str(raw.get("reasoning_summary") or "detailed"),
        system_prompt=str(raw.get("system") or "").strip(),
        user_template=str(raw.get("user_template") or "").strip(),
    )


# ---------------------------------------------------------------------------
# Prompt rendering helpers
# ---------------------------------------------------------------------------


def _render_sheet_headers(
    sheets: Sequence[str],
    frames: Mapping[str, pd.DataFrame],
    *,
    max_cols: int = 40,
) -> str:
    parts: List[str] = []
    for s in sheets:
        df = frames.get(s)
        if df is None:
            continue
        cols = list(df.columns)[:max_cols]
        more = (
            f"  (+{len(df.columns) - len(cols)} more)"
            if len(df.columns) > len(cols) else ""
        )
        parts.append(f"  - {s!r} ({len(df)} rows): {[str(c) for c in cols]}{more}")
    return "\n".join(parts) or "  (no sheets)"


def _render_unionable_block(
    groups: Sequence[UnionableGroupCandidate],
) -> str:
    if not groups:
        return "  (none)"
    parts: List[str] = []
    for g in groups:
        parts.append(
            f"  group_id={g.group_id} sheets={list(g.sheets)} "
            f"avg_jaccard={g.avg_jaccard:.2f} "
            f"shared={g.n_shared_headers}/{g.n_canonical_headers}"
        )
        for sheet, headers in g.sheet_headers:
            parts.append(f"    - {sheet!r}: {[str(h) for h in headers]}")
    return "\n".join(parts)


def _render_joinable_block(
    fk_candidates: Sequence[FKCandidate],
) -> str:
    if not fk_candidates:
        return "  (none)"
    parts: List[str] = []
    for i, fk in enumerate(fk_candidates, start=1):
        parts.append(
            f"  group_id=j{i} kind=join "
            f"parent={fk.parent!r}.{fk.parent_column!r} "
            f"child={fk.child!r}.{fk.child_column!r} "
            f"coverage={fk.coverage:.2f} "
            f"normalized_coverage={fk.normalized_coverage:.2f} "
            f"orphans={len(fk.orphan_row_indices)} "
            f"step={fk.best_normalization_step}"
        )
    return "\n".join(parts)


def _render_row_samples(
    sheets: Sequence[str],
    frames: Mapping[str, pd.DataFrame],
    *,
    n_rows: int = 3,
) -> str:
    parts: List[str] = []
    for s in sheets:
        df = frames.get(s)
        if df is None or df.empty:
            continue
        parts.append(f"=== {s!r} ===")
        try:
            parts.append(df.head(n_rows).to_string(index=False))
        except Exception:  # noqa: BLE001
            parts.append("(failed to render)")
        parts.append("")
    return "\n".join(parts) or "(no samples)"


def _speculative_union_candidate(
    sheets: Sequence[str],
    frames: Mapping[str, pd.DataFrame],
) -> UnionableGroupCandidate:
    """Synthesise an all-sheets union candidate when priors find none.

    Used by the recall-first fallback in :func:`resolve_workbook`:
    when structural Jaccard / FK detectors emit no candidate but the
    workbook is small enough to be worth one LLM call, this helper
    builds a single ``UnionableGroupCandidate`` that names every
    sheet so the LLM can confirm or reject from row samples alone.
    Stats are filled in conservatively (avg_jaccard=0.0,
    n_shared=0); they exist for the audit trail and do not influence
    the LLM's decision.
    """
    sheets_t = tuple(sheets)
    sheet_headers: List[Tuple[str, Tuple[str, ...]]] = []
    canonical_union: set = set()
    for s in sheets_t:
        df = frames.get(s)
        if df is None:
            continue
        raw = tuple(str(c) for c in df.columns)
        sheet_headers.append((s, raw))
        canonical_union |= {c for c in raw}
    return UnionableGroupCandidate(
        group_id="u_spec",
        sheets=sheets_t,
        n_sheets=len(sheets_t),
        n_canonical_headers=len(canonical_union),
        n_shared_headers=0,
        avg_jaccard=0.0,
        sheet_headers=tuple(sheet_headers),
    )


# ---------------------------------------------------------------------------
# LLM response parsing
# ---------------------------------------------------------------------------


def _parse_response(text: str) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """Parse the LLM JSON response into a list of group dicts.

    Returns ``(groups, parse_error)``. ``parse_error`` is None on
    success; otherwise the groups list is whatever could be salvaged
    (typically empty).
    """
    if not text:
        return [], "empty response"
    body = text.strip()
    # Strip markdown fence if present.
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
    groups = obj.get("groups")
    if not isinstance(groups, list):
        return [], "top-level 'groups' missing or not a list"
    return groups, None


def _validate_group(
    raw: Mapping[str, Any],
    *,
    workbook: str,
    valid_sheets: set,
    sheet_columns: Mapping[str, set],
) -> Tuple[Optional[ResolvedGroup], Optional[str]]:
    """Sanity-check one LLM group dict and convert to ResolvedGroup.

    Returns ``(group, reason)``. ``group`` is None when the entry is
    structurally invalid; ``reason`` is a short explanation logged
    by the caller. confirmed=false groups still pass validation -
    they're returned so the audit trail is complete.
    """
    gid = str(raw.get("group_id") or "").strip()
    if not gid:
        return None, "missing group_id"
    kind = str(raw.get("kind") or "").strip().lower()
    if kind not in {"union", "join"}:
        return None, f"invalid kind={kind!r}"
    confirmed = bool(raw.get("confirmed"))
    rationale = str(raw.get("rationale") or "").strip()
    sheets_raw = raw.get("sheets") or []
    if not isinstance(sheets_raw, list):
        return None, "'sheets' is not a list"
    sheets = tuple(str(s) for s in sheets_raw if str(s) in valid_sheets)
    if not sheets:
        return None, "no recognised sheets"

    column_mapping: Dict[str, Tuple[ColumnMappingEntry, ...]] = {}
    if confirmed:
        cm_raw = raw.get("column_mapping") or {}
        if not isinstance(cm_raw, dict):
            return None, "'column_mapping' is not an object"
        for canonical, entries in cm_raw.items():
            if not isinstance(entries, list):
                continue
            cleaned: List[ColumnMappingEntry] = []
            for e in entries:
                if not isinstance(e, dict):
                    continue
                sh = str(e.get("sheet") or "").strip()
                col = str(e.get("column") or "").strip()
                if sh not in valid_sheets:
                    continue
                if col not in sheet_columns.get(sh, set()):
                    # Reject hallucinated column names.
                    continue
                cleaned.append(ColumnMappingEntry(sheet=sh, column=col))
            if cleaned:
                column_mapping[str(canonical)] = tuple(cleaned)

    pk_raw = raw.get("primary_key") or []
    if not isinstance(pk_raw, list):
        pk: Tuple[str, ...] = ()
    else:
        pk = tuple(
            str(k) for k in pk_raw
            if str(k) in column_mapping
        )

    join_spec: Optional[JoinSpec] = None
    if kind == "join" and confirmed:
        j_raw = raw.get("join") or {}
        if isinstance(j_raw, dict):
            ps = str(j_raw.get("parent_sheet") or "").strip()
            cs = str(j_raw.get("child_sheet") or "").strip()
            pcs = j_raw.get("parent_columns") or []
            ccs = j_raw.get("child_columns") or []
            if (
                ps in valid_sheets and cs in valid_sheets
                and isinstance(pcs, list) and isinstance(ccs, list)
            ):
                pcs_clean = tuple(
                    str(c) for c in pcs
                    if str(c) in sheet_columns.get(ps, set())
                )
                ccs_clean = tuple(
                    str(c) for c in ccs
                    if str(c) in sheet_columns.get(cs, set())
                )
                if pcs_clean and ccs_clean and len(pcs_clean) == len(ccs_clean):
                    join_spec = JoinSpec(
                        parent_sheet=ps,
                        parent_columns=pcs_clean,
                        child_sheet=cs,
                        child_columns=ccs_clean,
                    )

    return ResolvedGroup(
        workbook=workbook,
        group_id=gid,
        kind=kind,
        confirmed=confirmed,
        rationale=rationale,
        sheets=sheets,
        column_mapping=column_mapping,
        primary_key=pk,
        join=join_spec,
    ), None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def resolve_workbook(
    *,
    workbook: str,
    sheets: Sequence[str],
    frames: Mapping[str, pd.DataFrame],
    digests: Mapping[str, TableDigest],
    llm: LLMClient,
    model: str,
    prompt_cfg: ConfirmPromptConfig,
    fk_candidate_cap: int = 12,
    speculative_max_sheets: int = 8,
) -> ResolvedWorkbook:
    """Run priors -> LLM confirmation for one workbook.

    Returns an empty (no-error) ``ResolvedWorkbook`` when there are
    no candidate groups at all AND the workbook is too large for a
    speculative all-sheets fallback.

    Recall-first fallback: when the structural priors return no
    unionable group and no FK candidate, but the workbook has at
    least two sheets and at most ``speculative_max_sheets``, we
    synthesise ONE speculative union candidate spanning every
    sheet and let the LLM decide. This rescues workbooks where
    multi-line / parenthetical / acronym header drift collapses
    every Jaccard pair to zero, but the sheets are clearly siblings
    when the LLM looks at row samples (the user's framing: "show
    metadata and a few rows, ask the LLM whether they are
    join/unionable").
    """
    sheets_t = tuple(sheets)
    if len(sheets_t) < 2:
        return ResolvedWorkbook(
            workbook=workbook,
            sheets=sheets_t,
            error="workbook has <2 sheets",
        )

    priors = compute_table_priors(
        sheets=sheets_t,
        frames=frames,
        digests=digests,
        error_type="inter_table_inconsistency",
        workbook=workbook,
    )
    union_candidates = priors.unionable_groups
    fk_top = priors.fk_candidates[:fk_candidate_cap]
    speculative = False
    if not union_candidates and not fk_top:
        if len(sheets_t) <= speculative_max_sheets:
            union_candidates = (
                _speculative_union_candidate(sheets_t, frames),
            )
            speculative = True
            logger.debug(
                "Workbook %r: no structural priors -> emitting speculative "
                "all-sheets union candidate (%d sheets) for LLM judgment.",
                workbook, len(sheets_t),
            )
        else:
            return ResolvedWorkbook(
                workbook=workbook, sheets=sheets_t, priors=priors,
            )

    # Build prompt.
    sheet_headers_block = _render_sheet_headers(sheets_t, frames)
    union_block = _render_unionable_block(union_candidates)
    join_block = _render_joinable_block(fk_top)
    row_samples_block = _render_row_samples(sheets_t, frames)

    prompt = prompt_cfg.user_template.format(
        workbook=workbook,
        sheet_headers_block=sheet_headers_block,
        unionable_groups_block=union_block,
        joinable_pairs_block=join_block,
        row_samples_block=row_samples_block,
    )

    response: LLMResponse = llm.query(
        model,
        prompt,
        reasoning_effort=prompt_cfg.reasoning_effort,
        reasoning_summary=prompt_cfg.reasoning_summary,
        system_prompt=prompt_cfg.system_prompt,
    )
    out = ResolvedWorkbook(
        workbook=workbook,
        sheets=sheets_t,
        priors=priors,
        prompt=prompt,
        raw_response=response.text,
        reasoning_summary=response.reasoning_summary,
        endpoint=response.endpoint,
        attempts=response.attempts,
        error=response.error,
    )
    if not response.ok:
        return out

    # Parse + validate.
    valid_sheets = set(sheets_t)
    sheet_columns = {
        s: set(str(c) for c in frames[s].columns)
        for s in sheets_t if s in frames
    }
    raw_groups, parse_err = _parse_response(response.text)
    if parse_err is not None:
        out.parse_error = parse_err
        return out
    seen_ids: set = set()
    for raw in raw_groups:
        if not isinstance(raw, dict):
            continue
        group, reason = _validate_group(
            raw, workbook=workbook,
            valid_sheets=valid_sheets,
            sheet_columns=sheet_columns,
        )
        if group is None:
            logger.debug(
                "Dropping malformed group from workbook=%s: %s; raw=%r",
                workbook, reason, raw,
            )
            continue
        if group.group_id in seen_ids:
            continue
        seen_ids.add(group.group_id)
        out.groups.append(group)

    return out


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def write_resolved_workbooks_jsonl(
    resolved: Sequence[ResolvedWorkbook],
    path: Path,
) -> int:
    """Persist one JSON line per workbook. Returns line count."""
    with Path(path).open("w", encoding="utf-8") as fh:
        for r in resolved:
            fh.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
    return len(resolved)


def _resolved_group_from_dict(d: Mapping[str, Any]) -> ResolvedGroup:
    """Round-trip a single ``ResolvedGroup`` dict back to the dataclass.

    Used by :func:`read_resolved_workbooks_jsonl` so step-3/step-4
    consumers can re-load a discovery run's
    ``multi_table_groups.jsonl`` without recomputing priors.
    """
    cm_raw: Mapping[str, Any] = d.get("column_mapping") or {}
    column_mapping: Dict[str, Tuple[ColumnMappingEntry, ...]] = {}
    for canonical, entries in cm_raw.items():
        mapped: List[ColumnMappingEntry] = []
        for e in entries or []:
            if isinstance(e, Mapping):
                mapped.append(
                    ColumnMappingEntry(
                        sheet=str(e.get("sheet", "")),
                        column=str(e.get("column", "")),
                    )
                )
        column_mapping[str(canonical)] = tuple(mapped)
    join_raw = d.get("join")
    join: Optional[JoinSpec] = None
    if isinstance(join_raw, Mapping):
        join = JoinSpec(
            parent_sheet=str(join_raw.get("parent_sheet", "")),
            parent_columns=tuple(
                str(c) for c in (join_raw.get("parent_columns") or [])
            ),
            child_sheet=str(join_raw.get("child_sheet", "")),
            child_columns=tuple(
                str(c) for c in (join_raw.get("child_columns") or [])
            ),
        )
    return ResolvedGroup(
        workbook=str(d.get("workbook", "")),
        group_id=str(d.get("group_id", "")),
        kind=str(d.get("kind", "union")),
        confirmed=bool(d.get("confirmed", False)),
        rationale=str(d.get("rationale", "")),
        sheets=tuple(str(s) for s in (d.get("sheets") or [])),
        column_mapping=column_mapping,
        primary_key=tuple(str(c) for c in (d.get("primary_key") or [])),
        join=join,
    )


def read_resolved_workbooks_jsonl(path: Path) -> List[ResolvedWorkbook]:
    """Round-trip ``multi_table_groups.jsonl`` back to dataclasses.

    Lossy: ``priors`` / ``prompt`` / ``raw_response`` /
    ``reasoning_summary`` are not persisted by
    :func:`write_resolved_workbooks_jsonl`, so they come back empty.
    Step 3 (semantic detector / compiler) and step 4 (verifier) only
    consume ``workbook``, ``sheets``, ``groups`` and ``error``, so
    the round-trip is sufficient for re-running the detection phase
    against groups confirmed in a previous run.
    """
    out: List[ResolvedWorkbook] = []
    p = Path(path)
    if not p.is_file():
        return out
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(d, Mapping):
                continue
            groups_raw = d.get("groups") or []
            groups = [
                _resolved_group_from_dict(g)
                for g in groups_raw
                if isinstance(g, Mapping)
            ]
            out.append(
                ResolvedWorkbook(
                    workbook=str(d.get("workbook", "")),
                    sheets=tuple(str(s) for s in (d.get("sheets") or [])),
                    groups=groups,
                    error=d.get("error"),
                    parse_error=d.get("parse_error"),
                    endpoint=str(d.get("endpoint", "")),
                    attempts=int(d.get("attempts", 0) or 0),
                )
            )
    return out
