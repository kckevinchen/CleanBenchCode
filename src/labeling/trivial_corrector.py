"""LLM cell-correction for trivially-tagged labels marked PARTIAL.

When the operator converts trivially-tagged cells (whitespace / punctuation)
to ``partial`` -- meaning "this IS a real error, but the mechanical fix is off"
-- they can ask an LLM to generate a better ``corrected_value`` for each cell.

This module is the framework-free core: it collects the eligible labels, builds
a focused single-cell correction prompt, calls the shared :class:`LLMClient`
(with bounded concurrency), parses the result, and writes the improved fix back
as a new ``partial`` label. The async job wrapper + HTTP endpoints live in
``routers/trivial_corrections.py``; keeping the logic here makes it unit-testable
with a stub LLM (no live service).

Provenance: a corrected label's note gains the literal marker ``[LLM-fix]`` so
re-runs skip it (idempotent) and the source of the correction is auditable.
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from ..core.llm_client import LLMClient
from .schema import Label, LabelStatus, LabelTarget

# Marker appended to a label's note once an LLM fix has been generated for it,
# so ``collect_partial_trivial_targets`` skips it on a re-run (idempotent).
LLM_FIX_MARKER = "[LLM-fix]"

DEFAULT_MODEL = "gpt-5.2"
DEFAULT_REASONING_EFFORT = "medium"
DEFAULT_CONCURRENCY = 8
# Cap on distinct column sample values included in a prompt for context.
_COLUMN_SAMPLE_CAP = 20


SYSTEM_PROMPT = (
    "You are a meticulous data-cleaning assistant. You are given ONE cell "
    "value from a CSV column that an automated check flagged as containing a "
    "likely error, along with the other cells in its row and a sample of other "
    "values from the same column for context.\n\n"
    "Produce the CORRECTED value for this one cell:\n"
    "1. Fix typos, spacing, punctuation, casing and formatting so the value is "
    "internally correct and consistent with the column's conventions.\n"
    "2. PRESERVE the real meaning / identity of the value. Never invent facts, "
    "add information that is not implied by the value, or swap it for a "
    "different entity.\n"
    "3. If the value is already correct, return it unchanged.\n"
    "4. Return ONLY the cell's value -- not an explanation, not the whole row.\n\n"
    'Output ONLY a JSON object of the form '
    '{"corrected_value": "<the corrected cell value>"}.'
)


@dataclass
class CellCorrectionTarget:
    """One (label, column) cell eligible for an LLM correction."""

    target_id: str
    target: str          # "cell" | "cluster"
    source: str
    row: int
    column: str
    original_value: str


@dataclass
class CorrectionResult:
    """Outcome of one LLM correction attempt for a single cell."""

    target_id: str
    target: str
    source: str
    row: int
    column: str
    original_value: str
    corrected_value: Optional[str]   # None on error / no parse
    changed: bool
    error: Optional[str] = None
    raw: str = ""


# ---------------------------------------------------------------------------
# Target collection
# ---------------------------------------------------------------------------


def _has_trivial_marker(note: Optional[str]) -> bool:
    return "TRIVIAL" in (note or "").upper()


def _already_llm_fixed(note: Optional[str]) -> bool:
    return LLM_FIX_MARKER.lower() in (note or "").lower()


def collect_partial_trivial_targets(
    ctx: Any, *, source_filter: Optional[str] = None,
) -> List[CellCorrectionTarget]:
    """Return one target per (label, column) that is a TRIVIAL-noted ``partial``
    label not yet carrying the ``[LLM-fix]`` marker.

    CLUSTER labels contribute one target per column; a column missing from the
    live frame (deleted / renamed) is skipped. The cell's live value is read via
    ``ctx.load_frame`` so the prompt sees the TRUE current value.
    """
    cluster_by_id: Dict[str, Dict[str, Any]] = {
        c["cluster_id"]: c
        for c in getattr(ctx, "clusters", []) or []
        if isinstance(c.get("cluster_id"), str)
    }
    latest = ctx.store.latest_by_target()
    frame_cache: Dict[str, Any] = {}

    def _frame(src: str):
        if src not in frame_cache:
            try:
                frame_cache[src] = ctx.load_frame(src)
            except (FileNotFoundError, KeyError):
                frame_cache[src] = None
        return frame_cache[src]

    targets: List[CellCorrectionTarget] = []
    items = sorted(
        latest.items(),
        key=lambda kv: (
            kv[1].source or "",
            kv[1].row if kv[1].row is not None else -1,
            kv[1].column or "",
            kv[0],
        ),
    )
    for tid, lab in items:
        if lab.status != LabelStatus.PARTIAL:
            continue
        if not _has_trivial_marker(lab.note) or _already_llm_fixed(lab.note):
            continue
        if source_filter is not None and (lab.source or "") != source_filter:
            continue

        src = lab.source
        if lab.target == LabelTarget.CELL:
            row_idx = lab.row
            cols = [lab.column] if lab.column else []
        elif lab.target == LabelTarget.CLUSTER:
            rec = cluster_by_id.get(lab.target_id)
            row_idx = lab.row if lab.row is not None else (
                rec.get("row") if rec else None
            )
            cols = [
                c for c in (lab.columns or (rec.get("columns") if rec else []) or [])
                if isinstance(c, str)
            ]
        else:
            continue
        if not src or row_idx is None or not cols:
            continue
        frame = _frame(src)
        if frame is None:
            continue
        try:
            row_int = int(row_idx)
        except (TypeError, ValueError):
            continue
        if not (0 <= row_int < len(frame)):
            continue

        for col in cols:
            if col not in frame.columns:
                continue
            v = frame.iat[row_int, frame.columns.get_loc(col)]
            original = "" if v is None else str(v)
            targets.append(CellCorrectionTarget(
                target_id=tid,
                target=lab.target.value,
                source=src,
                row=row_int,
                column=col,
                original_value=original,
            ))
    return targets


def collect_fn_targets(
    ctx: Any, *, source_filter: Optional[str] = None,
    skip_already_fixed: bool = True,
) -> List[CellCorrectionTarget]:
    """Return one target per operator-marked false-negative (FN) cell.

    FN labels are always single-cell (``target=CELL``). Unlike
    :func:`collect_partial_trivial_targets` this is NOT gated on a ``TRIVIAL``
    note -- every false negative the operator (or an auto-tagger) recorded is
    eligible for a suggested fix, whether or not it already carries a
    deterministic ``corrected_value``. ``skip_already_fixed`` (default True)
    drops labels whose note already carries the ``[LLM-fix]`` marker so a
    re-run is idempotent; pass ``False`` to force regeneration. The cell's
    live value is read via ``ctx.load_frame`` so the prompt sees the TRUE
    current value (after any post-run dataset normalization).
    """
    latest = ctx.store.latest_by_target()
    frame_cache: Dict[str, Any] = {}

    def _frame(src: str):
        if src not in frame_cache:
            try:
                frame_cache[src] = ctx.load_frame(src)
            except (FileNotFoundError, KeyError):
                frame_cache[src] = None
        return frame_cache[src]

    targets: List[CellCorrectionTarget] = []
    items = sorted(
        latest.items(),
        key=lambda kv: (
            kv[1].source or "",
            kv[1].row if kv[1].row is not None else -1,
            kv[1].column or "",
            kv[0],
        ),
    )
    for tid, lab in items:
        if lab.target != LabelTarget.CELL or lab.status != LabelStatus.FN:
            continue
        if skip_already_fixed and _already_llm_fixed(lab.note):
            continue
        if source_filter is not None and (lab.source or "") != source_filter:
            continue
        src = lab.source
        if not src or lab.row is None or not lab.column:
            continue
        frame = _frame(src)
        if frame is None:
            continue
        try:
            row_int = int(lab.row)
        except (TypeError, ValueError):
            continue
        if not (0 <= row_int < len(frame)) or lab.column not in frame.columns:
            continue
        v = frame.iat[row_int, frame.columns.get_loc(lab.column)]
        original = "" if v is None else str(v)
        targets.append(CellCorrectionTarget(
            target_id=tid,
            target=lab.target.value,
            source=src,
            row=row_int,
            column=lab.column,
            original_value=original,
        ))
    return targets


# ---------------------------------------------------------------------------
# Prompt + parse
# ---------------------------------------------------------------------------


def _column_samples(frame: Any, column: str, *, exclude: str, cap: int) -> List[str]:
    """Up to ``cap`` distinct non-empty values from ``column`` for context."""
    out: List[str] = []
    seen = {exclude}
    try:
        series = frame.iloc[:, frame.columns.get_loc(column)].tolist()
    except (KeyError, IndexError, ValueError):
        return out
    for v in series:
        s = "" if v is None else str(v)
        if not s.strip() or s in seen:
            continue
        seen.add(s)
        out.append(s)
        if len(out) >= cap:
            break
    return out


def build_cell_prompt(
    *, column: str, original_value: str,
    row_values: Optional[Dict[str, str]] = None,
    column_samples: Optional[List[str]] = None,
) -> str:
    """Build the single-cell correction prompt."""
    lines: List[str] = [
        f"Column: {column}",
        f"Original cell value: {original_value!r}",
    ]
    if row_values:
        lines.append("")
        lines.append("Other cells in this row:")
        for k, v in row_values.items():
            if k == column:
                continue
            lines.append(f"  {k}: {v!r}")
    if column_samples:
        lines.append("")
        lines.append(f"Sample of other values in the '{column}' column:")
        for s in column_samples:
            lines.append(f"  - {s!r}")
    lines.append("")
    lines.append(
        'Return the corrected value for the cell as JSON: '
        '{"corrected_value": "..."}'
    )
    return "\n".join(lines)


_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def parse_corrected_value(raw: str) -> Optional[str]:
    """Extract the corrected value from an LLM response.

    Accepts a fenced or bare ``{"corrected_value": "..."}`` object (the
    contracted shape), a bare JSON string, or -- as a last resort -- the
    trimmed text. Returns ``None`` when nothing usable is present.
    """
    if not raw or not raw.strip():
        return None
    text = _FENCE_RE.sub("", raw.strip()).strip()

    # Preferred: a JSON object with the contracted key.
    for candidate in (text, _first_json_object(text)):
        if not candidate:
            continue
        try:
            obj = json.loads(candidate)
        except (ValueError, TypeError):
            continue
        if isinstance(obj, dict):
            for key in ("corrected_value", "value", "corrected"):
                if key in obj and obj[key] is not None:
                    return str(obj[key])
        if isinstance(obj, str):
            return obj
    return None


def _first_json_object(text: str) -> Optional[str]:
    """Return the first ``{...}`` balanced-brace substring, or None."""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


# ---------------------------------------------------------------------------
# Generation (LLM dispatch) + write-back
# ---------------------------------------------------------------------------


def _correct_one(
    target: CellCorrectionTarget, *, llm_client: LLMClient, model: str,
    reasoning_effort: str, frame: Any,
) -> CorrectionResult:
    row_values: Dict[str, str] = {}
    if frame is not None and 0 <= target.row < len(frame):
        for col_obj in frame.columns:
            cv = frame.iat[target.row, frame.columns.get_loc(col_obj)]
            row_values[str(col_obj)] = "" if cv is None else str(cv)
    samples = (
        _column_samples(frame, target.column, exclude=target.original_value,
                        cap=_COLUMN_SAMPLE_CAP)
        if frame is not None else []
    )
    prompt = build_cell_prompt(
        column=target.column, original_value=target.original_value,
        row_values=row_values, column_samples=samples,
    )
    try:
        resp = llm_client.query(
            model, prompt,
            reasoning_effort=reasoning_effort,
            reasoning_summary="auto",
            system_prompt=SYSTEM_PROMPT,
        )
    except Exception as exc:  # noqa: BLE001
        return CorrectionResult(
            target_id=target.target_id, target=target.target,
            source=target.source, row=target.row, column=target.column,
            original_value=target.original_value, corrected_value=None,
            changed=False, error=f"call_failed: {exc}",
        )
    raw = getattr(resp, "text", "") or ""
    fixed = parse_corrected_value(raw)
    if fixed is None:
        return CorrectionResult(
            target_id=target.target_id, target=target.target,
            source=target.source, row=target.row, column=target.column,
            original_value=target.original_value, corrected_value=None,
            changed=False, error="parse_failed", raw=raw,
        )
    return CorrectionResult(
        target_id=target.target_id, target=target.target,
        source=target.source, row=target.row, column=target.column,
        original_value=target.original_value, corrected_value=fixed,
        changed=(fixed != target.original_value), raw=raw,
    )


def generate_corrections(
    ctx: Any, targets: List[CellCorrectionTarget], *,
    llm_client: LLMClient, model: str = DEFAULT_MODEL,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    concurrency: int = DEFAULT_CONCURRENCY,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> List[CorrectionResult]:
    """Call the LLM once per target (bounded concurrency) and return results.

    Pure with respect to the label store -- writing is :func:`write_corrections`
    -- so a dry run / preview can reuse this without mutating state.
    """
    total = len(targets)
    if total == 0:
        return []
    frame_cache: Dict[str, Any] = {}

    def _frame(src: str):
        if src not in frame_cache:
            try:
                frame_cache[src] = ctx.load_frame(src)
            except (FileNotFoundError, KeyError):
                frame_cache[src] = None
        return frame_cache[src]

    results: List[CorrectionResult] = []
    done = 0
    workers = max(1, min(concurrency, total))
    with ThreadPoolExecutor(max_workers=workers,
                            thread_name_prefix="trivial-fix") as ex:
        futs = {
            ex.submit(
                _correct_one, t, llm_client=llm_client, model=model,
                reasoning_effort=reasoning_effort, frame=_frame(t.source),
            ): t
            for t in targets
        }
        for fut in as_completed(futs):
            results.append(fut.result())
            done += 1
            if progress_cb is not None:
                progress_cb(done, total)
    return results


def write_corrections(
    ctx: Any, results: List[CorrectionResult], *, annotator: str,
) -> Dict[str, int]:
    """Persist improved fixes as new labels (one per target_id).

    Each successfully-corrected label is re-appended with its ``corrected_value``
    (single column) / ``corrected_values`` (multi column) updated to the LLM
    output and ``[LLM-fix]`` appended to its note. The label's EXISTING status
    is preserved (an FN cell stays ``fn``; a PARTIAL stays ``partial``) so this
    is shared by both the partial-correction and FN-fix flows. Failed / unparsed
    targets are left untouched so a re-run can retry them. Returns counts.
    """
    latest = ctx.store.latest_by_target()
    # Group successful results by target_id (a multi-column cluster yields one
    # result per column, but a single Label carries all of them).
    by_target: Dict[str, List[CorrectionResult]] = {}
    failed = 0
    unchanged = 0
    for r in results:
        if r.error is not None or r.corrected_value is None:
            failed += 1
            continue
        if not r.changed:
            unchanged += 1
            # Still record so the label is marked [LLM-fix] and not reprocessed.
        by_target.setdefault(r.target_id, []).append(r)

    batch: List[Label] = []
    corrected_cells = 0
    for tid, rs in by_target.items():
        lab = latest.get(tid)
        if lab is None:
            continue
        new_corrected_values = dict(lab.corrected_values) if lab.corrected_values else {}
        primary_value = lab.corrected_value
        for r in rs:
            new_corrected_values[r.column] = r.corrected_value
            if r.changed:
                corrected_cells += 1
            if r.column == (lab.column or r.column):
                primary_value = r.corrected_value
        # Single-column label: keep the scalar corrected_value in sync.
        if lab.column and lab.column in new_corrected_values:
            primary_value = new_corrected_values[lab.column]
        note = (lab.note or "").strip()
        if LLM_FIX_MARKER.lower() not in note.lower():
            note = (note + " " + LLM_FIX_MARKER).strip()
        batch.append(Label(
            target=lab.target,
            target_id=lab.target_id,
            source=lab.source,
            row=lab.row,
            column=lab.column,
            columns=list(lab.columns) if lab.columns else [],
            status=lab.status,
            corrected_value=primary_value,
            corrected_values=new_corrected_values,
            confirmed_canonical=lab.confirmed_canonical,
            value_corrections=(
                dict(lab.value_corrections) if lab.value_corrections else {}
            ),
            note=note,
            annotator=annotator,
        ))
    if batch:
        ctx.store.append_many(batch)
    return {
        "labels_written": len(batch),
        "cells_corrected": corrected_cells,
        "unchanged": unchanged,
        "failed": failed,
    }
