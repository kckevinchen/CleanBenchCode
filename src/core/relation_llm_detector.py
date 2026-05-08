"""LLM-based row-level violation detector for non-code-verifiable
relations.

The validation pass (:mod:`.relation_validator`) lets the LLM refuse
to write a ``check(df)`` for rules that depend on world-knowledge,
free-text semantics, or external context. Those relations carry
``verifiable=False`` and the deterministic compiler in
:mod:`.relation_violations` ignores them. This module is the LLM
fallback for that subset.

Strategy: **one call per (relation, source)**, passing the full
table (with row indices) and asking the model to enumerate violating
rows. This is much cheaper than the per-row chunk detector — a
30-row table costs 1 call here vs ~30 calls in chunk detection — at
the price of a context-window cap. For tables that exceed the cap
we shrink ``max_rows_per_call`` and surface a warning rather than
silently chunking, because chunking destroys the LLM's ability to
spot inconsistencies that span rows (FD-style violations need the
full table to be obvious).

Output is wrapped in ``RelationViolation`` records (same shape as
the code-verifier compiler) so the downstream ``cluster_issues``
pipeline does not care which path produced a finding.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import pandas as pd
import yaml

from .jsonl_io import extract_json_payloads as _extract_json_payloads
from .llm_client import LLMClient
from .relation_violations import RelationViolation, _full_row_values, _row_values

logger = logging.getLogger(__name__)


def load_detect_prompt(path: Path) -> Dict[str, Any]:
    """Load ``prompts/detect_relation_violation.yaml``."""
    with Path(path).open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"detect-relation-violation prompt yaml must be a mapping: {path}")
    return raw


def _render_table_md(df: pd.DataFrame, *, max_rows: int) -> str:
    """Render the frame as a Markdown table with a leading ``__row__``
    column carrying the integer position the LLM should cite.

    The renderer uses the same conventions as elsewhere in the project
    (NA-safe stringification; truncation to ``max_rows`` rows). Rows
    beyond the cap are dropped — the caller is responsible for
    deciding whether that is acceptable for a given relation.
    """
    if df.empty:
        return "(empty table)"
    capped = df.head(max_rows)
    work = capped.astype(object).copy()
    work.insert(0, "__row__", list(work.index))
    # ``to_markdown`` requires ``tabulate``; fall back to CSV when it
    # is not installed so the prompt still renders in slim
    # environments.
    try:
        return work.to_markdown(index=False)
    except Exception:  # noqa: BLE001
        return work.to_csv(index=False)


def _build_user_prompt(
    template: str,
    *,
    relation: Mapping[str, Any],
    refusal_reason: str,
    table_md: str,
    n_rows: int,
) -> str:
    return template.format(
        source=relation.get("source", "") or "",
        error_type=relation.get("error_type", "") or "",
        kind=relation.get("kind", "") or "",
        columns=", ".join(relation.get("columns") or []) or "(unspecified)",
        predicate=relation.get("predicate", "") or "(no predicate)",
        rationale=relation.get("rationale", "") or "(no rationale)",
        confidence=relation.get("confidence", "") or "unknown",
        refusal_reason=refusal_reason or "(none recorded)",
        table_md=table_md,
        n_rows=n_rows,
    )


def _parse_violations_response(text: str) -> Dict[str, Any]:
    """Parse the LLM response into ``{applicable, reason, violations}``.

    The body may include surrounding markdown / fences (the
    instructions ban them but real models occasionally still emit
    them). Falls back to a JSON-payload scan over the original text.
    """
    if not text:
        raise ValueError("empty LLM response")
    body = text.strip()
    if body.startswith("```"):
        body = body.strip("`")
        nl = body.find("\n")
        if nl > 0 and body[:nl].strip().isalpha():
            body = body[nl + 1:]
    obj: Optional[Dict[str, Any]] = None
    for candidate in (body.strip(), *_extract_json_payloads(text)):
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            obj = parsed
            break
    if obj is None:
        raise ValueError("no JSON object found in LLM response")
    # Tolerate alternative spellings; downstream only needs three
    # canonical fields.
    applicable = bool(obj.get("applicable", True))
    reason = str(obj.get("reason") or "")
    raw_violations = obj.get("violations") or []
    if not isinstance(raw_violations, list):
        raw_violations = []
    violations: List[Dict[str, Any]] = []
    for v in raw_violations:
        if not isinstance(v, Mapping):
            continue
        row = v.get("row")
        try:
            row_int = int(row)
        except (TypeError, ValueError):
            continue
        violations.append({
            "row": row_int,
            "reason": str(v.get("reason") or "").strip(),
        })
    return {"applicable": applicable, "reason": reason, "violations": violations}


def detect_violations_llm(
    relation: Mapping[str, Any],
    df: pd.DataFrame,
    *,
    refusal_reason: str,
    llm: LLMClient,
    model: str,
    prompt_cfg: Mapping[str, Any],
    max_rows_per_call: int = 300,
    max_violations_per_rule: int = 50,
) -> List[RelationViolation]:
    """Run one whole-table LLM call against ``relation`` over ``df``.

    The relation must reference at least one column that exists in
    ``df`` — otherwise we short-circuit with an empty result rather
    than wasting an LLM call on a guaranteed refusal.
    """
    columns = tuple(relation.get("columns") or [])
    if columns and not any(c in df.columns for c in columns):
        logger.info(
            "Skipping LLM detection for relation on %s: no referenced columns present",
            relation.get("source", "?"),
        )
        return []
    if df.empty:
        return []
    if len(df) > max_rows_per_call:
        logger.warning(
            "Table %s has %d rows; capping to %d for whole-table LLM scan. "
            "Some violations beyond the cap will not be surfaced.",
            relation.get("source", "?"), len(df), max_rows_per_call,
        )

    table_md = _render_table_md(df, max_rows=max_rows_per_call)
    user_prompt = _build_user_prompt(
        prompt_cfg.get("user_template") or "",
        relation=relation,
        refusal_reason=refusal_reason,
        table_md=table_md,
        n_rows=min(int(len(df)), max_rows_per_call),
    )
    system_prompt = prompt_cfg.get("system_prompt") or ""
    reasoning_effort = prompt_cfg.get("reasoning_effort") or "medium"

    try:
        resp = llm.query(
            model=model,
            prompt=user_prompt,
            reasoning_effort=reasoning_effort,
            system_prompt=system_prompt,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "LLM detection call failed for relation %s on %s: %s",
            relation.get("species_key", "?"),
            relation.get("source", "?"),
            exc,
        )
        return []

    try:
        decision = _parse_violations_response(resp.text)
    except ValueError as exc:
        logger.warning(
            "LLM detection parse failed for relation %s on %s: %s",
            relation.get("species_key", "?"),
            relation.get("source", "?"),
            exc,
        )
        return []

    if not decision.get("applicable"):
        logger.info(
            "Relation %s declared non-applicable on %s by detector: %s",
            relation.get("species_key", "?"),
            relation.get("source", "?"),
            decision.get("reason"),
        )
        return []

    raw_hits = decision.get("violations") or []
    if len(raw_hits) > max_violations_per_rule:
        logger.warning(
            "LLM returned %d violations for relation %s on %s; capping to %d. "
            "A high count usually indicates the rule is wrong, not that the "
            "data is broken.",
            len(raw_hits),
            relation.get("species_key", "?"),
            relation.get("source", "?"),
            max_violations_per_rule,
        )
        raw_hits = raw_hits[:max_violations_per_rule]

    species_key = str(relation.get("species_key") or "")
    source = str(relation.get("source") or "")
    error_type = str(relation.get("error_type") or "")
    error_type_name = str(relation.get("error_type_name") or error_type)
    error_type_category = str(relation.get("error_type_category") or "")
    kind = str(relation.get("kind") or "")
    predicate = str(relation.get("predicate") or "")
    rationale = str(relation.get("rationale") or "")
    confidence = str(relation.get("confidence") or "unknown").lower()

    out: List[RelationViolation] = []
    for hit in raw_hits:
        row = hit.get("row")
        reason = hit.get("reason") or ""
        if row is None or row < 0 or row >= len(df):
            continue
        out.append(RelationViolation(
            violation_id=f"rel__{species_key}#{source}#r{row:05d}",
            relation_species_key=species_key,
            source=source,
            error_type=error_type,
            error_type_name=error_type_name,
            error_type_category=error_type_category,
            kind=kind,
            row=int(row),
            columns=columns,
            row_values=_row_values(df, int(row), columns),
            full_row_values=_full_row_values(df, int(row)),
            predicate=predicate,
            rationale=rationale,
            rule_confidence=confidence,
            # Pass-rate is unknown for LLM-detected violations: we did
            # not run a deterministic check. Encode that as 0.0 so it
            # is visually distinct from the code-verifier path (which
            # only emits when pass_rate >= 0.85).
            rule_pass_rate=0.0,
            detection_method="llm_table_scan",
            peer_rows=[],
            llm_reason=str(reason),
        ))
    return out


def detect_violations_for_relations(
    relations: Sequence[Mapping[str, Any]],
    verifications: Mapping[str, Mapping[str, Any]],
    source_frames: Mapping[str, pd.DataFrame],
    *,
    llm: LLMClient,
    model: str,
    prompt_cfg: Mapping[str, Any],
    max_rows_per_call: int = 300,
    max_violations_per_rule: int = 50,
) -> List[RelationViolation]:
    """Walk every relation whose verification carries ``verifiable=False``
    and run :func:`detect_violations_llm` against it.

    Relations that the validator never reached, that are multi-table,
    or that produced a verifier error are skipped (the latter usually
    means the LLM returned malformed code; we'd rather not double-down
    on it with another LLM call).
    """
    out: List[RelationViolation] = []
    for rel in relations:
        species_key = str(rel.get("species_key") or "")
        v = verifications.get(species_key)
        if not v:
            continue
        if v.get("verifiable"):
            # Code-verifiable rules go through the deterministic
            # compiler; do not double-emit.
            continue
        if v.get("error"):
            # The validator hit a runtime/parse error; we have no
            # signal that the rule is real. Skip.
            continue
        if rel.get("tables"):
            # Multi-table fallback would need cross-sheet rendering
            # and is handled by the multi_table semantic detector.
            continue
        source = str(rel.get("source") or "")
        df = source_frames.get(source)
        if df is None:
            continue
        out.extend(detect_violations_llm(
            rel, df,
            refusal_reason=str(v.get("reason") or ""),
            llm=llm,
            model=model,
            prompt_cfg=prompt_cfg,
            max_rows_per_call=max_rows_per_call,
            max_violations_per_rule=max_violations_per_rule,
        ))
    logger.info(
        "LLM table-scan produced %d violation(s) across %d non-code-verifiable rule(s)",
        len(out),
        sum(1 for r in relations if not verifications.get(r.get("species_key", ""), {}).get("verifiable")),
    )
    return out
