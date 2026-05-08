"""Compile per-row violations from validated relations.

Discovery + validation produce ``relations*.jsonl`` (predicates) and
``relation_verifications.jsonl`` (LLM-written ``check()`` code +
empirical pass-rate). What this module adds is the missing third step:
turn each validated relation into the **specific rows** that violate
it, so the labelling UI can surface them as cluster items.

The compiler is deliberately conservative. The expectation is that a
*well discovered* relation should hold on the vast majority of rows
(``pass_rate >= τ``); a relation with a low pass-rate is more likely
a hallucinated rule than a real constraint, so we drop it instead of
flooding the queue with spurious findings. Likewise, a single
``check()`` flagging more than a few hundred rows is almost always a
sign that the rule itself is wrong (or the LLM mis-coded the
predicate), so we cap with ``max_violations_per_rule``.

Two detection methods are supported:

* ``code_verifier`` — re-execute the LLM-written ``check()`` against
  the source frame and emit one violation per ``False`` row. This
  uses the SAME sandbox the validator used (no imports, no I/O).
* ``llm_table_scan`` — for relations the LLM declared
  ``verifiable=False`` (free-text rules, world-knowledge rules,
  etc.), the caller can supply violations produced by
  :mod:`.relation_llm_detector` (one whole-table LLM call per
  relation per source) and the conversion to clusters is identical.

Multi-table relations are out of scope here: the cross-sheet pipeline
already has its own compiler in :mod:`.multi_table_violations`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from .relation_validator import _exec_check_code, _run_check_with_timeout

logger = logging.getLogger(__name__)


# Confidence ordering for the quality-gate. Higher rank = stricter rule.
_CONFIDENCE_RANK = {"high": 3, "medium": 2, "low": 1, "unknown": 0}


@dataclass
class RelationViolation:
    """One row flagged as violating a discovered relation.

    The ``violation_id`` is unique across a run and is used as the
    cluster id when these are promoted into ``clusters.json``. The
    ``peer_rows`` field carries sibling rows that participate in the
    same incident (other rows that share the same FD left-hand-side
    values, for example) — analogous to ``affected_rows`` on a
    ``row_duplication`` cluster.
    """

    violation_id: str
    relation_species_key: str
    source: str
    error_type: str
    error_type_name: str
    error_type_category: str
    kind: str
    row: int
    columns: Tuple[str, ...]
    row_values: Dict[str, str]
    full_row_values: Dict[str, str]
    predicate: str
    rationale: str
    rule_confidence: str
    rule_pass_rate: float
    detection_method: str  # "code_verifier" | "llm_table_scan"
    peer_rows: List[int] = field(default_factory=list)
    llm_reason: str = ""  # populated by llm_table_scan only

    def as_dict(self) -> Dict[str, Any]:
        return {
            "violation_id": self.violation_id,
            "relation_species_key": self.relation_species_key,
            "source": self.source,
            "error_type": self.error_type,
            "error_type_name": self.error_type_name,
            "error_type_category": self.error_type_category,
            "kind": self.kind,
            "row": self.row,
            "columns": list(self.columns),
            "row_values": dict(self.row_values),
            "full_row_values": dict(self.full_row_values),
            "predicate": self.predicate,
            "rationale": self.rationale,
            "rule_confidence": self.rule_confidence,
            "rule_pass_rate": self.rule_pass_rate,
            "detection_method": self.detection_method,
            "peer_rows": list(self.peer_rows),
            "llm_reason": self.llm_reason,
        }

    def to_flat_issue(self) -> Dict[str, Any]:
        """Render as a flat-issue dict consumable by
        :func:`src.core.aggregator.cluster_issues`.

        Sibling rows go into ``affected_rows`` so the queue groups
        together every row participating in the same FD violation
        (mirrors the ``row_duplication`` UI bucketing).
        """
        all_rows: List[int] = [self.row, *self.peer_rows]
        # Stable order, dedup-preserve.
        seen: set = set()
        affected: List[int] = []
        for r in all_rows:
            if r not in seen:
                seen.add(r)
                affected.append(r)
        # Build a human-friendly explanation that always carries the
        # discovered predicate so labellers can judge whether the row
        # genuinely violates the LLM's reading of the rule.
        explanation_lines = [
            f"Discovered relation: {self.predicate}",
        ]
        if self.rationale:
            explanation_lines.append(f"Rationale: {self.rationale}")
        if self.detection_method == "llm_table_scan" and self.llm_reason:
            explanation_lines.append(f"Reviewer reason: {self.llm_reason}")
        explanation_lines.append(
            f"Detection: {self.detection_method} "
            f"(rule pass-rate {self.rule_pass_rate:.0%}, "
            f"confidence {self.rule_confidence})"
        )
        observed = (
            self.llm_reason
            if self.detection_method == "llm_table_scan" and self.llm_reason
            else f"Row violates relation: {self.predicate}"
        )
        return {
            "id": self.violation_id,
            "row": self.row,
            "source": self.source,
            "error_type": self.error_type,
            "error_type_name": self.error_type_name,
            "error_type_category": self.error_type_category,
            "observed": observed,
            "explanation": "\n".join(explanation_lines),
            "suggested_fix": None,
            "columns": list(self.columns),
            "affected_rows": affected if len(affected) > 1 else [],
            "row_values": dict(self.row_values),
            "full_row_values": dict(self.full_row_values),
            "verdict": "",
            "discovery_strategy": f"relation:{self.detection_method}",
        }


def _row_values(df: pd.DataFrame, row: int, columns: Sequence[str]) -> Dict[str, str]:
    """Project ``columns`` from row ``row``, stringified and NA-safe."""
    if row < 0 or row >= len(df):
        return {}
    series = df.iloc[row]
    out: Dict[str, str] = {}
    for c in columns:
        if c not in df.columns:
            continue
        v = series[c]
        out[c] = "" if pd.isna(v) else str(v)
    return out


def _full_row_values(df: pd.DataFrame, row: int) -> Dict[str, str]:
    return _row_values(df, row, list(df.columns))


def _peer_rows_for_fd(
    df: pd.DataFrame,
    row: int,
    lhs: Sequence[str],
    fail_idx: Sequence[int],
) -> List[int]:
    """For a functional-dependency rule (LHS → RHS), return sibling
    violators that share the same LHS values as ``row``.

    Two rows form a "real" FD violation only when their LHS matches
    but their RHS disagrees — surfacing them together in the cluster
    queue lets the labeller see the actual conflict.
    """
    lhs_real = [c for c in lhs if c in df.columns]
    if not lhs_real:
        return []
    lhs_vals = {c: df.iloc[row].get(c) for c in lhs_real}
    peers: List[int] = []
    for other in fail_idx:
        if other == row:
            continue
        if all(df.iloc[other].get(c) == lhs_vals[c] for c in lhs_real):
            peers.append(int(other))
    return peers


def _passes_quality_gate(
    relation: Mapping[str, Any],
    verification: Mapping[str, Any],
    *,
    min_pass_rate: float,
    max_violations_per_rule: int,
    min_confidence: str,
) -> Optional[str]:
    """Return ``None`` if the rule should be compiled, else a reason
    string for the skip log."""
    if not verification.get("verifiable"):
        return "verifiable=False"
    if not verification.get("code"):
        return "no code"
    if verification.get("error"):
        return f"verifier error: {verification['error']}"
    pass_rate = verification.get("pass_rate")
    if pass_rate is None:
        return "no pass_rate"
    if pass_rate < min_pass_rate:
        return f"pass_rate={pass_rate:.2%} < {min_pass_rate:.0%}"
    n_violating = int(verification.get("n_violating") or 0)
    if n_violating > max_violations_per_rule:
        return (
            f"n_violating={n_violating} > {max_violations_per_rule} "
            "(rule likely wrong)"
        )
    confidence = str(relation.get("confidence") or "unknown").lower()
    if _CONFIDENCE_RANK.get(confidence, 0) < _CONFIDENCE_RANK.get(min_confidence, 2):
        return f"rule confidence={confidence} < {min_confidence}"
    if relation.get("tables"):
        # Multi-table relations have their own pipeline.
        return "multi-table relation (handled by multi_table pipeline)"
    return None


def compile_code_violations(
    relations_by_species: Mapping[str, Mapping[str, Any]],
    verifications: Sequence[Mapping[str, Any]],
    source_frames: Mapping[str, pd.DataFrame],
    *,
    min_pass_rate: float = 0.85,
    max_violations_per_rule: int = 200,
    min_confidence: str = "medium",
    timeout_seconds: float = 8.0,
) -> Tuple[List[RelationViolation], List[Dict[str, Any]]]:
    """Re-run validated ``check()`` codes and emit per-row violations.

    Parameters
    ----------
    relations_by_species
        ``{species_key: relation_dict}`` — needed for predicate /
        rationale / confidence / column metadata that the verification
        record alone does not carry.
    verifications
        Sequence of ``RelationVerification.as_dict()`` records, as
        loaded from ``relation_verifications.jsonl``.
    source_frames
        ``{source_key: DataFrame}`` — the same frames the validator
        ran against (basename keys, mirrors the cleaning runner).

    Returns
    -------
    (violations, skip_log)
        ``violations`` is the per-row violation set.
        ``skip_log`` is a list of ``{species_key, source, reason}``
        records describing every relation that was *not* compiled, so
        the run summary can report rule-level coverage.
    """
    violations: List[RelationViolation] = []
    skip_log: List[Dict[str, Any]] = []

    for v in verifications:
        species_key = v.get("species_key", "")
        rel = relations_by_species.get(species_key, {})
        if not rel:
            skip_log.append({
                "species_key": species_key,
                "source": v.get("source", ""),
                "reason": "no matching relation record",
            })
            continue
        skip_reason = _passes_quality_gate(
            rel, v,
            min_pass_rate=min_pass_rate,
            max_violations_per_rule=max_violations_per_rule,
            min_confidence=min_confidence,
        )
        if skip_reason:
            skip_log.append({
                "species_key": species_key,
                "source": v.get("source", ""),
                "reason": skip_reason,
            })
            continue

        source = v.get("source", "")
        df = source_frames.get(source)
        if df is None:
            skip_log.append({
                "species_key": species_key,
                "source": source,
                "reason": f"source frame {source!r} not loaded",
            })
            continue

        try:
            fn = _exec_check_code(v["code"])
            mask = _run_check_with_timeout(fn, df, timeout_seconds)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Re-execution failed for relation %s on %s: %s",
                species_key, source, exc,
            )
            skip_log.append({
                "species_key": species_key,
                "source": source,
                "reason": f"re-exec error: {exc}",
            })
            continue

        if mask is None or len(mask) == 0:
            continue

        # ``mask`` is True=passes, False=violates. NaN → treat as
        # violation (the validator already does the same in its
        # sample extraction) so silent data quality issues do not
        # masquerade as rule satisfaction.
        fail_mask = (~mask).fillna(True)
        if not fail_mask.any():
            continue

        fail_positions = [int(i) for i, bad in enumerate(fail_mask.values) if bool(bad)]

        columns = tuple(rel.get("columns") or [])
        lhs = tuple(rel.get("lhs") or [])
        kind = str(rel.get("kind") or v.get("kind") or "")

        for pos in fail_positions:
            row_vals = _row_values(df, pos, columns)
            full_vals = _full_row_values(df, pos)
            peers: List[int] = []
            # Any FD-shaped relation (kind in {fd, composite_fd, ...})
            # benefits from sibling-row bucketing: rows that share the
            # LHS but disagree on the RHS form the actual conflict.
            if kind.endswith("fd") and lhs:
                peers = _peer_rows_for_fd(df, pos, lhs, fail_positions)
            violations.append(RelationViolation(
                violation_id=f"rel__{species_key}#{source}#r{pos:05d}",
                relation_species_key=species_key,
                source=source,
                error_type=str(rel.get("error_type") or v.get("error_type") or ""),
                error_type_name=str(rel.get("error_type_name") or rel.get("error_type") or ""),
                error_type_category=str(rel.get("error_type_category") or ""),
                kind=kind,
                row=pos,
                columns=columns,
                row_values=row_vals,
                full_row_values=full_vals,
                predicate=str(rel.get("predicate") or ""),
                rationale=str(rel.get("rationale") or ""),
                rule_confidence=str(rel.get("confidence") or "unknown").lower(),
                rule_pass_rate=float(v.get("pass_rate") or 0.0),
                detection_method="code_verifier",
                peer_rows=peers,
            ))

    logger.info(
        "Compiled %d row-level violations from %d verifications "
        "(%d rules skipped by quality gate)",
        len(violations), len(verifications), len(skip_log),
    )
    return violations, skip_log


def relation_violations_to_flat_issues(
    violations: Iterable[RelationViolation],
) -> List[Dict[str, Any]]:
    """Render every violation as a flat-issue dict.

    The result is consumable by :func:`src.core.aggregator.cluster_issues`
    so the caller can interleave relation-derived findings with
    detection-pass findings using the existing clustering machinery.
    """
    return [v.to_flat_issue() for v in violations]
