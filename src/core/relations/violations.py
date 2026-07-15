"""Compile per-row violations from validated relations.

Discovery + validation produce ``relations*.jsonl`` (predicates) and
``relation_verifications.jsonl`` (LLM-written ``check()`` code +
empirical pass-rate). This module turns each validated relation
into the **specific rows** that violate it, so the labelling UI can
surface them as cluster items.

The pipeline is intentionally simple and recall-focused:

1. The LLM proposes candidate relations (discovery).
2. The LLM writes ``check(df)`` and we run it against the source
   frame (validation → ``pass_rate``).
3. This compiler classifies each ``(rule, source)`` pair into one
   of three outcomes:

   * ``compile``  — ``min_pass_rate ≤ pass_rate < max_pass_rate``:
     re-execute the check and emit ``code_verifier`` violation
     record(s). Non-FD rules emit ONE record per ``(rule, source)``
     that carries every failing row in ``peer_rows``; FD-shaped
     rules emit one record per LHS-collision group so each
     independent conflict surfaces as its own cluster.
   * ``reject``   — ``pass_rate < min_pass_rate`` (false rule) or
     ``pass_rate >= max_pass_rate`` (rule holds; nothing to flag).
     The rule is dropped; nothing reaches the labelling UI.
   * ``needs_review`` — the verifier could not score the rule
     (``verifiable=False``, missing code, verifier crash, or
     ``pass_rate`` is absent). We emit a single placeholder per
     ``(rule, source)`` so a human can review the rule against
     the table; the placeholder's ``detection_method`` is
     ``"needs_review"`` so renderers can distinguish it.

There is no LLM table-scan path: rules with no pass-rate signal
go straight to needs-review for human verification.

Multi-table relations are out of scope here; the cross-sheet pipeline
has its own compiler in :mod:`.multi_table_violations`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from ._check_runner import run_check_subprocess

logger = logging.getLogger(__name__)


# Classification outcomes returned by :func:`_classify_relation_outcome`.
# Compilation routes every ``(rule, source)`` pair into exactly one of
# these. ``compile`` re-executes the check and emits per-row
# violations; ``needs_review`` emits a single placeholder so the
# labelling UI surfaces the unverified rule; ``reject`` and
# ``drop_silent`` produce nothing visible. The difference between the
# latter two is purely audit-log noise: ``reject`` is a confident
# verifier signal (pass_rate in {<min, >=max}); ``drop_silent`` is
# bookkeeping (multi-table handled elsewhere, missing source frame,
# missing relation record, legacy verifier_kind).
OUTCOME_COMPILE = "compile"
OUTCOME_REJECT = "reject"
OUTCOME_NEEDS_REVIEW = "needs_review"
OUTCOME_DROP_SILENT = "drop_silent"


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
    detection_method: str  # "code_verifier" | "needs_review"
    peer_rows: List[int] = field(default_factory=list)
    llm_reason: str = ""  # populated by needs_review placeholders only
    # Machine-evaluable expression of the rule, when the LLM emitted one
    # at discovery time (e.g. ``total == unit_price * quantity``,
    # ``StartDate <= EndDate``). Empty for rules the LLM only described
    # in natural language. Carried through so the label UI can show the
    # proposed formula alongside the predicate.
    executable: str = ""

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
            "executable": self.executable,
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
        if self.executable:
            explanation_lines.append(f"Formula: {self.executable}")
        if self.rationale:
            explanation_lines.append(f"Rationale: {self.rationale}")
        if self.detection_method == "needs_review" and self.llm_reason:
            explanation_lines.append(f"Why needs review: {self.llm_reason}")
        explanation_lines.append(
            f"Detection: {self.detection_method} "
            f"(rule pass-rate {self.rule_pass_rate:.0%}, "
            f"confidence {self.rule_confidence})"
        )
        if self.detection_method == "needs_review":
            observed = (
                "Rule could not be auto-verified on this table — "
                "please review the rule against the rows."
            )
        else:
            observed = f"Row violates relation: {self.predicate}"
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
            # Tell ``cluster_issues`` that this issue is rule-level
            # evidence, not per-row evidence. Without this hint the
            # clusterer would fan a single 50-row violation into 50
            # buckets (one per row), creating 50 near-identical
            # clusters in the labelling queue. With it set, the
            # violation lands in exactly one cluster keyed by the
            # primary row; affected_rows are still preserved on the
            # cluster's detection so the UI can navigate to each row.
            "cluster_scope": "rule",
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


def _group_fd_incidents(
    df: pd.DataFrame,
    lhs: Sequence[str],
    fail_idx: Sequence[int],
) -> List[List[int]]:
    """Partition ``fail_idx`` into groups that share the same LHS
    values.

    Each group is one independent FD conflict: rows with the same
    LHS but disagreeing RHS. Two LHS-collision groups (e.g. zip
    10001 across rows {0,1,3} vs zip 99999 across rows {7,11}) MUST
    stay separate so the labeller can act on each conflict
    independently.

    Returned groups are sorted by their smallest row index so the
    output is deterministic; each group's members are sorted too.
    NaN LHS values are bucketed under a sentinel so two rows with
    NaN don't accidentally collide (pandas treats NaN != NaN).
    """
    lhs_real = [c for c in lhs if c in df.columns]
    if not lhs_real:
        # No usable LHS → fall back to "everyone is one incident".
        return [sorted(int(r) for r in fail_idx)]
    groups: Dict[Tuple[Any, ...], List[int]] = {}
    for pos in fail_idx:
        key: List[Any] = []
        for c in lhs_real:
            v = df.iloc[pos].get(c)
            if pd.isna(v):
                # Use row index as a unique sentinel so two NaN
                # rows never collide. ``id(pos)`` would be wrong:
                # CPython caches small ints, so ``id(0) == id(0)``.
                key.append(("__nan__", int(pos)))
            else:
                key.append(v)
        groups.setdefault(tuple(key), []).append(int(pos))
    return sorted(
        (sorted(g) for g in groups.values()),
        key=lambda g: g[0],
    )


def make_needs_review_placeholder(
    relation: Mapping[str, Any],
    verification: Mapping[str, Any],
    df: pd.DataFrame,
    *,
    skip_reason: str,
) -> RelationViolation:
    """Build a single-record placeholder for a rule that could not be
    auto-verified, so the labelling UI still surfaces the (rule,
    source) pair for human review.

    The placeholder anchors at row 0 with an empty ``peer_rows`` list
    so the clusterer produces exactly ONE cluster per (rule, source).
    The label UI loads the full source table for context anyway, so
    enumerating every other row in ``peer_rows`` (the previous
    behavior) just exploded one rule into N row-level clusters --
    overwhelming the review queue.

    ``detection_method`` is ``"needs_review"`` so renderers and
    downstream metrics can distinguish it from confirmed violations.

    ``rule_pass_rate`` is set to NaN (encoded as 0.0 here) and the
    skip reason is forwarded via ``llm_reason`` so the explanation
    block tells the labeller exactly why the rule landed in their
    queue.
    """
    species_key = str(relation.get("species_key") or "")
    source = str(relation.get("source") or verification.get("source") or "")
    columns = tuple(relation.get("columns") or [])
    n = int(len(df))
    if n == 0:
        # Cannot place even a single anchor row; caller should treat
        # this as drop_silent. We still return a record so the API is
        # uniform; the run_relation_discovery wrapper filters empty
        # frames upstream.
        anchor = 0
    else:
        anchor = 0
    # Empty peer_rows: needs-review placeholders represent
    # "(rule, source) needs human review", not "every row of this
    # table needs review". One cluster per rule/source is the
    # correct labelling granularity.
    peers: List[int] = []
    return RelationViolation(
        violation_id=f"rel__{species_key}#{source}#needs_review",
        relation_species_key=species_key,
        source=source,
        error_type=str(relation.get("error_type") or verification.get("error_type") or ""),
        error_type_name=str(relation.get("error_type_name") or relation.get("error_type") or ""),
        error_type_category=str(relation.get("error_type_category") or ""),
        kind=str(relation.get("kind") or verification.get("kind") or ""),
        row=anchor,
        columns=columns,
        row_values=_row_values(df, anchor, columns) if n > 0 else {},
        full_row_values=_full_row_values(df, anchor) if n > 0 else {},
        predicate=str(relation.get("predicate") or ""),
        rationale=str(relation.get("rationale") or ""),
        rule_confidence=str(relation.get("confidence") or "unknown").lower(),
        rule_pass_rate=0.0,
        detection_method="needs_review",
        peer_rows=peers,
        llm_reason=(
            f"Rule could not be auto-verified ({skip_reason}). "
            "Human review required."
        ),
        executable=str(relation.get("executable") or ""),
    )


def _classify_relation_outcome(
    relation: Mapping[str, Any],
    verification: Mapping[str, Any],
    *,
    min_pass_rate: float,
    max_pass_rate: float = 1.0 - 1e-9,
) -> Tuple[str, str]:
    """Classify a ``(relation, verification)`` pair into one of four
    outcomes (see module docstring).

    Returns ``(outcome, reason)`` where ``outcome`` is one of:

    * :data:`OUTCOME_COMPILE`  \u2014 ``min_pass_rate \u2264 pass_rate < max_pass_rate``;
      caller re-runs the check and emits per-row violations.
    * :data:`OUTCOME_REJECT`   \u2014 verifier ran and produced a confident
      reject signal (``pass_rate < min_pass_rate`` = false rule, or
      ``pass_rate >= max_pass_rate`` = rule holds, nothing to flag).
    * :data:`OUTCOME_NEEDS_REVIEW` \u2014 verifier could not score the rule
      (``verifiable=False``, missing code, sandbox error, no
      ``pass_rate``). Caller emits a needs-review placeholder so the
      labelling UI still surfaces the rule for human verification.
    * :data:`OUTCOME_DROP_SILENT` \u2014 bookkeeping (multi-table relation,
      legacy ``verifier_kind``). Caller logs but emits nothing.

    ``reason`` is a short human-readable string suitable for the
    audit log; it is empty for ``OUTCOME_COMPILE``.
    """
    if relation.get("tables"):
        # Multi-table relations have their own pipeline.
        return (OUTCOME_DROP_SILENT, "multi-table relation (handled by multi_table pipeline)")
    verifier_kind = str(verification.get("verifier_kind") or "code")
    if verifier_kind != "code":
        # Legacy artifact (older ``llm_sample`` validation runs).
        return (OUTCOME_DROP_SILENT, f"verifier_kind={verifier_kind!r}")
    if not verification.get("verifiable"):
        return (OUTCOME_NEEDS_REVIEW, "verifiable=False")
    if not verification.get("code"):
        return (OUTCOME_NEEDS_REVIEW, "no code")
    if verification.get("error"):
        return (OUTCOME_NEEDS_REVIEW, f"verifier error: {verification['error']}")
    pass_rate = verification.get("pass_rate")
    if pass_rate is None:
        return (OUTCOME_NEEDS_REVIEW, "no pass_rate")
    if pass_rate < min_pass_rate:
        return (OUTCOME_REJECT, f"pass_rate={pass_rate:.2%} < {min_pass_rate:.0%} (false rule)")
    if pass_rate >= max_pass_rate:
        return (OUTCOME_REJECT, f"pass_rate={pass_rate:.2%} >= {max_pass_rate:.0%} (rule holds)")
    return (OUTCOME_COMPILE, "")


def compile_code_violations(
    relations_by_species: Mapping[str, Mapping[str, Any]],
    verifications: Sequence[Mapping[str, Any]],
    source_frames: Mapping[str, pd.DataFrame],
    *,
    min_pass_rate: float = 0.90,
    max_pass_rate: float = 1.0 - 1e-9,
    timeout_seconds: float = 8.0,
) -> Tuple[List[RelationViolation], List[Dict[str, Any]]]:
    """Compile per-row violations from validated relations.

    For every ``(relation, source)`` pair this function classifies
    the rule via :func:`_classify_relation_outcome` and:

    * ``compile``       \u2192 re-executes the sandboxed ``check(df)`` and
      appends one :class:`RelationViolation` per failing row
      (``detection_method=\"code_verifier\"``).
    * ``needs_review``  \u2192 appends a single
      :func:`make_needs_review_placeholder` record so the labelling
      UI surfaces the rule for human verification
      (``detection_method=\"needs_review\"``).
    * ``reject``        \u2192 nothing is emitted; the rule is in the
      skip log so the run summary can report rule-level coverage.
    * ``drop_silent``   \u2192 same as reject but tagged for audit-only.

    Returns ``(violations, skip_log)``. ``skip_log`` entries carry
    ``{species_key, source, outcome, reason}`` so callers can split
    them by outcome without re-parsing reason strings.
    """
    violations: List[RelationViolation] = []
    skip_log: List[Dict[str, Any]] = []

    def _log(species_key: str, source: str, outcome: str, reason: str) -> None:
        skip_log.append({
            "species_key": species_key,
            "source": source,
            "outcome": outcome,
            "reason": reason,
        })

    for v in verifications:
        species_key = str(v.get("species_key") or "")
        source = str(v.get("source") or "")
        rel = relations_by_species.get(species_key) or {}
        if not rel:
            _log(species_key, source, OUTCOME_DROP_SILENT, "no matching relation record")
            continue

        outcome, reason = _classify_relation_outcome(
            rel, v,
            min_pass_rate=min_pass_rate,
            max_pass_rate=max_pass_rate,
        )

        if outcome == OUTCOME_REJECT:
            _log(species_key, source, OUTCOME_REJECT, reason)
            continue
        if outcome == OUTCOME_DROP_SILENT:
            _log(species_key, source, OUTCOME_DROP_SILENT, reason)
            continue

        df = source_frames.get(source)
        if df is None:
            _log(species_key, source, OUTCOME_DROP_SILENT, f"source frame {source!r} not loaded")
            continue

        if outcome == OUTCOME_NEEDS_REVIEW:
            if len(df) == 0:
                _log(species_key, source, OUTCOME_DROP_SILENT, "source frame empty")
                continue
            placeholder = make_needs_review_placeholder(
                rel, v, df, skip_reason=reason,
            )
            violations.append(placeholder)
            _log(species_key, source, OUTCOME_NEEDS_REVIEW, reason)
            continue

        # outcome == OUTCOME_COMPILE: re-execute the sandboxed check in a
        # killable, resource-capped subprocess (a runaway rule must not be
        # able to hang the detector).
        try:
            mask = run_check_subprocess(v["code"], df, timeout_seconds)
        except Exception as exc:  # noqa: BLE001
            # The validator scored the rule fine but our re-run
            # crashed. Don't lose recall: surface for human review.
            logger.warning(
                "Re-execution failed for relation %s on %s: %s",
                species_key, source, exc,
            )
            placeholder = make_needs_review_placeholder(
                rel, v, df, skip_reason=f"re-exec error: {exc}",
            )
            violations.append(placeholder)
            _log(species_key, source, OUTCOME_NEEDS_REVIEW, f"re-exec error: {exc}")
            continue

        if mask is None or len(mask) == 0:
            continue

        # ``mask`` is True=passes, False=violates. NaN \u2192 treat as
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

        # Aggregate failing rows into "incidents" so one rule firing
        # on N rows produces a SMALL number of clusters, not N. Each
        # incident becomes one ``RelationViolation`` whose ``row`` is
        # the smallest member (representative) and whose ``peer_rows``
        # carries the rest. Downstream the labeller sees a single
        # "rule X violated by rows {5, 23, 25, ...}" card instead of
        # 42 near-identical cards.
        #
        # FD-shaped rules (``kind`` ending in ``"fd"``) partition by
        # shared LHS values because each LHS-collision is an
        # independent conflict (zip 10001 disagrees on city across
        # rows {0, 1, 3} is distinct from zip 99999 disagreeing
        # across rows {7, 11}). Non-FD rules collapse to a SINGLE
        # incident per (rule, source).
        if kind.endswith("fd") and lhs:
            incidents = _group_fd_incidents(df, lhs, fail_positions)
        else:
            incidents = [list(fail_positions)]

        for group in incidents:
            if not group:
                continue
            primary = min(group)
            peers = sorted(r for r in group if r != primary)
            row_vals = _row_values(df, primary, columns)
            full_vals = _full_row_values(df, primary)
            violations.append(RelationViolation(
                violation_id=(
                    f"rel__{species_key}#{source}#r{primary:05d}"
                    if not (kind.endswith("fd") and lhs)
                    else f"rel__{species_key}#{source}#fd{primary:05d}"
                ),
                relation_species_key=species_key,
                source=source,
                error_type=str(rel.get("error_type") or v.get("error_type") or ""),
                error_type_name=str(rel.get("error_type_name") or rel.get("error_type") or ""),
                error_type_category=str(rel.get("error_type_category") or ""),
                kind=kind,
                row=primary,
                columns=columns,
                row_values=row_vals,
                full_row_values=full_vals,
                predicate=str(rel.get("predicate") or ""),
                rationale=str(rel.get("rationale") or ""),
                rule_confidence=str(rel.get("confidence") or "unknown").lower(),
                rule_pass_rate=float(v.get("pass_rate") or 0.0),
                detection_method="code_verifier",
                peer_rows=peers,
                executable=str(rel.get("executable") or ""),
            ))

    n_code = sum(1 for v in violations if v.detection_method == "code_verifier")
    n_needs_review = sum(1 for v in violations if v.detection_method == "needs_review")
    by_outcome: Dict[str, int] = {}
    for entry in skip_log:
        by_outcome[entry["outcome"]] = by_outcome.get(entry["outcome"], 0) + 1
    logger.info(
        "Compiled %d violation(s) from %d verifications: "
        "code=%d, needs_review=%d, rejects=%d, drop_silent=%d",
        len(violations), len(verifications), n_code, n_needs_review,
        by_outcome.get(OUTCOME_REJECT, 0),
        by_outcome.get(OUTCOME_DROP_SILENT, 0),
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
