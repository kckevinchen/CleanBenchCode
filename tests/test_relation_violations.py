"""Tests for ``src.core.relations.violations`` (the per-row violation
compiler that turns validated relations into cluster-ready findings).

Cover:

* ``_classify_relation_outcome`` — every routing branch returns the
  correct outcome ("compile", "reject", "needs_review", "drop_silent")
  and accompanying reason string.
* ``compile_code_violations`` end-to-end against a synthetic frame
  with a known FD violation; verifies row positions, peer_rows,
  ``detection_method == "code_verifier"``, that pass-rate at/above
  ``max_pass_rate`` and below ``min_pass_rate`` reject the rule, and
  that verifier-unscored rules surface as needs-review placeholders.
* ``relation_violations_to_flat_issues`` produces dicts the existing
  ``cluster_issues`` aggregator accepts (round-trip into Cluster).
* The compiled flat-issue contract carries every field downstream
  expects (``id, row, source, error_type, observed, explanation,
  columns, affected_rows, row_values, full_row_values, verdict,
  discovery_strategy``).
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.core.aggregator import cluster_issues
from src.core.relations.violations import (
    OUTCOME_COMPILE,
    OUTCOME_DROP_SILENT,
    OUTCOME_NEEDS_REVIEW,
    OUTCOME_REJECT,
    RelationViolation,
    _classify_relation_outcome,
    _peer_rows_for_fd,
    _row_values,
    compile_code_violations,
    relation_violations_to_flat_issues,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fd_frame() -> pd.DataFrame:
    """Frame with a clean FD violation: rows 1 and 3 share zip=10001
    but disagree on city. Row 2 has zip=20002 with consistent city.
    """
    return pd.DataFrame({
        "zip":  ["10001", "10001", "20002", "10001"],
        "city": ["NYC",   "Brooklyn", "Boston", "NYC"],
        "_x":   [0, 1, 2, 3],  # row marker for assertion stability
    })


@pytest.fixture
def fd_relation() -> dict:
    return {
        "species_key": "fd_zip_city",
        "source": "addresses.csv",
        "error_type": "fd_violation",
        "kind": "fd",
        "lhs": ["zip"],
        "rhs": ["city"],
        "columns": ["zip", "city"],
        "predicate": "zip determines city",
        "rationale": "Postal code uniquely identifies the city.",
        "confidence": "high",
    }


@pytest.fixture
def fd_check_code() -> str:
    """``check(df)`` that flags rows where zip→city is violated.
    Returns True for rows that satisfy the FD."""
    return (
        "def check(df):\n"
        "    return df.groupby('zip', dropna=False)['city']"
        ".transform('nunique').le(1)\n"
    )


@pytest.fixture
def fd_verification(fd_check_code) -> dict:
    return {
        "species_key": "fd_zip_city",
        "source": "addresses.csv",
        "error_type": "fd_violation",
        "kind": "fd",
        "verifiable": True,
        "code": fd_check_code,
        "pass_rate": 0.25,  # 1 of 4 rows passes the FD
        "n_evaluated": 4,
        "n_passing": 1,
        "n_violating": 3,
        "layer": "originals",
    }


# ---------------------------------------------------------------------------
# Outcome classifier
# ---------------------------------------------------------------------------


class TestClassifyRelationOutcome:
    def test_compile_when_pass_rate_in_band(self, fd_relation, fd_verification) -> None:
        fd_verification["pass_rate"] = 0.7
        outcome, reason = _classify_relation_outcome(
            fd_relation, fd_verification,
            min_pass_rate=0.5, max_pass_rate=1.0 - 1e-9,
        )
        assert outcome == OUTCOME_COMPILE
        assert reason == ""

    def test_reject_when_pass_rate_above_max(self, fd_relation, fd_verification) -> None:
        # Rule holds — nothing to flag.
        fd_verification["pass_rate"] = 1.0
        outcome, reason = _classify_relation_outcome(
            fd_relation, fd_verification,
            min_pass_rate=0.5, max_pass_rate=1.0 - 1e-9,
        )
        assert outcome == OUTCOME_REJECT
        assert "rule holds" in reason

    def test_reject_when_pass_rate_below_min(self, fd_relation, fd_verification) -> None:
        # False rule — too many rows violate it for it to be a real pattern.
        fd_verification["pass_rate"] = 0.20
        outcome, reason = _classify_relation_outcome(
            fd_relation, fd_verification,
            min_pass_rate=0.5, max_pass_rate=1.0 - 1e-9,
        )
        assert outcome == OUTCOME_REJECT
        assert "false rule" in reason

    def test_needs_review_when_non_verifiable(self, fd_relation, fd_verification) -> None:
        fd_verification["verifiable"] = False
        outcome, reason = _classify_relation_outcome(
            fd_relation, fd_verification,
            min_pass_rate=0.5, max_pass_rate=1.0 - 1e-9,
        )
        assert outcome == OUTCOME_NEEDS_REVIEW
        assert reason == "verifiable=False"

    def test_needs_review_when_verifier_error(self, fd_relation, fd_verification) -> None:
        fd_verification["error"] = "sandbox: NameError: foo"
        fd_verification["pass_rate"] = None
        outcome, reason = _classify_relation_outcome(
            fd_relation, fd_verification,
            min_pass_rate=0.5, max_pass_rate=1.0 - 1e-9,
        )
        assert outcome == OUTCOME_NEEDS_REVIEW
        assert "verifier error" in reason

    def test_needs_review_when_no_pass_rate(self, fd_relation, fd_verification) -> None:
        fd_verification["pass_rate"] = None
        outcome, reason = _classify_relation_outcome(
            fd_relation, fd_verification,
            min_pass_rate=0.5, max_pass_rate=1.0 - 1e-9,
        )
        assert outcome == OUTCOME_NEEDS_REVIEW
        assert reason == "no pass_rate"

    def test_drop_silent_for_multi_table(self, fd_relation, fd_verification) -> None:
        fd_relation["tables"] = ["sheet_a", "sheet_b"]
        fd_verification["pass_rate"] = 0.7
        outcome, reason = _classify_relation_outcome(
            fd_relation, fd_verification,
            min_pass_rate=0.5, max_pass_rate=1.0 - 1e-9,
        )
        assert outcome == OUTCOME_DROP_SILENT
        assert "multi-table" in reason


# ---------------------------------------------------------------------------
# Peer-row computation
# ---------------------------------------------------------------------------


class TestPeerRows:
    def test_finds_siblings_sharing_lhs(self, fd_frame) -> None:
        # Row 0 and 3 both have zip=10001. fail_idx = [0, 1, 3] (the FD
        # violators). Peers of row 0 should be [1, 3].
        peers = _peer_rows_for_fd(fd_frame, 0, ["zip"], [0, 1, 3])
        assert sorted(peers) == [1, 3]

    def test_returns_empty_when_lhs_missing(self, fd_frame) -> None:
        peers = _peer_rows_for_fd(fd_frame, 0, ["nonexistent_col"], [0, 1, 3])
        assert peers == []

    def test_excludes_self(self, fd_frame) -> None:
        peers = _peer_rows_for_fd(fd_frame, 1, ["zip"], [0, 1, 3])
        assert 1 not in peers


# ---------------------------------------------------------------------------
# Row-value projection
# ---------------------------------------------------------------------------


class TestRowValues:
    def test_projects_named_columns(self, fd_frame) -> None:
        out = _row_values(fd_frame, 1, ["zip", "city"])
        assert out == {"zip": "10001", "city": "Brooklyn"}

    def test_skips_missing_columns(self, fd_frame) -> None:
        out = _row_values(fd_frame, 1, ["zip", "missing_col"])
        assert out == {"zip": "10001"}

    def test_handles_nan(self) -> None:
        df = pd.DataFrame({"a": [1, None]})
        out = _row_values(df, 1, ["a"])
        assert out == {"a": ""}

    def test_out_of_range_returns_empty(self, fd_frame) -> None:
        assert _row_values(fd_frame, 99, ["zip"]) == {}


# ---------------------------------------------------------------------------
# End-to-end compile_code_violations
# ---------------------------------------------------------------------------


class TestCompileCodeViolations:
    def test_compiles_fd_violation_with_peers(
        self, fd_frame, fd_relation, fd_verification,
    ) -> None:
        violations, skip_log = compile_code_violations(
            relations_by_species={"fd_zip_city": fd_relation},
            verifications=[fd_verification],
            source_frames={"addresses.csv": fd_frame},
            min_pass_rate=0.20,
            max_pass_rate=1.0 - 1e-9,
        )
        # Rows 0, 1, 3 violate (the three rows with zip=10001 disagree
        # on city); they form ONE LHS-collision incident, so the
        # compiler emits a single ``RelationViolation`` with the
        # smallest row (0) as the representative and the other two
        # rows as ``peer_rows``. Row 2 is consistent (only zip=20002).
        assert len(violations) == 1
        v = violations[0]
        assert v.row == 0
        assert sorted(v.peer_rows) == [1, 3]
        assert v.detection_method == "code_verifier"
        assert v.predicate == "zip determines city"
        assert v.rationale.startswith("Postal code")
        assert v.rule_pass_rate == pytest.approx(0.25)
        assert v.rule_confidence == "high"
        # Violation id encodes that this is an FD-incident bucket
        # rather than a single failing row, so two LHS groups in the
        # same source remain disambiguated.
        assert v.violation_id == "rel__fd_zip_city#addresses.csv#fd00000"
        assert skip_log == []

    def test_rejects_rule_below_min_pass_rate(
        self, fd_frame, fd_relation, fd_verification,
    ) -> None:
        # min_pass_rate=0.85 against pass_rate=0.25 → false rule, rejected.
        violations, skip_log = compile_code_violations(
            relations_by_species={"fd_zip_city": fd_relation},
            verifications=[fd_verification],
            source_frames={"addresses.csv": fd_frame},
            min_pass_rate=0.85,
            max_pass_rate=1.0 - 1e-9,
        )
        assert violations == []
        assert len(skip_log) == 1
        assert skip_log[0]["outcome"] == OUTCOME_REJECT
        assert skip_log[0]["species_key"] == "fd_zip_city"
        assert "false rule" in skip_log[0]["reason"]

    def test_rejects_rule_that_holds(
        self, fd_frame, fd_relation, fd_verification,
    ) -> None:
        # pass_rate>=max_pass_rate → rule holds, nothing to flag.
        fd_verification["pass_rate"] = 1.0
        violations, skip_log = compile_code_violations(
            relations_by_species={"fd_zip_city": fd_relation},
            verifications=[fd_verification],
            source_frames={"addresses.csv": fd_frame},
            min_pass_rate=0.5,
            max_pass_rate=1.0 - 1e-9,
        )
        assert violations == []
        assert len(skip_log) == 1
        assert skip_log[0]["outcome"] == OUTCOME_REJECT
        assert "rule holds" in skip_log[0]["reason"]

    def test_needs_review_when_verifier_could_not_score(
        self, fd_frame, fd_relation, fd_verification,
    ) -> None:
        # Verifier could not score the rule → emit a needs-review
        # placeholder so the UI still surfaces it.
        fd_verification["verifiable"] = False
        fd_verification["pass_rate"] = None
        violations, skip_log = compile_code_violations(
            relations_by_species={"fd_zip_city": fd_relation},
            verifications=[fd_verification],
            source_frames={"addresses.csv": fd_frame},
            min_pass_rate=0.5,
            max_pass_rate=1.0 - 1e-9,
        )
        # One placeholder violation, exactly one needs_review log entry.
        assert len(violations) == 1
        assert violations[0].detection_method == "needs_review"
        assert len(skip_log) == 1
        assert skip_log[0]["outcome"] == OUTCOME_NEEDS_REVIEW

    def test_skips_when_source_frame_missing(
        self, fd_relation, fd_verification,
    ) -> None:
        # Bump pass_rate into the compile band so the gate doesn't
        # short-circuit the lookup.
        fd_verification["pass_rate"] = 0.7
        violations, skip_log = compile_code_violations(
            relations_by_species={"fd_zip_city": fd_relation},
            verifications=[fd_verification],
            source_frames={},  # empty
            min_pass_rate=0.5,
            max_pass_rate=1.0 - 1e-9,
        )
        assert violations == []
        assert any("not loaded" in s["reason"] for s in skip_log)


# ---------------------------------------------------------------------------
# Flat-issue contract + cluster_issues integration
# ---------------------------------------------------------------------------


class TestFlatIssueContract:
    def _make_violation(self, *, row: int = 1, peers=None) -> RelationViolation:
        return RelationViolation(
            violation_id=f"rel__fd_zip_city#addresses.csv#r{row:05d}",
            relation_species_key="fd_zip_city",
            source="addresses.csv",
            error_type="fd_violation",
            error_type_name="FD violation",
            error_type_category="schema",
            kind="fd",
            row=row,
            columns=("zip", "city"),
            row_values={"zip": "10001", "city": "Brooklyn"},
            full_row_values={"zip": "10001", "city": "Brooklyn", "_x": "1"},
            predicate="zip determines city",
            rationale="Postal code uniquely identifies the city.",
            rule_confidence="high",
            rule_pass_rate=0.25,
            detection_method="code_verifier",
            peer_rows=list(peers or []),
        )

    def test_flat_issue_carries_required_fields(self) -> None:
        v = self._make_violation(row=1, peers=[0, 3])
        issues = relation_violations_to_flat_issues([v])
        assert len(issues) == 1
        issue = issues[0]
        for key in (
            "id", "row", "source", "error_type", "observed",
            "explanation", "columns", "affected_rows", "row_values",
            "full_row_values", "verdict", "discovery_strategy",
        ):
            assert key in issue, f"flat-issue missing key {key!r}"
        assert issue["discovery_strategy"] == "relation:code_verifier"
        assert issue["row"] == 1
        # affected_rows includes the primary + peers (deduped).
        assert sorted(issue["affected_rows"]) == [0, 1, 3]
        # Explanation embeds the predicate verbatim.
        assert "Discovered relation: zip determines city" in issue["explanation"]

    def test_solo_violation_omits_affected_rows(self) -> None:
        v = self._make_violation(row=2, peers=[])
        issues = relation_violations_to_flat_issues([v])
        # Empty affected_rows when the row has no peers (no incident
        # group to bucket against).
        assert issues[0]["affected_rows"] == []

    def test_cluster_issues_round_trip(self) -> None:
        # Build a small set of violations and feed them through the
        # real aggregator. Expect one cluster per violating row.
        violations = [
            self._make_violation(row=0, peers=[1, 3]),
            self._make_violation(row=1, peers=[0, 3]),
            self._make_violation(row=3, peers=[0, 1]),
        ]
        flat = relation_violations_to_flat_issues(violations)
        clusters = cluster_issues(flat)
        # cluster_issues buckets the primary row + each entry of
        # affected_rows into its own cluster, so we get a cluster
        # for each unique row in the union (0, 1, 3 here). Every
        # cluster must carry the relation discovery_strategy on its
        # underlying detection.
        cluster_rows = sorted({c.row for c in clusters if c.row is not None})
        assert cluster_rows == [0, 1, 3]
        for c in clusters:
            assert any(
                d.discovery_strategy == "relation:code_verifier"
                for d in c.detections
            )


# ---------------------------------------------------------------------------
# Affected-rows fingerprint refinement
# ---------------------------------------------------------------------------


class TestAffectedRowsFingerprintSplit:
    """``_cluster_row_group`` must sub-split column-overlap components
    by ``affected_rows`` fingerprint when at least one rule-scoped
    issue is present, so a precise rule (small affected-row set) is
    not bundled with a noisy rule that happens to share a column.
    """

    def _rule_issue(
        self,
        *,
        iid: str,
        row: int,
        columns: tuple,
        affected_rows: list,
        error_type: str = "row_calculation_inconsistency",
    ) -> dict:
        return {
            "id": iid,
            "row": row,
            "source": "f.csv",
            "error_type": error_type,
            "error_type_name": error_type.replace("_", " ").title(),
            "error_type_category": "logic",
            "observed": "violation",
            "explanation": f"rule {iid}",
            "suggested_fix": None,
            "columns": list(columns),
            "affected_rows": list(affected_rows),
            "row_values": {},
            "full_row_values": {},
            "verdict": "",
            "discovery_strategy": "relation:code_verifier",
            "cluster_scope": "rule",
        }

    def test_disjoint_affected_rows_split_into_separate_clusters(self) -> None:
        # Two rule-scoped issues that share a column (``B``) but
        # disagree on which rows they flag must NOT merge.
        precise = self._rule_issue(
            iid="precise",
            row=4,
            columns=("A", "B"),
            affected_rows=[4, 5],
        )
        noisy = self._rule_issue(
            iid="noisy",
            row=4,
            columns=("B", "C"),
            affected_rows=[0, 3, 4, 5],
        )
        clusters = cluster_issues([precise, noisy])
        # One cluster per fingerprint, both at primary row 4.
        assert len(clusters) == 2
        rows_by_aff = {
            tuple(sorted(set().union(*(d.affected_rows or [] for d in c.detections)))): c
            for c in clusters
        }
        assert (4, 5) in rows_by_aff
        assert (0, 3, 4, 5) in rows_by_aff
        # Each sub-cluster carries exactly one detection.
        for c in clusters:
            assert len(c.detections) == 1

    def test_identical_affected_rows_stay_merged(self) -> None:
        # Two rule-scoped issues that share a column AND share the
        # same affected_rows fingerprint must still merge (they
        # describe corroborating evidence for the same defect set).
        rule_a = self._rule_issue(
            iid="a", row=4, columns=("A", "B"), affected_rows=[4, 5],
        )
        rule_b = self._rule_issue(
            iid="b", row=4, columns=("B", "C"), affected_rows=[4, 5],
        )
        clusters = cluster_issues([rule_a, rule_b])
        assert len(clusters) == 1
        assert len(clusters[0].detections) == 2

    def test_non_rule_scoped_issues_are_not_split_by_fingerprint(self) -> None:
        # Issues without ``cluster_scope == "rule"`` must NOT be
        # subdivided by affected_rows fingerprint - their column-overlap
        # clustering behaviour predates this refinement and must stay
        # unchanged. Two cell-level detections sharing a column but
        # with disjoint (non-empty) affected_rows lists would normally
        # fan out via the existing per-row bucketing in
        # ``cluster_issues``, so to isolate just the fingerprint gate
        # we hand-build a (source, row) bucket containing both and
        # call ``_cluster_row_group`` directly.
        from src.core.aggregator import _cluster_row_group

        def cell(iid: str, columns: tuple, affected: list) -> dict:
            issue = self._rule_issue(
                iid=iid, row=4, columns=columns, affected_rows=affected,
                error_type="row_duplication",
            )
            issue.pop("cluster_scope")  # not rule-scoped
            return issue

        entries = [
            (cell("d1", columns=("A", "B"), affected=[4, 5]), True),
            (cell("d2", columns=("B", "C"), affected=[4, 6]), True),
        ]
        groups = _cluster_row_group(4, entries)
        # Column overlap on ``B`` merges them; the fingerprint gate is
        # closed because neither issue is rule-scoped, so the merge
        # survives.
        assert len(groups) == 1
        assert len(groups[0][1]) == 2

    def test_mixed_rule_and_cell_scope_splits_by_fingerprint(self) -> None:
        # When a rule-scoped issue is present in a component, the
        # whole component is refined - cell-level issues sharing the
        # column ride along with whichever sub-bucket their own
        # fingerprint puts them in.
        rule = self._rule_issue(
            iid="r1", row=4, columns=("A", "B"), affected_rows=[4, 5],
        )
        cell = self._rule_issue(
            iid="c1", row=4, columns=("B",), affected_rows=[],
            error_type="missing_values",
        )
        cell.pop("cluster_scope")  # cell scope, empty affected_rows fingerprint
        clusters = cluster_issues([rule, cell])
        # Two sub-clusters: one for the rule (fingerprint {4,5}) and
        # one for the cell-level detection (empty fingerprint).
        assert len(clusters) == 2
        sizes = sorted(len(c.detections) for c in clusters)
        assert sizes == [1, 1]

