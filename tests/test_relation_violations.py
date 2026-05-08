"""Tests for ``src.core.relation_violations`` (the per-row violation
compiler that turns validated relations into cluster-ready findings).

Cover:

* ``_passes_quality_gate`` — every rejection branch returns the
  correct skip-reason and the happy path returns ``None``.
* ``compile_code_violations`` end-to-end against a synthetic frame
  with a known FD violation; verifies row positions, peer_rows,
  ``detection_method == "code_verifier"``, and that the quality gate
  drops low-pass-rate / high-violator rules.
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
from src.core.relation_violations import (
    RelationViolation,
    _passes_quality_gate,
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
# Quality gate
# ---------------------------------------------------------------------------


class TestQualityGate:
    def test_passes_when_all_constraints_met(self, fd_relation, fd_verification) -> None:
        # Bump pass_rate to satisfy default τ=0.85 in the gate.
        fd_verification["pass_rate"] = 0.95
        fd_verification["n_violating"] = 1
        result = _passes_quality_gate(
            fd_relation, fd_verification,
            min_pass_rate=0.85, max_violations_per_rule=200,
            min_confidence="medium",
        )
        assert result is None

    def test_rejects_non_verifiable(self, fd_relation, fd_verification) -> None:
        fd_verification["verifiable"] = False
        reason = _passes_quality_gate(
            fd_relation, fd_verification,
            min_pass_rate=0.85, max_violations_per_rule=200,
            min_confidence="medium",
        )
        assert reason == "verifiable=False"

    def test_rejects_low_pass_rate(self, fd_relation, fd_verification) -> None:
        fd_verification["pass_rate"] = 0.20
        reason = _passes_quality_gate(
            fd_relation, fd_verification,
            min_pass_rate=0.85, max_violations_per_rule=200,
            min_confidence="medium",
        )
        assert reason is not None and "pass_rate" in reason

    def test_rejects_runaway_violator_count(self, fd_relation, fd_verification) -> None:
        fd_verification["pass_rate"] = 0.95
        fd_verification["n_violating"] = 999
        reason = _passes_quality_gate(
            fd_relation, fd_verification,
            min_pass_rate=0.85, max_violations_per_rule=200,
            min_confidence="medium",
        )
        assert reason is not None and "n_violating" in reason

    def test_rejects_low_confidence(self, fd_relation, fd_verification) -> None:
        fd_relation["confidence"] = "low"
        fd_verification["pass_rate"] = 0.95
        fd_verification["n_violating"] = 1
        reason = _passes_quality_gate(
            fd_relation, fd_verification,
            min_pass_rate=0.85, max_violations_per_rule=200,
            min_confidence="medium",
        )
        assert reason is not None and "confidence" in reason

    def test_rejects_multi_table(self, fd_relation, fd_verification) -> None:
        fd_relation["tables"] = ["sheet_a", "sheet_b"]
        fd_verification["pass_rate"] = 0.95
        fd_verification["n_violating"] = 1
        reason = _passes_quality_gate(
            fd_relation, fd_verification,
            min_pass_rate=0.85, max_violations_per_rule=200,
            min_confidence="medium",
        )
        assert reason is not None and "multi-table" in reason

    def test_rejects_verifier_error(self, fd_relation, fd_verification) -> None:
        fd_verification["error"] = "sandbox: NameError: foo"
        fd_verification["pass_rate"] = None
        reason = _passes_quality_gate(
            fd_relation, fd_verification,
            min_pass_rate=0.85, max_violations_per_rule=200,
            min_confidence="medium",
        )
        assert reason is not None and "error" in reason


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
            max_violations_per_rule=200,
            min_confidence="medium",
        )
        # Rows 0, 1, 3 violate (the three rows with zip=10001 disagree
        # on city). Row 2 is consistent (only zip=20002).
        rows = sorted(v.row for v in violations)
        assert rows == [0, 1, 3]
        # Detection method is the deterministic path.
        for v in violations:
            assert v.detection_method == "code_verifier"
            assert v.predicate == "zip determines city"
            assert v.rationale.startswith("Postal code")
            assert v.rule_pass_rate == pytest.approx(0.25)
            assert v.rule_confidence == "high"
        # Each violator carries its FD siblings as peers.
        by_row = {v.row: v for v in violations}
        assert sorted(by_row[0].peer_rows) == [1, 3]
        assert sorted(by_row[1].peer_rows) == [0, 3]
        assert sorted(by_row[3].peer_rows) == [0, 1]
        assert skip_log == []

    def test_quality_gate_drops_rule_below_threshold(
        self, fd_frame, fd_relation, fd_verification,
    ) -> None:
        # Default min_pass_rate=0.85 against pass_rate=0.25 → drop.
        violations, skip_log = compile_code_violations(
            relations_by_species={"fd_zip_city": fd_relation},
            verifications=[fd_verification],
            source_frames={"addresses.csv": fd_frame},
            min_pass_rate=0.85,
            max_violations_per_rule=200,
            min_confidence="medium",
        )
        assert violations == []
        assert len(skip_log) == 1
        assert "pass_rate" in skip_log[0]["reason"]
        assert skip_log[0]["species_key"] == "fd_zip_city"

    def test_skips_when_source_frame_missing(
        self, fd_relation, fd_verification,
    ) -> None:
        # Bump pass_rate so the gate doesn't fire first.
        fd_verification["pass_rate"] = 0.95
        fd_verification["n_violating"] = 1
        violations, skip_log = compile_code_violations(
            relations_by_species={"fd_zip_city": fd_relation},
            verifications=[fd_verification],
            source_frames={},  # empty
            min_pass_rate=0.85,
            max_violations_per_rule=200,
            min_confidence="medium",
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
