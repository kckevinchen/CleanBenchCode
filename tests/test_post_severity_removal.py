"""Regression tests for the severity-removal pass.

These exercise the four code paths that touched ``severity`` /
``severity_max`` last:

* ``aggregator.cluster_issues`` and ``ClusterDetection`` no longer carry
  a severity field, but legacy issue dicts that still contain
  ``"severity"`` must be ingested without error.
* ``aggregator.summarise_clusters`` no longer emits
  ``clusters_by_severity`` and is keyless on missing severity.
* ``detector.summarise_reports`` no longer emits ``issues_by_severity``.
* ``report_builder.flatten_issues`` drops the ``severity`` field from
  the output and surfaces the new optional ``affected_rows_total``
  passthrough.
* ``report_builder._top_clusters`` orders by ``-evidence_count`` and
  emits cluster summaries without ``severity_max``.
* The labeling-app ``_cluster_summary`` projection no longer leaks
  ``severity_max``.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List

import pandas as pd
import pytest

from src.core.aggregator import (
    Cluster,
    ClusterDetection,
    cluster_issues,
    summarise_clusters,
)
from src.core.detector import ChunkReport, summarise_reports
from src.core.report_builder import (
    _top_clusters,
    build_summary,
    flatten_issues,
)
from src.labeling.routers.clusters import _cluster_summary


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _flat_issue(
    *,
    iid: str,
    source: str = "a.csv",
    row: int = 1,
    columns: List[str] | None = None,
    etype: str = "typo",
    affected_rows: List[int] | None = None,
    affected_rows_total: int | None = None,
    severity: str | None = None,
) -> Dict[str, Any]:
    """Build the dict shape that ``cluster_issues`` consumes."""
    d: Dict[str, Any] = {
        "id": iid,
        "source": source,
        "row": row,
        "columns": list(columns) if columns is not None else ["name"],
        "error_type": etype,
        "error_type_name": etype.title(),
        "error_type_category": "Cat",
        "observed": "obs",
        "explanation": "ex",
        "suggested_fix": "fix",
        "row_values": {"name": "x"},
    }
    if affected_rows is not None:
        d["affected_rows"] = affected_rows
    if affected_rows_total is not None:
        d["affected_rows_total"] = affected_rows_total
    # Legacy field that may still be present on disk / from older runs.
    if severity is not None:
        d["severity"] = severity
    return d


def _chunk_report(*, source: str = "a.csv", issues: List[Dict[str, Any]]) -> ChunkReport:
    return ChunkReport(
        source=source,
        start_row=0,
        end_row=10,
        model="test",
        prompt_name="test",
        issues=issues,
        error_type="typo",
        error_type_name="Typo",
        error_type_category="Cat",
    )


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------


class TestClusterIssues:
    def test_cluster_detection_has_no_severity_attribute(self) -> None:
        clusters = cluster_issues([_flat_issue(iid="i1")])
        assert len(clusters) == 1
        det = clusters[0].detections[0]
        assert not hasattr(det, "severity")
        # asdict must not contain the field either.
        assert "severity" not in asdict(det)

    def test_cluster_has_no_severity_max_attribute(self) -> None:
        clusters = cluster_issues([_flat_issue(iid="i1")])
        assert not hasattr(clusters[0], "severity_max")
        assert "severity_max" not in clusters[0].as_dict()

    def test_legacy_severity_field_is_ignored(self) -> None:
        """Old run dirs may still carry severity; ingestion must not crash."""
        clusters = cluster_issues([_flat_issue(iid="i1", severity="high")])
        assert len(clusters) == 1
        det = clusters[0].detections[0]
        # Field is silently dropped, not surfaced anywhere.
        assert not hasattr(det, "severity")

    def test_overlapping_columns_cluster(self) -> None:
        issues = [
            _flat_issue(iid="i1", columns=["name"]),
            _flat_issue(iid="i2", columns=["name", "age"], etype="missing_value"),
        ]
        clusters = cluster_issues(issues)
        assert len(clusters) == 1
        c = clusters[0]
        assert sorted(c.error_types) == ["missing_value", "typo"]
        assert c.is_multi_type
        assert c.evidence_count == 2

    def test_disjoint_columns_split_into_separate_clusters(self) -> None:
        issues = [
            _flat_issue(iid="i1", columns=["name"]),
            _flat_issue(iid="i2", columns=["zip"], etype="missing_value"),
        ]
        clusters = cluster_issues(issues)
        assert len(clusters) == 2

    def test_different_sources_do_not_cluster(self) -> None:
        issues = [
            _flat_issue(iid="i1", source="a.csv", columns=["name"]),
            _flat_issue(iid="i2", source="b.csv", columns=["name"]),
        ]
        clusters = cluster_issues(issues)
        assert len(clusters) == 2
        sources = {c.source for c in clusters}
        assert sources == {"a.csv", "b.csv"}


# ---------------------------------------------------------------------------
# Aggregator summary
# ---------------------------------------------------------------------------


class TestSummariseClusters:
    def test_summary_has_no_severity_key(self) -> None:
        clusters = cluster_issues(
            [_flat_issue(iid="i1"), _flat_issue(iid="i2", row=2)]
        )
        summary = summarise_clusters(clusters)
        assert "clusters_by_severity" not in summary
        assert summary["clusters_total"] == 2
        # The non-severity counters survived.
        assert "clusters_by_primary_error_type" in summary
        assert "clusters_by_confidence" in summary

    def test_empty_input(self) -> None:
        summary = summarise_clusters([])
        assert summary["clusters_total"] == 0
        assert summary["clusters_likely_false_positives"] == 0


# ---------------------------------------------------------------------------
# Detector summary
# ---------------------------------------------------------------------------


class TestSummariseReports:
    def test_no_issues_by_severity_key(self) -> None:
        report = _chunk_report(issues=[{"row": 1, "columns": ["name"]}])
        summary = summarise_reports([report])
        assert "issues_by_severity" not in summary
        assert summary["total_issues"] == 1
        assert summary["chunks"] == 1

    def test_issue_with_legacy_severity_does_not_break_aggregation(self) -> None:
        report = _chunk_report(
            issues=[{"row": 1, "columns": ["name"], "severity": "high"}]
        )
        summary = summarise_reports([report])
        # Severity is silently ignored; counters must still aggregate.
        assert summary["total_issues"] == 1
        assert summary["issues_by_error_type"]["typo"] == 1

    def test_chunk_dedup_across_per_type_calls(self) -> None:
        # Two reports for the same chunk should count as one chunk but
        # two LLM calls (per-type detector mode).
        r1 = _chunk_report(issues=[])
        r2 = _chunk_report(issues=[])
        summary = summarise_reports([r1, r2])
        assert summary["chunks"] == 1
        assert summary["llm_calls"] == 2


# ---------------------------------------------------------------------------
# Report-builder flattening + summary
# ---------------------------------------------------------------------------


class TestFlattenIssues:
    def _make_frame(self) -> pd.DataFrame:
        return pd.DataFrame({"name": ["a", "b", "c"], "age": [1, 2, 3]})

    def test_flat_record_has_no_severity(self) -> None:
        report = _chunk_report(
            issues=[{"row": 1, "columns": ["name"], "severity": "high"}]
        )
        flat = flatten_issues([report], {"a.csv": self._make_frame()})
        assert len(flat) == 1
        rec = flat[0]
        assert "severity" not in rec
        assert rec["error_type"] == "typo"

    def test_affected_rows_total_passthrough(self) -> None:
        report = _chunk_report(
            issues=[
                {
                    "row": 0,
                    "columns": ["name"],
                    "affected_rows": [0, 1, 2],
                    "affected_rows_total": 137,
                }
            ]
        )
        flat = flatten_issues([report], {"a.csv": self._make_frame()})
        assert flat[0]["affected_rows"] == [0, 1, 2]
        assert flat[0]["affected_rows_total"] == 137

    def test_affected_rows_total_omitted_when_absent(self) -> None:
        report = _chunk_report(issues=[{"row": 0, "columns": ["name"]}])
        flat = flatten_issues([report], {"a.csv": self._make_frame()})
        assert "affected_rows_total" not in flat[0]

    def test_affected_rows_total_garbage_silently_dropped(self) -> None:
        report = _chunk_report(
            issues=[
                {
                    "row": 0,
                    "columns": ["name"],
                    "affected_rows_total": "not-a-number",
                }
            ]
        )
        flat = flatten_issues([report], {"a.csv": self._make_frame()})
        # Bad input is dropped; flatten does not raise.
        assert "affected_rows_total" not in flat[0]


class TestTopClusters:
    def test_top_clusters_summary_keys_have_no_severity(self) -> None:
        clusters = cluster_issues(
            [_flat_issue(iid="i1"), _flat_issue(iid="i2", row=2)]
        )
        top = _top_clusters(clusters)
        assert top
        for entry in top:
            assert "severity_max" not in entry
            assert {"cluster_id", "row", "evidence_count", "primary_error_type"} <= set(
                entry
            )


class TestBuildSummary:
    def test_full_pipeline_summary_has_no_severity_keys(self) -> None:
        report = _chunk_report(
            issues=[{"row": 1, "columns": ["name"], "severity": "high"}]
        )
        flat = flatten_issues(
            [report], {"a.csv": pd.DataFrame({"name": ["a", "b"]})}
        )
        clusters = cluster_issues(flat)
        summary = build_summary([report], flat, clusters, config={})
        assert "issues_by_severity" not in summary
        assert "clusters_by_severity" not in summary
        # Sanity: structural counters survived.
        assert summary["total_issues"] == 1
        assert summary["clusters_total"] == 1


# ---------------------------------------------------------------------------
# Labeling-app projection
# ---------------------------------------------------------------------------


class TestClusterSummaryProjection:
    def test_no_severity_max_in_response(self) -> None:
        # Simulate a stored cluster dict that may carry a legacy
        # severity_max from an older run.
        c = {
            "cluster_id": "x",
            "source": "a.csv",
            "row": 5,
            "columns": ["name"],
            "error_types": ["typo"],
            "evidence_count": 1,
            "synthesis": None,
            "severity_max": "high",  # legacy
        }
        out = _cluster_summary(c)
        assert "severity_max" not in out
        assert out["cluster_id"] == "x"
        assert out["primary_error_type"] is None
        assert out["is_multi_type"] is False
