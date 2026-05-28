"""Tests for the ``include_singletons`` knob on ``ClusterSynthesiser``.

The Phase-B synthesiser used to skip every cluster with a single
detection (``is_multi_type=False``). We now run it on those clusters
too by default so the LLM emits a clean ``canonical_fix`` /
``canonical_explanation`` for every flagged cell. The legacy behaviour
is still reachable via ``include_singletons=False``.

These tests use a canned fake :class:`LLMClient` so no real LLM call is
made; only the gating logic in :meth:`ClusterSynthesiser.run` is
exercised.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Optional

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.aggregator import (  # noqa: E402
    Cluster,
    ClusterDetection,
    ClusterSynthesiser,
    SynthesisPrompt,
)
from src.core.error_taxonomy import ErrorTaxonomy, ErrorType  # noqa: E402
from src.core.llm_client import LLMResponse  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _RecordingLLM:
    """Counts ``query`` calls and returns a canned synthesis JSON."""

    def __init__(self, primary: str = "typo") -> None:
        self.endpoints = ["http://stub-1"]
        self.calls: List[str] = []
        self._primary = primary

    def query(
        self,
        model,
        prompt,
        *,
        reasoning_effort,
        reasoning_summary,
        system_prompt,
    ) -> LLMResponse:
        self.calls.append(prompt)
        body = json.dumps({
            "primary_error_type": self._primary,
            "supporting_error_types": [],
            "root_cause": "stub",
            "canonical_explanation": "stub explanation",
            "canonical_fix": "stub fix",
            "confidence": "medium",
            "is_likely_false_positive": False,
            "reviewer_note": "",
        })
        return LLMResponse(
            text=body,
            endpoint=self.endpoints[0],
            model=model,
            attempts=1,
        )


def _taxonomy() -> ErrorTaxonomy:
    types = (
        ErrorType(
            key="typo",
            name="Typo",
            category="single-column",
            description="d-typo",
            example="ex",
            notes="",
        ),
        ErrorType(
            key="casing",
            name="Casing",
            category="single-column",
            description="d-cas",
            example="ex",
            notes="",
        ),
    )
    return ErrorTaxonomy(
        system_prompt="SYS",
        user_template="t",
        types=types,
    )


def _detection(
    *,
    iid: str,
    error_type: str = "typo",
) -> ClusterDetection:
    return ClusterDetection(
        issue_id=iid,
        error_type=error_type,
        error_type_name=error_type.title(),
        error_type_category="single-column",
        observed="obs",
        explanation="ex",
        suggested_fix=None,
        columns=["name"],
        affected_rows=[],
    )


def _cluster(
    *,
    cid: str,
    error_types: List[str],
    detections: List[ClusterDetection],
    inferred: bool = False,
) -> Cluster:
    return Cluster(
        cluster_id=cid,
        row=1,
        source="a.csv",
        columns=["name"],
        row_values={"name": "x"},
        error_types=error_types,
        error_type_categories=["single-column"],
        evidence_count=len(detections),
        detections=detections,
        inferred=inferred,
    )


def _prompt() -> SynthesisPrompt:
    # ``user_template`` uses the same placeholders as the real one but
    # collapsed for speed - render_synthesis_prompt only needs the keys
    # it formats with to be present in the template.
    return SynthesisPrompt(
        system="SYS",
        user_template=(
            "C={cluster_id} R={row} N={detection_count} K={available_keys}\n"
            "ROWS:\n{row_values}\nDETS:\n{detections}\nTAX:\n{taxonomy_refs}"
        ),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestIncludeSingletonsDefault:
    """``include_singletons=True`` (default) must synthesise singletons."""

    def test_singleton_cluster_is_synthesised(self) -> None:
        llm = _RecordingLLM()
        synth = ClusterSynthesiser(
            llm=llm,  # type: ignore[arg-type]
            taxonomy=_taxonomy(),
            prompt=_prompt(),
            model="m",
            concurrency=1,
        )

        c = _cluster(
            cid="r1-c0",
            error_types=["typo"],
            detections=[_detection(iid="i1")],
        )
        out = synth.run([c])

        assert len(llm.calls) == 1, (
            "single-detection cluster must trigger one LLM call by default"
        )
        assert out[0].synthesis is not None
        assert out[0].synthesis.canonical_fix == "stub fix"
        assert out[0].synthesis.primary_error_type == "typo"

    def test_inferred_cluster_is_skipped(self) -> None:
        """Inferred clusters have no primary detection - never synthesise."""
        llm = _RecordingLLM()
        synth = ClusterSynthesiser(
            llm=llm,  # type: ignore[arg-type]
            taxonomy=_taxonomy(),
            prompt=_prompt(),
            model="m",
            concurrency=1,
        )

        c = _cluster(
            cid="r1-c0",
            error_types=["typo"],
            detections=[_detection(iid="i1")],
            inferred=True,
        )
        out = synth.run([c])

        assert llm.calls == [], "inferred clusters must not be synthesised"
        assert out[0].synthesis is None

    def test_multi_type_and_singleton_both_run(self) -> None:
        llm = _RecordingLLM()
        synth = ClusterSynthesiser(
            llm=llm,  # type: ignore[arg-type]
            taxonomy=_taxonomy(),
            prompt=_prompt(),
            model="m",
            concurrency=1,
        )

        single = _cluster(
            cid="r1-c0",
            error_types=["typo"],
            detections=[_detection(iid="i1", error_type="typo")],
        )
        multi = _cluster(
            cid="r2-c0",
            error_types=["typo", "casing"],
            detections=[
                _detection(iid="i2", error_type="typo"),
                _detection(iid="i3", error_type="casing"),
            ],
        )
        out = synth.run([single, multi])

        assert len(llm.calls) == 2
        assert all(c.synthesis is not None for c in out)


class TestIncludeSingletonsOff:
    """Legacy behaviour: only multi-type clusters get an LLM call."""

    def test_singleton_cluster_is_skipped(self) -> None:
        llm = _RecordingLLM()
        synth = ClusterSynthesiser(
            llm=llm,  # type: ignore[arg-type]
            taxonomy=_taxonomy(),
            prompt=_prompt(),
            model="m",
            concurrency=1,
            include_singletons=False,
        )

        single = _cluster(
            cid="r1-c0",
            error_types=["typo"],
            detections=[_detection(iid="i1")],
        )
        out = synth.run([single])

        assert llm.calls == []
        assert out[0].synthesis is None

    def test_multi_type_still_runs(self) -> None:
        llm = _RecordingLLM()
        synth = ClusterSynthesiser(
            llm=llm,  # type: ignore[arg-type]
            taxonomy=_taxonomy(),
            prompt=_prompt(),
            model="m",
            concurrency=1,
            include_singletons=False,
        )

        multi = _cluster(
            cid="r1-c0",
            error_types=["typo", "casing"],
            detections=[
                _detection(iid="i1", error_type="typo"),
                _detection(iid="i2", error_type="casing"),
            ],
        )
        out = synth.run([multi])

        assert len(llm.calls) == 1
        assert out[0].synthesis is not None


class TestBuildDetectionArgvForwardsFlag:
    """``--no-synthesise-singletons`` is forwarded by the labeling app."""

    def test_payload_flag_appears_in_argv(self) -> None:
        from src.labeling.runs import build_detection_argv

        argv = build_detection_argv({
            "no_synthesise_singletons": True,
        })
        assert "--no-synthesise-singletons" in argv

    def test_default_omits_flag(self) -> None:
        from src.labeling.runs import build_detection_argv

        argv = build_detection_argv({})
        assert "--no-synthesise-singletons" not in argv

    def test_falsy_payload_omits_flag(self) -> None:
        from src.labeling.runs import build_detection_argv

        argv = build_detection_argv({
            "no_synthesise_singletons": False,
        })
        assert "--no-synthesise-singletons" not in argv


class TestSummariseCountsLLMSyntheticOnSingletons:
    """``summarise_clusters`` distinguishes LLM-synth from auto-synth via
    the synthesis ``endpoint`` field, so single-detection clusters that
    went through Phase B are now counted in ``clusters_synthesised``
    rather than ``clusters_auto_synthesised``."""

    def test_llm_endpoint_counts_as_synthesised(self) -> None:
        from src.core.aggregator import summarise_clusters

        llm = _RecordingLLM()
        synth = ClusterSynthesiser(
            llm=llm,  # type: ignore[arg-type]
            taxonomy=_taxonomy(),
            prompt=_prompt(),
            model="m",
            concurrency=1,
        )
        c = _cluster(
            cid="r1-c0",
            error_types=["typo"],
            detections=[_detection(iid="i1")],
        )
        out = synth.run([c])
        summary = summarise_clusters(out)

        assert summary["clusters_synthesised"] == 1
        assert summary["clusters_auto_synthesised"] == 0
        assert summary["clusters_single_type"] == 1
        assert summary["clusters_synthesis_failed"] == 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-v"]))
