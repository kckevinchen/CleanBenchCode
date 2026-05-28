"""Tests for the column-clustering proposer + adaptive aggregator.

Slice 4 of the column-clusters labelling subsystem. Covers the
LLM-free helpers (parser + aggregator); the end-to-end LLM smoke
test lives in ``tests/test_column_clustering_e2e.py`` and runs
manually (``RUN_LLM_TESTS=1``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from src.core.value_cluster.proposer import (
    ProposedCluster,
    _strip_markdown_fences,
    parse_proposed_clusters,
)
from src.core.value_cluster.adaptive import (
    DraftCluster,
    _aggregate_drafts,
)
from src.labeling.schema import ClusterKind


# ---------------------------------------------------------------------------
# parse_proposed_clusters
# ---------------------------------------------------------------------------


class TestParseProposedClusters:
    def test_happy_path(self) -> None:
        text = (
            '{"clusters": ['
            '  {"values": ["BBB", "ABC HS"], "kind": "text",'
            '   "canonical": "BBB"},'
            '  {"values": ["EEE", "DDD"], "kind": "text",'
            '   "canonical": null}'
            ']}'
        )
        distinct = {"BBB", "ABC HS", "EEE", "DDD"}
        clusters, err = parse_proposed_clusters(text, distinct=distinct)
        assert err is None
        assert len(clusters) == 2
        c1 = next(c for c in clusters if "BBB" in c.values)
        assert c1.values == ("ABC HS", "BBB")  # sorted
        assert c1.kind == ClusterKind.TEXT
        assert c1.canonical == "BBB"
        c2 = next(c for c in clusters if "EEE" in c.values)
        assert c2.values == ("DDD", "EEE")
        assert c2.canonical is None

    def test_strip_markdown_fences(self) -> None:
        text = (
            '```json\n'
            '{"clusters": [{"values": ["a","b"], "kind": "untagged", '
            '"canonical": null}]}\n'
            '```'
        )
        clusters, err = parse_proposed_clusters(text, distinct={"a", "b"})
        assert err is None
        assert len(clusters) == 1
        assert clusters[0].values == ("a", "b")

    def test_empty_response(self) -> None:
        clusters, err = parse_proposed_clusters("", distinct={"a"})
        assert clusters == []
        assert err is not None
        assert "empty" in err

    def test_invalid_json(self) -> None:
        clusters, err = parse_proposed_clusters("not json", distinct={"a"})
        assert clusters == []
        assert err is not None

    def test_missing_clusters_key(self) -> None:
        clusters, err = parse_proposed_clusters(
            '{"groups": []}', distinct={"a"},
        )
        assert clusters == []
        assert err is not None

    def test_empty_clusters_list_is_ok(self) -> None:
        clusters, err = parse_proposed_clusters(
            '{"clusters": []}', distinct={"a", "b"},
        )
        assert err is None
        assert clusters == []

    def test_hallucinated_value_dropped(self) -> None:
        text = (
            '{"clusters": [{"values": ["a", "b", "ghost"], '
            '"kind": "untagged", "canonical": null}]}'
        )
        clusters, err = parse_proposed_clusters(
            text, distinct={"a", "b"},
        )
        assert err is None
        assert len(clusters) == 1
        assert "ghost" not in clusters[0].values

    def test_hallucinated_canonical_dropped(self) -> None:
        text = (
            '{"clusters": [{"values": ["a", "b"], '
            '"kind": "untagged", "canonical": "ghost"}]}'
        )
        clusters, err = parse_proposed_clusters(
            text, distinct={"a", "b"},
        )
        assert err is None
        assert clusters[0].canonical is None

    def test_single_value_cluster_dropped(self) -> None:
        text = (
            '{"clusters": ['
            '  {"values": ["a"], "kind": "untagged", "canonical": null},'
            '  {"values": ["b", "c"], "kind": "untagged", "canonical": null}'
            ']}'
        )
        clusters, err = parse_proposed_clusters(
            text, distinct={"a", "b", "c"},
        )
        assert err is None
        assert len(clusters) == 1
        assert clusters[0].values == ("b", "c")

    def test_unknown_kind_normalised(self) -> None:
        text = (
            '{"clusters": [{"values": ["a", "b"], '
            '"kind": "BOGUS_KIND", "canonical": "a"}]}'
        )
        clusters, err = parse_proposed_clusters(
            text, distinct={"a", "b"},
        )
        assert err is None
        assert clusters[0].kind == ClusterKind.UNTAGGED

    def test_target_style_parsed(self) -> None:
        text = (
            '{"clusters": [{"values": ["a", "b"], '
            '"kind": "pattern", "canonical": null, '
            '"target_style": "spelled-out form, as in \'Apple Inc.\'"}]}'
        )
        clusters, err = parse_proposed_clusters(
            text, distinct={"a", "b"},
        )
        assert err is None
        assert clusters[0].target_style == (
            "spelled-out form, as in 'Apple Inc.'"
        )

    def test_target_style_missing_is_none(self) -> None:
        text = (
            '{"clusters": [{"values": ["a", "b"], '
            '"kind": "text", "canonical": "a"}]}'
        )
        clusters, err = parse_proposed_clusters(
            text, distinct={"a", "b"},
        )
        assert err is None
        assert clusters[0].target_style is None

    def test_target_style_empty_string_normalised_to_none(self) -> None:
        text = (
            '{"clusters": [{"values": ["a", "b"], '
            '"kind": "pattern", "canonical": null, "target_style": "   "}]}'
        )
        clusters, err = parse_proposed_clusters(
            text, distinct={"a", "b"},
        )
        assert err is None
        assert clusters[0].target_style is None

    def test_target_style_non_string_dropped(self) -> None:
        # LLM occasionally emits non-string types (numbers, objects);
        # parser must coerce to None rather than crash.
        text = (
            '{"clusters": [{"values": ["a", "b"], '
            '"kind": "pattern", "canonical": null, "target_style": 42}]}'
        )
        clusters, err = parse_proposed_clusters(
            text, distinct={"a", "b"},
        )
        assert err is None
        assert clusters[0].target_style is None

    def test_legacy_kind_remapped(self) -> None:
        # Pre-migration LLM responses (and prompts that drift back to
        # the old vocabulary) emit ``entity_variant`` etc. The parser
        # must remap them to the new taxonomy rather than collapsing
        # to UNTAGGED, otherwise the kind-vote at the aggregator
        # would underweight legacy responses.
        text = (
            '{"clusters": ['
            '  {"values": ["a", "b"], "kind": "entity_variant", "canonical": "a"},'
            '  {"values": ["c", "d"], "kind": "style_convention", "canonical": null},'
            '  {"values": ["e", "f"], "kind": "template_shape", "canonical": "e"},'
            '  {"values": ["g", "h"], "kind": "encoding_convention", "canonical": null}'
            ']}'
        )
        clusters, err = parse_proposed_clusters(
            text, distinct={"a", "b", "c", "d", "e", "f", "g", "h"},
        )
        assert err is None
        by_values = {c.values: c.kind for c in clusters}
        assert by_values[("a", "b")] == ClusterKind.TEXT
        assert by_values[("c", "d")] == ClusterKind.CASING
        assert by_values[("e", "f")] == ClusterKind.PATTERN
        assert by_values[("g", "h")] == ClusterKind.NUMERICAL_FORMAT

    def test_duplicate_value_across_clusters_disjoint(self) -> None:
        """If the LLM emits ``a`` in two clusters, the second one
        drops it -- so the resulting record is disjoint by
        construction."""
        text = (
            '{"clusters": ['
            '  {"values": ["a", "b"], "kind": "untagged", "canonical": null},'
            '  {"values": ["a", "c"], "kind": "untagged", "canonical": null}'
            ']}'
        )
        clusters, err = parse_proposed_clusters(
            text, distinct={"a", "b", "c"},
        )
        assert err is None
        # First cluster keeps {a, b}; second loses ``a`` -> {c} only
        # -> dropped as a singleton.
        assert len(clusters) == 1
        assert clusters[0].values == ("a", "b")

    def test_strip_markdown_fences_helper(self) -> None:
        assert _strip_markdown_fences("```json\nfoo\n```") == "foo"
        assert _strip_markdown_fences("```\nbar\n```") == "bar"
        assert _strip_markdown_fences("plain") == "plain"
        assert _strip_markdown_fences("  ") == ""


# ---------------------------------------------------------------------------
# _aggregate_drafts
# ---------------------------------------------------------------------------


class TestAggregateDrafts:
    def _pc(self, *values: str,
            kind: ClusterKind = ClusterKind.UNTAGGED,
            canonical: str | None = None,
            target_style: str | None = None) -> ProposedCluster:
        return ProposedCluster(
            values=tuple(sorted(values)),
            kind=kind, canonical=canonical,
            target_style=target_style,
        )

    def test_perfect_agreement(self) -> None:
        c = self._pc("a", "b", canonical="a",
                     kind=ClusterKind.TEXT)
        species_to_runs = {("a", "b"): [0, 1, 2]}
        drafts = _aggregate_drafts(
            species_to_runs=species_to_runs,
            proposals_by_run=[[c], [c], [c]],
            runs_completed=3,
        )
        assert len(drafts) == 1
        d = drafts[0]
        assert d.values == ("a", "b")
        assert d.kind == ClusterKind.TEXT
        assert d.canonical == "a"
        assert d.support_runs == [0, 1, 2]
        assert d.agreement == 1.0

    def test_partial_agreement(self) -> None:
        c1 = self._pc("a", "b", canonical="a")
        species_to_runs = {("a", "b"): [0, 2]}  # appears in 2 of 3
        drafts = _aggregate_drafts(
            species_to_runs=species_to_runs,
            proposals_by_run=[[c1], [], [c1]],
            runs_completed=3,
        )
        assert len(drafts) == 1
        assert drafts[0].agreement == pytest.approx(2 / 3)
        assert drafts[0].support_runs == [0, 2]

    def test_kind_vote_most_common(self) -> None:
        c1 = self._pc("a", "b", kind=ClusterKind.TEXT)
        c2 = self._pc("a", "b", kind=ClusterKind.CASING)
        c3 = self._pc("a", "b", kind=ClusterKind.TEXT)
        drafts = _aggregate_drafts(
            species_to_runs={("a", "b"): [0, 1, 2]},
            proposals_by_run=[[c1], [c2], [c3]],
            runs_completed=3,
        )
        assert drafts[0].kind == ClusterKind.TEXT

    def test_canonical_vote_skips_none(self) -> None:
        c1 = self._pc("a", "b", canonical="a")
        c2 = self._pc("a", "b", canonical=None)
        c3 = self._pc("a", "b", canonical="a")
        drafts = _aggregate_drafts(
            species_to_runs={("a", "b"): [0, 1, 2]},
            proposals_by_run=[[c1], [c2], [c3]],
            runs_completed=3,
        )
        assert drafts[0].canonical == "a"

    def test_canonical_all_none_stays_none(self) -> None:
        c1 = self._pc("a", "b", canonical=None)
        drafts = _aggregate_drafts(
            species_to_runs={("a", "b"): [0, 1]},
            proposals_by_run=[[c1], [c1]],
            runs_completed=2,
        )
        assert drafts[0].canonical is None

    def test_target_style_vote_most_common(self) -> None:
        c1 = self._pc("a", "b", target_style="spelled-out form")
        c2 = self._pc("a", "b", target_style="full unit name")
        c3 = self._pc("a", "b", target_style="spelled-out form")
        drafts = _aggregate_drafts(
            species_to_runs={("a", "b"): [0, 1, 2]},
            proposals_by_run=[[c1], [c2], [c3]],
            runs_completed=3,
        )
        assert drafts[0].target_style == "spelled-out form"

    def test_target_style_tie_breaks_alphabetically(self) -> None:
        c1 = self._pc("a", "b", target_style="zeta form")
        c2 = self._pc("a", "b", target_style="alpha form")
        drafts = _aggregate_drafts(
            species_to_runs={("a", "b"): [0, 1]},
            proposals_by_run=[[c1], [c2]],
            runs_completed=2,
        )
        assert drafts[0].target_style == "alpha form"

    def test_target_style_vote_skips_none(self) -> None:
        c1 = self._pc("a", "b", target_style="spelled-out form")
        c2 = self._pc("a", "b", target_style=None)
        c3 = self._pc("a", "b", target_style="spelled-out form")
        drafts = _aggregate_drafts(
            species_to_runs={("a", "b"): [0, 1, 2]},
            proposals_by_run=[[c1], [c2], [c3]],
            runs_completed=3,
        )
        assert drafts[0].target_style == "spelled-out form"

    def test_target_style_all_none_stays_none(self) -> None:
        c1 = self._pc("a", "b", canonical="a")
        drafts = _aggregate_drafts(
            species_to_runs={("a", "b"): [0, 1]},
            proposals_by_run=[[c1], [c1]],
            runs_completed=2,
        )
        assert drafts[0].target_style is None

    def test_to_dict_includes_target_style(self) -> None:
        c1 = self._pc("a", "b", target_style="spelled-out form")
        drafts = _aggregate_drafts(
            species_to_runs={("a", "b"): [0]},
            proposals_by_run=[[c1]],
            runs_completed=1,
        )
        d = drafts[0].to_dict()
        assert d["target_style"] == "spelled-out form"

    def test_multiple_species_sort_by_agreement_desc(self) -> None:
        ab = self._pc("a", "b")
        cd = self._pc("c", "d")
        drafts = _aggregate_drafts(
            species_to_runs={
                ("a", "b"): [0, 1, 2],     # 100%
                ("c", "d"): [0],           # 33%
            },
            proposals_by_run=[[ab, cd], [ab], [ab]],
            runs_completed=3,
        )
        assert [d.values for d in drafts] == [("a", "b"), ("c", "d")]

    def test_zero_runs(self) -> None:
        drafts = _aggregate_drafts(
            species_to_runs={},
            proposals_by_run=[],
            runs_completed=0,
        )
        assert drafts == []


# ---------------------------------------------------------------------------
# Integration: proposer + adaptive driver using a MOCK proposer.
# Avoids the real LLM but exercises run_adaptive end-to-end.
# ---------------------------------------------------------------------------


class _FakeProposer:
    """A stand-in :class:`ValueClusterProposer` that returns a
    pre-baked list of clusters per call, ignoring inputs.

    Captures ``variant_idx`` per call so tests can assert prompt-
    ensemble wiring (run i passes variant_idx=i down to the proposer).
    """

    def __init__(self, scripted: List[List[ProposedCluster]]) -> None:
        self._scripted = list(scripted)
        self._idx = 0
        self.variant_idx_calls: List[int] = []

    def propose_once(self, *, source: str, column: str,
                     value_counts, value_order=None,
                     variant_idx: int = 0,
                     ) -> List[ProposedCluster]:
        self.variant_idx_calls.append(variant_idx)
        if self._idx >= len(self._scripted):
            return []
        out = self._scripted[self._idx]
        self._idx += 1
        return out


class TestColumnClusteringAdaptiveIntegration:
    def test_run_with_fake_proposer(self) -> None:
        from src.adaptive.scoped_gt import ArmConfig
        from src.core.value_cluster.adaptive import ColumnClusteringAdaptive

        c_ab = ProposedCluster(values=("a", "b"),
                               kind=ClusterKind.TEXT,
                               canonical="a")
        c_cd = ProposedCluster(values=("c", "d"),
                               kind=ClusterKind.UNTAGGED,
                               canonical=None)
        # First run finds both; subsequent runs only re-find {a,b}.
        scripted = [
            [c_ab, c_cd],   # run 0
            [c_ab],         # run 1
            [c_ab],         # run 2
            [c_ab],         # run 3
            [c_ab],         # run 4
        ]
        proposer = _FakeProposer(scripted)
        # ``arm_config`` requires every field set after merge_into(
        # default()); use default() directly.
        adaptive = ColumnClusteringAdaptive(
            proposer=proposer,  # type: ignore[arg-type]
            arm_config=ArmConfig.default(),
        )
        result = adaptive.run(
            source="x.csv", column="X",
            value_counts=[("a", 5), ("b", 2), ("c", 1), ("d", 1)],
        )
        assert result.runs_completed >= 2
        species = [d.values for d in result.draft_clusters]
        assert ("a", "b") in species
        assert ("c", "d") in species
        # {a,b} should have higher agreement than {c,d}.
        ab = next(d for d in result.draft_clusters if d.values == ("a", "b"))
        cd = next(d for d in result.draft_clusters if d.values == ("c", "d"))
        assert ab.agreement > cd.agreement

    def test_run_skips_trivial_column(self) -> None:
        from src.adaptive.scoped_gt import ArmConfig
        from src.core.value_cluster.adaptive import ColumnClusteringAdaptive

        proposer = _FakeProposer([])
        adaptive = ColumnClusteringAdaptive(
            proposer=proposer,  # type: ignore[arg-type]
            arm_config=ArmConfig.default(),
        )
        result = adaptive.run(
            source="x.csv", column="X",
            value_counts=[("only_one", 10)],
        )
        assert result.runs_completed == 0
        assert result.stopped_reason == "trivial"
        assert result.draft_clusters == []

    def test_adaptive_threads_run_idx_into_variant_idx(self) -> None:
        # Prompt-ensemble wiring: the adaptive driver must call the
        # proposer with ``variant_idx=run_idx`` so run 0 gets the
        # seed prompt and runs 1..K-1 rotate through paraphrases.
        from src.adaptive.scoped_gt import ArmConfig
        from src.core.value_cluster.adaptive import ColumnClusteringAdaptive

        c = ProposedCluster(
            values=("a", "b"),
            kind=ClusterKind.TEXT,
            canonical="a",
        )
        proposer = _FakeProposer([[c], [c], [c], [c], [c]])
        adaptive = ColumnClusteringAdaptive(
            proposer=proposer,  # type: ignore[arg-type]
            arm_config=ArmConfig.default(),
        )
        result = adaptive.run(
            source="x.csv", column="X",
            value_counts=[("a", 5), ("b", 2)],
        )
        # First call: variant_idx==0 (seed). Subsequent calls:
        # variant_idx > 0 (paraphrases) and monotonically increasing.
        assert proposer.variant_idx_calls[0] == 0
        for i, vi in enumerate(proposer.variant_idx_calls):
            assert vi == i, (
                f"variant_idx at run {i} should be {i}, got {vi}"
            )
        assert result.runs_completed == len(proposer.variant_idx_calls)


class TestValueClusterProposerVariants:
    """``ValueClusterProposer.variants`` lets the cleaning runner
    install paraphrased prompts; calls with ``variant_idx=i`` then
    route through those paraphrases instead of the seed. We don't
    touch the network -- a fake LLM captures the prompts so we can
    assert the right variant fired."""

    @staticmethod
    def _make_proposer(
        variants=None,
        *,
        tmp_path: Path,
    ):
        from src.core.value_cluster.proposer import ValueClusterProposer

        prompt_path = tmp_path / "seed.yaml"
        prompt_path.write_text(
            "system: SEED SYSTEM\n"
            "user_template: |\n"
            "  seed src={source} col={column} vals:\n"
            "  {value_list}\n",
            encoding="utf-8",
        )

        class _CapturingLLM:
            def __init__(self) -> None:
                self.calls: List[Dict[str, Any]] = []

            def query(self, model, prompt, *, reasoning_effort,
                      reasoning_summary, system_prompt):
                self.calls.append({
                    "system_prompt": system_prompt,
                    "prompt": prompt,
                })
                return _ResponseStub(text='{"clusters": []}')

        @dataclass
        class _ResponseStub:
            text: str
            ok: bool = True
            error: Optional[str] = None

        llm = _CapturingLLM()
        return ValueClusterProposer(
            llm=llm,  # type: ignore[arg-type]
            model="gpt-test",
            prompt_path=prompt_path,
            variants=variants,
        ), llm

    def test_variant_idx_0_uses_seed_prompt(self, tmp_path: Path) -> None:
        proposer, llm = self._make_proposer(
            variants=[{"system": "VAR1 SYS", "user_template": (
                "v1 src={source} col={column} vals:\n{value_list}"
            )}],
            tmp_path=tmp_path,
        )
        proposer.propose_once(
            source="s.csv", column="C",
            value_counts=[("a", 2), ("b", 1)],
            variant_idx=0,
        )
        call = llm.calls[-1]
        assert call["system_prompt"] == "SEED SYSTEM"
        assert "seed src=s.csv" in call["prompt"]
        assert "v1 src" not in call["prompt"]

    def test_variant_idx_1_uses_first_variant(self, tmp_path: Path) -> None:
        proposer, llm = self._make_proposer(
            variants=[{"system": "VAR1 SYS", "user_template": (
                "v1 src={source} col={column} vals:\n{value_list}"
            )}],
            tmp_path=tmp_path,
        )
        proposer.propose_once(
            source="s.csv", column="C",
            value_counts=[("a", 2), ("b", 1)],
            variant_idx=1,
        )
        call = llm.calls[-1]
        assert call["system_prompt"] == "VAR1 SYS"
        assert "v1 src=s.csv" in call["prompt"]
        assert "seed src" not in call["prompt"]

    def test_variant_idx_wraps_modulo_len(self, tmp_path: Path) -> None:
        # variant_idx > len(variants) wraps around so the proposer
        # never indexes out of range when the adaptive loop runs
        # more passes than variants were generated.
        proposer, llm = self._make_proposer(
            variants=[
                {"system": "A SYS", "user_template":
                    "A {source} {column} {value_list}"},
                {"system": "B SYS", "user_template":
                    "B {source} {column} {value_list}"},
            ],
            tmp_path=tmp_path,
        )
        proposer.propose_once(
            source="s.csv", column="C",
            value_counts=[("a", 2), ("b", 1)],
            variant_idx=3,  # (3-1) % 2 == 0 -> first variant
        )
        assert llm.calls[-1]["system_prompt"] == "A SYS"

    def test_no_variants_always_uses_seed(self, tmp_path: Path) -> None:
        proposer, llm = self._make_proposer(
            variants=None, tmp_path=tmp_path,
        )
        proposer.propose_once(
            source="s.csv", column="C",
            value_counts=[("a", 2), ("b", 1)],
            variant_idx=5,
        )
        assert llm.calls[-1]["system_prompt"] == "SEED SYSTEM"

    def test_malformed_variant_falls_back_to_seed(self, tmp_path: Path) -> None:
        # Defensive: a variant slot missing ``system`` falls back to
        # the seed rather than calling the LLM with an empty
        # system prompt.
        proposer, llm = self._make_proposer(
            variants=[{"system": "", "user_template": "x {source} {column} {value_list}"}],
            tmp_path=tmp_path,
        )
        proposer.propose_once(
            source="s.csv", column="C",
            value_counts=[("a", 2), ("b", 1)],
            variant_idx=1,
        )
        assert llm.calls[-1]["system_prompt"] == "SEED SYSTEM"
