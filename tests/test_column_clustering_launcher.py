"""Launcher-wiring tests for the column-clustering auto-trigger.

The "Launch new run" page does NOT take an opt-in flag for column
clustering. Instead, the cleaning detection runner auto-triggers the
pass whenever ``inconsistency`` is in the resolved error-type list.
These tests pin down two invariants that together implement that
contract end-to-end:

1. ``build_detection_argv`` does not inject any column-clustering
   CLI flag (no new flag, no per-error-type wiring on the launcher
   side -- the trigger lives entirely inside ``main()``).
2. ``run_cleaning_detection.py`` declares no ``--no-column-clustering``
   (or similar) flag. This is a guard against accidentally adding
   back a flag the user explicitly asked us not to add.
3. ``main()`` of the cleaning runner imports
   ``run_for_candidates`` and ``enumerate_candidates_from_frames``
   from ``experiments.run_column_clustering`` inside an
   ``inconsistency``-gated block. We verify the gate exists in the
   file source so a future refactor cannot silently drop it.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.labeling.runs import build_detection_argv  # noqa: E402


class TestBuildDetectionArgvNoColumnClusteringFlag:
    def test_inconsistency_in_error_types_adds_no_column_clustering_flag(
        self,
    ) -> None:
        argv = build_detection_argv({"error_types": "inconsistency"})
        # Hard guard: no flag is forwarded for column clustering --
        # the trigger is internal to the cleaning runner.
        for token in argv:
            assert "column-clustering" not in token, (
                f"build_detection_argv must not emit column-clustering "
                f"flags; got: {argv}"
            )
        # And the error_types value still gets through unchanged.
        assert "--error-types" in argv
        assert "inconsistency" in argv

    def test_default_payload_adds_no_column_clustering_flag(self) -> None:
        argv = build_detection_argv({})
        for token in argv:
            assert "column-clustering" not in token


class TestCleaningRunnerHasNoColumnClusteringArg:
    def test_argparse_does_not_define_no_column_clustering(self) -> None:
        # The user's contract: no new launcher-level arg. Reading
        # the source guards against accidentally re-adding one.
        src = (
            PROJECT_ROOT / "experiments" / "run_cleaning_detection.py"
        ).read_text(encoding="utf-8")
        assert "--no-column-clustering" not in src
        assert "no_column_clustering" not in src


class TestCleaningRunnerHasInconsistencyGate:
    def test_inconsistency_gate_present_in_main(self) -> None:
        # The auto-trigger lives behind an ``inconsistency`` gate.
        # We grep the source for the literal predicate so the
        # contract is fail-fast if someone widens or narrows the
        # condition without updating the test.
        src = (
            PROJECT_ROOT / "experiments" / "run_cleaning_detection.py"
        ).read_text(encoding="utf-8")
        assert 'et.key == "inconsistency"' in src, (
            "auto-trigger gate ``any(et.key == \"inconsistency\" "
            "for et in error_types)`` must be present in main()"
        )
        # And the gate body must import the column-clustering helpers.
        assert "enumerate_candidates_from_frames" in src
        assert "run_for_candidates" in src


class TestColumnClusteringHonorsStoppingRule:
    """The column-clustering pass must honor the same ``--gt-*`` CLI
    flags and ``good_turing.per_arm`` YAML block as the detection
    adaptive loop. We do not run the full pipeline here -- we only
    pin the wiring contract by checking that the gate block calls
    ``_resolve_per_arm_config`` with the right fallbacks and reads
    the ``inconsistency`` per-error-type override.
    """

    def test_resolve_per_arm_config_called_with_cli_fallbacks(
        self,
    ) -> None:
        src = (
            PROJECT_ROOT / "experiments" / "run_cleaning_detection.py"
        ).read_text(encoding="utf-8")
        # The column-clustering block (between the gate and the
        # exception handler) must call ``_resolve_per_arm_config``
        # so YAML defaults + ``--gt-per-arm-*`` knobs propagate.
        gate_idx = src.index('et.key == "inconsistency"')
        # Search a generous window past the gate; the call site is
        # inside the try/except block. The window must be wider than
        # the prompt-ensemble setup since that lives between the
        # stopping-rule resolution and the call into the driver.
        window = src[gate_idx: gate_idx + 10000]
        assert "_resolve_per_arm_config(" in window, (
            "stopping rule wiring missing: the column-clustering "
            "block must call _resolve_per_arm_config so user "
            "--gt-* flags / YAML good_turing.per_arm propagate"
        )
        # Loop-level k_min / k_max / epsilon fallbacks plumbed
        # through so the column-clustering arm uses the same
        # min/num/target-missing-mass as the detection loop.
        for key in ("fallback_k_min", "fallback_k_max", "fallback_epsilon"):
            assert key in window, (
                f"stopping rule wiring missing: {key!r} not "
                "threaded into _resolve_per_arm_config"
            )
        # Per-error-type override must be looked up under
        # "inconsistency".
        assert '"inconsistency" in _cc_arm_by_type' in window
        # And the resolved config -- not ArmConfig.default() -- must
        # be the one handed to the driver.
        assert "arm_config=_cc_arm_config" in window
        assert "arm_config=ArmConfig.default()" not in window, (
            "must not hardcode ArmConfig.default() -- user --gt-* "
            "flags would be silently ignored"
        )

    def test_resolve_per_arm_config_signature_supports_overrides(
        self,
    ) -> None:
        # Sanity guard: the helper still accepts the fallback args
        # our wiring relies on, and a per-error-type override under
        # "inconsistency" merges through.
        from experiments.run_cleaning_detection import (
            _resolve_per_arm_config,
        )

        config = {
            "good_turing": {
                "per_arm": {
                    "defaults": {"epsilon": 0.04, "k_min": 4},
                    "per_error_type": {
                        "inconsistency": {"k_max": 7},
                    },
                },
            },
        }
        default, by_type, _ = _resolve_per_arm_config(config, args=None)
        assert default.epsilon == 0.04
        assert default.k_min == 4
        assert "inconsistency" in by_type
        merged = by_type["inconsistency"].merge_into(default)
        # Per-type overrides apply on top, defaults fill the rest.
        assert merged.k_max == 7
        assert merged.k_min == 4
        assert merged.epsilon == 0.04


class TestColumnClusteringHonorsPromptEnsemble:
    """The column-clustering pass must honor the same
    ``--gt-prompt-ensemble`` toggle as detection. When the user
    enables the ensemble (adaptive mode), the auto-trigger block
    should generate paraphrased prompts and hand them to the
    runner via ``variants=...``. We pin this contract by source
    inspection so the test is fast and offline.
    """

    def test_ensemble_block_is_gated_on_prompt_ensemble_setting(
        self,
    ) -> None:
        src = (
            PROJECT_ROOT / "experiments" / "run_cleaning_detection.py"
        ).read_text(encoding="utf-8")
        gate_idx = src.index('et.key == "inconsistency"')
        window = src[gate_idx: gate_idx + 12000]
        # Toggle: the existing adaptive_settings prompt_ensemble flag
        # must drive the column-clustering ensemble too -- no new
        # error-type-specific CLI flag.
        assert 'adaptive_settings.get("prompt_ensemble")' in window
        # Generator + persister are imported and used inside the
        # ensemble-on branch.
        assert "ClusterPromptVariantGenerator" in window
        assert "save_cluster_variant_records" in window
        # The hardcoded rewriter template path lives in the prompts
        # dir (no new CLI flag for the template either).
        assert "generate_cluster_prompt_variant.yaml" in window
        # And variants are forwarded into the driver call.
        assert "variants=_cc_variants" in window

    def test_generator_template_exists_and_has_base_block(self) -> None:
        import yaml

        path = (
            PROJECT_ROOT
            / "prompts"
            / "generate_cluster_prompt_variant.yaml"
        )
        assert path.is_file()
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert isinstance(raw, dict)
        assert "base" in raw
        base = raw["base"]
        assert "system" in base and "user_template" in base
        # The generator template itself uses Python str.format with
        # these placeholders; the test pins them so a refactor of
        # the generator wouldn't silently drop one.
        ut = base["user_template"]
        for ph in ("{base_system}", "{base_user_template}", "{pass_index}"):
            assert ph in ut, f"generator template missing {ph!r}"
