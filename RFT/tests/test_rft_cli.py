"""Smoke tests for the unified RFT CLI / engine (no network).

Run with: ``pytest RFT/tests/test_rft_cli.py`` (the RFT tree is untracked and
outside the default ``testpaths=tests`` glob, so it is opt-in).

These guard the refactor's contract: every registered spec builds a valid RFT
``method`` payload and a working argparse surface, the generic scorer honours
ambiguous FP-immunity, and run-state stays backward-compatible with the legacy
``state.json`` shape.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from RFT._common.engine import RftEngine
from RFT._common.scoring import make_cell_micro_f1_scorer, make_json_array_parser
from RFT._common.spec import Hparams
from RFT._common.state import RunState
from RFT import cli, registry

ALL_TASKS = sorted(registry.REGISTRY)
DUMMY_ENDPOINT = "https://example.openai.azure.com/"


def _engine(name: str, tmp_path: Path) -> RftEngine:
    return RftEngine(registry.get(name), endpoint=DUMMY_ENDPOINT, output_dir=tmp_path)


@pytest.mark.parametrize("name", ALL_TASKS)
def test_spec_builds_valid_method_payload(name, tmp_path):
    spec = registry.get(name)
    eng = _engine(name, tmp_path)
    for variant in spec.reward_variants:
        method = eng._build_method(
            variant=variant, grader_model=spec.default_grader_model, hparams=spec.hparams
        )
        rein = method["reinforcement"]
        assert method["type"] == "reinforcement"
        assert rein["grader"].get("type") in {"multi", "score_model", "python", "label_model"}
        assert rein["response_format"]["type"] in {"json_schema", "json_object"}
        # seed is omitted unless set
        assert "seed" not in rein["hyperparameters"]


@pytest.mark.parametrize("name", ALL_TASKS)
def test_cli_parser_builds_and_reflects_spec(name):
    spec = registry.get(name)
    parser = cli._build_parser(spec)
    # `submit` parses with spec defaults
    args = parser.parse_args(["submit"])
    assert args.base_model == spec.base_model
    assert args.grader_model == spec.default_grader_model
    # reward-variant flag exists iff the task has more than one variant
    has_variant_flag = len(spec.reward_variants) > 1
    assert hasattr(args, "reward_variant") == has_variant_flag
    if has_variant_flag:
        with pytest.raises(SystemExit):
            parser.parse_args(["submit", "--reward-variant", "__nope__"])


@pytest.mark.parametrize("name", ALL_TASKS)
def test_prepare_validates_real_splits(name, tmp_path):
    # The committed splits must exist and be non-empty JSONL.
    _engine(name, tmp_path).prepare()


def test_scorer_ambiguous_is_fp_immune():
    score = make_cell_micro_f1_scorer(gold_field="gold", ambiguous_field="amb")
    rows = [{"gold": json.dumps(["a", "b"]), "amb": json.dumps(["x"])}]
    preds = [{"a", "x", "z"}]  # a=TP, x=ambiguous(immune), z=FP, b=FN
    m = score(rows, preds)
    assert (m["tp"], m["fp"], m["fn"]) == (1, 1, 1)
    assert m["precision"] == 0.5 and m["recall"] == 0.5


def test_parser_handles_fences_and_keys():
    parse = make_json_array_parser("typos")
    assert parse('{"typos":[{"value":"teh"}]}') == {"teh"}
    assert parse('```json\n{"typos":[{"value":"a"}]}\n```') == {"a"}
    assert parse("not json") == set()


def test_runstate_backward_compatible(tmp_path):
    (tmp_path / "state.json").write_text(
        json.dumps({"train_file_id": "f1", "valid_file_id": "f2", "job_id": "j1"})
    )
    st = RunState.load(tmp_path)
    assert (st.train_file_id, st.job_id, st.deployment_name, st.attempts) == ("f1", "j1", None, 0)
    st.fine_tuned_model = "m1"
    st.save(tmp_path)
    reloaded = json.loads((tmp_path / "state.json").read_text())
    assert reloaded["fine_tuned_model"] == "m1" and reloaded["job_id"] == "j1"


def test_hparams_overrides_drop_none():
    base = Hparams(n_epochs=2)
    assert base.with_overrides(n_epochs=None, seed=7).n_epochs == 2
    assert base.with_overrides(n_epochs=None, seed=7).seed == 7


def test_template_spec_builds_valid_method():
    # The shipped template must stay importable and produce a valid grader.
    from RFT._template.spec import SPEC

    eng = RftEngine(SPEC, endpoint=DUMMY_ENDPOINT, output_dir="/tmp/rft_tmpl")
    method = eng._build_method(variant="default", grader_model=SPEC.default_grader_model, hparams=SPEC.hparams)
    grader = method["reinforcement"]["grader"]
    assert grader["type"] == "score_model"
    assert SPEC.prediction_parser is not None  # findings_key wired by make_spec


def test_shared_pr_judge_grader_embeds_rulebook():
    from RFT._common.graders import build_pr_judge_grader

    g = build_pr_judge_grader(
        name="t_pr", rulebook="RULEBOOK_SENTINEL", task_description="widgets",
        findings_key="findings",
    )
    assert g["type"] == "score_model" and g["range"] == [0.0, 5.0]
    assert "RULEBOOK_SENTINEL" in g["input"][0]["content"]
    assert "{{sample.output_text}}" in g["input"][1]["content"]


def test_scaffold_creates_runnable_task(tmp_path):
    from RFT import scaffold

    dst = scaffold.scaffold_task(
        "scaffolded", version="v3", findings_key="hits", dest_root=tmp_path,
    )
    assert dst == tmp_path / "scaffolded" / "v3"
    assert (tmp_path / "scaffolded" / "__init__.py").exists()  # error-type pkg
    spec_py = (dst / "spec.py").read_text()
    assert 'name="scaffolded_v3"' in spec_py
    assert 'error_type="scaffolded"' in spec_py
    assert 'version="v3"' in spec_py
    # imports were rewritten off the template package into the nested path
    assert "RFT._template" not in spec_py
    assert "RFT.scaffolded.v3" in spec_py
    assert (dst / "_prompt.py").read_text().count('FINDINGS_KEY = "hits"') == 1
