from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from external.gpt5_detect import run as detector_run  # noqa: E402
from external.gpt5_detect import tune_prompt, tuning  # noqa: E402


def _write_dataset(
    root: Path,
    name: str,
    *,
    error_type: str | None,
    value: str = "ordinary",
) -> dict:
    dataset = root / name
    dataset.mkdir(parents=True)
    (dataset / "dirty.csv").write_text(f"Feature,Context\n{value},peer\n", encoding="utf-8")
    records = []
    if error_type is not None:
        records.append(
            {
                "i": 0,
                "j": 0,
                "status": "error",
                "error_types": [error_type],
                "original_value": value,
            }
        )
    (dataset / "gt_cells.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    return {
        "dataset": name,
        "source": f"{name}.csv",
        "n_error_cells": len(records),
        "n_ambiguous_cells": 0,
    }


def _fake_benchmark(tmp_path: Path) -> Path:
    root = tmp_path / "datasets"
    root.mkdir()
    manifest = {}
    error_types = (
        "typo-spelling",
        "inconsistency",
        "semantic_incompatible",
        "numeric_outlier",
    )
    for index, error_type in enumerate(error_types):
        for duplicate in range(3):
            name = f"positive_{index}_{duplicate}"
            manifest[name] = _write_dataset(root, name, error_type=error_type)
    for index in range(48):
        name = f"clean_{index:02d}"
        manifest[name] = _write_dataset(root, name, error_type=None)
    (root / "_manifest.json").write_text(
        json.dumps({"datasets": manifest, "missing_sources": []}),
        encoding="utf-8",
    )
    return root


def _dossier() -> dict:
    return {
        "sampled_false_negatives": [
            {
                "dataset": "abc123def456",
                "source": "private_source.csv",
                "row": 17,
                "column": "Patient Code",
                "target_value": "ZX-9912",
                "nearby_rows": [
                    {"row": 16, "values": {"Patient Code": "ZX-9911", "Region": "Northwest"}}
                ],
            }
        ],
        "sampled_false_positives": [],
        "sampled_true_positives": [],
    }


def _valid_prompt() -> str:
    body = """
You audit arbitrary tabular data for typo, inconsistency, semantic_incompatible,
and numeric_outlier errors. Require evidence from the affected cell and peer
values, avoid unsupported assumptions, and ignore empty values. Return a JSON
object with an issues array. Every issue must contain row, column, error_type,
observed, explanation, corrected_value, and confidence. Use the exact source
row index and exact column label supplied to you. Emit no prose outside JSON.
"""
    return (body + "\nGeneral evidence must remain domain independent." * 20).strip()


def test_split_is_deterministic_disjoint_and_type_covered(tmp_path: Path) -> None:
    root = _fake_benchmark(tmp_path)
    first = tuning.create_dataset_split(root, dev_fraction=0.2, seed=41)
    second = tuning.create_dataset_split(root, dev_fraction=0.2, seed=41)

    assert first == second
    assert not set(first["development"]) & set(first["test"])
    assert set(first["development"]) | set(first["test"]) == set(
        json.loads((root / "_manifest.json").read_text())["datasets"]
    )
    assert len(first["development"]) == 12
    for counts in (
        first["summary"]["development_type_tables"],
        first["summary"]["test_type_tables"],
    ):
        assert set(counts) == {
            "inconsistency",
            "numeric_outlier",
            "semantic_incompatible",
            "typo-spelling",
        }
    tuning.validate_split(first, root)


def test_split_fingerprint_includes_table_contents(tmp_path: Path) -> None:
    root = _fake_benchmark(tmp_path)
    split = tuning.create_dataset_split(root, dev_fraction=0.2, seed=41)
    changed = root / split["development"][0] / "dirty.csv"
    changed.write_text(changed.read_text() + "new,content\n", encoding="utf-8")

    with pytest.raises(ValueError, match="fingerprint"):
        tuning.validate_split(split, root)


def test_round_artifact_tampering_is_detected(tmp_path: Path) -> None:
    run_dir = tmp_path.resolve()
    prompt = run_dir / "prompt.txt"
    predictions = run_dir / "predictions.jsonl"
    score = run_dir / "score.json"
    prompt.write_text("original", encoding="utf-8")
    predictions.write_text("", encoding="utf-8")
    score.write_text("{}", encoding="utf-8")
    state = {
        "rounds": [
            {
                "prompt_path": str(prompt),
                "prompt_sha256": tuning.file_sha256(prompt),
                "predictions_path": str(predictions),
                "predictions_sha256": tuning.file_sha256(predictions),
                "score_path": str(score),
                "score_sha256": tuning.file_sha256(score),
            }
        ]
    }
    tune_prompt._verify_round_artifacts(state, run_dir)
    prompt.write_text("modified", encoding="utf-8")

    with pytest.raises(ValueError, match="changed after scoring"):
        tune_prompt._verify_round_artifacts(state, run_dir)


def test_leak_scanner_allows_generic_feedback() -> None:
    feedback = {
        "summary": "Require stronger peer evidence before flagging plausible variation.",
        "suggestions": [
            {
                "category": "precision",
                "change": "Distinguish isolated defects from legitimate variation.",
                "rationale": "Generic evidence thresholds reduce unsupported flags.",
            }
        ],
    }

    assert tuning.scan_for_development_leak(feedback, _dossier())["passed"]


def test_leak_scanner_allows_generic_single_word_headers() -> None:
    dossier = _dossier()
    dossier["sampled_false_negatives"][0]["column"] = "Model"
    dossier["sampled_false_negatives"][0]["nearby_rows"][0]["values"].update(
        {"Row": "ordinary", "State": "ordinary", "Size": "ordinary"}
    )
    feedback = {
        "summary": "The model should reason at the row level before changing state.",
        "suggestions": [
            {
                "category": "reasoning",
                "change": "Use row context and model confidence without assuming size.",
                "rationale": "Generic context can improve the model decision.",
            }
        ],
    }

    assert tuning.scan_for_development_leak(feedback, dossier)["passed"]


def test_leak_scanner_defers_short_header_collisions_to_judge() -> None:
    dossier = _dossier()
    dossier["sampled_false_negatives"][0]["column"] = "Order"
    feedback = {
        "summary": "Order the generic checks by confidence.",
        "suggestions": [
            {
                "category": "reasoning",
                "change": "Use a stable reasoning order.",
                "rationale": "A generic order makes decisions consistent.",
            }
        ],
    }

    assert tuning.scan_for_development_leak(feedback, dossier)["passed"]


def test_leak_scanner_rejects_data_and_format_details() -> None:
    leaked = {
        "summary": "Use Patient Code from private_source.csv at row 17.",
        "suggestions": [
            {
                "category": "precision",
                "change": "Treat ZX-9912 as suspicious when the score is 0.25.",
                "rationale": "Northwest reveals the observed format.",
            }
        ],
    }

    report = tuning.scan_for_development_leak(leaked, _dossier())
    categories = {item["category"] for item in report["violations"]}
    assert not report["passed"]
    assert {"header", "identifier", "cell_value", "location_reference", "numeric_literal"} <= categories


def test_prompt_validation_permits_existing_contract_but_blocks_new_leak() -> None:
    previous = _valid_prompt()
    assert not tuning.validate_candidate_prompt(
        previous,
        previous_prompt=previous,
        dossier=_dossier(),
    )

    leaked = previous + "\nPrioritize the Patient Code field."
    errors = tuning.validate_candidate_prompt(
        leaked,
        previous_prompt=previous,
        dossier=_dossier(),
    )
    assert any("development header" in error for error in errors)


def test_best_round_uses_f1_then_precision_then_earliest() -> None:
    rounds = [
        {"round": 0, "metrics": {"f1": 0.2, "precision": 0.4, "recall": 0.1}},
        {"round": 1, "metrics": {"f1": 0.3, "precision": 0.2, "recall": 0.6}},
        {"round": 2, "metrics": {"f1": 0.3, "precision": 0.2, "recall": 0.6}},
    ]
    assert tuning.select_best_round(rounds)["round"] == 1


def test_exact_dataset_list_validation(tmp_path: Path) -> None:
    path = tmp_path / "split.json"
    path.write_text(json.dumps({"datasets": ["a", "b"]}), encoding="utf-8")
    assert detector_run._load_dataset_list(path, {"a", "b", "c"}) == ["a", "b"]

    path.write_text(json.dumps(["a", "missing"]), encoding="utf-8")
    try:
        detector_run._load_dataset_list(path, {"a"})
    except ValueError as exc:
        assert "unknown datasets" in str(exc)
    else:
        raise AssertionError("unknown dataset must be rejected")


def test_request_cache_hash_binds_prompt_and_reasoning_effort() -> None:
    baseline = detector_run._request_sha("system", "user", "medium")
    assert detector_run._request_sha("changed", "user", "medium") != baseline
    assert detector_run._request_sha("system", "changed", "medium") != baseline
    assert detector_run._request_sha("system", "user", "high") != baseline


def test_dossier_contains_development_data_only(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    root.mkdir()
    manifest = {
        "dev": _write_dataset(root, "dev", error_type="typo-spelling", value="dev-secret"),
        "test": _write_dataset(root, "test", error_type="typo-spelling", value="test-secret"),
    }
    (root / "_manifest.json").write_text(json.dumps({"datasets": manifest}), encoding="utf-8")
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text("", encoding="utf-8")
    issues = tmp_path / "issues.jsonl"
    issues.write_text("", encoding="utf-8")
    score = tmp_path / "score.json"
    score.write_text(
        json.dumps(
            {
                "results": {
                    "candidate": {
                        "precision": 0.0,
                        "recall": 0.0,
                        "f1": 0.0,
                        "per_type_recall": {},
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    dossier = tuning.build_evaluation_dossier(
        root,
        ["dev"],
        predictions,
        issues,
        score,
        sample_limit=2,
    )
    serialized = json.dumps(dossier)
    assert "dev-secret" in serialized
    assert "test-secret" not in serialized
    assert '"test"' not in serialized


def test_empty_predictions_are_scored(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    root.mkdir()
    manifest = {"sample": _write_dataset(root, "sample", error_type="typo-spelling")}
    (root / "_manifest.json").write_text(json.dumps({"datasets": manifest}), encoding="utf-8")
    predictions = tmp_path / "predictions_empty.jsonl"
    predictions.write_text("", encoding="utf-8")
    Path(str(predictions) + ".ran.json").write_text(
        json.dumps({"per_detector": {"empty": ["sample"]}, "all_loaded": ["sample"]}),
        encoding="utf-8",
    )
    neutral = tmp_path / "neutral.jsonl"
    neutral.write_text("", encoding="utf-8")
    output = tmp_path / "score.json"

    subprocess.run(
        [
            sys.executable,
            "external/rein/score.py",
            "--predictions",
            str(predictions),
            "--datasets-dir",
            str(root),
            "--neutral-file",
            str(neutral),
            "--out",
            str(output),
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )
    metrics = json.loads(output.read_text())["results"]["empty"]
    assert metrics["n_predicted"] == 0
    assert metrics["fn"] == 1
    assert metrics["f1"] == 0.0
