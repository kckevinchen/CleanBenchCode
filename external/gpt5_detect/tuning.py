"""Pure helpers for leakage-safe GPT-5.5 prompt tuning.

The live orchestrator is :mod:`tune_prompt`. This module contains deterministic
split, audit, dossier, and prompt-validation logic so those safety properties can
be tested without making LLM calls.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

CELL_TYPES = ("typo", "inconsistency", "semantic_incompatible", "numeric_outlier")
REQUIRED_PROMPT_FIELDS = ("issues", "row", "column", "error_type", "confidence")
GENERIC_HEADERS = {
    "address",
    "category",
    "city",
    "code",
    "country",
    "date",
    "description",
    "email",
    "field",
    "format",
    "id",
    "model",
    "name",
    "number",
    "record",
    "row",
    "size",
    "state",
    "status",
    "table",
    "type",
    "value",
    "year",
}
FORBIDDEN_FEEDBACK_PATTERNS = (
    (
        "benchmark_reference",
        re.compile(r"\b(?:benchmark|data\s*set|dataset|corpus|source\s+file)\b", re.I),
    ),
    (
        "location_reference",
        re.compile(r"\b(?:row|column|table|file|source)\s*(?:number|index|id)?\s*[#:=]?\s*\d+\b", re.I),
    ),
    ("file_reference", re.compile(r"\.(?:csv|xlsx?|jsonl?)\b", re.I)),
    ("hash_identifier", re.compile(r"\b[0-9a-f]{12,}\b", re.I)),
    ("numeric_literal", re.compile(r"(?<!\w)[-+]?\d+(?:\.\d+)?%?(?!\w)")),
    ("example", re.compile(r"\b(?:e\.g\.|for example|such as this)\b", re.I)),
)


def stable_digest(*parts: object) -> str:
    text = "\x1f".join(str(part) for part in parts)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def benchmark_fingerprint(ds_root: Path) -> str:
    """Fingerprint the manifest, ground truth, and table bytes in evaluation scope."""
    manifest_path = ds_root / "_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))["datasets"]
    digest = hashlib.sha256(manifest_path.read_bytes())
    for dataset in sorted(manifest):
        gt_path = ds_root / dataset / "gt_cells.jsonl"
        dirty_path = ds_root / dataset / "dirty.csv"
        digest.update(dataset.encode("utf-8"))
        if gt_path.exists():
            digest.update(gt_path.read_bytes())
        if dirty_path.exists():
            digest.update(dirty_path.read_bytes())
    return digest.hexdigest()


def _dataset_error_types(ds_root: Path, dataset: str) -> set[str]:
    types: set[str] = set()
    path = ds_root / dataset / "gt_cells.jsonl"
    if not path.exists():
        return types
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("status") != "error":
            continue
        types.update(str(value) for value in record.get("error_types", []))
    return types


def _stratified_dev_counts(
    strata: dict[str, list[str]],
    fraction: float,
) -> dict[str, int]:
    """Allocate one exact rounded global target using deterministic remainders."""
    total = sum(len(values) for values in strata.values())
    target = min(total - 1, max(1, round(total * fraction)))
    raw = {name: len(values) * fraction for name, values in strata.items()}
    minimum = {name: int(len(values) > 1) for name, values in strata.items()}
    maximum = {
        name: len(values) - int(len(values) > 1) for name, values in strata.items()
    }
    counts = {
        name: min(maximum[name], max(minimum[name], math.floor(raw[name])))
        for name in strata
    }
    while sum(counts.values()) < target:
        candidates = [name for name in strata if counts[name] < maximum[name]]
        if not candidates:
            raise ValueError("cannot allocate requested development split")
        chosen = max(candidates, key=lambda name: (raw[name] - counts[name], name))
        counts[chosen] += 1
    while sum(counts.values()) > target:
        candidates = [name for name in strata if counts[name] > minimum[name]]
        if not candidates:
            raise ValueError("cannot reduce development split to requested size")
        chosen = max(candidates, key=lambda name: (counts[name] - raw[name], name))
        counts[chosen] -= 1
    return counts


def _ranked(values: list[str], seed: int, stratum: str) -> list[str]:
    return sorted(values, key=lambda value: stable_digest(seed, stratum, value))


def _coverage_by_type(ds_root: Path, datasets: set[str]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for dataset in datasets:
        for error_type in _dataset_error_types(ds_root, dataset):
            counts[error_type] += 1
    return dict(sorted(counts.items()))


def create_dataset_split(
    ds_root: Path,
    *,
    dev_fraction: float = 0.2,
    seed: int = 20260713,
) -> dict[str, Any]:
    """Create a deterministic table-level development/test split.

    Error-bearing and no-error tables are ranked independently, preserving the
    benchmark's clean-table proportion. A small deterministic repair ensures each
    error type represented by at least two tables occurs on both sides.
    """
    if not 0.0 < dev_fraction < 1.0:
        raise ValueError("dev_fraction must be strictly between 0 and 1")
    manifest = json.loads((ds_root / "_manifest.json").read_text(encoding="utf-8"))[
        "datasets"
    ]
    positive = [
        dataset
        for dataset, entry in manifest.items()
        if int(entry.get("n_error_cells", 0)) > 0
    ]
    positive_set = set(positive)
    no_error = [dataset for dataset in manifest if dataset not in positive_set]

    strata = {"positive": positive, "no_error": no_error}
    dev_counts = _stratified_dev_counts(strata, dev_fraction)
    dev = set(_ranked(positive, seed, "positive")[: dev_counts["positive"]])
    dev.update(_ranked(no_error, seed, "no_error")[: dev_counts["no_error"]])
    all_datasets = set(manifest)
    test = all_datasets - dev

    types_by_dataset = {
        dataset: _dataset_error_types(ds_root, dataset) for dataset in positive
    }
    holders: dict[str, set[str]] = defaultdict(set)
    for dataset, error_types in types_by_dataset.items():
        for error_type in error_types:
            holders[error_type].add(dataset)

    def removable(pool: set[str], required_side: set[str]) -> list[str]:
        candidates = []
        for dataset in pool:
            error_types = types_by_dataset.get(dataset, set())
            if all(len(holders[t] & required_side) > 1 for t in error_types):
                candidates.append(dataset)
        return _ranked(candidates, seed, "coverage-swap")

    for error_type, type_holders in sorted(holders.items()):
        if len(type_holders) < 2:
            continue
        if not (type_holders & dev):
            incoming = _ranked(list(type_holders & test), seed, f"dev-{error_type}")[0]
            outgoing_options = removable(dev & positive_set, dev)
            if not outgoing_options:
                outgoing_options = _ranked(list(dev & set(no_error)), seed, "dev-no-error-swap")
            if not outgoing_options:
                raise ValueError(f"cannot preserve development coverage for {error_type}")
            outgoing = outgoing_options[0]
            dev.remove(outgoing)
            test.add(outgoing)
            test.remove(incoming)
            dev.add(incoming)
        if not (type_holders & test):
            incoming = _ranked(list(type_holders & dev), seed, f"test-{error_type}")[0]
            outgoing_options = removable(test & positive_set, test)
            if not outgoing_options:
                outgoing_options = _ranked(list(test & set(no_error)), seed, "test-no-error-swap")
            if not outgoing_options:
                raise ValueError(f"cannot preserve test coverage for {error_type}")
            outgoing = outgoing_options[0]
            test.remove(outgoing)
            dev.add(outgoing)
            dev.remove(incoming)
            test.add(incoming)

    if dev & test or dev | test != all_datasets:
        raise AssertionError("development and test splits must be disjoint and exhaustive")

    dev_positive = dev & positive_set
    test_positive = test & positive_set
    return {
        "schema_version": 1,
        "seed": seed,
        "dev_fraction": dev_fraction,
        "benchmark_sha256": benchmark_fingerprint(ds_root),
        "development": sorted(dev),
        "test": sorted(test),
        "summary": {
            "total": len(all_datasets),
            "development": len(dev),
            "test": len(test),
            "development_error_bearing": len(dev_positive),
            "development_no_error": len(dev) - len(dev_positive),
            "test_error_bearing": len(test_positive),
            "test_no_error": len(test) - len(test_positive),
            "development_type_tables": _coverage_by_type(ds_root, dev),
            "test_type_tables": _coverage_by_type(ds_root, test),
        },
    }


def validate_split(split: dict[str, Any], ds_root: Path) -> None:
    manifest = json.loads((ds_root / "_manifest.json").read_text(encoding="utf-8"))[
        "datasets"
    ]
    development = split.get("development")
    test = split.get("test")
    if not isinstance(development, list) or not isinstance(test, list):
        raise ValueError("split must contain development and test lists")
    if len(development) != len(set(development)) or len(test) != len(set(test)):
        raise ValueError("split contains duplicate dataset identifiers")
    dev_set, test_set = set(development), set(test)
    if dev_set & test_set:
        raise ValueError("development and test splits overlap")
    if dev_set | test_set != set(manifest):
        raise ValueError("split does not exactly cover the current manifest")
    if split.get("benchmark_sha256") != benchmark_fingerprint(ds_root):
        raise ValueError("split fingerprint does not match the current benchmark")


def write_split_files(run_dir: Path, split: dict[str, Any]) -> None:
    atomic_write_json(run_dir / "split.json", split)
    atomic_write_json(
        run_dir / "development_datasets.json",
        {"datasets": split["development"], "benchmark_sha256": split["benchmark_sha256"]},
    )
    atomic_write_json(
        run_dir / "test_datasets.json",
        {"datasets": split["test"], "benchmark_sha256": split["benchmark_sha256"]},
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _stable_sample(
    values: set[tuple[str, int, int]],
    limit: int,
    seed: str,
) -> list[tuple[str, int, int]]:
    return sorted(values, key=lambda value: stable_digest(seed, *value))[:limit]


def _table_context(
    ds_root: Path,
    manifest: dict[str, Any],
    key: tuple[str, int, int],
    *,
    radius: int = 1,
    max_columns: int = 12,
    max_value_chars: int = 120,
) -> dict[str, Any]:
    dataset, row, col = key
    dirty_path = ds_root / dataset / "dirty.csv"
    with dirty_path.open(encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        headers = next(reader)
        rows = list(reader)
    selected_indexes = list(range(min(len(headers), max_columns)))
    if col not in selected_indexes:
        selected_indexes[-1] = col
        selected_indexes = sorted(set(selected_indexes))

    def clip(value: str) -> str:
        return value if len(value) <= max_value_chars else value[:max_value_chars] + "..."

    nearby = []
    for index in range(max(0, row - radius), min(len(rows), row + radius + 1)):
        values = rows[index]
        nearby.append(
            {
                "row": index,
                "values": {
                    headers[j]: clip(values[j] if j < len(values) else "")
                    for j in selected_indexes
                },
            }
        )
    target_value = ""
    if 0 <= row < len(rows) and 0 <= col < len(rows[row]):
        target_value = clip(rows[row][col])
    return {
        "dataset": dataset,
        "source": manifest[dataset]["source"],
        "row": row,
        "column": headers[col] if 0 <= col < len(headers) else f"column_{col}",
        "target_value": target_value,
        "nearby_rows": nearby,
    }


def build_evaluation_dossier(
    ds_root: Path,
    datasets: list[str],
    predictions_path: Path,
    issues_path: Path,
    score_path: Path,
    *,
    sample_limit: int = 6,
    sample_seed: str = "prompt-tuning",
) -> dict[str, Any]:
    """Build a bounded development-only dossier for the evaluator agent."""
    manifest = json.loads((ds_root / "_manifest.json").read_text(encoding="utf-8"))[
        "datasets"
    ]
    scope = set(datasets)
    errors: set[tuple[str, int, int]] = set()
    ambiguous: set[tuple[str, int, int]] = set()
    types_by_cell: dict[tuple[str, int, int], list[str]] = {}
    for dataset in datasets:
        for record in _read_jsonl(ds_root / dataset / "gt_cells.jsonl"):
            key = (dataset, int(record["i"]), int(record["j"]))
            if record.get("status") == "error":
                errors.add(key)
                types_by_cell[key] = sorted(str(t) for t in record.get("error_types", []))
            else:
                ambiguous.add(key)

    predictions = {
        (str(record["dataset"]), int(record["row"]), int(record["col"]))
        for record in _read_jsonl(predictions_path)
        if record.get("dataset") in scope
    }
    scored_predictions = predictions - ambiguous
    true_positives = scored_predictions & errors
    false_positives = scored_predictions - errors
    false_negatives = errors - scored_predictions

    issue_by_cell: dict[tuple[str, int, int], dict[str, Any]] = {}
    for record in _read_jsonl(issues_path):
        key = (str(record["dataset"]), int(record["row"]), int(record["col"]))
        issue_by_cell.setdefault(key, record)

    def render(key: tuple[str, int, int], kind: str) -> dict[str, Any]:
        context = _table_context(ds_root, manifest, key)
        context["classification"] = kind
        if key in types_by_cell:
            context["gold_error_types"] = types_by_cell[key]
        issue = issue_by_cell.get(key)
        if issue:
            context["predicted_error_type"] = issue.get("error_type")
            context["model_explanation"] = issue.get("explanation")
            context["model_confidence"] = issue.get("confidence")
        return context

    score_payload = json.loads(score_path.read_text(encoding="utf-8"))
    if len(score_payload.get("results", {})) != 1:
        raise ValueError("prompt-tuning score must contain exactly one detector result")
    metrics = next(iter(score_payload["results"].values()))
    return {
        "scope": {
            "development_tables": len(datasets),
            "note": "Only development data is present. No test identifiers or data are included.",
        },
        "aggregate_metrics": metrics,
        "sampled_false_negatives": [
            render(key, "false_negative")
            for key in _stable_sample(false_negatives, sample_limit, sample_seed + ":fn")
        ],
        "sampled_false_positives": [
            render(key, "false_positive")
            for key in _stable_sample(false_positives, sample_limit, sample_seed + ":fp")
        ],
        "sampled_true_positives": [
            render(key, "true_positive")
            for key in _stable_sample(true_positives, sample_limit, sample_seed + ":tp")
        ],
    }


def _all_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [text for child in value.values() for text in _all_strings(child)]
    if isinstance(value, list):
        return [text for child in value for text in _all_strings(child)]
    return []


def _dossier_sensitive_values(dossier: dict[str, Any]) -> dict[str, set[str]]:
    values: dict[str, set[str]] = {
        "identifier": set(),
        "header": set(),
        "cell_value": set(),
    }

    def sensitive_header(value: str) -> bool:
        normalized = value.strip()
        if not normalized or normalized.casefold() in GENERIC_HEADERS:
            return False
        # Short one-word headers often collide with ordinary evaluator prose
        # ("group", "order", "no"). Leave those ambiguous cases to the LLM
        # judge; deterministically block only high-confidence schema phrases.
        return (
            len(normalized) >= 8
            or len(normalized.split()) > 1
            or bool(re.search(r"[_/\\()[\]{}]", normalized))
        )

    for group in (
        "sampled_false_negatives",
        "sampled_false_positives",
        "sampled_true_positives",
    ):
        for example in dossier.get(group, []):
            values["identifier"].update(
                str(example.get(key, "")).strip()
                for key in ("dataset", "source")
                if example.get(key)
            )
            header = str(example.get("column", "")).strip()
            if sensitive_header(header):
                values["header"].add(header)
            target = str(example.get("target_value", "")).strip()
            if len(target) >= 4:
                values["cell_value"].add(target)
            for row in example.get("nearby_rows", []):
                for raw_column, raw_cell in row.get("values", {}).items():
                    column = str(raw_column).strip()
                    if sensitive_header(column):
                        values["header"].add(column)
                    cell = str(raw_cell).strip()
                    if len(cell) >= 6:
                        values["cell_value"].add(cell)
    return values


def scan_for_development_leak(
    payload: Any,
    dossier: dict[str, Any],
    *,
    allowed_text: str = "",
) -> dict[str, Any]:
    """Fail-closed scan for direct data, identifier, location, or format leakage."""
    text = "\n".join(_all_strings(payload))
    lowered = text.casefold()
    allowed = allowed_text.casefold()
    violations: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def add(category: str, message: str) -> None:
        key = (category, message)
        if key not in seen:
            seen.add(key)
            violations.append({"category": category, "message": message})

    for category, pattern in FORBIDDEN_FEEDBACK_PATTERNS:
        new_matches = [
            match.group(0)
            for match in pattern.finditer(text)
            if match.group(0).casefold() not in allowed
        ]
        if new_matches:
            add(category, f"contains a forbidden {category.replace('_', ' ')}")

    for category, phrases in _dossier_sensitive_values(dossier).items():
        for phrase in phrases:
            normalized = phrase.casefold()
            if normalized in allowed or normalized not in lowered:
                continue
            if category == "cell_value" and len(normalized) < 6:
                continue
            add(category, f"contains an exact development {category.replace('_', ' ')}")
            break

    return {"passed": not violations, "violations": violations}


def validate_evaluator_feedback(payload: Any) -> list[str]:
    errors = []
    if not isinstance(payload, dict):
        return ["feedback must be a JSON object"]
    if not isinstance(payload.get("summary"), str) or not payload["summary"].strip():
        errors.append("summary must be a non-empty string")
    suggestions = payload.get("suggestions")
    if not isinstance(suggestions, list) or not 1 <= len(suggestions) <= 6:
        errors.append("suggestions must contain between one and six entries")
        return errors
    allowed_categories = {"precision", "recall", "taxonomy", "reasoning", "output_contract"}
    for index, suggestion in enumerate(suggestions):
        if not isinstance(suggestion, dict):
            errors.append(f"suggestion {index} must be an object")
            continue
        if suggestion.get("category") not in allowed_categories:
            errors.append(f"suggestion {index} has an invalid category")
        for field in ("change", "rationale"):
            if not isinstance(suggestion.get(field), str) or not suggestion[field].strip():
                errors.append(f"suggestion {index}.{field} must be a non-empty string")
    return errors


def validate_judge_verdict(payload: Any) -> list[str]:
    if not isinstance(payload, dict):
        return ["judge verdict must be a JSON object"]
    errors = []
    if not isinstance(payload.get("passed"), bool):
        errors.append("passed must be a boolean")
    violations = payload.get("violations")
    if not isinstance(violations, list):
        errors.append("violations must be a list")
    else:
        allowed = {
            "benchmark_reference",
            "data_identifier",
            "data_value",
            "domain_specificity",
            "exact_format",
            "example",
            "location",
            "metric",
            "none",
        }
        if not all(isinstance(value, str) and value in allowed for value in violations):
            errors.append("each violation must be an allowed generic category")
    if payload.get("passed") is True and violations:
        errors.append("a passing verdict must have no violations")
    if payload.get("passed") is False and not violations:
        errors.append("a failing verdict must include at least one violation category")
    return errors


def validate_candidate_prompt(
    candidate: str,
    *,
    previous_prompt: str,
    dossier: dict[str, Any] | None = None,
) -> list[str]:
    errors = []
    stripped = candidate.strip()
    if len(stripped) < 500:
        errors.append("prompt is too short to retain the task and output contract")
    if len(stripped) > 60_000:
        errors.append("prompt exceeds the 60,000-character safety limit")
    lowered = stripped.casefold()
    for field in REQUIRED_PROMPT_FIELDS:
        if field.casefold() not in lowered:
            errors.append(f"prompt is missing required output field {field!r}")
    for error_type in CELL_TYPES:
        if error_type.casefold() not in lowered:
            errors.append(f"prompt is missing error type {error_type!r}")
    leak_report = scan_for_development_leak(
        candidate,
        dossier or {},
        allowed_text=previous_prompt,
    )
    errors.extend(item["message"] for item in leak_report["violations"])
    return errors


def select_best_round(rounds: list[dict[str, Any]]) -> dict[str, Any]:
    """Select by development F1, then precision, recall, then earliest round."""
    if not rounds:
        raise ValueError("at least one completed round is required")

    def key(record: dict[str, Any]) -> tuple[float, float, float, int]:
        metrics = record["metrics"]
        return (
            float(metrics["f1"]),
            float(metrics["precision"]),
            float(metrics["recall"]),
            -int(record["round"]),
        )

    return max(rounds, key=key)


def score_summary(score_path: Path) -> dict[str, Any]:
    payload = json.loads(score_path.read_text(encoding="utf-8"))
    results = payload.get("results", {})
    if len(results) != 1:
        raise ValueError(f"{score_path} must contain exactly one detector result")
    return next(iter(results.values()))


def estimated_dev_calls(ds_root: Path, datasets: list[str], chunk_size: int) -> int:
    """Estimate target-model calls from CSV row counts without loading cell values."""
    total = 0
    for dataset in datasets:
        path = ds_root / dataset / "dirty.csv"
        with path.open(encoding="utf-8", newline="") as stream:
            row_count = max(0, sum(1 for _ in csv.reader(stream)) - 1)
        total += max(1, math.ceil(row_count / chunk_size))
    return total
