"""Regression: an UNSURE inconsistency cluster whose value is dropped via
Per-value corrections (an absent key in ``value_corrections``) -- or whose
LLM consensus says it is already-canonical -- must still export as
``ambiguous`` (FP immunity), NOT vanish from the benchmark.

Background: ``value_corrections`` on a column-cluster label maps each surface
form to its correction; an ABSENT key means "drop this value from the export".
For a TP cluster a drop is correct (the canonical form of a confirmed error is
clean). But for an UNSURE cluster the verdict is ``ambiguous`` -- the reviewer
could not confidently call it an error, so a model that flags ANY of its values
must not be penalised. The per-value UI defaults a value's action to *Drop*
whenever the LLM said it is already-canonical, so an operator can drop such a
value accidentally; dropping it used to strip the cell's FP immunity.

See ``src.labeling.inconsistency_export.materialize_inconsistency_benchmark``
Pass 3.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from src.labeling.inconsistency_export import (
    materialize_inconsistency_benchmark,
)


def _cc_target_id(source: str, column: str, local_id: str) -> str:
    return f"colclust\x00{source}\x00{column}\x00{local_id}"


def _build_run(
    tmp_path: Path, *, status: str, drop_value: str,
) -> Path:
    """A run with one column ``Fund`` holding two variants of the same
    entity. ``value`` A is the surface variant, ``value`` B the canonical
    form. The operator saves ``value_corrections`` that map A -> B and
    DROP B (absent key). ``status`` is the cluster verdict (tp / unsure).
    """
    run_dir = tmp_path / "run"
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    csv = data_dir / "funds.csv"
    # value_a on rows 0,2 (the variant); value_b (canonical) on rows 1,3.
    csv.write_text(
        "id,Fund\n"
        "1,Transportation Imactp Fee\n"      # row 0 - variant
        "2,Transportation Impact Fee\n"      # row 1 - canonical (to be dropped)
        "3,Transportation Imactp Fee\n"      # row 2 - variant
        "4,Transportation Impact Fee\n",     # row 3 - canonical (to be dropped)
        encoding="utf-8",
    )
    run_dir.mkdir(parents=True)
    (run_dir / "summary.json").write_text(
        json.dumps({"config": {"csv_paths": [str(csv)]}}),
    )
    # One draft cluster grouping the two forms.
    (run_dir / "column_cluster_drafts.jsonl").write_text(
        json.dumps({
            "source": "funds.csv", "column": "Fund",
            "draft_clusters": [
                {"kind": "text", "canonical": "Transportation Impact Fee",
                 "values": ["Transportation Imactp Fee",
                            "Transportation Impact Fee"]},
            ],
        }) + "\n",
    )
    # Operator verdict + value_corrections that DROP ``drop_value`` (its key
    # is absent). The variant is corrected to the canonical form.
    (run_dir / "labels.jsonl").write_text(
        json.dumps({
            "target": "column_cluster",
            "target_id": _cc_target_id("funds.csv", "Fund", "c1"),
            "source": "funds.csv", "column": "Fund", "columns": ["Fund"],
            "status": status,
            "value_corrections": {
                "Transportation Imactp Fee": "Transportation Impact Fee",
            },
            "note": "[llm-suggested]", "annotator": "t",
        }) + "\n",
    )
    return run_dir


def test_unsure_dropped_value_still_exports_ambiguous(tmp_path: Path) -> None:
    stats: Counter = Counter()
    cells = materialize_inconsistency_benchmark(
        _build_run(tmp_path, status="unsure",
                   drop_value="Transportation Impact Fee"),
        dry_run=True, stats=stats,
    )
    by_val = {}
    for c in cells:
        by_val.setdefault(c.original_value, []).append(c)

    # The dropped canonical value must STILL be present, as ambiguous with
    # no correction (ambiguous cells carry no ground-truth fix).
    dropped = by_val.get("Transportation Impact Fee", [])
    assert len(dropped) == 2, "dropped unsure value must not vanish"
    assert all(c.label == "ambiguous" for c in dropped)
    assert all(c.corrected_value is None for c in dropped)

    # The variant also exports as ambiguous (unsure verdict).
    variant = by_val.get("Transportation Imactp Fee", [])
    assert len(variant) == 2
    assert all(c.label == "ambiguous" for c in variant)

    assert stats.get("ambiguous_groups_kept_despite_drop", 0) >= 1


def test_tp_dropped_value_is_still_dropped(tmp_path: Path) -> None:
    """The fix is scoped to unsure: for a TP cluster, dropping the canonical
    value must STILL remove it (it is genuinely clean, not an error)."""
    stats: Counter = Counter()
    cells = materialize_inconsistency_benchmark(
        _build_run(tmp_path, status="tp",
                   drop_value="Transportation Impact Fee"),
        dry_run=True, stats=stats,
    )
    by_val = {}
    for c in cells:
        by_val.setdefault(c.original_value, []).append(c)

    # The dropped canonical value is absent (clean) for a TP cluster.
    assert "Transportation Impact Fee" not in by_val
    # The variant exports as an error with the canonical correction.
    variant = by_val.get("Transportation Imactp Fee", [])
    assert len(variant) == 2
    assert all(c.label == "error" for c in variant)
    assert all(
        c.corrected_value == "Transportation Impact Fee" for c in variant
    )
    assert stats.get("ambiguous_groups_kept_despite_drop", 0) == 0
