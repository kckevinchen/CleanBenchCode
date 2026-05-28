"""Schema-deduplication within grouped (multi-table) workbooks.

When ``layout=grouped`` and a single workbook contains two CSVs whose
post-clean column headers are identical, only the first occurrence is
kept. The duplicate is recorded in the manifest and counted under
``skip_reason_counts['duplicate_schema_within_group']``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.data_prep.curate import curate


def _write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _make_rows(n: int, prefix: str) -> pd.DataFrame:
    return pd.DataFrame({
        "name":  [f"{prefix}-name-{i}" for i in range(n)],
        "city":  [f"{prefix}-city-{i}" for i in range(n)],
        "score": list(range(n)),
    })


@pytest.fixture
def grouped_corpus(tmp_path: Path) -> Path:
    src = tmp_path / "corpus"
    # workbook_a: two tables sharing the same schema, one distinct
    _write_csv(src / "workbook_a" / "a1.csv", _make_rows(20, "a1"))
    _write_csv(src / "workbook_a" / "a2.csv", _make_rows(25, "a2"))  # dup of a1
    _write_csv(src / "workbook_a" / "a3.csv", pd.DataFrame({
        "product": [f"p{i}" for i in range(15)],
        "price":   [i * 1.5 for i in range(15)],
    }))
    # workbook_b: two tables, same schema. After dedup only one remains.
    _write_csv(src / "workbook_b" / "b1.csv", _make_rows(12, "b1"))
    _write_csv(src / "workbook_b" / "b2.csv", _make_rows(14, "b2"))
    return src


def _curate_kwargs(out: Path) -> dict:
    return dict(
        output_dir=out,
        num_targets=10,
        min_rows=5, max_rows=200,
        min_columns=2, max_columns=10,
        min_tables=1, max_tables=10,
        seed=0,
        language_check="off",
        max_non_latin_chars=10_000,
        max_empty_ratio=1.0,
        max_cell_chars=None, max_cell_words=None,
        max_oversized_row_ratio=1.0,
        # The within-group dedup tests below predate the cross-group
        # feature; pin it off so workbook_b's identical-schema table
        # isn't swallowed by workbook_a. The dedicated cross-group
        # tests at the bottom of this file exercise that path.
        dedupe_across_groups="off",
    )


def test_dedup_drops_duplicate_schema_within_group(grouped_corpus, tmp_path):
    out = tmp_path / "out"
    manifest = curate(
        source_dir=grouped_corpus,
        dedupe_schema_within_group=True,
        **_curate_kwargs(out),
    )

    assert manifest["layout"] == "grouped"
    by_id = {g["id"]: g for g in manifest["selected"]}

    # workbook_a kept a1 + a3 (a2 dropped as duplicate of a1)
    wa = by_id["workbook_a"]
    kept_a = sorted(t["filename"] for t in wa["tables"])
    assert kept_a == ["a1.csv", "a3.csv"]
    assert wa["dropped_duplicate_schema"] == [
        {"filename": "a2.csv", "duplicate_of": "a1.csv"}
    ]
    # only the kept files were written
    assert (out / "workbook_a" / "a1.csv").exists()
    assert (out / "workbook_a" / "a3.csv").exists()
    assert not (out / "workbook_a" / "a2.csv").exists()

    # workbook_b kept just b1 (b2 dropped)
    wb = by_id["workbook_b"]
    kept_b = sorted(t["filename"] for t in wb["tables"])
    assert kept_b == ["b1.csv"]
    assert wb["dropped_duplicate_schema"] == [
        {"filename": "b2.csv", "duplicate_of": "b1.csv"}
    ]
    assert not (out / "workbook_b" / "b2.csv").exists()

    # skip_reason_counts surfaces the dedup activity
    assert manifest["skip_reason_counts"].get("duplicate_schema_within_group") == 2
    assert manifest["filters"]["dedupe_schema_within_group"] is True


def test_dedup_disabled_keeps_all_duplicates(grouped_corpus, tmp_path):
    out = tmp_path / "out"
    manifest = curate(
        source_dir=grouped_corpus,
        dedupe_schema_within_group=False,
        **_curate_kwargs(out),
    )

    by_id = {g["id"]: g for g in manifest["selected"]}
    kept_a = sorted(t["filename"] for t in by_id["workbook_a"]["tables"])
    assert kept_a == ["a1.csv", "a2.csv", "a3.csv"]
    kept_b = sorted(t["filename"] for t in by_id["workbook_b"]["tables"])
    assert kept_b == ["b1.csv", "b2.csv"]
    assert by_id["workbook_a"]["dropped_duplicate_schema"] == []
    assert by_id["workbook_b"]["dropped_duplicate_schema"] == []
    assert "duplicate_schema_within_group" not in manifest["skip_reason_counts"]
    assert manifest["filters"]["dedupe_schema_within_group"] is False


def test_dedup_below_min_tables_rejects_workbook(tmp_path):
    src = tmp_path / "corpus"
    # Both tables in workbook_x share schema; with min_tables=2 the
    # workbook becomes ineligible after dedup collapses it to 1 table.
    _write_csv(src / "workbook_x" / "t1.csv", _make_rows(10, "t1"))
    _write_csv(src / "workbook_x" / "t2.csv", _make_rows(10, "t2"))

    out = tmp_path / "out"
    kwargs = _curate_kwargs(out)
    kwargs["min_tables"] = 2
    manifest = curate(
        source_dir=src,
        dedupe_schema_within_group=True,
        **kwargs,
    )

    assert manifest["num_targets_selected"] == 0
    assert manifest["skip_reason_counts"].get("too_few_tables_after_dedup") == 1
    assert not (out / "workbook_x").exists() or not any((out / "workbook_x").iterdir())


def test_dedup_noop_in_flat_layout(tmp_path):
    src = tmp_path / "flat"
    src.mkdir()
    # Two flat CSVs with identical schemas — flat layout treats each as
    # its own group, so cross-group duplicates are NOT deduped.
    _write_csv(src / "f1.csv", _make_rows(8, "f1"))
    _write_csv(src / "f2.csv", _make_rows(8, "f2"))

    out = tmp_path / "out"
    manifest = curate(
        source_dir=src,
        dedupe_schema_within_group=True,
        **_curate_kwargs(out),
    )

    assert manifest["layout"] == "flat"
    kept = sorted(t["filename"] for g in manifest["selected"] for t in g["tables"])
    assert kept == ["f1.csv", "f2.csv"]
    assert "duplicate_schema_within_group" not in manifest["skip_reason_counts"]
    assert manifest["filters"]["dedupe_schema_within_group"] is None


# ---------------------------------------------------------------------------
# Cross-group dedup
# ---------------------------------------------------------------------------


def test_dedup_across_groups_schema_skips_duplicate_workbooks(tmp_path):
    """Two workbooks contribute identical-schema single tables.

    Under ``dedupe_across_groups='schema'`` only the first survives; the
    second workbook is rejected wholesale because all its surviving
    tables collide with previously accepted signatures. Models the
    forked-repo / Spark-partition duplication seen in ``curated_git_500``.
    """
    src = tmp_path / "corpus"
    _write_csv(src / "fork_a" / "part.csv", _make_rows(10, "a"))
    _write_csv(src / "fork_b" / "part.csv", _make_rows(10, "b"))
    _write_csv(src / "fork_c" / "part.csv", pd.DataFrame({
        "x": list(range(10)), "y": list(range(10)),
    }))

    out = tmp_path / "out"
    kwargs = _curate_kwargs(out)
    kwargs["dedupe_across_groups"] = "schema"
    manifest = curate(source_dir=src, **kwargs)

    kept_ids = sorted(g["id"] for g in manifest["selected"])
    # ``fork_a`` and ``fork_c`` survive; ``fork_b`` is a schema dup of
    # ``fork_a``. (Order depends on the deterministic shuffle, but the
    # ``fork_*`` ids are stable across runs given seed=0.)
    assert "fork_c" in kept_ids
    assert len(kept_ids) == 2
    assert manifest["skip_reason_counts"].get("all_tables_duplicate_across_groups") == 1
    assert manifest["skip_reason_counts"].get("duplicate_across_groups") == 1
    assert manifest["filters"]["dedupe_across_groups"] == "schema"


def test_dedup_across_groups_content_keeps_same_schema_different_rows(tmp_path):
    """``content`` mode hashes the cleaned body, so two tables with the
    same headers but different rows survive (where ``schema`` would drop
    the second).
    """
    src = tmp_path / "corpus"
    _write_csv(src / "g1" / "t.csv", _make_rows(10, "g1"))
    _write_csv(src / "g2" / "t.csv", _make_rows(10, "g2"))  # same schema, diff rows
    _write_csv(src / "g3" / "t.csv", _make_rows(10, "g1"))  # byte-identical to g1

    out = tmp_path / "out"
    kwargs = _curate_kwargs(out)
    kwargs["dedupe_across_groups"] = "content"
    manifest = curate(source_dir=src, **kwargs)

    kept_ids = sorted(g["id"] for g in manifest["selected"])
    # g1 and g2 survive (different rows); g3 is byte-identical to one of
    # them and is dropped.
    assert len(kept_ids) == 2
    assert manifest["skip_reason_counts"].get("all_tables_duplicate_across_groups") == 1


def test_dedup_across_groups_off_keeps_all(tmp_path):
    src = tmp_path / "corpus"
    _write_csv(src / "g1" / "t.csv", _make_rows(10, "g1"))
    _write_csv(src / "g2" / "t.csv", _make_rows(10, "g2"))  # same schema

    out = tmp_path / "out"
    kwargs = _curate_kwargs(out)
    kwargs["dedupe_across_groups"] = "off"
    manifest = curate(source_dir=src, **kwargs)

    assert sorted(g["id"] for g in manifest["selected"]) == ["g1", "g2"]
    assert "duplicate_across_groups" not in manifest["skip_reason_counts"]


def test_dedup_across_groups_schema_is_fuzzy_and_orderless(tmp_path):
    """Schema dedup normalizes column tokens and ignores order.

    All three groups carry the SAME logical schema but with cosmetic
    differences: snake_case vs camelCase vs space-separated, and
    reordered columns. The fuzzy ``schema`` match should collapse them.
    """
    src = tmp_path / "corpus"
    n = 10
    _write_csv(src / "g1" / "t.csv", pd.DataFrame({
        "user_id": list(range(n)),
        "full_name": [f"a{i}" for i in range(n)],
        "city":      [f"x{i}" for i in range(n)],
    }))
    _write_csv(src / "g2" / "t.csv", pd.DataFrame({
        # Reordered, camelCased, and one column renamed with a space.
        "City":      [f"p{i}" for i in range(n)],
        "userId":    list(range(n, 2 * n)),
        "Full Name": [f"d{i}" for i in range(n)],
    }))
    _write_csv(src / "g3" / "t.csv", pd.DataFrame({
        # Hyphenated + uppercase.
        "USER-ID":   list(range(2 * n, 3 * n)),
        "FULL_NAME": [f"g{i}" for i in range(n)],
        "city":      [f"s{i}" for i in range(n)],
    }))

    out = tmp_path / "out"
    kwargs = _curate_kwargs(out)
    kwargs["dedupe_across_groups"] = "schema"
    manifest = curate(source_dir=src, **kwargs)

    kept = [g["id"] for g in manifest["selected"]]
    # Only one survives - all three normalize to {userid, fullname, city}.
    assert len(kept) == 1, f"expected fuzzy dedup to collapse to 1; got {kept}"
    assert manifest["skip_reason_counts"].get("all_tables_duplicate_across_groups") == 2


def test_orphans_pruned_when_overwrite(tmp_path):
    """With ``overwrite=True``, subdirs from a prior run that are not
    in the new manifest are removed so the on-disk corpus matches.
    """
    src = tmp_path / "corpus"
    _write_csv(src / "g1" / "t.csv", _make_rows(10, "g1"))
    _write_csv(src / "g2" / "t.csv", pd.DataFrame({
        "x": list(range(10)), "y": list(range(10)),
    }))

    out = tmp_path / "out"
    # Simulate a previous run that produced an extra group dir that
    # the new curate won't select (no matching source).
    (out / "stale_leftover").mkdir(parents=True)
    (out / "stale_leftover" / "leftover.csv").write_text("a,b\n1,2\n")

    kwargs = _curate_kwargs(out)
    kwargs["overwrite"] = True
    manifest = curate(source_dir=src, **kwargs)

    assert not (out / "stale_leftover").exists(), "orphan dir should be pruned"
    assert manifest["pruned_orphan_groups"] == ["stale_leftover"]
    # Sanity: the two real groups still got written.
    assert sorted(g["id"] for g in manifest["selected"]) == ["g1", "g2"]


def test_orphans_preserved_without_overwrite(tmp_path):
    """Without ``overwrite``, the curator must not touch pre-existing
    output -- the conservative legacy behaviour.
    """
    src = tmp_path / "corpus"
    _write_csv(src / "g1" / "t.csv", _make_rows(10, "g1"))

    out = tmp_path / "out"
    (out / "stale_leftover").mkdir(parents=True)
    (out / "stale_leftover" / "leftover.csv").write_text("a,b\n1,2\n")

    kwargs = _curate_kwargs(out)
    kwargs["overwrite"] = False
    manifest = curate(source_dir=src, **kwargs)

    assert (out / "stale_leftover" / "leftover.csv").exists()
    assert manifest["pruned_orphan_groups"] == []
