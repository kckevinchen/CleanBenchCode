"""Smoke test: feed the gpt-5.5 reason-high prompt-simple TSV through the
``build_per_cell_view`` pipeline with the ``curated_bing21_200`` dataset
attached, and report how the ``unmatched_prediction`` bucket splits into
``bench_gap`` vs ``unmatched_hallucination``.

Run from the project root::

    venv/bin/python3 scripts/smoke_bench_gap.py
"""
from __future__ import annotations

from pathlib import Path

from experiments.compare_external_typo_run import (
    build_per_cell_view,
    load_benchmark,
    load_predictions,
)
from src.core.data_loader import load_csv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCH = PROJECT_ROOT / "label_result" / "benchmark_typo_20260507T215150Z_gpt5-family.canonical.jsonl"
PREDS = PROJECT_ROOT / "external" / "GPT_score" / "output_20260509_060447_gpt-5.5_reason-high_prompt-simple_n1335_typo_detection.scoring.tsv"
DATASET_DIR = PROJECT_ROOT / "data" / "curated_bing21_200"


def main() -> None:
    bench = load_benchmark(BENCH)
    preds = load_predictions(PREDS)
    print(f"bench rows: {len(bench)}, pred rows: {len(preds)}, dataset dir exists: {DATASET_DIR.is_dir()}")

    _cache: dict = {}

    def resolver(source: str):
        if source in _cache:
            return _cache[source]
        # Mirror src/labeling/app.py::_resolve_source_csv: rglob so we
        # find files nested under e.g. ``curated_bing21_200/<hash>.xlsx/<src>``.
        frame = None
        try:
            for p in DATASET_DIR.rglob(source):
                if p.is_file() and p.suffix.lower() == ".csv":
                    try:
                        frame = load_csv(p)
                    except Exception as exc:  # noqa: BLE001
                        print(f"  load_csv({p}) failed: {exc}")
                        frame = None
                    break
        except OSError:
            frame = None
        _cache[source] = frame
        return frame

    view_no_csv = build_per_cell_view(bench, preds, include_trivial=False)
    print("\n== WITHOUT csv_resolver ==")
    for k, v in sorted(view_no_csv["states"].items(), key=lambda kv: -kv[1]):
        print(f"  {k:30s} {v}")

    view = build_per_cell_view(bench, preds, include_trivial=False, csv_resolver=resolver)
    print("\n== WITH csv_resolver (dataset=curated_bing21_200) ==")
    for k, v in sorted(view["states"].items(), key=lambda kv: -kv[1]):
        print(f"  {k:30s} {v}")
    print(f"  csv_resolver_used = {view['summary'].get('csv_resolver_used')}")
    print(f"  bench_gap_candidates = {view['summary'].get('bench_gap_candidates')}")

    # Show a handful of bench_gap cells so the operator can sanity-check
    # that the value really does occur in the CSV at the reported rows.
    gaps = [
        c for c in view["cells"]
        if c.get("state") == "unmatched_prediction" and c.get("csv_match_status") == "in_csv"
    ][:5]
    print("\nSample bench_gap candidates:")
    for c in gaps:
        rows = c.get("csv_match_rows", [])
        print(
            f"  src={c['source']} col={c['column']!r} val={c.get('original_value')!r}"
            f"  csv_rows={rows[:5]}{'…' if len(rows) > 5 else ''}"
            f"  method={c.get('csv_match_method')}"
        )

    halls = [
        c for c in view["cells"]
        if c.get("state") == "unmatched_prediction" and c.get("csv_match_status") == "not_in_csv"
    ][:5]
    print("\nSample hallucinations (value not in CSV):")
    for c in halls:
        print(f"  src={c['source']} col={c['column']!r} val={c.get('original_value')!r}")


if __name__ == "__main__":
    main()
