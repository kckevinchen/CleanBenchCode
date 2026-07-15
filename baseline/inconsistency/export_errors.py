"""Export per-cell TP / FP / FN (with context) for a prompt-opt run, as CSV.

For a chosen run + split (default: the held-out ``test`` split) and prompt
(``best`` optimized rulebook and/or ``seed``), this re-runs the detector — using
the run's ``detector_cache`` so it spends **no** API calls unless ``--allow-live``
— and writes one CSV row per ``(source, column, value)`` unit, classified:

  TP    predicted value that is a gold error
  FP    predicted value that is neither gold-error nor ambiguous (over-flag)
  FN    gold-error value the detector missed
  AMBIG predicted value on an ``ambiguous`` gold cell (FP-immune; context only)

Rich context per row: dataset, source, column, col_kind (labeled/clean), the
value and its row-count / share, the column's size + distinct count, the
detector's predicted kind/correction/rationale, the gold acceptable
correction(s), and a sample of the column's value distribution (so an
inconsistency call can be judged against its peers).

    python -m baseline.inconsistency.export_errors --run outputs/inconsistency_promptopt/hardened10v2
    python -m baseline.inconsistency.export_errors --run .../lean --split val --prompt both
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baseline.inconsistency import data  # noqa: E402
from baseline.inconsistency._task import norm, parse_findings  # noqa: E402
from baseline.inconsistency.evaluator import (  # noqa: E402
    Gpt55Detector,
    _read_csv,
    column_values,
)

API_VERSION = "2025-04-01-preview"
DEFAULT_ENDPOINT = "https://dmxaz-test-aoai-eu2.openai.azure.com/"

CSV_FIELDS = [
    "prompt_variant",
    "run",
    "split",
    "category",
    "dataset",
    "source",
    "column",
    "col_kind",
    "value",
    "value_count",
    "value_share",
    "column_n_rows",
    "column_n_distinct",
    "predicted_kind",
    "predicted_correction",
    "predicted_rationale",
    "gold_corrections",
    "column_top_values",
]


def _split_sources(split_meta: dict, split: str) -> set[str]:
    if split not in split_meta:
        raise SystemExit(f"split {split!r} not in split.json (have: dev/val/test)")
    return set(split_meta[split])


def _top_values(values_with_counts: list[tuple[str, int]], k: int = 12) -> str:
    return " | ".join(f"{v}({c})" for v, c in values_with_counts[:k])


def _base_row(ctx: dict, category: str, value: str, counts: dict, n_rows: int) -> dict:
    """One CSV row from per-column context ``ctx`` plus this value's stats."""
    return {
        **ctx,
        "category": category,
        "value": value,
        "value_count": counts.get(value, ""),
        "value_share": (round(counts[value] / n_rows, 4) if value in counts else ""),
    }


def _prompt_text(run_dir: Path, which: str) -> str:
    if which == "best":
        return (run_dir / "best_prompt.md").read_text(encoding="utf-8")
    if which == "seed":
        return (ROOT / "baseline" / "prompts" / "inconsistency" / "rulebook.md").read_text(
            encoding="utf-8"
        )
    raise SystemExit(f"unknown prompt {which!r}")


def _rows_for_prompt(
    *,
    which,
    prompt,
    run_dir,
    run_name,
    split,
    sources,
    cols,
    source_dataset,
    csv_map,
    detector,
    allow_live,
    exclusions=None,
):
    rows = []
    stats = {
        "TP": 0,
        "FP": 0,
        "FN": 0,
        "AMBIG": 0,
        "uncached_cols": 0,
        "cols": 0,
        "excluded_cols": 0,
    }
    labeled_keys = set(cols)
    EMPTY = {"error": {}, "amb": set()}

    # All (source, column) jobs for the split: labeled gold columns + clean columns.
    frames: dict[str, object] = {}

    def frame(src):
        if src not in frames:
            p = csv_map.get(src)
            frames[src] = _read_csv(p) if p else None
        return frames[src]

    jobs: list[tuple[str, str, dict, str]] = []
    for (src, col), gold in sorted(cols.items()):
        if src in sources:
            jobs.append((src, col, gold, "labeled"))
    for src in sorted(sources):
        df = frame(src)
        if df is None:
            continue
        for col in df.columns:
            if (src, str(col)) in labeled_keys:
                continue
            jobs.append((src, str(col), EMPTY, "clean"))

    for src, col, gold, col_kind in jobs:
        if exclusions and exclusions.drops_column(src, col):
            stats["excluded_cols"] += 1
            continue
        df = frame(src)
        if df is None or col not in df.columns:
            continue
        vals = column_values(df, col)
        if len(vals) < 2:
            continue
        stats["cols"] += 1
        n_rows = int(len(df))
        n_distinct = int(df[col].nunique(dropna=False))
        counts = {norm(v): c for v, c in vals}
        top = _top_values(vals)

        cached = detector.is_cached(prompt, src, col)
        if not cached and not allow_live:
            stats["uncached_cols"] += 1
            continue
        raw = detector(prompt, src, col, vals)
        pred = parse_findings(raw)
        gerr = gold["error"]
        # Skipped-cell values are FP-immune (folded into the ambiguous set).
        gamb = set(gold["amb"])
        if exclusions:
            gamb |= {v for v in exclusions.immune_values(src, col) if v not in gerr}

        ctx = {
            "prompt_variant": which,
            "run": run_name,
            "split": split,
            "dataset": source_dataset.get(src, "?"),
            "source": src,
            "column": col,
            "col_kind": col_kind,
            "column_n_rows": n_rows,
            "column_n_distinct": n_distinct,
            "column_top_values": top,
        }

        for nv, info in pred.items():
            if nv in gerr:
                cat = "TP"
            elif nv in gamb:
                cat = "AMBIG"
            else:
                cat = "FP"
            stats[cat] += 1
            row = _base_row(ctx, cat, nv, counts, n_rows)
            row["predicted_kind"] = info.get("kind")
            row["predicted_correction"] = info.get("corrected_value")
            row["predicted_rationale"] = info.get("rationale")
            row["gold_corrections"] = " | ".join(sorted(gerr[nv])) if nv in gerr else ""
            rows.append(row)

        for nv, acc in gerr.items():
            if nv not in pred:
                stats["FN"] += 1
                row = _base_row(ctx, "FN", nv, counts, n_rows)
                row["gold_corrections"] = " | ".join(sorted(acc))
                rows.append(row)

    return rows, stats


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--run", required=True, help="Run dir (has split.json, best_prompt.md, detector_cache/)."
    )
    ap.add_argument("--split", default="test", choices=["dev", "val", "test"])
    ap.add_argument("--prompt", default="best", choices=["best", "seed", "both"])
    ap.add_argument("--benchmark", default=str(ROOT / data.DEFAULT_BENCHMARK))
    ap.add_argument("--data-dirs", default=",".join(data.DEFAULT_DATA_DIRS))
    ap.add_argument("--model", default="gpt-5.5")
    ap.add_argument("--effort", default="none")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--endpoints", default="")
    ap.add_argument(
        "--allow-live",
        action="store_true",
        help="Permit live API calls on cache miss (default: skip uncached columns).",
    )
    ap.add_argument(
        "--out", default="", help="Output CSV path (default: <run>/error_cells_<split>.csv)."
    )
    ap.add_argument(
        "--exclusions",
        default=str(ROOT / data.DEFAULT_EXCLUSIONS),
        help="ignored.jsonl with curator-ignored CSVs/columns/rows/cells (FP-immune, dropped).",
    )
    ap.add_argument(
        "--no-exclusions",
        action="store_true",
        help="Audit everything, including curator-ignored columns/files (for comparison).",
    )
    args = ap.parse_args()

    run_dir = Path(args.run)
    run_name = run_dir.name
    split_meta = json.loads((run_dir / "split.json").read_text())
    sources = _split_sources(split_meta, args.split)

    cols, source_dataset = data.load_gold(args.benchmark)
    csv_map = data.csv_lookup([d.strip() for d in args.data_dirs.split(",") if d.strip()])
    exclusions = None
    if not args.no_exclusions:
        exclusions = data.load_exclusions(args.exclusions, csv_map=csv_map)
        print(f"exclusions: {exclusions.stats()}")

    clients = None
    if args.allow_live:
        from baseline._common import build_client

        eps = [e.strip() for e in (args.endpoints or DEFAULT_ENDPOINT).split(",") if e.strip()]
        clients = [build_client(ep, API_VERSION) for ep in eps]
    detector = Gpt55Detector(
        clients or [],
        model=args.model,
        effort=args.effort,
        temperature=args.temperature,
        cache_dir=run_dir / "detector_cache",
    )

    variants = ["best", "seed"] if args.prompt == "both" else [args.prompt]
    all_rows = []
    for which in variants:
        prompt = _prompt_text(run_dir, which)
        rows, stats = _rows_for_prompt(
            which=which,
            prompt=prompt,
            run_dir=run_dir,
            run_name=run_name,
            split=args.split,
            sources=sources,
            cols=cols,
            source_dataset=source_dataset,
            csv_map=csv_map,
            detector=detector,
            allow_live=args.allow_live,
            exclusions=exclusions,
        )
        all_rows.extend(rows)
        tp, fp, fn = stats["TP"], stats["FP"], stats["FN"]
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        print(
            f"[{which}] cols={stats['cols']} excluded_cols={stats['excluded_cols']} "
            f"TP={tp} FP={fp} FN={fn} AMBIG={stats['AMBIG']} "
            f"| P={prec:.3f} R={rec:.3f} F1={f1:.3f} uncached={stats['uncached_cols']}"
        )

    out = Path(args.out) if args.out else run_dir / f"error_cells_{args.split}.csv"
    # Sort for easy investigation: variant, category, dataset, source, column.
    cat_order = {"FP": 0, "FN": 1, "TP": 2, "AMBIG": 3}
    all_rows.sort(
        key=lambda r: (
            r["prompt_variant"],
            cat_order.get(r["category"], 9),
            r["dataset"],
            r["source"],
            r["column"],
            str(r["value"]),
        )
    )
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nwrote {len(all_rows)} rows -> {out}")


if __name__ == "__main__":
    main()
