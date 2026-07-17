"""Measure the deterministic oracle against the inconsistency benchmark.

Answers the question that gates any post-training: how far does a non-LLM,
structural-only detector get on its own, and how big is the *semantic* residual
that genuinely needs a model?

For a split (default ``test``) it runs the oracle over every labeled + clean
column (honouring exclusions), scores per distinct ``(column, value)`` against
gold (ambiguous = FP-immune), and reports:

  * overall precision / recall / F1, vs the GPT-5.5 best+seed numbers;
  * a per-dataset breakdown;
  * the oracle's findings by sub-pattern kind;
  * **FN decomposition** — of the gold errors the oracle misses, how many have a
    structural twin in their column (catchable, a fixable miss) vs none
    (semantic: alias/abbreviation/qualifier — the irreducible LLM residual).

    python -m external.baseline.inconsistency.measure_oracle --split test
    python -m external.baseline.inconsistency.measure_oracle --split full --no-exclusions
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from external.baseline.inconsistency import data  # noqa: E402
from external.baseline.inconsistency._task import parse_findings  # noqa: E402
from external.baseline.inconsistency.evaluator import (  # noqa: E402
    _build_jobs,
    _prf,
    _read_csv,
    column_values,
    score_column,
)
from external.baseline.inconsistency.oracle import (  # noqa: E402
    is_structurally_justified,
    make_oracle_detector,
)


def _split_sources(split_meta, split):
    if split == "full":
        return None  # all sources
    if split not in split_meta:
        raise SystemExit(f"split {split!r} not in split.json")
    return set(split_meta[split])


def _run_verifier(args, run_dir, jobs, csv_map):
    """Gate GPT-5.5's flags through the oracle justification predicate.

    Loads ``error_cells_<split>.csv`` (best prompt), and for every model flag
    (TP or FP) asks whether the oracle would justify it structurally. Cross-tabs
    against the gold category and reports the resulting oracle-gated precision /
    recall — i.e. the effect of forcing the model to only emit structurally
    defensible flags.
    """
    import csv as _csv

    # Column value cache (value lists per (source, column)).
    val_cache: dict[tuple[str, str], list] = {}
    frames: dict[str, object] = {}

    def vals_for(source, column):
        key = (source, column)
        if key not in val_cache:
            if source not in frames:
                p = csv_map.get(source)
                frames[source] = _read_csv(p) if p else None
            df = frames[source]
            val_cache[key] = column_values(df, column) if (df is not None and column in df.columns) else []
        return val_cache[key]

    csv_path = run_dir / f"error_cells_{args.split}.csv"
    if not csv_path.exists():
        raise SystemExit(f"need {csv_path} (run export_errors first)")

    # Gold error total for recall denominator.
    gold_err = sum(1 for (_s, _c, g, _v, ck, _d) in jobs if ck == "labeled" for _ in g["error"])

    cells = collections.Counter()  # (category, justified) -> n
    killed_tp = []
    with open(csv_path) as _fh:
        rows = [r for r in _csv.DictReader(_fh)
                if r["prompt_variant"] == "best" and r["category"] in ("TP", "FP")]
    for r in rows:
        vals = vals_for(r["source"], r["column"])
        just = is_structurally_justified(r["value"], vals) if vals else False
        cells[(r["category"], just)] += 1
        if r["category"] == "TP" and not just:
            killed_tp.append((r["column"], r["value"]))

    tp_keep = cells[("TP", True)]
    tp_kill = cells[("TP", False)]
    fp_keep = cells[("FP", True)]
    fp_kill = cells[("FP", False)]
    tp0, fp0 = tp_keep + tp_kill, fp_keep + fp_kill

    print(f"=== Oracle as FP-VERIFIER on '{args.split}' (gating GPT-5.5 best flags) ===")
    print(f"  model flags: TP={tp0}  FP={fp0}")
    print("  oracle justification cross-tab:")
    print(f"    TP justified (kept)   = {tp_keep:3}   TP NOT justified (killed) = {tp_kill}")
    print(f"    FP justified (slips)  = {fp_keep:3}   FP NOT justified (VETOED) = {fp_kill}")
    veto_total = tp_kill + fp_kill
    if veto_total:
        print(f"  veto precision (vetoed that are truly FP) = {fp_kill}/{veto_total} = "
              f"{fp_kill/veto_total:.0%}")
    if fp0:
        print(f"  FP coverage (FPs the oracle vetoes)       = {fp_kill}/{fp0} = {fp_kill/fp0:.0%}")
    # Resulting metrics if the model only emitted justified flags.
    p0, r0, f0 = _prf(tp0, fp0, gold_err - tp0)
    p1, r1, f1 = _prf(tp_keep, fp_keep, gold_err - tp_keep)
    print(f"\n  GPT-5.5 best            : P={p0:.3f} R={r0:.3f} F1={f0:.3f}")
    print(f"  oracle-gated (keep just): P={p1:.3f} R={r1:.3f} F1={f1:.3f}  "
          f"(killed {tp_kill} real errors)")
    print("  sample TPs the gate would wrongly kill:", killed_tp[:8])


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--split", default="test", choices=["dev", "val", "test", "full"])
    ap.add_argument("--run", default="outputs/inconsistency_promptopt/hardened10v2",
                    help="Run dir for the split.json and the GPT-5.5 comparison numbers.")
    ap.add_argument("--benchmark", default=str(ROOT / data.DEFAULT_BENCHMARK))
    ap.add_argument("--data-dirs", default=",".join(data.DEFAULT_DATA_DIRS))
    ap.add_argument("--exclusions", default=str(ROOT / data.DEFAULT_EXCLUSIONS))
    ap.add_argument("--no-exclusions", action="store_true")
    ap.add_argument("--min-dominance", type=float, default=0.0,
                    help="Require the canonical form to hold >= this fraction of the group.")
    ap.add_argument("--dump-fp", default="", help="Write oracle FP rows to this CSV for inspection.")
    ap.add_argument("--verify", action="store_true",
                    help="VERIFIER mode: gate GPT-5.5's flags (from error_cells_<split>.csv) "
                         "through the oracle's structural-justification predicate and report "
                         "how many FPs it vetoes vs how many TPs it wrongly kills.")
    args = ap.parse_args()

    cols, source_dataset = data.load_gold(args.benchmark)
    csv_map = data.csv_lookup([d.strip() for d in args.data_dirs.split(",") if d.strip()])
    exclusions = None if args.no_exclusions else data.load_exclusions(args.exclusions, csv_map=csv_map)

    run_dir = Path(args.run)
    split_meta = json.loads((run_dir / "split.json").read_text())
    sel = _split_sources(split_meta, args.split)
    sources = set(sel) if sel is not None else {s for s, _ in cols} | {
        p.name for p in csv_map.values()
    }
    # For 'full' restrict to sources that actually have gold (keep it meaningful).
    if sel is None:
        sources = {s for s, _ in cols}

    detector = make_oracle_detector(min_dominance=args.min_dominance)
    jobs = _build_jobs(sources, cols, source_dataset, csv_map,
                       include_clean=True, exclusions=exclusions)

    if args.verify:
        _run_verifier(args, run_dir, jobs, csv_map)
        return

    results = []
    fn_structural = fn_semantic = 0
    fn_samples = {"structural": [], "semantic": []}
    fp_rows = []
    kind_tally = {"TP": {}, "FP": {}}
    for source, column, gold, vals, col_kind, ds in jobs:
        raw = detector(None, source, column, vals)
        pred = parse_findings(raw)
        res = score_column(pred, gold, source=source, column=column, dataset=ds, col_kind=col_kind)
        results.append(res)
        # by-kind tallies from predictions
        for nv, info in pred.items():
            if nv in gold["error"]:
                bucket = "TP"
            elif nv in gold["amb"]:
                bucket = None
            else:
                bucket = "FP"
            if bucket:
                k = info.get("kind", "?")
                kind_tally[bucket][k] = kind_tally[bucket].get(k, 0) + 1
        for x in res["fps"]:
            fp_rows.append(x)
        # FN decomposition: can the oracle's structural signal (twin OR
        # minority-format-signature) justify flagging the missed gold value?
        for nv in gold["error"]:
            if nv not in pred:
                if is_structurally_justified(nv, vals):
                    fn_structural += 1
                    if len(fn_samples["structural"]) < 12:
                        fn_samples["structural"].append((column, nv))
                else:
                    fn_semantic += 1
                    if len(fn_samples["semantic"]) < 12:
                        fn_samples["semantic"].append((column, nv))

    tp = sum(r["tp"] for r in results if r["col_kind"] == "labeled")
    fp = sum(r["fp"] for r in results)
    fn = sum(r["fn"] for r in results if r["col_kind"] == "labeled")
    p, r, f1 = _prf(tp, fp, fn)

    print(f"=== Deterministic oracle on '{args.split}' "
          f"(exclusions={'off' if args.no_exclusions else 'on'}, "
          f"min_dominance={args.min_dominance}) ===")
    print(f"  TP={tp}  FP={fp}  FN={fn}")
    print(f"  precision={p:.3f}  recall={r:.3f}  F1={f1:.3f}")

    # GPT-5.5 comparison
    rescored = run_dir / "rescored_metrics.json"
    if rescored.exists() and args.split in ("dev", "val", "test"):
        m = json.loads(rescored.read_text())["splits"][args.split]
        for who in ("best", "seed"):
            d = m[who]
            print(f"  [GPT-5.5 {who:4}] P={d['precision_incl_clean']:.3f} "
                  f"R={d['recall']:.3f} F1={d['f1_incl_clean']:.3f}")

    # by dataset
    print("  by dataset:")
    for dsname in sorted({r["dataset"] for r in results}):
        d_lab = [x for x in results if x["dataset"] == dsname and x["col_kind"] == "labeled"]
        d_all = [x for x in results if x["dataset"] == dsname]
        dtp = sum(x["tp"] for x in d_lab)
        dfp = sum(x["fp"] for x in d_all)
        dfn = sum(x["fn"] for x in d_lab)
        dp, dr, df = _prf(dtp, dfp, dfn)
        print(f"    {dsname:12} P={dp:.3f} R={dr:.3f} F1={df:.3f} (TP={dtp} FP={dfp} FN={dfn})")

    print(f"  oracle findings by kind:  TP={kind_tally['TP']}  FP={kind_tally['FP']}")

    total_fn = fn_structural + fn_semantic
    print("\n=== Recall-miss decomposition (where the structural ceiling is) ===")
    print(f"  FN total (labeled) = {total_fn}")
    print(f"   - structural (a normalised twin EXISTS -> fixable): {fn_structural}")
    print(f"   - semantic   (no twin: alias/abbrev/qualifier)    : {fn_semantic}")
    if total_fn:
        print(f"  => semantic share of misses = {fn_semantic/total_fn:.0%} "
              f"(this is the irreducible LLM residual)")
    print("  sample structural misses:", fn_samples["structural"][:6])
    print("  sample semantic   misses:", fn_samples["semantic"][:6])

    if args.dump_fp and fp_rows:
        import csv as _csv
        with open(args.dump_fp, "w", newline="", encoding="utf-8") as fh:
            w = _csv.DictWriter(fh, fieldnames=["source", "column", "dataset", "value",
                                                "kind", "pred_correction", "rationale"])
            w.writeheader()
            for x in fp_rows:
                w.writerow({k: x.get(k, "") for k in w.fieldnames})
        print(f"\nwrote {len(fp_rows)} oracle FP rows -> {args.dump_fp}")


if __name__ == "__main__":
    main()
