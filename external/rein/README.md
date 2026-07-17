# `external/rein/` — REIN error-detection baselines on `benchmark/merged_v2`

Runs detectors from the **REIN** data-cleaning benchmark
([mohamedyd/rein-benchmark](https://github.com/mohamedyd/rein-benchmark),
*REIN: A Comprehensive Benchmark Framework for Data Cleaning Methods*, PVLDB 2023)
as external baselines on our cell-level benchmark, scored with **cell-level micro
P/R/F1** — the same metric REIN uses internally and that `RFT.md` mandates.

This directory holds only our integration code (tracked). The REIN repo itself,
the materialized `datasets/`, the `venv/`, and `results/` are gitignored.

```
external/rein/
  common.py            shared paths + REIN-faithful CSV loader + value-equality helpers
  prepare_datasets.py  benchmark  ->  datasets/<ds>/{dirty,clean,actual_errors,...}
  run_rein.py          run detectors -> predictions JSONL  (no PostgreSQL)
  score.py             predictions  -> DETECTION micro P/R/F1 + per-error-type recall
  score_repair.py      valued preds / repaired.csv -> CORRECTION & END-TO-END micro P/R/F1
  setup.sh             clone REIN + make venv + install pinned deps
  requirements.txt     pinned detector deps (raha needs pandas 2.2.x)
  rein-benchmark/      third-party clone            (gitignored)
  datasets/ results/ venv/   generated artifacts    (gitignored)
```

## Why this design

REIN normally needs PostgreSQL + per-tool native installs (Picket, FAHES, KATARA,
HoloClean, ed2…) and drives an ML-pipeline impact study. We only need **detection
accuracy**, so the integration is deliberately lighter and postgres-free:

1. **Inject our labels as ground truth.** REIN's `benchmark.py` honors a
   precomputed `datasets/<ds>/actual_errors.csv` (`i,j,dummy`) + `error_rate.json`
   instead of diffing dirty vs clean. `prepare_datasets.py` writes these from the
   benchmark, so detectors are graded against *our* `error` cells.
2. **Call detectors directly.** REIN's `rein/detectors.py` eagerly imports the
   whole heavy stack (torch/tensorflow/holoclean/cleanlab-1.0/…), none of which
   install on Python 3.12. So `run_rein.py` **reimplements the lightweight
   detectors faithfully** (ported from `rein/detectors.py`) and calls the
   underlying tools directly.
3. **One REIN dataset per benchmark source CSV** (215 of 217 sources are present
   locally under `data/_merged_v2_all_sources/`).

## Setup

```bash
bash external/rein/setup.sh           # clone REIN, create venv, install deps
source external/rein/venv/bin/activate
```

The dedicated venv pins **pandas 2.2.3** (raha 1.x calls `DataFrame.applymap`,
removed in pandas 3.0) and is isolated from the project's main `venv`.

## Run

```bash
# 1) materialize the benchmark into REIN dataset dirs (215 labeled datasets)
python external/rein/prepare_datasets.py            # --limit N for a smoke test
#   --include-clean-universe   ALSO materialize every other CSV in --sources-dir
#                              as a CLEAN dataset (no labeled errors) -> 1002 total.
#                              REQUIRED for honest precision (flags on clean tables
#                              are false positives); labeled-only scoring inflates it.

# 1b) (optional) export the skipped/excluded NEUTRAL set so scoring can ignore it
python -m experiments.export_neutral_cells          # -> benchmark/merged_v2/neutral_cells.jsonl

# 2) run detector(s) -> predictions JSONL
python external/rein/run_rein.py --detector raha --tag raha_fewshot20   # FEW-SHOT: see note
python external/rein/run_rein.py --detector outlierdetector --outlier-method IF
python external/rein/run_rein.py --detector mvdetector --detector outlierdetector
#   raha is few-shot: it labels whole ROWS with benchmark ground truth and writes a
#   <predictions>.labeled_rows.json sidecar so the scorer can hold those rows out.
#     --raha-labeling-budget 20    fixed tuples/table (default; ~28% cell-leak here)
#     --raha-budget-frac 0.05      per-table = ceil(0.05*n_rows) -> ~5% cell-leak
#     --tag raha_fewshot5          write to its own predictions file
#   measure the leakage: python external/rein/measure_raha_leakage.py

# 3) score -> micro P/R/F1 + per-error-type recall
python external/rein/score.py --predictions external/rein/results/predictions_raha_fewshot20.jsonl
#   --ambiguous {exclude|fp|error}   (default exclude: ambiguous cells are neutral)
#   --neutral-file PATH / --no-neutral   (skipped+excluded cells; default on if the file exists)
#   held-out eval is ON by default when a .labeled_rows.json sidecar exists (raha);
#   add --no-holdout-labeled-rows to see the raw, leaky number instead.

# 3b) score VALUE-level repair (correction / end-to-end) -> micro P/R/F1
#   Needs proposed VALUES, not just locations: either a predictions JSONL carrying
#   a value field (corrected_value/value/new_value/repaired/correction), or a
#   directory of full repaired tables (<dataset>[suffix]/repaired.csv).
python external/rein/score_repair.py --task correction \
    --predictions external/rein/results/predictions_<method>.jsonl   # oracle-location (§5.5)
python external/rein/score_repair.py --task e2e \
    --predictions external/rein/results/predictions_<method>.jsonl   # detect+fix (§5.6)
python external/rein/score_repair.py --task e2e --detector-name holoclean \
    --repaired-dir external/holoclean/cache/v8 --repaired-suffix "__<commit>" --diff-casefold
#   --casefold / --no-numeric   value-match knobs (default: case-sensitive, numeric-aware)
#   --diff-casefold             with --repaired-dir: ignore pure re-casing/whitespace
#                               so pipeline preprocessing artifacts aren't counted as edits
```

### Value-level repair scoring (`score_repair.py`)

`score.py` grades **location** (was a cell flagged?). `score_repair.py` grades
the **proposed value** against the gold `clean.csv`, in two tasks:

- **`--task correction`** (oracle-location, §5.5): the method is handed the true
  error cells and only proposes replacements. Scored over the gold error set —
  `precision = correct repairs / attempts`, `recall = correct repairs / |errors|`.
- **`--task e2e`** (end-to-end, §5.6): the method both finds and fixes over the
  full corpus. A *change* is any predicted cell with a usable value (or, with
  `--repaired-dir`, any cell whose repaired value differs from dirty). A change is
  a TP only if it lands on a real error **and** the value matches gold.
  `precision = TP / changes`, `recall = TP / |errors|`; `valAcc` = value accuracy
  among correctly-located edits.

Value equality mirrors the main pipeline (`common.values_equal`): whitespace-
normalized string match plus numeric-surface-form equivalence (`1,000`==`1000`).
It reuses `score.py`'s neutral / held-out-labeled-row / synthetic-cell machinery,
so few-shot correctors (e.g. Baran) are held out identically to detection.
Grading math is covered by `_selftest_score_repair.py`.

## Evaluation protocol

The benchmark is **exhaustively labeled** (unlabeled cell == clean == true
negative), so scoring runs over the **entire CSV**:
- **Positives** = labeled `error` cells only (recall is measured against these).
- **Neutral** (never clean/FP) = `ambiguous` (from the benchmark) plus `skipped`
  and `excluded` cells from `benchmark/merged_v2/neutral_cells.jsonl`
  (`python -m experiments.export_neutral_cells`, which reuses the same
  `_materialize_run` that built the benchmark). A labeled **error wins** over any
  neutral rule. The scorer's `neut` column reports how many predictions were
  neutralized.
- Everything else = clean; a prediction there is a false positive. Precision is
  therefore exact.

## Detectors

| detector | scope | deps | status |
|---|---|---|---|
| `raha` | all cell error types (config-free ensemble) | `pip raha` (vendored venv) | ✅ working — **few-shot\*\*** |
| `baran` | correction-as-detection (Raha detect → Baran correct) | `pip raha` (vendored venv) | ✅ working — **few-shot\*\*\*** |
| `outlierdetector` | numeric outliers (SD / IQR / IsolationForest) | pure pandas+sklearn | ✅ working |
| `mvdetector` | explicit missing values (empty string) | pure pandas | ✅ working |
| `dboost` | histogram/gaussian outliers | vendored dBoost | ⚠️ experimental* |

\* The dBoost copy vendored in `mohamedyd/rein-benchmark` is **incomplete**
(its `dboost/utils/` package is missing), so `dboost` raises a clear error until
`utils/` is vendored from upstream `cpitclaudel/dBoost`. It is also an *oracle*
(REIN picks the best config by F1 against ground truth). Use `outlierdetector`
for a working, non-oracle outlier baseline.

\*\* **raha is few-shot, not zero-shot.** It samples a few rows per table and labels
every cell in them with the benchmark **ground truth** (via `clean.csv`), then
trains a per-table classifier. For its labeled rows it emits the ground-truth error
cells *verbatim* as detections (`detection.py` line 366) — guaranteed TPs it was
handed — so the runner records `<predictions>.labeled_rows.json` and the scorer
**holds those rows out** of evaluation by default (don't test a few-shot learner on
its own training examples). Held-out, the standard-budget F1 is 0.66 (raw-with-leak
0.78); shrinking the budget (`--raha-budget-frac`) drops it further (0.39 at ~5%
leak, 0.27 at ~2.4%). Keep raha in a **separate few-shot section** from the
zero-shot baselines.

\*\*\* **baran is few-shot, like raha** (it's raha's correction engine, the
"Baran" baseline in the TabClean paper). We run the realistic **Raha detect →
Baran correct** pipeline and count a cell as a detection iff Baran actually
*changes* its value (a no-op correction is not a detection) — so it is a
high-precision subset of raha's flags. It labels whole rows with ground truth
(detection budget ∪ correction budget), so the runner records those rows and the
scorer holds them out (same machinery as raha). Two upstream bugs are worked
around in `run_rein.py`: raha forks `multiprocessing.Pool`s after numpy/sklearn
spawn BLAS threads (deadlock — fixed by pinning thread counts to 1 before numpy
loads) and `Correction.run()` loops forever when a table has fewer rows than the
labeling budget (fixed by capping the budget at the row count). Held-out, Baran
scores P≈0.98 / R≈0.45 / F1≈0.62 — the most precise baseline here.

To run it:
```bash
python external/rein/run_rein.py --detector baran --tag baran   # ~80s over 1002 tables
python external/rein/score.py --predictions external/rein/results/predictions_baran.jsonl
```

## Results

Consolidated, up-to-date results for **all** baselines (full 1002-CSV universe,
zero-shot vs few-shot separated) live in **`external/RESULTS.md`** and
**`external/BASELINES.md`**. raha's supervision trend is in `RESULTS.md` §1b;
measure its leakage with `python external/rein/measure_raha_leakage.py`.

Notes:
- raha fails on some datasets with "probabilities contain NaN" (a known numerical
  edge case in raha's few-shot sampling); those are skipped and excluded
  from its scope. Coverage is reported per detector.
- `outlierdetector`/`mvdetector` target error types that are rare/absent here, so
  low scores are expected — they are faithful REIN detectors, not tuned to this
  benchmark.

## Known limitations / extensions

- **Out-of-bounds labels:** ~30 benchmark cells reference a row/column not found
  in the local source CSV (≈3%); they are skipped (see `prepare`'s summary).
- **Missing sources:** 2 of 217 sources are absent locally and skipped.
- **Row-level benchmark** (`benchmark_row.json`: fd_violation, row_duplication,
  row/string calculation) is **not** covered here — different granularity; a
  natural follow-up (REIN's `nadeef`/`duplicatesdetector` map to it).
- **Heavy REIN detectors** (HoloClean, ed2, Picket, KATARA, ZeroER, FAHES) need
  PostgreSQL + native builds and per-dataset rules; they are out of scope but can
  be added behind the same `prepare -> run -> score` flow.
