# data-quality-llm

A modular research harness for running data-quality experiments against
Azure-hosted LLMs (currently the GPT-5 family).

The flagship experiment, `data_cleaning_error_detection`, asks the model
to scan a CSV **one narrow error type at a time** and merges the
findings via deterministic clustering plus an optional LLM-based
synthesis step. A 20-type taxonomy ships in
[`prompts/error_types.yaml`](prompts/error_types.yaml).

---

## Quickstart

```bash
# Setup
./create_venv.sh
source venv/bin/activate
pip install -r requirements.txt
az login                     # Azure OpenAI uses AzureCliCredential

# Smoke test (no LLM call - just renders the first prompt)
python -m experiments.run_cleaning_detection --dry-run

# Default focused run on the shipped 100-row sample
python -m experiments.run_cleaning_detection

# Open the interactive dashboard
xdg-open outputs/<timestamp>_<model>/report.html
```

For a web UI that launches runs, browses outputs, and labels clusters:

```bash
./scripts/run_server.sh start --port 8000
# -> http://127.0.0.1:8000/
```

---

## How it works

1. **Per-type detection.** For each CSV chunk we issue one LLM call per
   error type. Tight, well-defined prompts beat one monolithic prompt.
   Calls run in parallel across an Azure endpoint pool.
2. **Phase A — deterministic clustering.** Detections sharing a row and
   overlapping columns are merged into one cluster.
3. **Phase B — LLM synthesis** *(optional, on by default)*. For every
   multi-type cluster we fire one extra LLM call that picks a
   `primary_error_type`, classifies the rest as supporting, and emits a
   canonical explanation + fix + confidence + false-positive flag.

Total detection calls =
`(rows / chunk_size) × (1 if column_mode=all else len(columns)) × len(error_types)`.

---

## Error-type taxonomy

The 20 keys in [`prompts/error_types.yaml`](prompts/error_types.yaml)
mirror `error_type.csv` row-for-row, grouped by Scope:

| Scope | Types | column_mode |
|---|---|---|
| **Single-Column** *(default)* | typo, text_inconsistency, casing_inconsistency, pattern_inconsistency, punctuation_inconsistency, number_format_inconsistency, semantic_incompatible, numeric_outlier | `single` ok |
| **Multi-Column** *(intra-row)* | missing_value, embedded_value_error, cross_column_inconsistency, fd_violation, row_calculation_inconsistency, business_rule_violation | `all` required |
| **Multi-Row** *(intra-table)* | column_aggregation_inconsistency, row_duplication, cross_record_inconsistency | `all` required |
| **Multi-Table** *(rare in single-table chunks)* | cross_table_error, cross_source_naming_conflict, value_representation_conflict | `all` required |

Subset via `prompts.error_types` in `config/experiment.yaml` or
`--error-types` on the CLI. List the active set with
`--list-error-types`.

---

## Project layout

```
config/experiment.yaml              experiment settings
prompts/                            taxonomy + synth + variant prompts
experiments/                        CLI entry points (zero logic)
src/                                library code
  core/                             detection pipeline (runtime)
  adaptive/                         Good-Turing / multi-pass extension
  data_prep/                        offline curation library
  labeling/                         FastAPI workbench
frontend/                           HTML dashboard + labeling UI
scripts/run_server.sh               detached server start/stop wrapper
outputs/                            timestamped run dirs (gitignored)
```

Code (`src/`), prompts (`prompts/`), and configuration (`config/`) are
separated on purpose — iterate on the taxonomy without touching Python.

---

## Common runs

```bash
# Single column, single type — one LLM call
python -m experiments.run_cleaning_detection \
  --error-types typo --columns 'Implementing Agency Name' --max-chunks 1

# All Single-Column types in one all-columns sweep (fast overview)
python -m experiments.run_cleaning_detection --column-mode all

# Cross-column / cross-row types (require column_mode=all)
python -m experiments.run_cleaning_detection \
  --error-types typo,cross_column_inconsistency,fd_violation,row_duplication \
  --column-mode all --chunk-size 20

# Run on a folder of CSVs (multi-CSV mode; per-CSV tabs in the dashboard)
python -m experiments.run_cleaning_detection --csv-dir data/curated

# Skip Phase-B synthesis
python -m experiments.run_cleaning_detection --no-synthesis
```

Every flag is also a YAML key in `config/experiment.yaml`.

<details>
<summary><b>Multi-CSV mode details</b></summary>

`--csv-dir <DIR>` (or pointing `--csv` at a directory) loads every file
matching `--csv-glob` (default `*.csv`) and runs **one** detection pass
with shared concurrency.

- Phase-A clustering is source-aware: row 5 of `a.csv` and row 5 of
  `b.csv` are unrelated.
- Detections and clusters carry a source filename.
- The dashboard shows a tab strip (one tab per CSV plus "All sources").
  Each tab scopes every metric, chart, and table to that file.
- `summary.json` gains `clusters_by_source` and `issues_by_source`.

Set `data.csv_dir` / `data.csv_glob` in YAML to make `python -m
experiments.run_cleaning_detection` default to a folder.

</details>

---

## Outputs

Each run creates `outputs/<timestamp>_<model>/` containing:

| File | What |
|---|---|
| `report.html` | self-contained interactive dashboard (open directly) |
| `report.md` | human-readable summary |
| `summary.json` | aggregated metrics |
| `clusters.json(l)` | one cluster per line (Phase A + B output) |
| `issues.json(l)` | one raw detection per line |
| `chunk_reports.json(l)` | one (chunk × error_type) result per line |
| `raw_responses/` | raw LLM text + reasoning summary per call |
| `runs/`, `good_turing/` | adaptive-mode artifacts (when `--mode adaptive`) |

<details>
<summary><b>Dashboard tour</b></summary>

The HTML report is a single self-contained file with three views behind
one toggle:

- **Clusters** *(default)* — one row per underlying defect; metric
  cards, charts, filters, sortable table, drill-down with the Phase-B
  synthesis + full source-row values + every underlying detection.
- **Raw detections** — one row per per-type verdict, for auditing.
- **Source data** — the raw CSV with flagged cells highlighted; click
  any cell to record a label or note (designed for spotting
  false negatives).

The **My review** panel lets you label rows directly in the report:
`-1` = error, `0` = ambiguous, `+1` = no error (default for any
unlabeled row). For `-1` you can adopt the LLM's explanation verbatim
or write your own. Annotations persist to `localStorage` per `run_id`
and can be exported / imported as JSON.

The **Run parameters** panel shows the exact configuration that
produced the report (model, reasoning effort, chunk size, column mode,
error types, sources, …) so you have full provenance at a glance.

To iterate on the dashboard, edit
[`frontend/report_template.html`](frontend/report_template.html); the
`/*__REPORT_DATA_JSON__*/` placeholder is filled by
`src/core/report_builder.py`.

</details>

<details>
<summary><b>Viewing report.html on a remote VM</b></summary>

```bash
./serve_report.sh                   # serves the newest run on :8000
./serve_report.sh -p 8123           # custom port
./serve_report.sh outputs/<run>/    # specific run
```

VS Code / Cursor auto-forwards the port. `report.html` is fully
self-contained, so you can also right-click → Download and open it
locally.

</details>

---

## Workbench

`experiments.workbench` spins up a FastAPI web app that does everything
from one port: **launch experiments** via forms (cleaning detection,
schema/relation discovery, dataset curation, report builds), **browse
every run** under `outputs/`, **view embedded HTML reports**, and
**label clusters** to materialise a per-cell ground-truth benchmark
(`benchmark.jsonl`).

```bash
./scripts/run_server.sh start --port 8000     # detached; survives SSH drops
./scripts/run_server.sh status
./scripts/run_server.sh stop
tail -f logs/workbench.log
```

Job state is persisted to `logs/jobs/<job_id>.json`; running detector
children are spawned in their own session group, so a server restart
reattaches live jobs and finalises the rest from disk.

<details>
<summary><b>Labeling UI keys + export</b></summary>

| Key | Action |
|---|---|
| `j` / `k` | next / previous cluster |
| `1`-`5` | verdict = tp / fp / partial / unsure / skip |
| `c` / `n` | focus corrected-value / note fields |
| `Enter` | save + advance |

Every POST appends one line to `<run_dir>/labels.jsonl`. The
`Benchmark` link materialises `<run_dir>/benchmark.jsonl`, one row per
`(source, row, column)` cell:

```jsonl
{"source":"foo.csv","row":39,"column":"...","original_value":"3500",
 "label":"error","corrected_value":"0","error_types":[...], ...}
```

Mapping: `tp`/`partial` → `error`, `fp` → `clean`, `unsure` →
`ambiguous`, `skip` → omitted. Two clusters disagreeing on the same
cell collapse to `ambiguous`.

</details>

---

## Subsampling and curation

```bash
# Reproducible random subsample (sorted back into original row order)
python -m experiments.data_prep.subsample \
  --input data/<big>.csv --output data/sample_100.csv \
  --rows 100 --seed 42

# Curate a few clean tables from a large corpus (e.g. data/Excel50KTables)
python -m experiments.data_prep.curate_tables \
  --source-dir data/Excel50KTables --output-dir data/curated \
  --num-tables 5 --min-columns 3 --max-columns 10 \
  --min-rows 30 --max-rows 100 --max-empty-ratio 0.1 --seed 42
```

`curate_tables` filters by dimensions + empty-cell ratio, drops
all-empty columns, then drops rows with any remaining empty cell, and
finally subsamples rows. Output tables are guaranteed to contain **zero
empty cells**, so downstream detectors can focus on substantive errors.

---

## Adaptive Good-Turing mode (FN-rate estimation)

A single pass tells you nothing about defects you missed. Running `K`
perturbed passes (row / column shuffles, optional per-pass prompt
rewrites) lets us compute a Good-Turing missing-mass and a Chao1
estimate of total defects, giving a **point + 95% log-normal CI on the
false-negative rate**.

```bash
python -m experiments.run_cleaning_detection \
  --csv-dir data/curated --error-types pattern_inconsistency \
  --mode adaptive --gt-num-runs 5 --gt-base-seed 42
```

<details>
<summary><b>Adaptive-mode knobs</b></summary>

All also settable under `good_turing:` in `config/experiment.yaml`.

| Flag | Meaning |
|---|---|
| `--gt-mode {adaptive,fixed}` | adaptive stops early once missing-mass < target |
| `--gt-num-runs K` | max passes (more = tighter CI) |
| `--gt-min-runs` / `--gt-target-missing-mass` | adaptive stop condition |
| `--gt-base-seed` | reproducibility root |
| `--gt-prompt-ensemble` | per-pass LLM-rewritten prompt variants |
| `--gt-shuffle-rows` / `--gt-shuffle-columns` | frame-level perturbation |
| `--gt-skip-identity-run` | by default run 0 is unshuffled + seed prompts |

Artifacts (under the same `outputs/<...>/`):

- `runs/run_<i>/detections.jsonl`, `runs/run_<i>/run.json`
- `prompt_variants/run_<i>/<type_key>.json`
- `good_turing/species.jsonl` — canonical species `(source, column, row)`
- `good_turing/estimates.json` — overall + per-type + per-source
- `good_turing/progression.jsonl` — snapshot per pass
- `good_turing/report.md` — interpretation cheat-sheet

The HTML dashboard's **Good-Turing** tab surfaces all of the above.

**Caveats.** Both estimators assume species are drawn roughly
independently across runs. Row-shuffle perturbation is a reasonable but
not airtight surrogate. Chao1 is a *lower* bound — true FN rate may be
higher if the detector has systematic blind spots no reshuffling
exposes. `K = 3` is the practical minimum; `K = 5` is the sweet spot;
`K >= 8` tightens further at proportional LLM cost.

</details>

---

## Configuration

Endpoints (default pool ships in `src/core/llm_client.py`):

```bash
export AZURE_OPENAI_ENDPOINTS="https://foo.openai.azure.com/,https://bar.openai.azure.com/"
```

Concurrency defaults to `min(8, #endpoints)`. Detection and Phase-B
synthesis use independent knobs (`--concurrency` /
`--synthesis-concurrency`).

---

## Adding or editing an error type

1. Add (or edit) a key under `types:` in
   [`prompts/error_types.yaml`](prompts/error_types.yaml) with `name`,
   `category`, `description`, `example`, and `notes` (both **WHAT TO
   FLAG** and **WHAT NOT TO FLAG**, cross-referencing sibling type keys
   so detectors stay in their lane).
2. Rerun. The new type is picked up automatically and appears in the
   HTML filter + summary.

To iterate on the prompt template shared by every type, edit
`base.system` and `base.user_template` in the same file. Numeric
thresholds (peer minimum, dominant-baseline percentage, affected-row
cap, …) live under `defaults:` and are substituted into the templates
at load time — change them in one place.

---

## Further reading

- [`docs/experiment_summary.md`](docs/experiment_summary.md) — design
  notes and experiment write-ups.
- [`docs/pipeline.html`](docs/pipeline.html) — visual pipeline diagram.
