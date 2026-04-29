# data-quality-llm

Tiny, modular research harness for running data-quality experiments against
Azure-hosted LLMs (currently the GPT-5 family). The first experiment,
`data_cleaning_error_detection`, is **per error type**: it asks the model
to scan a CSV one narrow error class at a time and then merges the
findings.

A single canonical taxonomy ships in the repo: `prompts/error_types.yaml`,
the **20 fine-grained error types** from `error_type.csv`, grouped by
Scope:

* **Single-Column** (8 types — the default subset; work in
  `column_mode: single`): `typo`, `text_inconsistency`,
  `casing_inconsistency`, `pattern_inconsistency`,
  `punctuation_inconsistency`, `number_format_inconsistency`,
  `semantic_incompatible`, `numeric_outlier`.
* **Multi-Column / intra-row** (6 types; require `column_mode: all`):
  `missing_value`, `embedded_value_error`,
  `cross_column_inconsistency`, `fd_violation`,
  `row_calculation_inconsistency`, `business_rule_violation`.
* **Multi-Row / intra-table** (3 types; cross-row): `column_aggregation_inconsistency`,
  `row_duplication`, `cross_record_inconsistency`.
* **Multi-Table / cross-source** (3 types; rare findings inside a
  single-table chunk): `cross_table_error`,
  `cross_source_naming_conflict`, `value_representation_conflict`.

Pick any subset via `prompts.error_types` in `config/experiment.yaml`
(or `--error-types` on the CLI). The default subset is the 8
Single-Column types — they are cheap, work in `column_mode: single`,
and cover the bulk of cell-local data-quality defects.

For each CSV chunk we issue **one LLM call per error type**. The per-type
prompts give the model a tight, well-defined target and measurably reduce
the "miscellaneous grab-bag" problem you get from a single monolithic
prompt. Calls run in parallel across the endpoint pool.

A *chunk* has two independent dimensions, both configurable:

* **`chunk_size`** - number of CSV rows per call (the row slice).
* **`column_mode`** - whether one call sees every column for that row
  slice (`all`) or just one column at a time (`single`).

Total detection calls = `(rows / chunk_size) * (1 if column_mode=all
else len(columns)) * len(error_types)`.

The 8 default Single-Column types only need one column at a time, so
the default config uses `column_mode: single` with a `chunk_size` of
100. Switch on Multi-Column / Multi-Row / Multi-Table types and the
config must move to `column_mode: all`; see the per-key column-mode
notes in `prompts/error_types.yaml`.

Because the error types are **not** fully independent (one underlying
defect can trip several of them on the same row), the pipeline then runs
a two-phase aggregation step:

* **Phase A - deterministic clustering.** Detections sharing the same row
  and overlapping columns are merged into one "cluster" record
  (`cluster_id`, `evidence_count`, union of columns, set
  of distinct error types).
* **Phase B - LLM synthesis** *(optional, on by default)*. For every
  multi-type cluster we fire one additional LLM call that picks a single
  `primary_error_type`, classifies the rest as `supporting_error_types`,
  and emits a canonical explanation + fix + confidence score + an
  `is_likely_false_positive` flag. Disable with `--no-synthesis`.

The interactive dashboard defaults to the **Clusters** view (one row per
underlying defect) with a single-click toggle to **Raw detections** (one
row per per-type verdict) for auditing.

## Project layout

```
data-quality-llm/
  config/
    experiment.yaml                          # experiment settings (model, chunking, paths, concurrency)
  data/
    0a0a2ebffde940e2a232f26ba3ceb327.csv     # full input CSV (1195 rows)
    sample_100.csv                           # 100-row reproducible subsample (seed=42)
  experiments/                               # user-facing entry points (CLI only; zero logic)
    run_cleaning_detection.py                # `python -m experiments.run_cleaning_detection`
    build_report.py                          # `python -m experiments.build_report`
    label_server.py                          # `python -m experiments.label_server` (web workbench)
    data_prep/                               # thin wrappers around src/data_prep/
      subsample.py                           # `python -m experiments.data_prep.subsample`
      curate_tables.py                       # `python -m experiments.data_prep.curate_tables`
      curate_pbi.py                          # `python -m experiments.data_prep.curate_pbi`
      bing21_reconstruct.py                  # `python -m experiments.data_prep.bing21_reconstruct`
  scripts/
    run_server.sh                            # detached start/stop/status wrapper for label_server
  frontend/
    report_template.html                     # self-contained HTML dashboard template
    labeling/                                # static assets for experiments.label_server
      index.html                             # Runs dashboard + launcher form + jobs panel
      runs.js / runs.css                     # Runs/launcher page logic + styles
      label.html                             # Per-run labelling UI (?run=<id>)
      label.js                               # Labelling logic
      styles.css                             # Shared palette
  prompts/
    error_types.yaml                         # canonical 20-type taxonomy (mirrors error_type.csv)
    synthesise_cluster.yaml                  # Phase-B synthesis prompt (shared system + user template)
    generate_prompt_variant.yaml             # rewriter prompt for the prompt-ensemble perturbation
  src/                                       # library code - NOT invoked directly by users
    core/                                    # detection pipeline (runtime)
      llm_client.py                          # Azure OpenAI pool + retries + reasoning summaries (thread-safe)
      data_loader.py                         # CSV loader + chunking
      error_taxonomy.py                      # loads error_types.yaml + renders per-type prompts
      detector.py                            # PerTypeCleaningDetector: (chunk x type) -> issues, in parallel
      aggregator.py                          # Phase A clustering + Phase B LLM synthesis of multi-type clusters
      report_builder.py                      # multi-format artifact writer (JSON/MD/HTML)
    adaptive/                                # Good-Turing / multi-pass extension (mode=adaptive only)
      good_turing.py                         # Good-Turing + Chao1 FN-rate estimator
      perturbations.py                       # row / column shuffling per pass
      prompt_variants.py                     # LLM-based prompt rewriter for the prompt ensemble
    data_prep/                               # offline curation library (called via experiments/data_prep/)
      subsample.py                           # reproducible CSV subsampler
      curate_tables.py                       # pick a few small tables from a corpus
      curate_pbi.py                          # pick multi-table spreadsheets from the PBI corpus
      bing21_reconstruct.py                  # rebuild per-table CSVs from the Bing21 column-level TSV
    labeling/                                # FastAPI workbench (called via experiments/label_server.py)
      schema.py                              # Label pydantic models + LabelStatus enum
      store.py                               # append-only labels.jsonl
      export.py                              # labels.jsonl -> per-cell benchmark.jsonl
      runs.py                                # Run discovery + subprocess job manager
      app.py                                 # FastAPI factory (workspace-scoped)
  outputs/                                   # timestamped run dirs (gitignored)
  requirements.txt
  create_venv.sh
  serve_report.sh                            # optional: serve report.html via http.server
  error_type.xlsx                            # human-readable taxonomy reference
```

The separation of **code** (`src/`), **prompts** (`prompts/`), and
**configuration** (`config/`) is intentional: iterate on the taxonomy or
the base prompt template without touching Python, and share `src/` across
future experiments.

## Error-type taxonomy

`prompts/error_types.yaml` is the single canonical taxonomy file,
loaded by `src/core/error_taxonomy.py`:

* `base.system` - the system prompt shared by every per-type call
  (conservative-by-default, "whole column wrong = false positive",
  majority-rule guard).
* `base.user_template` - the per-type user prompt template
  (placeholders: `type_name`, `type_category`, `type_description`,
  `type_example`, `type_notes`, `headers`, `rows`, `start_row`, `end_row`).
* `types.<key>` - one entry per error type with `name`, `category`,
  `description`, `example`, and `notes` (WHAT TO FLAG +
  WHAT NOT TO FLAG, with explicit cross-references to sibling types
  so detectors stay in their lane).

The 20 keys mirror `error_type.csv` row-for-row. See the YAML's
header comment for the full grouped list and per-key column-mode
requirements.

### Listing + subsetting

List the keys of the currently-configured taxonomy:

```bash
python -m experiments.run_cleaning_detection --list-error-types
```

Subset to just a few types:

```bash
python -m experiments.run_cleaning_detection \
  --error-types typo,pattern_inconsistency
```

## Setup

```bash
cd /datadrive/kaiwen/data-quality-llm
./create_venv.sh
source venv/bin/activate
pip install -r requirements.txt
az login   # required: Azure OpenAI calls use AzureCliCredential
```

## Subsampling for manual review

Running on the full 1195-row CSV produces far too many issues to hand-verify.
`src/data_prep/subsample.py` generates a reproducible random subsample with a fixed
seed (sorted back into original row order for easier top-to-bottom reading):

```bash
python -m experiments.data_prep.subsample \
  --input data/0a0a2ebffde940e2a232f26ba3ceb327.csv \
  --output data/sample_100.csv \
  --rows 100 \
  --seed 42
```

The shipped `config/experiment.yaml` already points at `data/sample_100.csv`
(100 rows, seed 42) so that `python -m experiments.run_cleaning_detection`
gives a manageable set of issues by default. Point at the full CSV with
`--csv data/0a0a2ebffde940e2a232f26ba3ceb327.csv` for the final run.

### Curating tables from a large corpus

When the corpus has thousands of tables (e.g. `data/Excel50KTables`,
50 000 CSVs), `src/data_prep/curate_tables.py` picks a few that are small enough
to hand-verify, **cleans them so they contain zero empty cells**, and
subsamples their rows if needed. Selection is reproducible under
`--seed`, and a `manifest.json` records the full provenance (original
dims, empty-cell ratio, how many columns / rows were dropped during
cleaning, whether the output was subsampled).

Selection + cleaning pipeline:

1. Column count must be in `[--min-columns, --max-columns]`.
2. Row count must be >= `--min-rows`.
3. Fraction of empty cells in the raw CSV must be <=
   `--max-empty-ratio` (default **0.1**) - mostly-blank tables are
   filtered out up front so we don't waste a slot on unhelpful data.
4. All-empty columns are dropped.
5. Any row that still has at least one empty cell is dropped.
6. If > `--max-rows` rows remain, subsample deterministically from
   the cleaned rows using `--seed`.
7. Re-check dimensions against `[--min-columns, --max-columns]` and
   `--min-rows`; skip if the clean table no longer satisfies them.
8. Assert zero empty cells, then write to `--output-dir`.

```bash
python -m experiments.data_prep.curate_tables \
  --source-dir data/Excel50KTables \
  --output-dir data/curated \
  --num-tables 5 \
  --min-columns 3 --max-columns 10 \
  --min-rows 30  --max-rows 100 \
  --max-empty-ratio 0.1 \
  --seed 42
```

Each curated table is written as `<output-dir>/<original-stem>.csv`
and is guaranteed to contain **no empty cells at all** - the
downstream experiment can therefore focus on substantive errors
(typos, inconsistencies, …) without having to route around blanks.
Point the experiment at any of them:

```bash
python -m experiments.run_cleaning_detection \
  --csv data/curated/7b27acc938824579b9d99b68fb9e27bc.csv
```

Or process every curated table in one go (see the multi-CSV section
below):

```bash
python -m experiments.run_cleaning_detection --csv-dir data/curated
```

## Run the experiment

Dry run (render the first `(chunk, error_type)` prompt, no LLM call):

```bash
python -m experiments.run_cleaning_detection --dry-run
```

Default focused run (1 row chunk x 33 columns x 4 error types = 132
detection calls, plus a handful of Phase-B synthesis calls for any
multi-type clusters):

```bash
python -m experiments.run_cleaning_detection
```

Smoke-test on a single column with one type (one LLM call):

```bash
python -m experiments.run_cleaning_detection \
  --error-types typo \
  --columns 'Implementing Agency Name' \
  --max-chunks 1
```

Run the default Single-Column subset as a single all-columns sweep
(8 calls per chunk instead of 8 × #columns - useful for a fast first
pass):

```bash
python -m experiments.run_cleaning_detection --column-mode all
```

Enable cross-column / cross-row types (require `column_mode=all`):

```bash
python -m experiments.run_cleaning_detection \
  --error-types typo,text_inconsistency,cross_column_inconsistency,fd_violation,row_duplication \
  --column-mode all \
  --chunk-size 20
```

Common overrides:

```bash
python -m experiments.run_cleaning_detection \
  --model gpt-5.2 \
  --chunk-size 50 \
  --column-mode single \
  --columns 'Status (from list),Place name (Community or Evacuation Centre or School),Implementing Agency Name' \
  --max-chunks 6 \
  --concurrency 8 \
  --reasoning-effort medium
```

Skip Phase-B synthesis (deterministic Phase-A clustering only):

```bash
python -m experiments.run_cleaning_detection --no-synthesis
```

### Run on a folder of CSVs (multi-CSV mode)

Three equivalent ways to feed multiple CSVs into a single run:

```bash
python -m experiments.run_cleaning_detection --csv-dir data/curated
python -m experiments.run_cleaning_detection --csv data/curated
python -m experiments.run_cleaning_detection \
  --csv-dir data/curated --csv-glob 'sample_*.csv'
```

What it does:

- Loads every CSV matching `--csv-glob` (default `*.csv`) in the given
  directory.
- Generates `(row, column)` chunks per file using the same
  `--chunk-size` / `--column-mode` / `--columns` knobs (column names
  missing from a particular CSV are silently skipped for that file).
- Runs **one** detection pass with shared concurrency, so the LLM
  endpoints are saturated across files instead of per file.
- Tags every detection and cluster with its **source filename**.
  Phase-A clustering is source-aware: row 5 of `a.csv` and row 5 of
  `b.csv` are treated as unrelated even when their column overlap matches.
- Writes a single set of artifacts (one `report.html`, one
  `summary.json`, one `clusters.jsonl`, ...) into the run's output
  directory. The dashboard shows:
  - A **tab strip at the top with one tab per CSV** (plus an "All
    sources" tab). Switching tabs scopes every metric card, every
    chart, and both the Clusters and Raw detections tables to that
    single file — so each CSV effectively has its own mini-dashboard
    inside the single `report.html`.
  - A "Source" column in the Clusters and Raw tables (drops out of
    sight for single-CSV runs).
  - A "by source" chart that only appears on the "All sources" tab.

The summary file gains two extra fields for multi-CSV runs:

```json
"clusters_by_source": { "a.csv": 12, "b.csv": 7 },
"issues_by_source":   { "a.csv": 31, "b.csv": 19 }
```

You can also set the directory in YAML so plain
`python -m experiments.run_cleaning_detection` runs over the whole folder:

```yaml
data:
  csv_dir: data/curated
  csv_glob: "*.csv"
```

### Cost / latency estimate

Total LLM calls (Phase A) =
`(rows / chunk_size) * (1 if column_mode=all else len(columns)) * len(error_types)`,
plus one additional call for every **multi-type cluster** (Phase B
synthesis). Single-type clusters skip Phase B entirely.

* **Focused taxonomy (default).** `chunk_size=100`,
  `column_mode=single` on `sample_100.csv` (100 rows x 33 columns)
  gives `1 * 33 * 4 = 132` detection calls. Each call sees the full
  100-row column distribution but only a few hundred bytes of data,
  so prompts are tiny and concurrency hides most latency. End-to-end
  is typically a couple of minutes with `concurrency=8`.
  Switch to `column_mode=all` (`1 * 1 * 4 = 4` calls) for a near-
  instant overview that trades depth for speed.
* **Exhaustive taxonomy.** Use `column_mode=all` (the cross-column
  types need to see neighbouring columns). With `chunk_size=20`, that
  is `5 * 1 * 16 = 80` detection calls; we typically see ~60-80 raw
  detections collapsing into 30-50 clusters, of which 15-30 are
  multi-type, so Phase B adds on the order of 15-30 extra calls.
  Total wall time ~2-5 minutes at `concurrency=8`.

Tip: with `column_mode=single`, set `chunk_size` to the **total number
of rows you want analysed** so the model sees the entire distribution
of each column in one shot. Lower it only if a single column's text
becomes too large to fit in the context window.

## Labelling + experiment launcher (web app)

`experiments.label_server` spins up a FastAPI workbench that does three
things from one port:

1. **Launch new detection runs** via a form (no CLI typing).
2. **Browse every run** under `outputs/` and build the HTML report.
3. **Label clusters** to materialise a per-cell ground-truth benchmark
   (`benchmark.jsonl`).

```bash
source venv/bin/activate
python -m experiments.label_server
# -> UI  http://127.0.0.1:8000/
# -> API http://127.0.0.1:8000/api/docs
```

Defaults to binding on `127.0.0.1`. When running on a remote VM via VS
Code Remote-SSH the editor auto-forwards the port - just open
`http://localhost:8000/` in your local browser. For a plain SSH session:
`ssh -L 8000:127.0.0.1:8000 user@vm`.

### Surviving SSH drops

Foreground `python -m experiments.label_server` dies when its terminal
(tmux-less SSH session, VS Code integrated terminal) goes away.
`scripts/run_server.sh` wraps it with `nohup setsid` so the server is
fully detached from the login session:

```bash
./scripts/run_server.sh start --port 8000     # detached; pid -> logs/label_server.pid
./scripts/run_server.sh status                # show pid + last 10 log lines
./scripts/run_server.sh stop                  # SIGTERM the server
tail -f logs/label_server.log                 # live server output
```

**Job state is persisted to disk.** Every launched detection/report job
writes metadata to `logs/jobs/<job_id>.json` on every state change.
The detector child is spawned in its own session (`setsid`) so it keeps
running even if the server dies. On startup, the server:

1. Loads every `logs/jobs/*.json`.
2. For jobs last seen as `running`, probes `kill(pid, 0)`:
   - **pid alive** -> reattach, poll liveness every 5s, finalise when it exits.
   - **pid dead** -> mark `orphaned`, resolve the resulting `run_id` from `outputs/` mtimes.
3. Rehydrates completed / cancelled / failed jobs as-is.

Running jobs carry a `recovered: true` flag in the UI so you can tell
which ones survived a server restart. **Cancel still works** on
recovered jobs - the server sends SIGTERM to the saved process group.

### Launcher

The **Launch** tab turns every `run_cleaning_detection` flag into a
form control. The launcher:

* discovers candidate datasets under `data/` (directories with CSVs +
  top-level CSVs) and populates the dataset selector;
* reads `config/experiment.yaml` for placeholder defaults;
* emits scope + error-type chips from `prompts/error_types.yaml`;
* switches adaptive-mode controls (GT mode, num_runs, prompt ensemble,
  shuffles, ...) on when `mode=adaptive` is picked.

Submitting POSTs `/api/launcher/jobs` which spawns
`python -m experiments.run_cleaning_detection` as a child process. Logs
stream to `<project_root>/logs/jobs/<job_id>.log` and tail live in the
Jobs tab. Kill a running job via the Cancel button (sends SIGTERM to
the process group).

Once a job finishes, the new run dir is auto-detected and surfaced in
the Runs tab with direct **Label** / **Report** / **Benchmark** links.

### Labelling

Selecting a run from the Runs tab (or visiting
`/ui/label.html?run=<id>`) opens the annotation UI - cluster queue +
CSV viewer centred on the target row + verdict panel. Keyboard-driven:

| Key | Action |
|---|---|
| `j` / `k` | next / previous cluster |
| `1` / `2` / `3` / `4` / `5` | verdict = tp / fp / partial / unsure / skip |
| `c` / `n` | focus corrected-value / note fields |
| `Enter` | save current verdict + advance |

**Persistence.** Every POST appends one line to
`<run_dir>/labels.jsonl` (append-only, crash-safe, human-diffable).
Refreshing the browser or restarting the server rehydrates fully from
that file.

**Export.** `GET /api/runs/<run_id>/export/benchmark` (or the
`Benchmark` link on the run row) materialises
`<run_dir>/benchmark.jsonl`, one row per `(source, row, column)` cell:

```jsonl
{"source":"foo.csv","row":39,"column":"GASTO ...","original_value":"3500","label":"error","corrected_value":"0","error_types":[...],"detection_ids":[...],"cluster_ids":[...],"annotators":["kaiwen"],"note":""}
```

Mapping: `tp`/`partial` → `error`, `fp` → `clean`, `unsure` → `ambiguous`,
`skip` → omitted. Two clusters disagreeing on the same cell collapse
to `ambiguous`.

## Outputs

Each run creates a timestamped directory under `outputs/` with a full set
of human- and machine-readable artifacts. Each `ChunkReport` represents
one `(chunk, error_type)` pair; every raw detection is tagged with its
originating `error_type`; and every cluster aggregates the detections
that pertain to a single underlying defect.

```
outputs/20260421T120000Z_gpt-5.2/
  chunk_reports.jsonl     # one (chunk, error_type) result per line (slim)
  chunk_reports.json      # same data, pretty-printed JSON array
  issues.jsonl            # one raw detection per line, with error_type,
                          #   chunk metadata, and the source CSV values
                          #   for each flagged column
  issues.json             # same, pretty-printed
  clusters.jsonl          # one cluster per line; each cluster carries its
                          #   Phase-A metadata (row, columns,
                          #   error_types, evidence_count, row_values,
                          #   detections) and - for multi-type clusters -
                          #   a Phase-B `synthesis` object with
                          #   primary_error_type, supporting_error_types,
                          #   root_cause, canonical_explanation,
                          #   canonical_fix, confidence, and
                          #   is_likely_false_positive.
  clusters.json           # same, pretty-printed
  summary.json            # raw-detection aggregates (error_type
                          #   / taxonomy category / top columns / top rows)
                          #   PLUS cluster-level metrics (total clusters,
                          #   multi-type count, synthesised count, likely
                          #   false-positive count, clusters_by_primary_error_type /
                          #   clusters_by_primary_error_type /
                          #   clusters_by_confidence, top_clusters)
  report.md               # human-readable Markdown summary (clusters +
                          #   raw detections)
  report.html             # self-contained interactive dashboard with
                          #   Clusters (default) / Raw detections /
                          #   Source data views (latter for false-negative
                          #   review)
  raw_responses/
    0000__outliers.txt             # one file per (chunk, error_type) with
    0001__rule_violations.txt      #   the raw LLM text + reasoning summary
    ...
```

### Interactive dashboard (`report.html`)

The HTML report is a single self-contained file - open it directly in any
browser, no web server required.

Above the views you'll find two always-visible panels:

* **Run parameters** (collapsible, below the metric strip) - shows the
  exact experiment configuration that produced the report (model,
  reasoning effort, chunk size, column mode, column filter, detection
  call count, error-type keys, sources, Phase-B synthesis config).
  Expanding it gives you full provenance at a glance.
* **My review** - your personal annotation workspace.
  - Labels follow a **three-valued schema**: `-1` = *error in cell*,
    `0` = *ambiguous / interesting*, `+1` = *no error*. Any flagged
    item you don't touch is **treated as +1 by default**, so the
    typical workflow is "label real errors with -1, mark borderline
    cases with 0, and leave the rest alone".
  - For `-1`, you can choose whether to **adopt the LLM's explanation
    verbatim** or attach your own note. Clicking the `-1` button
    auto-selects the LLM comment (when one exists) as the annotation;
    clicking the pencil (`✎`) opens a prompt where an empty response
    means "keep the LLM's comment" and a non-empty one overrides it.
    Reports persist this choice as `use_llm_comment` in the stored
    record so downstream tooling can tell which reviews were a
    verbatim LLM adoption and which were reviewer-authored.
  - Live counts show `-1` / `0` / `+1` / user-created FNs.
  - **Filter dropdown** scopes the Clusters + Raw tables to one label
    (or to "unlabeled", i.e. implicitly +1).
  - **Export JSON** downloads every annotation for this run as a file
    you can commit to git or share with teammates.
  - **Import JSON** reloads annotations (merge or replace). Legacy
    exports that still use the older `tp`/`fp`/`highlight` enum are
    auto-migrated to `-1`/`+1`/`0` on import.
  - **Clear all** wipes annotations for *this* run (keyed by `run_id`).
  Annotations persist to `localStorage` automatically; different runs
  have independent state so reloading the same HTML always restores
  your review.

Then there are three views behind one toggle:

* **Clusters** (default) - one row per underlying defect.
  - Metric cards: total / multi-type / single-type / likely-FP clusters
    and likely false-positive count.
  - Charts: clusters by primary error type, by synthesis
    confidence, top columns.
  - Filters: search, primary error type, confidence, hide /
    only show false positives, multi- vs single-type.
  - Sortable table. Click any row to drill in: the Phase-B synthesis
    (primary / supporting types, root cause, canonical explanation, fix,
    confidence, FP flag, reviewer note), the **full** source-row values,
    and every underlying per-type detection with its columns,
    observed text, explanation, and suggested fix.
* **Raw detections** - one row per per-type verdict, preserved for
  auditing. Same filter / search / sort / drill-down ergonomics as
  before.

Both the Clusters and Raw tables now have a **Review** column with
  compact `−1 / 0 / +1 / ✎` buttons. Clicking any value toggles that
  label on the row (clicking it again clears back to the default +1);
  `✎` opens a note prompt whose placeholder depends on the current
  label. The row is tinted red (−1, error) / yellow (0, ambiguous) /
  green (+1, no error) for an at-a-glance audit trail.
* **Source data** - the raw CSV as a table, with every flagged cell
  highlighted and annotated on hover with the detection(s).
  Designed for manually spotting **false negatives**: unflagged cells
  render plain, so errors the model missed stand out as plain text next
  to tinted ones. **Click any cell** to open the review popover:
  - On a flagged cell it lists all linked detections with the same
    `−1 / 0 / +1 / ✎` controls as the tables, and shows the LLM's
    explanation inline so you can decide whether to adopt or override
    it with `use_llm_comment`.
  - On an unflagged cell (or a flagged one you also want to annotate
    as a miss), it lets you record a user **false negative** with a
    note. Cells with a saved FN get a blue tint so you
    can find them on re-review.
  - Metric cards: rows, flagged rows / cells, flagged-cell ratio.
  - Row filter: `All / Only flagged / Only unflagged`.
  - Column filter: focus the table on a single column.
  - **Column values** side panel: for a chosen column, lists every
    distinct value with its frequency, highlighting those the model
    flagged. Switch between "most frequent", "rarest first", and "A-Z"
    to find variants, typos, or pattern outliers - rare unflagged
    values next to a flagged look-alike are prime FN candidates.
  - Multi-CSV runs: the view follows the per-CSV tab strip at the top
    of the page. On the **All sources** tab you get a quick-jump index
    of files with flagged / unflagged row counts instead of a combined
    table.

To iterate on the dashboard look and feel, edit
`frontend/report_template.html`. The template contains a single
`/*__REPORT_DATA_JSON__*/` placeholder that `src/core/report_builder.py`
substitutes with the run payload
(`{run_metadata, summary, issues, clusters}`).

### Viewing `report.html` from a remote workspace

If your workspace is on a remote machine (e.g. an Azure VM accessed via
Cursor/VS Code Remote), clicking `report.html` inside the editor tries
to open a `vscode-remote://` URL that your local OS cannot handle ("get
an app to open this link"). Two ways around it:

1. **Serve over HTTP** (recommended). Use the helper script:

   ```bash
   cd /datadrive/kaiwen/data-quality-llm
   ./serve_report.sh                   # serves the newest run on :8000
   ./serve_report.sh -p 8123           # or a custom port
   ./serve_report.sh outputs/<run>/    # or a specific run
   ```

   Cursor/VS Code auto-forwards the port; open the forwarded
   `http://localhost:<port>/report.html` from the editor's Ports panel.

2. **Download the file.** `report.html` is fully self-contained (data is
   embedded inline, no external fetches), so you can right-click it in
   the remote file explorer -> Download, then double-click locally.

## Adding or editing error types

1. Open `prompts/error_types.yaml`.
2. Add (or edit) a key under `types:` with `name`, `category`,
   `description`, `example`, and `notes` covering both WHAT TO FLAG
   and WHAT NOT TO FLAG (cross-reference sibling type keys so the
   detector stays in its lane).
3. Rerun the experiment. The new type is picked up automatically and
   appears in the HTML filter + summary.

To iterate on the system/user template shared by every type, edit
`base.system` and `base.user_template` in the same file. The template
uses `str.format`-style placeholders - the available names are listed
in the comment at the top of the file.

### Group-level findings

The focused taxonomy's base template accepts an optional
`affected_rows` field in the JSON output. `text_inconsistency` and
`pattern_inconsistency` use it to report ALL rows participating in a
cross-row variant group with a single issue record (the headline `row`
is the smallest affected row). `src/core/report_builder.py` passes the list
through to `issues.json(l)` and the HTML drill-down so reviewers can
see the full span of each group-level defect.

## Configuring endpoints

`src/core/llm_client.py` ships with the default Azure endpoint pool used in the
original script. Override at runtime via:

```bash
export AZURE_OPENAI_ENDPOINTS="https://foo.openai.azure.com/,https://bar.openai.azure.com/"
```

Concurrency defaults to `min(8, #endpoints)`. The client itself is
thread-safe (round-robin endpoint selection is lock-protected), so
concurrency is achieved by having `PerTypeCleaningDetector` (detection
phase) and `ClusterSynthesiser` (Phase-B synthesis) each submit jobs to a
`ThreadPoolExecutor`. The two phases use independent concurrency knobs
(`--concurrency` vs `--synthesis-concurrency`) so you can throttle them
separately.

## Aggregation (Phase A + Phase B) internals

`src/core/aggregator.py` is the single source of truth for the clustering
pipeline:

* `cluster_issues(flat_issues)` - deterministic Phase A. Groups issues by
  row and applies connected-component union-find over column overlap.
  Issues with no `row` become singleton clusters. Returns a list of
  `Cluster` dataclasses (cluster_id, row, columns, row_values,
  error_types, detections, ...).
* `ClusterSynthesiser(...).run(clusters)` - Phase B. For every cluster
  with `evidence_count >= 2`, it renders the Phase-B prompt from
  `prompts/synthesise_cluster.yaml` (with the full source-row values
  inlined for context), fires one LLM call per cluster in parallel, and
  attaches a `ClusterSynthesis` object (`primary_error_type`,
  `supporting_error_types`, `root_cause`, `canonical_explanation`,
  `canonical_fix`, `confidence`, `is_likely_false_positive`,
  `reviewer_note`, plus `endpoint` / `attempts` / `raw_text` /
  `error` / `parse_error` provenance).

Single-type clusters skip Phase B by design (there is nothing to
reconcile). The dashboard transparently falls back to the lone
detection's explanation/fix for those.

## Estimating the false-negative rate (Good-Turing / Chao1)

A single detection pass tells us nothing about defects we **missed**.
The trick is to run the detector `K` times on *slightly perturbed*
views of the same data (row shuffles, optionally column shuffles) and
treat each distinct defect as a "species". A defect found by all K
runs is abundant; one found by a single run is a "singleton". Good-
Turing (1953) says the fraction of singletons bounds the probability
that the next run surfaces something brand new; Chao1 (1984) turns
that same frequency-of-frequencies distribution into a lower-bound
estimate of the TOTAL number of defects, which in turn gives an
estimated false-negative rate plus a 95 % log-normal CI (Chao & Yang
1993) - the statistical guarantee.

The row index is preserved end-to-end (via the frame's index through
`iter_chunks` and `to_markdown`), so detections from shuffled runs
come back already grounded in the original CSV row-id space. No
remapping is needed.

### Running a multi-pass experiment

The adaptive Good-Turing loop is integrated into the main
`run_cleaning_detection` entry point behind `--mode adaptive`:

```bash
python -m experiments.run_cleaning_detection \
    --csv-dir data/curated \
    --error-types pattern_inconsistency \
    --mode adaptive \
    --gt-num-runs 5 \
    --gt-base-seed 42
```

Knobs (all also settable under `good_turing:` in `config/experiment.yaml`):

* `--gt-mode {adaptive,fixed}` - `adaptive` stops early once the
  Good-Turing missing-mass drops below the target; `fixed` always
  runs `--gt-num-runs` passes.
* `--gt-num-runs K` - max passes. More runs tighten the CI at
  proportional LLM cost.
* `--gt-min-runs` / `--gt-target-missing-mass` - adaptive stop
  condition.
* `--gt-base-seed` - reproducibility root. Same seed + `K` yields the
  same perturbations file-by-file.
* `--gt-prompt-ensemble` / `--no-gt-prompt-ensemble` - use an LLM to
  rewrite each error type's description/notes/example per pass
  (recommended for cell-local error types where row shuffling does
  not move the detector).
* `--gt-shuffle-rows` / `--gt-shuffle-columns` - frame-level
  perturbation (column shuffle only observable in `column_mode=all`).
* `--gt-skip-identity-run` - by default run 0 uses the unshuffled
  frames + seed prompts (so you get the vanilla single-pass result as
  one of the K samples). Pass this to make every run perturbed.

Phase-B synthesis remains available in adaptive mode but the
estimator itself operates on raw per-type detections (synthesis would
partially merge species across runs and bias the estimate).

### Artifacts

Under the unified `outputs/<timestamp>_<model>/` run directory, in
addition to the normal single-pass artifacts:

* `runs/run_<i>/detections.jsonl` - flat detections from run `i`,
  already grounded in original CSV rows.
* `runs/run_<i>/run.json` - per-file effective seed + shuffle
  description.
* `prompt_variants/run_<i>/<type_key>.json` - seed vs. LLM-rewritten
  prompt content when `--gt-prompt-ensemble` is on.
* `good_turing/species.jsonl` - one line per canonical species
  `(source, column, row)` (cell-level identity - error_type is *not*
  part of the species key, see "Singletons are cells, not (cell,
  error_type) pairs" below) with `runs_found_in`, `runs_by_error_type`
  and the frequency `r`.
* `good_turing/estimates.json` - overall + per-`error_type` + per-
  source estimators (`good_turing_missing_mass`, `chao1.*`,
  `estimated_fn_rate` with 95 % CI).
* `good_turing/progression.jsonl` - estimator snapshot after each pass
  (used by the HTML dashboard's Good-Turing tab).
* `good_turing/report.md` - human-readable summary table + a
  cheat-sheet on how to interpret the numbers.

The HTML dashboard's **Good-Turing** tab surfaces all of the above
interactively whenever the run was produced in `mode=adaptive`.

### Reading the numbers

* **`good_turing_missing_mass`**: `n_1 / sum_r r*n_r`. The probability
  that a fresh perturbed run surfaces a never-before-seen defect.
  0 % means K runs have fully saturated the detector; 20 % means
  roughly one in five detections on the next run will be new.
* **`chao1.estimated_total`**: lower bound on the true number of
  defects. Minus the observed count gives the estimated "hidden"
  defects.
* **`estimated_fn_rate`**: point + 95 % CI on the fraction of defects
  the detector is missing. This is the headline "how worried should I
  be?" number.

Per-`error_type` tables tell you which detectors are most incomplete
(typically the ones with a long tail of singletons); per-source
tables pinpoint which CSV has the most hidden defects relative to
what was found.

### Caveats

* Both estimators assume species (defects) are drawn roughly
  independently across runs. Row-shuffle perturbation is a reasonable
  surrogate because chunking / ordering is the main thing that
  toggles marginal detections, but the assumption is not airtight.
  Treat the CI as "indicative, not adversarial".
* Chao1 is a *lower* bound on the total species count. The real FN
  rate may be higher if the detector has systematic blind spots that
  no amount of row reshuffling will expose.
* `K = 3` is the practical minimum (the bias-corrected Chao1 kicks in
  when `n_2 = 0`). `K = 5` gives comfortable doubleton counts on
  realistic data; `K >= 8` tightens the CI further if you can afford
  the LLM spend.
