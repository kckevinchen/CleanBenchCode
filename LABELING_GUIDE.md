# Label a new dataset with the workbench

This guide takes a dataset from local CSV files to a reviewed, cell-level
benchmark. The workbench can launch detection runs, display every candidate in
its source-table context, record human decisions, capture errors the detector
missed, and export an evaluation-ready bundle.

## What the labeling workflow produces

A normal session creates or updates files inside `outputs/<run_id>/`:

- `labels.jsonl`: append-only human verdict history.
- `exclusions.jsonl`: CSV, row, and column scopes intentionally left out.
- `benchmark.jsonl`: materialized cell-level ground truth.
- `benchmark.ignored.jsonl`: skipped or excluded cells that must be immune to
  false-positive and false-negative scoring.
- `benchmark_bundle_<run_id>.zip`: the recommended portable export, containing
  both JSONL files and `MANIFEST.txt` with the scoring contract.

The `data/` and `outputs/` directories are intentionally ignored by Git. They
remain untracked and are stored locally in the workspace, but detection sends
selected table content to the Azure OpenAI endpoints you configure. Optional
web verification can also send selected values to the configured Foundry Agent.
Review your organization's data-handling policy before using live services.

## 1. Prepare the environment

From the repository root:

```bash
./create_venv.sh
source venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
```

Live detection calls use Azure CLI authentication and endpoints from `.env`:

```bash
az login
cp .env.example .env
# Edit .env and set AZURE_OPENAI_ENDPOINTS.
```

If a completed run already exists under `outputs/` and you only need to review
it, the workbench does not make an Azure call during ordinary labeling.

## 2. Add the dataset

Put one CSV or a directory of CSVs under `data/`. Use a descriptive, stable
name because source filenames become part of benchmark cell identities.

```text
data/
  customer_review_2026/
    customers.csv
    orders.csv
```

Recommendations:

- Keep column headers in the first row.
- Give every CSV a unique filename within the dataset.
- Do not rename a CSV or reorder its rows after labeling starts. Benchmark rows
  use `(source, row, column)` identities and zero-based data-row numbers.
- Keep unrelated datasets in separate directories so a run has a clear scope.

## 3. Start the workbench

Use an annotator name that will remain meaningful in exported provenance:

```bash
./scripts/run_server.sh start --port 8000 --annotator your-name
./scripts/run_server.sh status
```

Open `http://localhost:8000/`. VS Code Remote SSH normally forwards the port
automatically. The server log is `logs/workbench.log`.

Useful server commands:

```bash
./scripts/run_server.sh restart --port 8000 --annotator your-name
./scripts/run_server.sh stop
```

## 4. Create candidate findings

If you already have a completed run under `outputs/`, skip to the next section.
The Runs page discovers run directories automatically.

To launch a run from the UI:

1. Open **Launch new run**.
2. For the first live run, choose one small dataset CSV in **Dataset**. Expand
   to directory mode after you have checked the output and token volume.
3. For directory mode, keep **CSV glob** as `*.csv` unless you need a narrower
   pattern.
4. Select only `typo`, set **Max chunks** to `1`, and enable **Skip Phase-B
   synthesis** for the smoke run. Dedicated inconsistency, row-duplication, and
   multi-table paths have separate call patterns and are not bounded by the
   generic chunk limit.
5. Choose **single** mode for the first pass. The configured adaptive mode can run
   up to several perturbed LLM passes per arm and is better suited to a later
   coverage study.
6. Select **Preview command** and confirm the resolved CSV path, error types,
   model, and output settings.
7. Select **Launch run**.
8. Return to **Runs** and wait for the job to complete.

The workbench persists background job state under `logs/jobs/`, so restarting
the server does not discard a detector process that is still running.

A run receives a **Label** button when it contains cluster, inconsistency, or
multi-table findings. If no Label button appears, inspect the run report and job
log first; the run may be incomplete or may contain no candidates.

## 5. Review the candidate queue

Select **Label** on the run. Start with the **Status: unlabeled** filter so the
queue contains only undecided items.

For each item:

1. Inspect the highlighted source row and surrounding values.
2. Read the proposed error type, explanation, affected columns, and suggested
   correction.
3. Choose a verdict.
4. Enter the corrected value when the finding is an error and the intended value
   is known.
5. Add a short note when the decision depends on domain knowledge or an
   exception that another reviewer should understand.
6. Press **Enter** to save and advance.

### Verdict meanings

| Verdict | Use when | Benchmark effect |
|---|---|---|
| `tp` | The candidate is a real data error. | `label=error` |
| `fp` | The value is valid and the detector over-fired. | `label=clean` |
| `partial` | The candidate contains a real error, but only part of the proposed scope or fix is correct. Edit the correction and note. | `label=error` |
| `unsure` | Evidence is insufficient to decide safely. | `label=ambiguous` |
| `skip` | This cell should not be judged, for example because it needs unavailable context. | Omitted from the benchmark and added to the ignored/immunity set |

Use `skip` for a specific deferred finding. Use an exclusion when an entire CSV,
row range, or column is outside benchmark scope.

### Keyboard shortcuts

| Key | Action |
|---|---|
| `j` / `k` | Next / previous queue item |
| `1` | `tp` |
| `2` | `fp` |
| `3` | `partial` |
| `4` | `unsure` |
| `5` | `skip` |
| `c` | Focus corrected value |
| `n` | Focus note |
| `/` | Focus queue search |
| `Enter` | Save and advance; if no verdict is selected, the UI defaults to `fp` |

The sidebar filters can combine source, status, error type, and adaptive column
status. Use search for a filename, column, value, or cluster identifier.

## 6. Add errors the detector missed

A candidate-only review measures precision but cannot establish recall. Record
false negatives whenever you find a real error that has no candidate.

### Missed cell errors

In the source table, select the missed cell and create a cell-level `fn` label.
Enter its corrected value and error type. The cell joins the same queue and
benchmark as detector-proposed findings.

### Missed inconsistency values

For systematic value-standardization review, open the **column board** from an
inconsistency item.

On the column board:

1. Review each LLM-proposed cluster.
2. Accept or reject it; before accepting, edit the canonical value, members, and
   inconsistency kind when necessary.
3. To add a missed cluster, select two or more distinct values and choose
   **Make cluster from selection**.
4. If one bad value has no canonical peer in the column, select its flag icon,
   enter the corrected value, and save it as a missed inconsistency (`fn`).
5. Finish the column with one status:
   - `labeled`: apply the accepted clusters.
   - `no_inconsistency`: the column contains nothing to standardize.
   - `dropped_overlapping`: another error category already covers the issue.
   - `dropped_messy`: the column cannot support a reliable benchmark decision.

The column board writes missed values immediately; no separate cluster is
required.

## 7. Exclude out-of-scope data

The exclusion bar on the Label page can exclude:

- the current CSV,
- specific zero-based row numbers or ranges,
- named columns.

Excluded slices remain visible and can be restored, but they are removed from
both the benchmark and sanitized source export. The eval bundle represents them
in `benchmark.ignored.jsonl` as wildcard immunity records.

Do not mark an out-of-scope column as `fp`: `fp` asserts that reviewed cells are
clean, while an exclusion asserts that they must not be scored.

## 8. Quality-check the labels

Before export:

1. Filter `unsure` and resolve any decisions for which new context is available.
2. Filter `partial` and confirm every corrected value and affected column.
3. Filter `fn` and ensure every missed error has the intended correction and
   error type.
4. Check that no false negative is already covered by a `tp` cluster.
5. Review excluded scopes and remove accidental exclusions.
6. Sample `fp` decisions to catch systematic over-rejection.
7. Review each inconsistency column that still has an unlabeled status.

The export action runs a conflict check. If the same value in the same source
column has contradictory labels across rows, the UI opens a conflict dialog
with **Jump** links. Resolve the conflict unless the row-specific difference is
intentional and documented.

## 9. Export the benchmark

Open **Export** and choose **Eval bundle (recommended)**. The ZIP contains:

- `benchmark.jsonl`: one authoritative record per `(source, row, column)`, with
   `label` in `error`, `clean`, or `ambiguous`. Row-level findings such as
   duplication use `column=""` as a sentinel.
- `benchmark.ignored.jsonl`: exact cells and wildcard row/column/CSV exclusions
  that receive false-positive and false-negative immunity.
- `MANIFEST.txt`: file descriptions and the cell-level micro-F1 scoring rules.

Other UI exports are useful for narrower workflows:

- **Export benchmark**: `benchmark.jsonl` only.
- **Per-type benchmark**: one JSONL per error type in a ZIP.
- **Single error type**: one selected error-type JSONL.
- **Export source CSVs**: source files with active exclusions removed. When
   rows are removed, `_ROW_MAP.json` maps each exported row position back to
   its original zero-based benchmark row.

Keep the eval bundle and source export together when handing the benchmark to a
training or evaluation pipeline. Apply `_ROW_MAP.json` before joining
predictions from a row-filtered source CSV to benchmark coordinates.

## Resume, duplicate, and back up work

Labels are append-only and persist after every save, so closing the browser does
not lose completed decisions. Reopen the same run from **Runs** to continue.

Use **Duplicate** on the Runs page before a risky bulk operation. A duplicate can
keep existing labels or start as a clean-slate copy.

For a direct backup, copy the complete run directory:

```bash
cp -a outputs/<run_id> /path/to/backup/
```

Do not copy only `benchmark.jsonl` while labeling is in progress; the append-only
label stores are the source of truth and the benchmark is regenerated on export.

## Troubleshooting

### Dataset is not listed

Confirm it is under `data/`, then refresh the Runs page or restart the workbench.
For directory mode, confirm the CSV glob matches at least one file.

### Launch fails before the first model call

Check `.env`, run `az login`, and inspect `logs/workbench.log`. A live run needs
at least one endpoint in `AZURE_OPENAI_ENDPOINTS`.

### Label button is missing

The run must finish with at least one cluster, inconsistency draft/label, or
multi-table finding. Open its report and job log to determine whether detection
completed successfully.

### Export is blocked by conflicts

Use the conflict dialog's **Jump** actions, align the contradictory verdicts,
and export again. Use **Export anyway** only when the same literal value is truly
row-dependent and the discrepancy is intentional.

### Port 8000 is unavailable

Start on another port:

```bash
./scripts/run_server.sh start --port 8001 --annotator your-name
```
