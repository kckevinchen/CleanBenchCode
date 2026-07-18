# CleanBench benchmark

CleanBench is the final cross-error-type benchmark produced from human decisions
in the labeling workbench. It combines cell-level, row-level, and cross-table
findings under one stable name and one stable directory. Refreshes replace the
current artifact in place.

## Current local artifact

The current build contains:

| Artifact | Records | Purpose |
|---|---:|---|
| `benchmark_cell.json` | 2,249 | Cell-level error and ambiguous records. |
| `benchmark_cell.csv` | 2,249 | CSV mirror of the cell JSONL. |
| `benchmark_row.json` | 46 | Row/relation/cross-table records. |
| `benchmark_row.csv` | 46 | CSV mirror of the row JSONL. |
| `ignored.jsonl` | 65 | Exclusion immunity sentinels. |
| `neutral_cells.jsonl` | 994 | Legacy skip/exclusion projection used by comparison harnesses. |
| `sanitized_sources.zip` | 1,000 CSVs | Source corpus with active exclusions applied. |
| `llm_explanation_cache.jsonl` | local cache | Content-keyed cell explanations. |
| `llm_row_cache.jsonl` | local cache | Content-keyed row explanations. |

Counts describe the local artifact on 2026-07-18. Rebuilds may change counts as
human labels or exclusions change.

The files named `*.json` are newline-delimited JSON, not JSON arrays. Parse them
one record per line.

## Cell benchmark

`benchmark_cell.json` is keyed by `(dataset, source, row, column)`.

| Field | Type | Meaning |
|---|---|---|
| `dataset` | string | `pbi_500` or `bing21_500`. |
| `source` | string | Source CSV basename. |
| `row` | integer | Original zero-based data-row index. |
| `column` | string | Column name. |
| `original_value` | string | Value observed in the source table. |
| `status` | string | `error` or `ambiguous`. Absence means unlabelled/clean under the exhaustive protocol. |
| `error_types` | string[] | One or more confirmed error types. Typo records may be split into spelling/formatting subtypes. |
| `corrected_value` | string, string[], or null | Canonical fix, acceptable bidirectional set, or no asserted fix. |
| `explanation` | string | Canonical explanation generated from the reviewed decision. |

Current status counts: 1,409 `error`, 840 `ambiguous`.

## Row benchmark

`benchmark_row.json` covers `fd_violation`, calculation inconsistencies,
`row_duplication`, and `inter_table_inconsistency`.

Common fields are `dataset`, `source`, `error_type`, `status`, `rows`, `columns`,
`rule`, `rationale`, `target_row`, `target_column`, `original_value`,
`corrected_value`, and `explanation`.

Cross-table records additionally identify the workbook, partner sheet, and
partner row. Row-duplication errors use `corrected_value="drop_row"` when the
human decision identifies a target row to remove.

Current status counts: 25 `error`, 21 `ambiguous`.

## Ignored and neutral records

`ignored.jsonl` contains active exclusion scopes. A matching prediction is
false-positive and false-negative immune. Records may represent:

- an entire CSV;
- a row in one CSV;
- a column in one CSV.

Exclusions are unioned across error-type runs before benchmark materialization.
An excluded slice cannot remain a positive label through a sibling run.

`neutral_cells.jsonl` is a compatibility projection for baseline scorers. New
evaluation code should prefer the benchmark plus `ignored.jsonl` contract.

## Source corpus

`sanitized_sources.zip` contains 1,000 CSV files under dataset directories plus
`MANIFEST.json` and, when rows are excluded, `_ROW_MAP.json`. It omits fully
excluded CSVs and removes excluded rows and columns.

Extract it to the canonical local source path:

```bash
rm -rf data/CleanBench
mkdir -p data/CleanBench
python - <<'PY'
from pathlib import Path
import zipfile
with zipfile.ZipFile('benchmark/CleanBench/sanitized_sources.zip') as archive:
    archive.extractall(Path('data/CleanBench'))
PY
```

When row exclusions exist, consumers must preserve original benchmark row
coordinates. For each mapped source, `_ROW_MAP.json` defines
`exported_to_source[i]` as the original benchmark row for exported data row
`i`. Do not assume that a filtered CSV's new positional index equals the
benchmark `row` field.

## Scoring contract

For cell-level micro-F1:

- **TP**: prediction on a CleanBench record with `status="error"`.
- **FN**: `error` record with no matching prediction.
- **FP**: prediction that does not match an error, excluding immune records.
- `ambiguous` records are FP-immune.
- `ignored.jsonl` scopes are FP/FN-immune.

Report supervision honestly. Zero-shot methods, synthetic few-shot methods, and
methods trained on real benchmark labels are not directly comparable. Rows used
as supervision must be held out of evaluation.

## Rebuild

Refresh the canonical directory in place:

```bash
az login
python -m scripts.export_cleanbench
```

Useful options:

```bash
# Reuse existing caches but make no new LLM calls.
python -m scripts.export_cleanbench --no-llm

# Restrict export to one registered source pool.
python -m scripts.export_cleanbench --pools excel_500

# Force explanation regeneration by dropping local caches first.
python -m scripts.export_cleanbench --fresh
```

The command composes:

1. `experiments.export_cell_benchmark`;
2. `experiments.export_row_benchmark`;
3. `experiments.export_cleanbench` for scoped immunity;
4. `experiments.export_neutral_cells`;
5. `experiments.export_sanitized_sources`.

The workbench run registry in `experiments.export_cleanbench` is the provenance
source for pool-to-run mappings. Generated benchmark data remains ignored and
must be backed up separately from Git.
