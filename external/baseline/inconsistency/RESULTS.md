# Inconsistency prompt-optimization — results

GPT-5.5 inconsistency **rulebook** optimized by the multi-agent loop
(`external/baseline/inconsistency/`). Reviewer sees dev results; a mechanical leakage
filter + editor-output scrub keep all data out of the prompt; the editor (which
never sees data) rewrites the rulebook; candidates are selected on a held-out
**val** split and the winner is scored **once** on a held-out **test** split.

- Detector: `gpt-5.5`, `reasoning_effort=none`, `temperature=0`.
- Agents (reviewer/editor): `gpt-5.5`, `reasoning_effort=medium`.
- Benchmark: `benchmark/merged_v2/benchmark_inconsistency_cells.jsonl`
  (pbi_500 + bing21_500; 422 error + 229 ambiguous), split **by source**:
  dev 28 / val 14 / test 42. *Corrected 2026-06-22:* 3 missed-FN cells were
  retagged `inconsistency` (2 phone dot→dash, 1 trailing comma); all numbers
  below are re-scored against this corrected gold (`rescored_metrics.json`).
- Scoring: per distinct `(column, value)`; `ambiguous` is FP-immune; clean
  columns audited for precision. Curator-ignored CSVs/columns/cells
  (`benchmark/merged_v1/ignored.jsonl`) are excluded from scoring.

## Run `hardened10v2` (12 iterations)

Held-out **TEST** (42 sources, scored once on the best prompt vs the seed):

| Prompt | Precision | Recall | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|
| Seed rulebook | 0.207 | **0.889** | 0.335 | 80 | 307 | 10 |
| **Optimized (iter 12)** | **0.312** | 0.800 | **0.449** | 72 | 159 | 18 |
| **Δ** | **+0.105** | −0.089 | **+0.114** | −8 | **−148** | +8 |

Test F1 **+0.114 (+34 % relative)**, driven by roughly **halving false positives**
(307 → 159) at a modest recall cost. Per dataset (optimized): bing21 F1 0.456,
pbi F1 0.389 — both well above the seed.

> **Exclusions matter:** ignoring the curator-marked columns/files removes 19
> columns from the test audit (incl. two of the largest false-positive columns)
> and lifts measured F1 from 0.402 → **0.449** for the optimized prompt and
> 0.306 → 0.335 for the seed. Re-run any export with `--no-exclusions` to compare.

## Run `lean` (10 iterations, length-budgeted)

Same protocol with a 6 KB prompt budget + schematic-only examples. Best prompt
is **7.0 KB** (vs 13 KB for `hardened10v2`) at comparable accuracy:

| Prompt | Precision | Recall | F1 |
|---|---|---|---|
| Seed rulebook | 0.207 | 0.889 | 0.335 |
| **Optimized (lean)** | **0.295** | 0.744 | **0.423** |

About half the prompt length for ~0.026 less test F1 than `hardened10v2`.

### Error CSV for investigation
`outputs/inconsistency_promptopt/<run>/error_cells_{dev,val,test}.csv` — one row
per `(source, column, value)` classified TP/FP/FN/AMBIG, with context (value
count/share, column size + value distribution, predicted kind/correction/
rationale, gold correction). Built by `python -m external.baseline.inconsistency.export_errors`
(uses the run's detector cache → no API calls). Curator-ignored columns/files
are excluded by default.

### Integrity (verified every iteration)
- **No data leak:** 0 dev values reached the editor input or any evolved prompt
  (13/13 prompts clean) — mechanical filter + editor-output scrub.
- **No test overfit:** selection on held-out val; test scored once.

### Known issue (addressed by the `lean` run)
- `hardened10v2`'s optimized prompt grew 4.2 KB → 13 KB (3.3×) with concrete
  domain examples. The `lean` run (length budget + schematic-only examples +
  consolidation) brings this to **7.0 KB** at ~0.026 lower test F1.

Artifacts: `outputs/inconsistency_promptopt/<run>/` (`best_prompt.md`,
`loop_summary.json`, `rescored_metrics.json`, per-iter `prompt.md` /
`decision.json` / `leakage_report.json`, `error_cells_{dev,val,test}.csv`).
