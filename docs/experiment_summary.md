# Experiment Summary: Heuristic + Synthetic Benchmark

Two complementary harnesses for measuring the LLM data-quality detector
against ground truth, run on `data/curated_pbi_full/154329618` (6 small
CSVs from the PowerBI corpus).

## Experiment A — Heuristic-as-oracle on clean data

Compares the LLM's flat issue list against deterministic heuristic
detectors (`whitespace`, `unicode_norm`, `casing_fingerprint`,
`pattern_minority`, `key_collision_typo`, `fd_violation`,
`numeric_outlier`) over the **un-perturbed** corpus.

- LLM run: [outputs/20260425T001101Z_gpt-5.2](outputs/20260425T001101Z_gpt-5.2)
  (288 LLM calls, single-pass, gpt-5.2)
- Heuristic run: [outputs/20260424T235931Z_heuristic](outputs/20260424T235931Z_heuristic)
- Comparison: [outputs/expA_clean_compare/comparison.md](outputs/expA_clean_compare/comparison.md)

### Result

| metric | value |
|---|---|
| LLM issues | **0** |
| Heuristic issues | **1** (one numeric outlier, low confidence) |
| LLM ∩ Heuristic | 0 |
| Heuristic-only (FN candidates) | 1 |

**Interpretation.** This curated slice is genuinely clean — both the
LLM and the heuristics find essentially nothing. The corpus does not
discriminate detector quality; **for a meaningful Experiment A we need
a slice with real, organic errors** (e.g. a non-curated `data/raw_pbi/`
spreadsheet, or a curated slice known to contain manually-verified
issues). The harness itself works end-to-end and is ready for re-run.

## Experiment B — Synthetic injection benchmark

Inject a controlled mix of perturbations into the same clean corpus,
then score each detector against the manifest by `(source, row,
column)`. Because the manifest carries the *injected* error type, this
gives **exact per-error-type precision / recall / F1**.

- Benchmark: [outputs/synth_smoke](outputs/synth_smoke) — 49 injections
  across 47 unique cells (rate 0.02, seed 17, 8 injectors).
- LLM run on injected: [outputs/20260425T001559Z_gpt-5.2](outputs/20260425T001559Z_gpt-5.2)
- Heuristic run on injected: [outputs/synth_smoke_heur](outputs/synth_smoke_heur)
- LLM score: [outputs/synth_smoke/score_llm/benchmark_score.md](outputs/synth_smoke/score_llm/benchmark_score.md)
- Heuristic score: [outputs/synth_smoke/score_heur/benchmark_score.md](outputs/synth_smoke/score_heur/benchmark_score.md)

### Cell-level recall (any issue flagged on injected cell)

| detector | cells flagged | cell-recall |
|---|---:|---:|
| LLM (gpt-5.2, single pass) | 28 | **0.60** |
| Heuristic stack | 32 | **0.68** |

### Per-error-type recall

| error_type | n_inj | LLM recall | LLM F1 | Heuristic recall | Heuristic F1 |
|---|---:|---:|---:|---:|---:|
| `casing_inconsistency` | 7 | **0.86** | 0.92 | **1.00** | 1.00 |
| `missing_value` | 6 | 0.00 | 0.00 | 0.00 | 0.00 |
| `number_format_inconsistency` | 7 | **0.43** | 0.50 | 0.00 | 0.00 |
| `numeric_outlier` | 7 | **1.00** | 0.93 | 0.43 | 0.55 |
| `pattern_inconsistency` | 6 | 0.00 | 0.00 | 0.00 | 0.00 |
| `punctuation_inconsistency` | 5 | **1.00** | 0.67 | **1.00** | 1.00 |
| `text_inconsistency` (whitespace) | 7 | 0.00 | 0.00 | **0.86** | 0.92 |
| `typo` | 2 | 0.50 | 0.67 | 0.00 | 0.00 |

### Findings

1. **LLM and heuristics are complementary, not redundant.**
   - LLM dominates on `numeric_outlier` (1.00 vs 0.43) and is the only
     detector that catches `number_format_inconsistency` and `typo`.
   - Heuristics dominate on `text_inconsistency` (whitespace anomalies:
     0.86 vs 0.00) and tie on `casing` and `punctuation`.
   - Cell-level union recall = **0.79** (37/47 cells caught by at least
     one detector). The two stacks together miss 10 of 47 injected
     cells — primarily `missing_value` and `pattern_inconsistency`.

2. **Type-level confusion is the LLM's main failure mode**, not
   blindness. The confusion matrix shows the LLM consistently flags
   injected `text_inconsistency` (whitespace) as
   `punctuation_inconsistency` (3 cells), and injected
   `pattern_inconsistency` as `number_format_inconsistency` /
   `punctuation_inconsistency`. The cell is detected, just labelled
   wrong. **This means cell-level recall (0.60) is a lower bound on
   the LLM's true detection rate — the labelling-aware F1 is harsher
   than the underlying detector quality.**

3. **`missing_value` is a systematic blind spot.** Neither stack
   catches it (heuristic has no missing-value detector enabled, and
   `missing_value` is not in the default LLM error-type list — only 8
   of the 20 taxonomy types ran). Easy fix: add `missing_value` to
   `--error-types`, plus a one-line heuristic.

4. **`pattern_inconsistency` is a blind spot for both.** The heuristic
   has a `pattern_minority` detector but its modal-pattern threshold
   (0.8) is too strict for short numeric/alphanumeric columns where
   the injection adds a single punctuation char. The LLM mis-labels.

### Threats to validity

- **Single dataset, 49 injections**, biased toward small spreadsheets.
  Per-type recall confidence intervals are wide (~±0.15 at this n).
- **Perfect ground truth on the manifest, but lossy alignment**: the
  scorer matches by `(source, row, column)` and counts a cell as
  caught even if the LLM flags it for a different reason than the
  injected one. The per-error-type metrics adjust for this; the
  cell-level recall does not.
- **Synthetic ≠ organic**. Real spreadsheets often contain semantic
  errors (`semantic_incompatible`, `business_rule_violation`) that
  cannot be mechanically injected; this benchmark scores the easy
  half of the taxonomy.

## Reproducing

```bash
# A: heuristic + LLM on a clean slice, then compare.
python -m experiments.run_heuristic_detection \
    --csv-dir data/curated_pbi_full/<id> --output-dir outputs --no-html
python -m experiments.run_cleaning_detection \
    --csv-dir data/curated_pbi_full/<id> --mode single --output-dir outputs
python -m experiments.compare_runs \
    --llm-run outputs/<llm_run> --heuristic-run outputs/<heur_run> \
    --output-dir outputs/expA_compare

# B: synth benchmark.
python -m experiments.build_synth_benchmark \
    --input data/curated_pbi_full/<id> --output-dir outputs/synth_<id> \
    --rate 0.02 --seed 17
python -m experiments.run_cleaning_detection \
    --csv-dir outputs/synth_<id>/injected --mode single --output-dir outputs
python -m experiments.score_synth_benchmark \
    --benchmark-dir outputs/synth_<id> --run-dir outputs/<llm_run> \
    --output-dir outputs/synth_<id>/score_llm
```

## What's next

- Re-run Experiment A on a less curated slice (`data/raw_pbi/...`) so
  the heuristic finds enough issues to make the FN-floor estimate
  non-degenerate.
- Add `missing_value` heuristic detector (1 line: `pd.isna(v)`-based).
- Hand-label a sample of the LLM's false negatives in the labeling UI
  to estimate true heuristic precision and bias-correct the FN-floor.
- Scale the synth benchmark to 200+ injections across 5 spreadsheets
  for tighter per-type CIs.
