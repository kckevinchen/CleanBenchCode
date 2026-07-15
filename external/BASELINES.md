# External baselines on `benchmark/merged_v2` (cell-level)

Consolidated results for the external/published baselines evaluated on the
cell-level benchmark, all scored identically by `external/rein/score.py`:
detect over the **entire CSV**, count only labeled **`error`** cells as positives,
**neutralize** `ambiguous` + `skipped`/`excluded` cells (never clean/FP), with
unlabeled cells = clean (exhaustively-labeled benchmark). Metric = cell-level
micro precision/recall/F1 (RFT.md protocol).

All baselines run on the full benchmark **universe of 1002 source CSVs** — **217
labeled** (≥1 error/ambiguous cell) + **785 clean** (no errors) — byte-identical
to `data/curated_bing21_500_dedup` + `data/curated_pbi_500_dedup`. Flags on clean
tables count as false positives, so precision is honest (scoring only the labeled
tables roughly **doubles** precision for recall-heavy methods — see
`external/RESULTS.md` §5).

## ⚠️ Two categories: zero-shot vs few-shot

**raha, Baran, and TabClean are FEW-SHOT, not zero-shot, and are reported
separately.** Each is handed benchmark **ground truth** on a sample of rows per
table (raha/Baran via `clean.csv`; TabClean via an adaptive dev set), learns from
it, and predicts the rest. Two consequences:

1. **Direct error leak.** A few-shot learner can echo the labels it was handed
   back as "detections". raha is the clearest case — for its labeled rows it emits
   the ground-truth error cells *verbatim* (`detection.py` line 366: `i in
   labeled_tuples and extended_labeled_cells[(i,j)]`). Those are **guaranteed true
   positives it was simply handed**, not detection. We therefore evaluate every
   few-shot method **held-out**: its labeled/dev rows are removed from *both*
   predictions and ground truth (you don't test a few-shot learner on its own
   training examples). For raha at the standard 20-tuple budget, **46% of its raw
   TPs are leaked** (raw F1 0.779 → held-out 0.656); Baran and TabClean are held
   out the same way.
2. **Supervision scales with table size.** These label whole rows, so on small
   tables (median 98 rows) even a "few" tuples is a large fraction; raha's budget
   sweep below trades supervision for fairness.

All other baselines are **zero-shot** (no labels). `measure_raha_leakage.py`
quantifies raha's leak.

## Zero-shot leaderboard — full universe (ambiguous = neutral), sorted by precision

No ground-truth labels are given to any method here.

| baseline | cov | P | R | **F1** | typo | inconsist. | semantic | numeric |
|---|---|---|---|---|---|---|---|---|
| **OpenRefine clustering** (classical) | 1002 | **0.245** | 0.040 | 0.069 | 0.006 | 0.088 | 0.000 | 0.000 |
| **Cocoon** (ICDEW'25, reimpl.) | 1002 | 0.217 | 0.663 | **0.327** | 0.644 | 0.691 | 0.692 | 1.000 |
| **GPT-5.5 single-prompt** | 1000 | 0.185 | **0.773** | 0.299 | 0.841 | 0.693 | 0.808 | 1.000 |
| pattern/format-outlier (classical) | 1002 | 0.039 | 0.063 | 0.048 | 0.010 | 0.112 | 0.231 | 0.250 |
| SymSpell (default) | 1002 | 0.029 | 0.398 | 0.054 | 0.670 | 0.102 | 0.000 | 0.000 |
| NeuSpell | 1002 | 0.018 | 0.672 | 0.035 | 0.835 | 0.516 | 0.154 | 0.000 |
| SymSpell (aggressive) | 1002 | 0.017 | 0.488 | 0.032 | 0.750 | 0.204 | 0.000 | 0.000 |
| SymSpell ∪ NeuSpell (union) | 1002 | 0.016 | 0.717 | 0.032 | **0.896** | 0.545 | 0.154 | 0.000 |
| fm_data_tasks (VLDB'23, per-row) | 1002 | 0.008 | 0.478 | 0.016 | 0.695 | 0.221 | 0.692 | 0.750 |
| CleanAgent (unparseable) | 1002 | 0.004 | 0.016 | 0.006 | 0.012 | 0.007 | 0.308 | 0.000 |
| CleanAgent (changed) | 1002 | 0.002 | 0.083 | 0.004 | 0.020 | 0.136 | 0.615 | 0.500 |

`cov` = datasets scored (GPT-5.5 excludes 2 that timed out). Per-type columns are
**recall**; per-type precision is in `external/RESULTS.md`. **By F1, Cocoon is the
best honest detector** (a zero-shot, no-label method beats every few-shot method
once they are denied real-label memorization — see the synthetic-trained section).
**By precision, the classical OpenRefine clustering wins** (0.245) — a label-free
n-gram/fingerprint clusterer is *more precise* than the best LLM, but at much lower
recall (it only fires on clear casing/punctuation/token-order variants).

> **Numeric-outlier detectors omitted (F1 ≈ 0).** `outlierdetector` (IsolationForest)
> and the `PyOD` ensemble (ECOD/COPOD/KNN/LOF) both score P≈0/F1≈0: the benchmark
> has only **4 numeric-outlier cells**, so statistical outlier detection has
> essentially nothing to find (its flags are valid-but-extreme values). The
> paradigm is covered for completeness in `external/classical/` but not tabulated.

**Classical paradigms** (`external/classical/`, no LLM cost) close the
non-LLM gaps: OpenRefine clustering and the pattern/format-outlier detector cover
**inconsistency** (the largest error type, previously only LLM/few-shot).

## Few-shot leaderboard (held-out — labeled/dev rows excluded), sorted by precision

All three are handed ground truth and scored **held-out**. Report them separately
from the zero-shot table — they are not directly comparable.

| baseline | cov | P | R | **F1** | typo | inconsist. | semantic | numeric |
|---|---|---|---|---|---|---|---|---|
| **Baran** (Raha-detect → Baran-correct) | 1002 | **0.980** | 0.452 | 0.619 | 0.598 | 0.377 | 0.005 | 0.000 |
| **TabClean** (multi-agent repair) | 258† | 0.879 | 0.531 | **0.662** | 0.449 | 0.430 | 0.000 | 0.000 |
| **raha** (detection, budget 20) | 1002 | 0.702 | 0.615 | 0.656 | 0.355 | 0.345 | 0.002 | 0.000 |

Per-type columns are **precision** (these are the high-precision methods, so
precision is the interesting axis; per-type recall is in `RESULTS.md`).

- **Baran** — raha's correction engine, the "Baran" baseline in the TabClean
  paper. We run the realistic **Raha detect → Baran correct** pipeline and count a
  cell as detected iff Baran actually *changes* its value, so it is a
  high-precision subset of raha's flags (P **0.98**, only ~4 false positives over
  1002 tables). No LLM cost.
- **TabClean** — the headline new baseline (arXiv:2606.25388). Multi-agent LLM
  pipeline that synthesizes a reusable Python cleaning *program*, run through our
  Azure `LLMClient` (gpt-5.5, ~$102 total). **Best few-shot F1 (0.662)**. Nearly
  all of its few false positives are on 3 clean tables, and the largest cluster is
  TabClean **catching genuine float-precision artifacts** (`14045.400000000001`
  → `14045.4`) that the benchmark never labeled — i.e. benchmark gaps, so its true
  precision is likely higher than 0.879.
- **raha** — see the budget sweep below.

† TabClean is LLM-expensive, so it ran on the **158 error-bearing tables + a
100-table clean sample** (258 total), not all 1002. Because conservative few-shot
methods barely flag clean tables (only 3/100 clean tables flagged here), this
labeled+sample precision closely tracks the full-universe value — the same
labeled-only ≈ full-universe behavior verified for raha in `RESULTS.md` §5.

### raha budget sweep (held-out)

raha across labeling budgets. `cell-leak` = average fraction of a table's cells
whose label raha is handed (`python external/rein/measure_raha_leakage.py`).

| raha budget | cell-leak | cov | P | R | F1 | (raw F1, w/ leak) |
|---|---|---|---|---|---|---|
| **fixed 20 tuples** (standard protocol) | ~28% | 1002 | 0.702 | 0.615 | **0.656** | 0.779 |
| frac 0.05 of rows | ~5% | 1002 | 0.483 | 0.321 | 0.386 | 0.423 |
| frac 0.02 of rows | ~2.4% | 1002 | 0.253 | 0.278 | 0.265 | 0.289 |

Even held-out and fair, raha's score falls steeply as supervision shrinks
(F1 0.66 → 0.39 → 0.27) — it remains few-shot at every budget. Reproduce:
```bash
python external/rein/run_rein.py --detector baran --tag baran            # Baran
python external/rein/run_rein.py --detector raha --tag raha_fewshot20     # raha
external/tabclean/venv/bin/python external/tabclean/run_tabclean.py \
    --scope sample --clean-sample 100 --dev-min-rows 20 --workers 8       # TabClean
python external/rein/score.py --predictions external/rein/results/predictions_baran.jsonl
#   held-out is the default; add --no-holdout-labeled-rows for the raw (leaky) number
```

## Reading the results

- **GPT-5.5 single-prompt** — best *zero-shot* precision **and** recall, most
  balanced per-type coverage (only baseline strong on all four types at once). One
  unified prompt (distilled from `prompts/error_types.yaml`) per whole table.
- **Typo specialists (SymSpell / NeuSpell / union)** — recall-heavy on typos
  (union 0.90) and useful spill into inconsistency, but precision is very low
  (they flag clean proper nouns). NeuSpell is contextual (BERT); union maximizes
  coverage.
- **CleanAgent** — narrow by design (5 standardizable types ≈ date columns here);
  genuine niche is `semantic_incompatible` (0.62) via unstandardizable values.
- **outlierdetector** — numeric-only; near-zero on this typo/inconsistency-heavy
  benchmark (expected).
- **OpenRefine clustering (classical)** — highest *precision* on the board
  (0.245), no labels, no LLM. Key-collision (fingerprint + n-gram) flags minority
  casing/punctuation/token-order variants of a column's canonical form. Precise
  but low recall — only the clearest inconsistency variants.
- **pattern / format-outlier (classical)** — per-column char-class signature;
  flags minority *formats* in format-regular columns (dates, codes, numbers).
  Recall-oriented coverage of `inconsistency`/`typo-formatting`; modest precision.
- **PyOD ensemble (classical, numeric)** — ECOD/COPOD/KNN/LOF consensus on numeric
  columns. Scores ~0 because the benchmark has only 4 numeric-outlier cells; like
  `outlierdetector`, it honestly shows statistical outlier detection has nothing
  to find here (its flags are legitimate extreme-but-valid values).
- **raha** — few-shot; competitive only because it is handed benchmark labels, and
  even held-out its score depends heavily on the labeling budget. Report it
  separately from zero-shot methods.

## Where each baseline lives

| baseline | dir | how |
|---|---|---|
| raha, baran, outlierdetector, mvdetector, dboost | `external/rein/` | REIN detectors, no PostgreSQL |
| SymSpell, NeuSpell, union | `external/typo_baselines/` | spell correctors as detectors |
| CleanAgent | `external/cleanagent/` | LLM annotate (`gpt-4o`) → `dataprep.clean` diff |
| GPT-5.5 single-prompt | `external/gpt5_detect/` | one unified prompt per table, all 4 cell types |
| Cocoon | `external/cocoon/` | zero-shot per-column regex/CASE-WHEN (ICDEW'25 reimpl.) |
| TabClean | `external/tabclean/` | few-shot multi-agent repair program (arXiv:2606.25388) |
| fm_data_tasks | `external/fm_data_tasks/` | per-row serialization + Yes/No recipe (VLDB'23 reimpl.) |
| ForestED | `external/forested/` | few-shot LLM-induced decision-tree + GNN ensemble (clone, LLMClient seam) |
| **pattern, openrefine, pyod** | **`external/classical/`** | **classical: format-outlier, key-collision clustering, PyOD ensemble** |

Reproduce any row:
```bash
python external/rein/score.py --predictions external/rein/results/predictions_<name>.jsonl
```

## Notes
- Scoring/neutral-set protocol and the shared harness are documented in
  `external/rein/README.md`.
- LLM baselines use the repo's Azure `LLMClient`: CleanAgent annotation =
  `gpt-4o` (paper's model); single-prompt detector = `gpt-5.5`, reasoning effort
  `medium`, one prompt per whole table (chunk-size 100).
