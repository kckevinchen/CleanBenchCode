# Classical (non-LLM) detectors on `benchmark/merged_v2` (cell-level)

Three zero-shot, label-free, deterministic detectors that fill the **classical /
statistical paradigms** the rest of the roster lacked (everything else is
spell-checkers, LLMs, or few-shot learners). All run over the full **1002-CSV
universe**, emit `(dataset, row, col)` cells in the shared schema, and are scored
by `external/rein/score.py` — no labels, no held-out, no LLM cost.

| detector | paradigm | targets | full-universe P / R / F1 |
|---|---|---|---|
| `openrefine` | key-collision clustering | inconsistency, casing/punct/token-order variants | **0.245** / 0.040 / 0.069 |
| `pattern` | char-class format profiling | inconsistency, typo-formatting | 0.039 / 0.063 / 0.048 |
| `pyod` | statistical outlier ensemble | numeric_outlier | 0.000 / 0.001 / 0.000 |

**OpenRefine clustering is the most *precise* baseline on the whole board** (0.245,
above Cocoon's 0.217 and GPT-5.5's 0.185) — a label-free clusterer beats the LLMs
on precision, though at far lower recall.

## What each does

- **`pattern`** — for each column, map every value to a collapsed **character-class
  signature** (uppercase→`A`, lowercase→`a`, digit→`9`, punctuation kept; runs
  collapsed, so `Dec 1`→`Aa 9` and `12/1/11`→`9/9/9`). In a column with a dominant
  format, cells whose signature is a clear minority are flagged as format outliers.
  Skips (a) free-text columns with no dominant pattern and (b) low-cardinality
  categorical *enums* (a rare label is a legitimate category, not a format error —
  those are left to clustering). The "Auto-Detect / Potter's Wheel" idea.
- **`openrefine`** — OpenRefine's **key-collision** clustering: group a column's
  values by a canonical key (`fingerprint` = lowercase, drop punctuation, sort
  unique tokens; plus `n-gram fingerprint`). Within a cluster sharing a key, the
  minority raw variants are flagged as inconsistent representations of the majority
  canonical form. Restricted to **textual** columns (so token-sorting can't merge
  genuinely-distinct numbers/dates like `3/4/2008` vs `4/3/2008`).
- **`pyod`** — for each numeric column, a **PyOD ensemble** (ECOD, COPOD, KNN, LOF)
  with a consensus vote flags outlier cells. Scores ~0 here because the benchmark
  has only 4 numeric-outlier cells — an honest "the paradigm has nothing to find"
  result, matching `outlierdetector`.

## Run

```bash
# rein venv (has sklearn/scipy/numpy + pyod): pip install pyod
source external/rein/venv/bin/activate
python external/classical/run_classical.py --detector pattern --detector openrefine --detector pyod
#   datasets come from external/rein/prepare_datasets.py --include-clean-universe

python external/rein/score.py --predictions external/rein/results/predictions_openrefine.jsonl
python external/rein/analyze_by_type.py        # consolidated per-type P/R/F1
```

`--limit N` runs a smoke test on the first N datasets; `--tag` overrides the
output filename (single detector). Predictions land in `external/rein/results/`
(gitignored) as `predictions_<detector>.jsonl` + a `.ran.json` scope sidecar.

## Layout / notes

```
external/classical/
  detectors.py     # detect_pattern / detect_openrefine / detect_pyod (pandas; pyod for pyod)
  run_classical.py # runner reusing external/rein/common.py paths + loader
  README.md
```

- **Zero-shot, full universe.** These belong on the *zero-shot* leaderboard with
  Cocoon, GPT-5.5, SymSpell/NeuSpell, and CleanAgent — never the few-shot section.
- **Deterministic, fast** (~1s for pattern/openrefine, ~14s for pyod over 1002
  tables). No network, no labels, no LLM budget.
- **Faithful, not vendored.** `pattern` and `openrefine` reimplement the documented
  classical methods (pattern profiling; OpenRefine key-collision); `pyod` uses the
  PyOD library directly.
