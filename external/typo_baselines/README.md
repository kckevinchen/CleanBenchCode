# `external/typo_baselines/` — SymSpell + NeuSpell on `benchmark/merged_v2`

Two classic spell/typo correctors repurposed as **cell-level error detectors** and
evaluated on the cell benchmark, reusing the REIN harness so results sit in one
table next to `raha`.

* **SymSpell** — dictionary + Symmetric-Delete edit distance (no ML, CPU, ~3s).
* **NeuSpell** — contextual BERT spell-corrector (`subwordbert-probwordnoise`, CPU).

A cell is flagged when the corrector judges it misspelled (SymSpell: a token has a
strictly-closer dictionary word; NeuSpell: the contextual correction differs from
the original). To maximize coverage they run over **every textual cell** (any cell
with a letter, all columns), and when both run together a **`union`** detector
(SymSpell OR NeuSpell) is also emitted.

```
external/typo_baselines/
  run_typo.py        the two detectors (+ union); writes predictions into external/rein/results/
  setup.sh           venv + deps + NeuSpell checkpoint download
  requirements.txt   pinned deps (transformers==4.38.2, gdown, ...)
  venv/              (gitignored; holds the 2.2GB NeuSpell checkpoint)
```

## Evaluation protocol (matches the benchmark's design)

The benchmark is **exhaustively labeled**: a cell is an error only if labeled
`error`; every **unlabeled** cell is **clean** (a true negative). So:

1. **Detect over the entire CSV** — all cells, not just labeled ones.
2. **Positives = labeled `error` cells only.** Recall is measured against these.
3. **Neutralize `ambiguous` + `skipped` + `excluded` cells** — never scored as
   clean/FP. `ambiguous` comes from the benchmark; `skipped`/`excluded` come from
   `benchmark/merged_v2/neutral_cells.jsonl` (see
   `python -m experiments.export_neutral_cells`). A labeled **error always wins**
   over a neutral rule (so the 20+21 error cells that live in excluded
   files/columns still count).

This is exactly what `external/rein/score.py` does (the `neut` column shows how
many predictions were neutralized); precision is therefore **exact**, and the
spellers' low precision is a real finding — they over-flag clean proper nouns.

## Setup

```bash
bash external/typo_baselines/setup.sh        # venv + deps + 2.2GB NeuSpell checkpoint
source external/typo_baselines/venv/bin/activate
```

### NeuSpell on Python 3.12 — the four gotchas (all handled)

NeuSpell is unmaintained; getting it to run on py3.12 needed:
1. **Checkpoint download** — NeuSpell's Google-Drive downloader returns an HTML
   "virus scan" page for the 2.2GB model; `setup.sh` fetches it with `gdown`.
2. **`torch.load`** — torch ≥2.6 defaults to `weights_only=True` and rejects the
   checkpoint; `run_typo.py` forces `weights_only=False` for it.
3. **`load_state_dict`** — old checkpoints carry an extra `embeddings.position_ids`
   buffer; loaded non-strictly.
4. **`transformers`** — v5 removed `tokenizer.encode_plus`; pinned to `4.38.2`
   (still has it, and ships py3.12 `tokenizers` wheels).

## Run

```bash
# datasets are materialized once by the REIN adapter:
python external/rein/prepare_datasets.py            # if not already done

python external/typo_baselines/run_typo.py --tool symspell --tool neuspell
#   --mode aggressive   lowers thresholds (min token len 3, keep ALL-CAPS) for more recall
#   --limit N           smoke test on N datasets

python external/rein/score.py --predictions external/rein/results/predictions_union.jsonl
python external/rein/score.py --predictions external/rein/results/predictions_neuspell_default.jsonl
```

NeuSpell over all 215 CSVs (~57k textual cells) takes ~17 min on CPU; SymSpell ~3s.

## Results (full benchmark, 922 error cells, neutral set applied)

| detector | coverage | P | R | F1 | typo R | inconsist. R | semantic R |
|---|---|---|---|---|---|---|---|
| `raha` (REIN, for reference) | 186/215 | **0.785** | 0.690 | **0.735** | 0.693 | 0.672 | 1.000 |
| SymSpell (default) | 215/215 | 0.054 | 0.399 | 0.095 | 0.671 | 0.102 | 0.000 |
| SymSpell (aggressive) | 215/215 | 0.033 | 0.489 | 0.062 | 0.752 | 0.204 | 0.000 |
| NeuSpell | 215/215 | 0.040 | 0.675 | 0.076 | 0.837 | 0.516 | 0.154 |
| **SymSpell ∪ NeuSpell** | 215/215 | 0.036 | **0.719** | 0.068 | **0.898** | **0.545** | 0.154 |

Takeaways:
- **Coverage maximized**: the union recovers **90% of typos** and — because a
  contextual speller treats casing/format slips like misspellings — **55% of
  inconsistency** errors and some semantic ones, despite being "just" spell tools.
- **Precision is genuinely low**: spellers flag thousands of clean proper-noun
  cells (firm names). raha stays the precision/F1 leader; the spellers are
  recall-oriented complements.
- **`raha` coverage caveat**: raha could not run on 29/215 datasets (a NaN edge
  case in its sampling) and is scored on its 186-dataset / 836-error scope; the
  spellers cover all 215.

## Limitations

- Numeric-outlier recall is ~0 (only 2 such cells; not a typo task).
- 2 excluded-column names in `neutral_cells.jsonl` don't match a `dirty.csv`
  header (duplicate-column rename / whitespace) and are skipped — negligible.
- Row-level skips (153) are ignored for cell scoring (they pertain to row-level
  error types, handled by the separate row benchmark).
