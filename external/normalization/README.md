# Normalization-key baseline

A deterministic baseline that groups distinct values in one column when they
become identical after lowercasing and removing punctuation, symbols,
whitespace, digits, and underscores.

Run it as a workbench-compatible detection job:

```bash
python -m external.normalization.run \
  --csv-dir <csv-or-directory> \
  --output-dir outputs
```

The resulting run contains standard issues, clusters, summary, and optional HTML
report artifacts, so it can be reviewed in the labeling workbench.

For a raw streaming export of normalization groups:

```bash
python -m external.normalization.detector \
  --data-dir <flat-csv-directory> \
  --out-dir outputs/normalization_baseline
```

This method is label-free, LLM-free, and project-authored.
