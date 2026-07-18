# Experiment commands

`experiments/` contains supported command-line orchestration for detection,
labeling, benchmark export, and evaluation. Reusable behavior belongs in `src/`;
temporary migrations and dataset-specific repair scripts should not be retained
here after they have served their purpose.

## Core workflow

- `run_cleaning_detection.py`: main candidate-generation orchestrator.
- `workbench.py`: labeling workbench entry point.
- `build_report.py`: rebuild a run report.
- `compare_runs.py`: compare two compatible detection runs.
- `run_heuristic_detection.py`: deterministic heuristic baseline runner.

## Column clustering

- `run_column_clustering.py`: generate whole-column inconsistency clusters.
- `score_column_clusters.py`: materialize and score column-cluster labels.
- `export_inconsistency_benchmark.py`: export reviewed inconsistency labels.
- `export_inconsistency_benchmark_with_llm.py`: export with canonical LLM fixes.

## Relations

- `run_relation_discovery.py`: discover and validate relation candidates.
- `run_relation_validation.py`: validate a saved relation run.
- `run_relation_detection.py`: compile relation violations into findings.
- `build_relation_report.py`: build the relation report.
- `revalidate_relations.py`: rerun downstream validation after rule changes.
- `recompile_relation_violations.py`: rerun violation compilation without discovery.
- `synthesise_relation_clusters.py`: add canonical fixes and explanations.

## Web verification

- `run_web_verify.py`: verify individual findings with the configured agent.
- `run_web_verify_column.py`: verify whole-column findings.

## CleanBench

The supported one-command entry point is `python -m scripts.export_cleanbench`.
It composes these modules:

- `export_cell_benchmark.py`
- `export_row_benchmark.py`
- `export_cleanbench.py`
- `export_neutral_cells.py`
- `export_sanitized_sources.py`

## Evaluation and analysis

- `analyze_stop_thresholds.py`: analyze adaptive stopping thresholds.
- `build_synth_benchmark.py` and `score_synth_benchmark.py`: synthetic benchmark pair.
- `compare_external_inconsistency_run.py`: compare external inconsistency findings.
- `score_typo_dir_benchmark.py` and `dump_typo_dir_mistakes.py`: typo baseline scoring and diagnostics.

## Data preparation

The thin wrappers under `data_prep/` delegate to reusable implementations in
`src.data_prep`:

- `data_prep/bing21_reconstruct.py`: `python -m experiments.data_prep.bing21_reconstruct`
- `data_prep/curate.py`: `python -m experiments.data_prep.curate`
- `data_prep/subsample.py`: `python -m experiments.data_prep.subsample`
