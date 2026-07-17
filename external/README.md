# External baselines

This directory contains project-authored harnesses for comparison methods. The
repository tracks wrapper source, setup manifests, upstream commit pins, and
method documentation. It does **not** track third-party clones, virtual
environments, model checkpoints, browser profiles, source datasets, caches,
predictions, or result files.

The wrappers are not all equally maintained or equally reproducible. Read the
status column and each package README before use.

Clone all required upstream repositories at their tested commits with:

```bash
bash external/setup_upstreams.sh
```

This does not install environments, checkpoints, databases, or browser
dependencies; follow each package README for those steps.

## Baseline catalog

| Method | Package / command | Supervision | Status and provenance |
|---|---|---:|---|
| GPT-5.5 typo prompt | `python -m external.baseline.label_gpt55` | zero-shot | Maintained, self-contained wrapper. |
| Inconsistency prompt optimization | `python -m external.baseline.inconsistency.optimize` | benchmark dev split | Maintained deterministic core plus optional Azure agents. |
| Blind inconsistency LLM | `python -m external.llm.blind_inconsistency` | zero-shot | Project-native prompt and scorer. |
| Conservative precision LLM | `python -m external.llm.precision` | zero-shot | Project-native per-type prompts; score with `external.llm.score_precision`. |
| GPT-5 prompt tuning | `python -m external.gpt5_detect.run` / `.tune_prompt` | zero-shot or dev-tuned | Project-native Azure wrapper; caches and tuning dossiers are ignored. |
| Normalization key collision | `python -m external.normalization.run` | none | Maintained deterministic baseline; no upstream clone. |
| OpenRefine-style, pattern, PyOD | `python external/classical/run_classical.py` | none | Project-authored classical implementations; uses REIN data harness. |
| Raha / Baran / REIN detectors | `python external/rein/run_rein.py` | few-shot for Raha/Baran | Project wrapper around an ignored REIN clone. Review copied-port provenance before redistribution outside this repository. |
| SymSpell / NeuSpell | `python external/typo_baselines/run_typo.py` | none | Wrapper is tracked; setup downloads a large NeuSpell checkpoint. Upstream packages are MIT. |
| Cocoon | `python external/cocoon/run_cocoon.py` | zero-shot | Wrapper around upstream Cocoon commit `410a4f7`; upstream is MIT. Clone separately. |
| fm_data_tasks | `python external/fm_data_tasks/run.py` | zero/few-shot | Wrapper around upstream commit `c954cfd`; upstream is Apache-2.0. |
| ForestED | `python external/forested/run_forested.py` | synthetic few-shot | Upstream commit `6696c8e`, CC BY-NC 4.0. Noncommercial use only. |
| CleanAgent | `python external/cleanagent/annotate.py` then `detect.py` | zero-shot column typing | Wrapper loads prompt constants from upstream commit `66c10c2` after setup; upstream repository did not expose a clear license during integration. |
| TabClean | `python external/tabclean/run_tabclean.py` | few-shot | Wrapper around upstream commit `f514ce5`; no clear upstream license was found during integration. |
| ADR | `python external/adr/run_adr.py` | none | Experimental heavy wrapper around commit `b3004a8`; mixed upstream subprojects and unresolved top-level licensing. |
| HoloClean | `python external/holoclean/run_holoclean.py` | none | Experimental legacy wrapper around Apache-2.0 commit `d4f5929`; requires Python 3.6 and PostgreSQL. |
| ED2 | `python external/ed2/run_ed2.py` | active learning | Source retained for provenance; upstream license/fidelity remains unresolved. Do not use as a headline baseline yet. |
| ZeroER | `python external/zeroer/run_zeroer.py` | unsupervised | Apache-2.0 upstream commit `4c1ea6a`; current cell projection is not an authoritative row-duplication evaluation. |
| Excel Copilot | `python external/excel_copilot_local/scripts/run_excel_copilot.py` | product black box | Windows/Edge/M365 automation. Review service terms and organizational data policy. |
| Google Sheets Cleanup | `python external/sheets_cleanup/scripts/run_all.py` | product black box | Browser automation with an authenticated Google account. Review service terms and data policy. |

## Shared benchmark harness

Many wrappers use the REIN-shaped local dataset universe and common scorer:

```bash
bash external/rein/setup.sh
source external/rein/venv/bin/activate
python external/rein/prepare_datasets.py
python external/rein/score.py --predictions <predictions.jsonl>
```

Benchmark data and materialized tables are intentionally ignored. A clean clone
contains the harness but not the data, predictions, or reported leaderboard
artifacts.

## Reproducibility rules

When adding or maintaining a baseline:

1. Track only project-authored adapters, runners, setup files, requirements,
   upstream commit pins, and documentation.
2. Keep upstream repositories in ignored package-local clone directories.
3. Pin the exact upstream commit used by the wrapper.
4. State upstream license and model substitution explicitly.
5. Never commit downloaded checkpoints, environments, caches, prompts copied
   from license-unclear upstream source, source data, predictions, or reports.
6. State whether the method is zero-shot, synthetic few-shot, or receives real
   benchmark labels. Hold out supervised rows during evaluation.
7. Use the same benchmark immunity contract: ambiguous cells are FP-immune and
   `benchmark.ignored.jsonl` records are FP/FN-immune.
8. A wrapper that cannot run from its documented setup should be marked
   experimental rather than maintained.

## Ignored content

`.gitignore` is authoritative. Typical ignored paths include:

```text
external/*/venv/
external/*/mamba_env/
external/*/cache/
external/*/results/
external/*/outputs/
external/*/<upstream-clone>/
external/typo_baselines/**/checkpoints/
external/excel_copilot_local/profile/
external/sheets_cleanup/profile/
```

Do not force-add ignored content to make a wrapper appear self-contained.
