# Coding agent guide

This file is the canonical repository context for coding agents. Human setup and
product usage belong in `README.md`; the complete annotation workflow belongs in
`LABELING_GUIDE.md`.

## Mission and system boundary

This repository is a research harness for creating data-quality benchmarks with
Azure-hosted language models:

```text
candidate generation -> human labeling workbench -> benchmark export -> RFT/eval
```

The benchmark, not the model output, is the final product. Prefer recall during
candidate generation, preserve human provenance, and never blur training data
with held-out evaluation data.

## Start here

Before changing code:

1. Read the nearest owning module and its focused tests.
2. Read `LABELING_GUIDE.md` for any labeling or export behavior.
3. Inspect `config/experiment.yaml` and `prompts/error_types.yaml` before adding
   Python configuration.
4. Check `git status`; the worktree may contain user changes. Never revert or
   stage unrelated work.
5. For RFT work, run `python -m RFT.cli --list-tasks` before editing. RFT is
   currently being flattened to unversioned task packages; a failing import is a
   blocker to repair, not a reason to reintroduce `vN` package dependencies.

## Environment and commands

```bash
./create_venv.sh
source venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt

# Live Azure calls use Entra/Azure CLI credentials, not committed API keys.
az login
cp .env.example .env
set -a; source .env; set +a

# Metadata and an offline prompt render.
python -m experiments.run_cleaning_detection --list-error-types
python -m experiments.run_cleaning_detection --csv /path/to/input.csv --dry-run

# Focused tests first.
python -m pytest -q tests/test_cluster_synthesiser_singletons.py
python -m pytest -q tests/test_export_benchmark.py
python -m pytest -q -m "not integration"

# Use Ruff on touched files. The full repository has historical lint debt.
python -m ruff check --select F,I path/to/touched.py tests/test_touched.py
python -m ruff format --check path/to/touched.py
```

Do not run live-service tests unless the task requires them. Tests marked
`integration` or gated by `RUN_LLM_TESTS=1` can spend tokens and depend on Azure
resources.

## Architecture

### Candidate generation

The taxonomy is defined in `prompts/error_types.yaml` and currently has nine
error types grouped by scope.

- `src/core/detector.py`: per-chunk, per-type detector used by the standard
  single-column path.
- `src/core/aggregator.py`: source-aware deterministic union-find clustering
  plus the optional Phase-B canonical explanation/fix pass.
- `src/adaptive/`: repeated perturbed passes plus Good-Turing/Chao1 coverage
  estimates.
- `src/core/relations/`: discovery, code verification, filtering, and violation
  detection for inter-column relations.
- `src/core/row_dup/`: row-duplication candidate generation and adjudication.
- `src/core/multi_table/`: cross-table grouping and inconsistency detection.
- `src/verifier/`: optional Foundry Agent web verification.

`experiments/run_cleaning_detection.py` is the main orchestrator. It contains
routing decisions that are easy to miss: for example, inconsistency findings
can be owned by the whole-column clustering path rather than the generic chunk
scanner. Follow the current dispatcher before assuming taxonomy scope equals one
implementation.

### Human labeling workbench

- Entry point: `experiments/workbench.py`.
- Backend: `src/labeling/` (FastAPI app, stores, routers, export materialization).
- Frontend: `frontend/labeling/` (static HTML, CSS, and ES modules).
- Run discovery: every compatible directory under `outputs/` should be visible
  without a separate database.

Label stores are append-only JSONL histories. An `unlabeled` record is a
tombstone, not a destructive delete. Preserve that model when adding undo or
bulk operations.

The main label vocabulary is:

- cluster `tp` / `partial` -> benchmark `error`;
- cluster `fp` -> benchmark `clean`;
- cluster `unsure` -> benchmark `ambiguous`;
- cluster `skip` and active exclusions -> ignored export with FP/FN immunity;
- cell `fn` -> human-added missed error.

Exports must stay aligned:

- `benchmark.jsonl`: authoritative labeled cells;
- `benchmark.ignored.jsonl`: skipped/excluded immunity records, including
  wildcard row/column/CSV scopes;
- eval bundle: both files plus `MANIFEST.txt` scoring rules;
- source export: source CSVs with the same exclusions applied; row-filtered
  files carry `_ROW_MAP.json` for original benchmark coordinates.

When changing labeling behavior, test both the store/API path and benchmark
materialization. Relevant suites include:

```bash
python -m pytest -q tests/test_export_benchmark.py
python -m pytest -q tests/test_column_cluster_endpoints.py
python -m pytest -q tests/test_column_cluster_export.py
python -m pytest -q tests/test_exclusion_cascade.py
```

### RFT

`RFT/` is a generic engine driven by task specs. The intended target task
surface is unversioned:

```text
RFT/typo/
RFT/inconsistency/
```

Do not create another copied launcher or import a historical peer version.
Shared mechanics belong under `RFT/_common/`; task-specific prompts, graders,
and split builders belong in the task package.

Non-negotiable evaluation rule: the eval split must never be uploaded as
training or validation data. Cell-level micro-F1 is computed from summed TP, FP,
and FN counts; ambiguous cells are FP-immune, while ignored cells are FP/FN-immune.

The RFT tree is mid-refactor and currently fails on stale versioned imports.
Always treat these checks as preconditions:

```bash
python -m RFT.cli --list-tasks
python -m pytest -q RFT/tests/test_rft_cli.py -k "not prepare_validates_real_splits"
```

Repair stale package imports before making feature changes. Do not describe RFT
as runnable in human documentation until these checks pass.

### External baselines

`external/README.md` is the catalog and provenance guide. The repository tracks
project-authored wrappers across multiple methods, including:

```bash
python -m external.baseline.prompt
python -m external.llm.precision --help
python external/rein/run_rein.py --help
python external/classical/run_classical.py --help
python -m external.normalization.run --help
```

Only wrappers, setup manifests, provenance pins, and documentation are tracked.
Upstream clones, environments, checkpoints, browser profiles, source data,
caches, predictions, and results remain ignored. Preserve each method's stated
supervision level and licensing/fidelity caveats when changing or reporting it.

## Ownership and layering rules

- Keep `experiments/*.py` thin: argument parsing and orchestration only. Put
  reusable logic in `src/`.
- Keep taxonomy/prompt language in `prompts/*.yaml`, runtime settings in
  `config/experiment.yaml`, and behavior in Python.
- Preserve YAML/CLI parity when adding configuration.
- Use structured JSON/JSONL/YAML APIs, not ad hoc string replacement.
- Reuse existing JSONL stores and router context rather than adding a second
  persistence path.
- Keep output directories compatible with the workbench run loader.
- Avoid unrelated refactors in large orchestration and frontend files.

## Generated and private data

The following are intentionally ignored and may be very large or sensitive:

```text
data/ datasets/ outputs/ benchmark/ label_result/ logs/ backups/
docs/ notebooks/ checkpoints/ runs/ state/ tmp/ external/*
```

The source-wrapper exceptions under `external/` are declared explicitly in
`.gitignore`.
Never force-add ignored datasets, benchmark rows, model outputs, logs, cloud job
state, `.env`, endpoint names, assistant IDs, or credentials.

Azure configuration rules:

- Configure OpenAI resources through `AZURE_OPENAI_ENDPOINTS`; some specialized
  commands also expose explicit endpoint arguments.
- Configure Foundry Agent resources through `AZURE_AGENTS_*`.
- Authentication is normally `AzureCliCredential`.
- `external/baseline` also honors `AZURE_OPENAI_KEY` when explicitly set and
  otherwise falls back to `AzureCliCredential`.
- Do not add organization-specific resource URLs or IDs as code defaults.

## Common change workflows

### Add or edit an error type

1. Edit `prompts/error_types.yaml`.
2. Include `name`, `category`, `description`, `example`, and notes with both
   `WHAT TO FLAG` and `WHAT NOT TO FLAG`.
3. Add any tunable thresholds to YAML, not Python literals.
4. Run `--list-error-types`, an offline `--dry-run`, prompt-template tests, and
   focused detector tests.

### Change the labeling UI

1. Find the owning router/store and the corresponding frontend module.
2. Keep direct API and UI behavior equivalent.
3. Preserve keyboard shortcuts and non-mouse operation.
4. Check text fitting and desktop/mobile layout.
5. Parse ES modules with:

   ```bash
   node --input-type=module --check < frontend/labeling/label.js
   ```

6. Run focused API/export tests. For visual changes, start the workbench and
   inspect the real page rather than relying only on static markup.

### Change benchmark export

1. Update materialization in `src/labeling/export.py` or the owning export
   router.
2. Preserve one record per `(source, row, column)`. Row-level findings use
  `column=""` as the sentinel key.
3. Keep exclusions and skipped-cell immunity synchronized across benchmark,
   ignored sidecar, bundle manifest, and source export.
4. Run `tests/test_export_benchmark.py` and conflict/exclusion tests.

### Change adaptive detection

Preserve canonical source/row/column identities across perturbations. Final
reports are built from the union of detections across passes; saturation is per
arm, not a global stop signal.

## Validation and completion

After the first substantive edit, run the cheapest check that can falsify it.
Before finishing:

1. Run focused tests for every touched subsystem.
2. Compile or import changed Python entry points.
3. Check changed JavaScript as ES modules.
4. Run `git diff --check`.
5. Inspect `git status` and keep unrelated changes uncommitted.
6. State any skipped live tests, existing unrelated failures, or known broken
   surfaces explicitly.

A full `ruff check .` is not currently a useful release gate because the
repository carries broad historical modernization debt. Do not fix thousands of
unrelated warnings as part of a focused task.
