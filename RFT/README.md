# Reinforcement fine-tuning

> **Current status: unavailable while package flattening is incomplete.**
>
> The intended task surface is unversioned (`RFT/typo/` and
> `RFT/inconsistency/`), but stale imports still reference deleted historical
> packages. Do not submit, resume, or deploy an RFT job until both preconditions
> below pass.

```bash
python -m RFT.cli --list-tasks
python -m pytest -q RFT/tests/test_rft_cli.py -k 'not prepare_validates_real_splits'
```

A failing import is part of the unfinished refactor. Repair it against the
unversioned packages; do not restore dependencies on `vN` directories.

## Intended architecture

RFT uses a generic Azure lifecycle engine driven by declarative task specs:

```text
RFT/
  _common/          shared engine, state, scoring, deployment, and graders
  typo/             current typo prompt, grader, split builder, and spec
  inconsistency/    current inconsistency prompt, grader, split builder, and spec
  cli.py            generic command surface
  registry.py       one registered spec per task
```

A task spec defines:

- trainee response format;
- grader and reward variants;
- train, validation, and evaluation split locations;
- model and hyperparameter defaults;
- optional prediction parsing and evaluation scoring.

The generic engine owns prepare, upload, submit, status, cancel, supervise,
deploy, run, delete, and model-listing behavior. Task packages must not copy
that lifecycle code.

## Non-negotiable data rules

- The evaluation split must never be uploaded as training or validation data.
- Human benchmark rows, generated splits, model outputs, cloud state, and logs
  are local ignored artifacts.
- Cell-level micro-F1 uses summed TP, FP, and FN counts.
- Ambiguous benchmark cells are false-positive immune.
- Records in `benchmark.ignored.jsonl` are false-positive and false-negative
  immune, including wildcard row, column, and CSV exclusions.
- Azure endpoints, job IDs, deployment IDs, and assistant IDs must come from
  environment variables, CLI arguments, or ignored state, never committed
  defaults.

## Completing the flattening

The repair is complete only when:

1. `RFT/registry.py` imports only unversioned task specs.
2. `RFT/typo/` imports only its own modules and `RFT/_common/`.
3. `RFT/inconsistency/` imports only its own modules and `RFT/_common/`.
4. The registry exposes one stable task name per error type.
5. Split builders do not read data from historical peer-version directories.
6. Both precondition commands at the top of this file pass from a clean clone.
7. Documentation and examples contain no `typo_vN`, `inconsistency_vN`, or
   `RFT/<task>/vN` command paths.

Historical experiment results should live outside the runtime package, for
example in a paper artifact or an ignored local archive. They should not remain
as importable peer packages.

## Intended command surface

The commands below describe the target interface. Run them only after the
preconditions pass:

```bash
az login

python -m RFT.cli --list-tasks
python -m RFT.cli --task typo prepare
python -m RFT.cli --task typo all --reward-variant r1 \
  --endpoint <azure-openai-endpoint> \
  --output-dir outputs/rft_typo
python -m RFT.cli --task typo status --follow
python -m RFT.cli --task typo deploy \
  --deployment-name <deployment-name> \
  --resource-group <resource-group>
python -m RFT.cli --task typo run --deployment <deployment-name>
```

State, generated splits, predictions, and logs belong under ignored output
folders. Before any live work, inspect the task's split paths and confirm that
the training grader cannot access held-out labels.

Coding agents must read [`../AGENTS.md`](../AGENTS.md) before changing this
subsystem.
