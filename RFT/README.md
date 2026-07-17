# RFT experiments

Reproducible source snapshots of each Reinforcement Fine-Tuning (RFT)
experiment run on Azure OpenAI. Versioned folders preserve prompts, grader
specifications, task configuration, and the scripts that produce each run.
Generated datasets, predictions, logs, cloud state, and resource identifiers
remain under ignored runtime directories and are not part of the repository.

## Unified RFT CLI (the way to run jobs now)

The whole Azure lifecycle is one generic engine driven by a declarative
**task spec**, instead of a per-version launcher. A task spec
(`RFT/<task>/spec.py`, a `RftTaskSpec`) captures the only things a version
varies -- the trainee output schema, the grader, the data splits, and the
defaults. Everything else (upload / submit / poll / supervise / deploy /
delete / run) lives once in `RFT/_common/engine.py` and is exposed by
`RFT/cli.py`.

```bash
az login                                   # required for any live call
python -m RFT.cli --list-tasks             # tasks grouped by error type

# Train: prepare -> upload -> submit -> follow, in one shot
python -m RFT.cli --task typo_v10 all --reward-variant r1 \
   --endpoint <azure-openai-endpoint> \
    --output-dir outputs/rft_typo_v10_r1 --suffix v10-r1

python -m RFT.cli --task typo_v10 status --follow            # monitor
python -m RFT.cli --task typo_v10 cancel                     # cancel

# Auto-resubmit to the next endpoint on terminal failure (was inconsistency-only)
python -m RFT.cli --task inconsistency_v1 supervise --attach-job-id <job>

# Deploy the fine-tuned model (management-plane via `az`), then evaluate it
python -m RFT.cli --task typo_v10 deploy --deployment-name o4-mini-v10-r1 \
    --resource-group <rg>          # or set $AZURE_RESOURCE_GROUP
python -m RFT.cli --task typo_v10 run --deployment o4-mini-v10-r1   # micro-F1 on eval split
python -m RFT.cli --task typo_v10 delete --target deployment --target files
```

## Layout: versions grouped by error type

Versions live nested under their error type, and **no version imports another
version** -- shared, rulebook-free typo primitives (the trainee schema + the
deterministic python sub-graders) live in `RFT/_common/typo_task.py`; the
version-specific rulebook + score-model judge stay in each version:

```
RFT/
  _common/      spec.py engine.py cli-helpers, typo_task.py (shared primitives)
  typo/         v1/ v2/ ... v9/ v10/      (each self-contained)
  inconsistency/ v1/
  cli.py registry.py scaffold.py _template/
```

**Add a new version -- change only data, prompt, grader:**

```bash
python -m RFT.cli new mytask v1 --findings-key findings    # -> RFT/mytask/v1/
```

scaffolds `RFT/mytask/v1/` from `RFT/_template/` (a working skeleton). Then edit
three things and register one line:

1. **prompt** — `_prompt.py` (`TRAINEE_PROMPT` + output schema);
2. **grader** — `_grader.py` (set `TASK_DESCRIPTION`; the default is the shared
   gold-free `5*F1` judge in `RFT/_common/graders.py`, so most tasks write no
   grader code);
3. **data** — `scripts/build_splits.py --sources <csv_dir>` writes
   `data/{train,valid,eval}.jsonl` (trainee messages baked in);
4. register `from RFT.mytask.v1.spec import SPEC as MYTASK_V1` in
   `RFT/registry.py`.

No launcher, no argparse, no lifecycle code -- the full
upload/submit/supervise/deploy/run/delete surface comes for free. `spec.py` is
built by `make_spec(...)`, which wires the `run`-verb parser/scorer from a
couple of field names. See `RFT/_template/README.md` for the full checklist.

The frozen `typo/v3` … `typo/v8` snapshots keep their own `scripts/*_rft_*.py`
launchers for historical reproducibility; only the live tasks (`typo/v9`,
`typo/v10`, `inconsistency/v1`) are migrated to specs and fully decoupled from
peer versions. (Their eval/data-prep scripts still reuse the shared v5 compare
harness + v8 label helpers -- analysis tooling, not the RFT runtime path.)

| Experiment        | Task                                                      | Status |
|-------------------|-----------------------------------------------------------|--------|
| [typo/v1](typo/v1) | Per-cell typo / clean / ambiguous classifier (3-way) | historical eval complete (label accuracy 0.483) |
| [typo/v2](typo/v2) | Per-cell typo / clean classifier (BINARY; no trivial errors) | historical run submitted |

The **live** task definition in `experiments/_typo_task.py` is **v2**.
The v1 task definition is preserved frozen at
`RFT/typo/v1/scripts/_typo_task.py`. See
[`typo/v2/README.md`](typo/v2/README.md) for the rationale behind the
v2 changes.

Add a new sibling folder (`RFT/<task>/`) per new RFT run rather than
overwriting an existing one -- the folders are versioned snapshots, not
the live working copies.
