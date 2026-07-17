# GitHub Copilot instructions

Read [`AGENTS.md`](../AGENTS.md) before changing this repository. It is the
canonical architecture, workflow, invariants, and validation guide for coding
agents.

Use the other documentation by audience:

- [`README.md`](../README.md): human setup and product overview.
- [`LABELING_GUIDE.md`](../LABELING_GUIDE.md): authoritative labeling and export
  workflow.
- [`external/baseline/README.md`](../external/baseline/README.md): maintained
  external baseline commands.

Highest-priority repository rules:

- Check `git status` first and preserve unrelated user changes.
- Keep `experiments/*.py` as thin orchestration; reusable behavior belongs in
  `src/`.
- Keep taxonomy and prompt policy in `prompts/*.yaml`, settings in
  `config/experiment.yaml`, and preserve YAML/CLI parity.
- Never commit ignored datasets, benchmark rows, model outputs, logs, `.env`,
  credentials, endpoint names, assistant IDs, or cloud job state.
- Azure authentication uses `AzureCliCredential`; the main detector reads
  endpoints from `AZURE_OPENAI_ENDPOINTS`, while some specialized commands also
  accept explicit endpoint arguments.
- Label stores are append-only. Keep benchmark, ignored-cell immunity, eval
  bundle, and source-export exclusion semantics aligned.
- Before RFT work, run `python -m RFT.cli --list-tasks`. The intended task
  surface is unversioned; repair stale imports instead of restoring `vN`
  dependencies.
- Run focused tests immediately after edits. Use Ruff on touched files because
  the full repository has historical lint debt.
