# Claude Code repository guide

Read [`AGENTS.md`](AGENTS.md) before making changes. It is the canonical coding
agent guide for this repository and covers architecture, ownership boundaries,
generated-data rules, subsystem workflows, and validation commands.

Human-facing documentation:

- [`README.md`](README.md): setup and project overview.
- [`LABELING_GUIDE.md`](LABELING_GUIDE.md): end-to-end human labeling workflow.

Critical reminders:

- Preserve unrelated worktree changes.
- Do not commit data, outputs, benchmarks, logs, secrets, or Azure resource IDs.
- Keep entry points thin and put reusable logic in `src/`.
- Keep prompt/taxonomy policy in YAML and preserve config/CLI parity.
- Run focused tests after each edit and report skipped live-service checks.
- Treat `python -m RFT.cli --list-tasks` as a precondition for RFT work; do not
  reintroduce historical version-package dependencies.
