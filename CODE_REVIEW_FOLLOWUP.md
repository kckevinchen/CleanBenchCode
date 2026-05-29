# Code-review follow-up — status: all three DONE

The three items deferred from the whole-codebase review (batches 1–7) are
now implemented on this branch, one commit each, full suite green between
them. This file is kept as the record; the commit messages carry the full
rationale.

| # | Item | Commit | Tests |
|---|------|--------|-------|
| 1 | `run_id` assigned at launch (kills concurrent misattribution + same-second dir collision) | `07887fc` | `tests/test_runs_run_id.py` (+15) |
| 2 | 6 JSONL stores collapsed onto a shared `JsonlStore` base (also extends out-of-band self-healing to the four stores that lacked it) | `db7f934` | `tests/test_jsonl_store.py` (+8) |
| 3 | LLM `check(df)` runs in a killable, resource-capped subprocess (`_sandbox.py` + `_check_runner.py`), replacing in-process exec on an uncancellable thread — fixes VERIFY *and* DETECT | `d9aa687` | `tests/test_check_runner.py` (+8) |

Suite at completion: **1561 passed, 6 skipped**.

## What each fix did, briefly

1. **`runs.py`** — `_mint_run_id` assigns `<ts>_<model-tag>_<rand8>` at launch
   and passes it via `--run-name`; `job.run_id` is set at Job construction.
   `_resolve_run_id` is demoted to a legacy fallback for jobs persisted by
   an older build. `build_detection_argv` now emits `--run-name`.
2. **`_jsonl_store.py`** — `JsonlStore[T, K]` owns the O_APPEND+fsync writes,
   the latest-per-key index, malformed-line tolerance, and the
   `(mtime,size)` reload-on-out-of-band-write. The six stores keep their
   own method names as thin wrappers and their distinct tombstone
   semantics. −511 duplicated lines → +147 wrappers + 175-line base.
3. **`_sandbox.py` / `_check_runner.py`** — sandbox primitives split out
   (dependency-light, so the child never imports the Azure/OpenAI SDKs);
   `run_check_subprocess` enforces a hard wall-clock timeout (SIGKILL),
   `RLIMIT_CPU` backstop, and `RLIMIT_AS` (4 GiB). Error labels
   (`timeout:` / `sandbox:` / `runtime:`) and masks are unchanged — proven
   by the existing relation tests routing through the subprocess.

## Remaining minor follow-ups (not blocking, low priority)

- **Relation DETECT re-exec reuse:** DETECT re-runs each verified
  `check(df)` against the full frame (now one subprocess spawn per verified
  relation). When VERIFY ran against the *full* frame (not a sample), its
  mask could be reused to skip the DETECT re-exec entirely. Minor; the
  spawn is dwarfed by the preceding LLM I/O.
- **`correction_suggester` dispatch layer:** `dispatch_groups` / `call_llm`
  remain untested by design (need a fake LLM client). Add if that code
  changes. The pure logic is covered (`tests/test_correction_suggester.py`).
