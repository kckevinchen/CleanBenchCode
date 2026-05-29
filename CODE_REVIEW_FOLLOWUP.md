# Code-review follow-up — 3 deferred items

Written after the whole-codebase review (batches 1–7, all committed on
`refactor/core-packages-and-router-extraction`). These three were
**explicitly deferred** to a fresh session because each is larger than a
bug-fix batch and carries real regression risk. They are ordered by
risk-to-reward: do them top-to-bottom, one PR each, full suite green
between them.

The contained concurrency bug (stores serving stale data on out-of-band
writes) was **already fixed** in batch 7 — `LabelStore` and
`ColumnClusterStore` now track `(mtime, size)` and reload on external
change. Item 2 below generalises that fix to the other four stores.

---

## 1. `run_id` misattribution across concurrent detection launches

**Where:** `src/labeling/runs.py`
- `_launch` — `runs.py:883`; snapshots `existing = sorted({r.run_id ...})`
  at `runs.py:888`, stored on the job as `existing_run_ids` (`runs.py:925`).
- `_resolve_run_id` — `runs.py:978`; runs in the reaper thread after the
  child exits (`_reap` calls it at `runs.py:963`) and also from the
  recovery path (`runs.py:1059`, `runs.py:1070`).

**The bug.** A detection subprocess creates `outputs/<timestamp>_<model>/`
itself; the parent never knows the name up front, so it *infers* it:

```python
existing = set(job.existing_run_ids)              # snapshot at launch
now_runs = {r.run_id for r in self._registry.list()}
new_runs = now_runs - existing
if len(new_runs) == 1:
    job.run_id = next(iter(new_runs))
elif new_runs:                                    # >1 new dir: guess newest by mtime
    newest = max(new_runs, key=mtime); job.run_id = newest.run_id
```

If **two detection jobs run concurrently** (the workbench allows it), job
A's reaper sees *both* A's and B's directories in `new_runs`. The
`len == 1` branch fails, and the `mtime` tiebreak attributes whichever dir
is newest — frequently B's — to job A. Result: the UI links job A to B's
report, and B may later get A's. It is a race purely in the *inference*;
the runs themselves are fine on disk.

**Fix (preferred): make the run id deterministic and pass it in.** Stop
inferring. Generate the run-dir name in `_launch` and hand it to the child:

1. In `_launch`, when `kind == "detection"`, build the run id the same way
   the experiment script does (timestamp + model) — or just a ULID — and
   pass it as an explicit CLI flag, e.g. `--run-id <id>` /
   `--output-dir outputs/<id>`. Set `job.run_id` immediately (no
   inference, no reaper dependency).
2. In `experiments/run_cleaning_detection.py` (and the relation-detection
   entry points), honour that flag instead of minting the dir name. Check
   whether a flag already exists — several entry points accept
   `--output-dir`; if so this is mostly wiring, not new surface.
3. Delete `_resolve_run_id` and the `existing_run_ids` snapshot, or reduce
   `_resolve_run_id` to a *fallback* used only for legacy jobs persisted
   without a `run_id` (recovery path at `runs.py:1030`+).

**Fix (smaller, if changing the child contract is undesirable): claim by
identity, not by recency.** Have the child write its chosen dir name to a
parent-known sentinel (`<log_dir>/<job_id>.runid`), and have
`_resolve_run_id` read that file instead of diffing the registry. Removes
the race but keeps the inference architecture.

**Risk:** medium. Touches the launch contract between the job manager and
every detection entry point. The recovery path (`recover_running` /
re-attach after a server restart, `runs.py:1030`+) also calls
`_resolve_run_id` and must keep working for jobs persisted by the old code.

**Tests:** in `tests/test_runs.py` (or wherever the job-manager tests
live), add a test that launches two fake "detection" jobs whose stub
subprocesses each create a distinct `outputs/<id>/` dir, let both reap,
and assert each `job.run_id` matches the dir *that job* created — not the
newest. Today that test fails; after the fix it passes. Keep a
legacy-job recovery test if the fallback path is retained.

---

## 2. Collapse the 6 JSONL stores onto a shared `JsonlStore` base

**The six stores** (all in `src/labeling/`, all the same shape):

| file | class | line | key |
|------|-------|------|-----|
| `store.py` | `LabelStore` | 26 | `target_id` |
| `column_cluster_store.py` | `ColumnClusterStore` | 32 | `(source, column)` |
| `cell_notes_store.py` | `CellNotesStore` | 29 | (cell key) |
| `exclusions_store.py` | `ExclusionStore` | 26 | (scope key) |
| `multi_table_label_store.py` | `MultiTableLabelStore` | 21 | (target key) |
| `relation_review_store.py` | `RelationReviewStore` | 79 | (relation key) |

Every one is: `__init__(path)` + `threading.Lock` + `_records: List[T]` +
`_latest: Dict[K, T]` + `reload()` + `append()` (identical
`O_WRONLY|O_CREAT|O_APPEND` + `os.write` + `os.fsync`) + per-key queries.
~779 lines total, mostly duplicated.

**Why now.** Batch 7 added the `_file_sig` / `_reload_if_changed`
self-healing only to `LabelStore` and `ColumnClusterStore`. The **other
four still serve stale data on out-of-band writes** — the exact bug we
just fixed, still latent in `CellNotesStore`, `ExclusionStore`,
`MultiTableLabelStore`, `RelationReviewStore`. A base class fixes all four
at once and prevents the next store from regressing.

**Approach.** Introduce `src/labeling/_jsonl_store.py`:

```python
class JsonlStore(Generic[T, K]):
    model: type[T]                      # pydantic model, set by subclass
    def _key(self, rec: T) -> K: ...    # subclass: key extraction
    # provides: __init__(path), reload, _reload_locked, _reload_if_changed,
    #           _current_sig, append, append_many, all, __iter__,
    #           latest_by_key -> Dict[K, T], latest_for(key), history(pred)
```

Each subclass shrinks to: declare `model`, implement `_key`, and add its
*specialised* queries with friendly names (e.g. `LabelStore.latest_by_target`
→ thin wrapper over `latest_by_key`). **Preserve the specialisations** —
they are not all identical:
- `LabelStore` filters `LabelStatus.UNLABELED` tombstones out of
  `latest_by_target` / `latest_for` (`store.py:135`, `store.py:156`) and
  has `append_many` (single-fsync batch, `store.py:99`).
- malformed-line skipping currently logs in `ColumnClusterStore`
  (`column_cluster_store.py:62`) but is silent elsewhere — pick one
  behaviour (logging at `debug` is the better default).

**Risk:** medium-high by blast radius (every labelling read/write path),
low by logic (behaviour-preserving). Do it as a pure refactor: no
behaviour change except that the four un-healed stores gain
reload-on-external-change.

**Tests:** the existing per-store test files
(`tests/test_label_store.py`, `tests/test_column_cluster_store.py`, …)
must pass unchanged. Add the batch-7 out-of-band-write regression test
(see `tests/test_label_store.py::test_reloads_on_out_of_band_write`) for
the four newly-healed stores. Consider a parametrised
`tests/test_jsonl_store.py` exercising the base contract once across all
six subclasses.

---

## 3. Real sandbox for the LLM `check(df)` predicate

**Where:** `src/core/relations/validator.py`
- `_exec_check_code` — `validator.py:189`: `exec(compile(code, ...), namespace)`
  **in-process**, with `__builtins__` swapped for `_SAFE_BUILTINS`
  (`validator.py:100`) and an AST guard (`_validate_code_ast`,
  `validator.py:141`) that rejects `eval`/`exec`/`open`/`__import__`/dunder
  access (batch 3 also dropped `format`/`%`-formatting).
- `_run_check_with_timeout` — `validator.py:252`: runs the predicate on a
  **daemon `threading.Thread`** with `t.join(timeout)` (`validator.py:279`).

**The gap.** The AST guard + restricted builtins stop *casual* escapes,
but two real problems remain:
1. **The timeout cannot actually stop the code.** `t.join(timeout)`
   returns, but a `check(df)` with `while True:` or a giant allocation
   keeps running on that daemon thread, holding the GIL and burning
   CPU/RAM until the *whole process* exits. On a batch of adversarial or
   buggy LLM rules this degrades the server.
2. **In-process `exec` shares the address space.** A determined payload
   that slips past the AST guard runs with the server's full memory and
   file descriptors.

**Approach: run `check(df)` in a killable, resource-capped subprocess.**

1. A tiny runner module (`src/core/relations/_check_runner.py`) invoked as
   `python -m ...`: reads `{code, dataframe}` from stdin (pickle/parquet),
   applies the **same** `_validate_code_ast` + `_SAFE_BUILTINS`, runs
   `check(df)`, writes the boolean mask back to stdout. Keep the AST guard
   — defence in depth.
2. Parent launches it with `subprocess.run(..., timeout=timeout_seconds)`;
   on `TimeoutExpired` it **kills the process** (own process group, like
   `_launch` already does at `runs.py:905`) and records
   `rec.error = "timeout"` — same record shape `_run_check_with_timeout`
   produces today, so `validate_relations` upstream is unchanged.
3. Before `exec`, the child sets `resource.setrlimit` for `RLIMIT_CPU`,
   `RLIMIT_AS` (memory), and `RLIMIT_NOFILE`; optionally `RLIMIT_FSIZE 0`.
4. Spawn cost: serialising the (sampled) df per rule adds overhead.
   Validation already samples rows (`sample_rows`, `validator.py:625`),
   and the LLM I/O dominates wall-clock (`validator.py:1108` notes the
   exec is not the bottleneck), so per-rule subprocess spawn is acceptable.
   If it bites, reuse a worker via `multiprocessing.Pool(1)` with a hard
   `terminate()` on timeout.

**Risk:** medium. Behaviour-preserving for *valid* rules; changes failure
semantics for runaway rules (now killed, previously leaked). Windows lacks
`resource` and `setsid` — guard with `hasattr`, matching the existing
`os.setsid` guard at `runs.py:905`. This box is Linux (`platform: linux`).

**Tests:** `tests/test_relation_validator.py` already covers the AST guard
and valid/invalid predicates — keep them green against the new path. Add:
(a) an infinite-loop `check` returns a timeout record within
`~timeout_seconds` and **leaves no live child** (assert via pid); (b) a
memory-bomb `check` is killed, not OOM-the-server; (c) a valid predicate
returns the identical mask through the subprocess path.

---

## Also noted during review (smaller, not blocking)

- **Efficiency:** `table_digest` value-count memoisation was done
  (batch 4). Not yet: relation DETECT re-executes each verified `check(df)`
  a second time against the full frame — could reuse the VERIFY mask when
  the sample was the full frame. Low priority.
- **Test gaps:** `correction_suggester` pure logic now covered (batch 5,
  `tests/test_correction_suggester.py`). The LLM-dispatch layer
  (`dispatch_groups` / `call_llm`) remains untested by design (needs a
  fake LLM client) — add if that code changes.
