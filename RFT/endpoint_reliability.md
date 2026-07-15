# RFT endpoint reliability log

Manual log of Azure RFT fine-tuning job outcomes per endpoint, so we can
track which endpoints fail (usually transient 500s during pre/post-processing)
and estimate a per-endpoint failure rate.

**Convention:** one row per submitted job. `Result` = `success` (reached
`succeeded`, model produced), `failed` (terminal `failed`), or `running`
(still in flight — update when it settles). Note `billed=0h` for failures that
cost nothing.

## Failure-rate summary

Update the counts when a `running` job settles.

| Endpoint | Submitted | Succeeded | Failed | Failure rate |
|---|---|---|---|---|
| dmxaz-test-aoai | 2 | 0 | 2 | 2/2 (100%) — both 500 in postprocessing |
| dmxaz-test-aoai-eu2 | 3 | 0 | 0 | 0/3 (3 running) |

## Observations

- `dmxaz-test-aoai` failed **both** v10 jobs at the postprocessing/model-packaging
  stage with `500 A system error was encountered` after 3 internal retries.
  Training itself completed (billed 0h each) — looks like an endpoint-side
  packaging problem, not a data/grader config issue (same config runs on eu2).
  Avoid this endpoint for RFT until it recovers.
- `dmxaz-test-aoai-eu2` **stalled** the v11 job: it failed once, retried, reached
  step 32, then "paused by system" (2026-06-04 06:49 UTC) and never resumed —
  stuck in `running` with zero progress for 3+ days, no model produced. Cancelled
  2026-06-08 and resubmitted on yeyehe. A silent stall (not a terminal `failed`)
  is NOT caught by `supervise` auto-rotation — only a runtime deadline stops it.

## Job log

| Date (UTC) | Variant | Job ID | Endpoint | Epochs | Result | Notes |
|---|---|---|---|---|---|---|
| 2026-06-01 | v10-r1 | ftjob-e4b232db2962482bb1ba5d3db199915d | dmxaz-test-aoai | 3 | failed | 500 during postprocessing after 3 retries; billed 0h |
| 2026-06-01 | v10-nogold | ftjob-cc2dd7062ab349e7a61960d879659a2e | dmxaz-test-aoai | 3 | failed | 500 during postprocessing after 3 retries; billed 0h |
| 2026-06-01 | v10-r1 (retry) | ftjob-8ce0dec485904d42add17f7df57bdbc7 | dmxaz-test-aoai-eu2 | 3 | running | resubmit after the dmxaz-test-aoai 500 |
| 2026-06-01 | v10-nogold (retry) | ftjob-406dc4ada69949f7aa32fc4a9cddf311 | dmxaz-test-aoai-eu2 | 3 | running | resubmit after the dmxaz-test-aoai 500; precision+trivial-only grader |
| 2026-06-01 | v10-nogold-recall | ftjob-f3e3687d91a34e3fbee274c0220738b4 | dmxaz-test-aoai-eu2 | 3 | running | new gold-free F1(precision,recall) judge — empty-output degeneracy fixed |
| 2026-06-04 | v11 (gold-free 2-judge) | ftjob-886cd2321f5c4c5280f8c2f8da1401eb | dmxaz-test-aoai-eu2 | 2 | failed | stalled in `running` after step 32 (system pause never resumed); cancelled 2026-06-08 after 3+ days no progress |
| 2026-06-08 | v11-r1 (retry) | ftjob-298af499ba3f469192d9ae6700fdb1b3 | yeyehe-aoai | 2 | running | resubmit after eu2 stall; supervised (yeyehe→eu2 rotation), full-benchmark held-out eval |
