# ADR'24 Automatic-Data-Repair runnability notes

Upstream clone: `external/adr/adr_repo` (unmodified, pinned in `PINNED_COMMIT.txt`).
Pinned commit: `b3004a8980104ae5412b8dc39e609b2c1c8e439d`.

Isolation: user-space micromamba is installed at `$HOME/bin/micromamba`; the ADR env is
`external/adr/mamba_env` with Python 3.8, pandas/numpy/tqdm, and OpenJDK from conda-forge.
No conda/system Java/sudo dependencies are required.

## Validated end-to-end method

**Horizon** was validated on the bundled hospital data with shipped FDs.  Command:

```bash
cd <repo>/external/adr/hospital_validation
<repo>/external/adr/mamba_env/bin/python \
  <repo>/external/adr/adr_repo/horizon/horizon.py \
  --task_name hospital1 \
  --rule_path <repo>/external/adr/adr_repo/data_with_rules/hospital/dc_rules-validate-fd-horizon.txt \
  --onlyed 0 --perfected 0 \
  --dirty_path <repo>/external/adr/adr_repo/data_with_rules/hospital/noise/hospital-inner_outer_error-01.csv \
  --clean_path <repo>/external/adr/adr_repo/data_with_rules/hospital/clean.csv
```

Output: `external/adr/hospital_validation/Repaired_res/horizon/hospital/repaired_hospital1-inner_outer_error-01.csv` (1000 x 19) plus metric files under `Exp_result/horizon/hospital/`.

## Rule discovery enabler

`external/adr/discover_rules.py` mines approximate functional dependencies from **dirty.csv only**.
For each candidate LHS -> RHS it groups rows by LHS, keeps the majority RHS per group, and accepts
rules above a confidence threshold while filtering near-unique keys and near-constant columns.  It emits:

- Horizon FD file: `dc_rules-validate-fd-horizon.txt` lines like `HospitalName ⇒ ZipCode`.
- HoloClean/DC file: `dc_rules_holoclean.txt` lines like `t1&t2&EQ(t1.HospitalName,t2.HospitalName)&IQ(t1.ZipCode,t2.ZipCode)`.

Hospital qualitative check (`dirty.csv`, default confidence 0.95) rediscovers plausible shipped-style FDs,
including `HospitalName ⇒ ZipCode`, `HospitalName ⇒ Address1`, `HospitalName ⇒ City`,
`HospitalName ⇒ PhoneNumber`, `HospitalName ⇒ ProviderNumber`, `MeasureName ⇒ MeasureCode`,
`MeasureName ⇒ Condition`, and `MeasureCode ⇒ Condition`.  It also emits noisy approximate FDs expected
from mining dirty data without ground truth.

## Per-method runnability roster

| ADR method | Implementation | Dependencies | Rules / detections needed | Upstream invocation and I/O | Verdict |
|---|---|---|---|---|---|
| Unified | Python (`Unified/Unified.py`) | Python 3.8, pandas/numpy/tqdm; no Java at runtime | **FD rules** (`dc_rules-validate-fd-horizon.txt`) | `python Unified/Unified.py --task_name <name> --rule_path <fd> --onlyed 0 --perfected 0 --dirty_path <dirty> --clean_path <clean>`; writes `Repaired_res/Unified/<task>/repaired_<task><error>.csv` and metrics under `Exp_result/Unified/`. `clean_path` is used for metrics/perfected mode; non-perfected repair is dirty+rules. | Integratable after adapter work; not yet smoke-validated. |
| Holistic | Python (`BigDansing_Holistic/holistic.py`) | Python 3.8, pandas/numpy/tqdm | **DC rules** (`dc_rules_holoclean.txt`) | `python BigDansing_Holistic/holistic.py --task_name <name> --rule_path <dc> --onlyed 0 --perfected 0 --dirty_path <dirty> --clean_path <clean>`; writes `Repaired_res/holistic/<task>/repaired_<task><error>.csv`. | Likely runnable; not selected for first validation. |
| NADEEF | Java/Ant (`NADEEF-master`) | JDK + Ant; console/dashboard app, custom rule classes | **DC/rule programs**; not a simple ADR CSV runner | Build with `ant`, then `./nadeef.sh console`; expects NADEEF project/rule setup, not the Python runner shape. | Not harnessed yet; needs Java/Ant build and a batch rule-project adapter. |
| SCARE | Python (`SCAREd/scared.py`) | Python 3.8 plus bundled `raha-master` import/deps | **Upstream detections**, not DCs; script runs Raha when `perfected=0` | `python SCAREd/scared.py --task_name <name> --rule_path <fd> --onlyed 0 --perfected 0 --dirty_path <dirty> --clean_path <clean>`; internally calls `raha.Detection()` and writes `Repaired_res/scared/...`. | Correction/end-to-end only for our detection roster unless using its Raha detections; needs separate Raha dependency handling. |
| Relative Trust | Python (`Relative_Trust/relative.py`) | Python 3.8, pandas/numpy/tqdm | **FD rules** (`dc_rules-validate-fd-horizon.txt`) | `python Relative_Trust/relative.py --task_name <name> --rule_path <fd> --onlyed 0 --perfected 0 --dirty_path <dirty> --clean_path <clean>`; writes `Repaired_res/Relative_trust/<task>/repaired_<task><error>.csv`. | Integratable after Horizon because input format is similar. |
| BigDansing | Python (`BigDansing_Holistic/bigdansing.py`) | Python 3.8, pandas/numpy/tqdm | **DC rules** (`dc_rules_holoclean.txt`) | `python BigDansing_Holistic/bigdansing.py --task_name <name> --rule_path <dc> --onlyed 0 --perfected 0 --dirty_path <dirty> --clean_path <clean>`; writes `Repaired_res/bigdansing/<task>/repaired_<task><error>.csv`. | Likely runnable; needs DC parser validation on discovered column names. |
| BoostClean | Python (`BoostClean/activedetect/experiments/Experiment.py`) | Python with scikit-learn/tqdm; older code | **Upstream detections / learned error modules**, not DCs; `rule_path` arg is unused | `python BoostClean/activedetect/experiments/Experiment.py --task_name <name> --rule_path <ignored> --onlyed 0 --perfected 0 --dirty_path <dirty> --clean_path <clean>`; treats last column as label, learns/repairs, writes `Repaired_res/boostclean/...`. | Needs separate harness; not a constraint-rule method and may be semantically mismatched for arbitrary tables. |
| Daisy | Python (`Daisy/daisy.py` + `Daisy/daisy/`) | Python 3.8, pandas/numpy | **DC rules** (`dc_rules_holoclean.txt`) | `python Daisy/daisy.py --task_name <name> --rule_path <dc> --onlyed 0 --perfected 0 --dirty_path <dirty> --clean_path <clean>`; writes `Repaired_res/daisy/<task>/repaired_<task><error>.csv`. | Likely runnable; not yet smoke-validated. |
| Horizon | Python (`horizon/horizon.py`) | Python 3.8, pandas/numpy | **FD rules** (`dc_rules-validate-fd-horizon.txt`) | `python horizon/horizon.py --task_name <name> --rule_path <fd> --onlyed 0 --perfected 0 --dirty_path <dirty> --clean_path <metric-only>`; writes `Repaired_res/horizon/<task>/repaired_<task><error>.csv`. | **Validated** and wired as `adr_horizon`. |
| MLNClean | Java (`mlnclean/code`) + Python dataset converters | OpenJDK, likely Maven/Gradle/classpath setup; shipped script references an ephemeral `/tmp/cp_*.jar` | **DC/first-order rules** converted from HoloClean DC (`mlnclean/dataset/aggre_process4data.py`) | Script calls `java -cp /tmp/cp_... main.Test <dataset_dir> trainData.csv testData.csv 1 0`; expected input is a dataset directory with `trainData.csv`, `testData.csv`, converted rules, and outputs later transferred by `res_transfer.py`. | Hard blocker: released run script points to a non-reproducible temp classpath jar; needs build reconstruction before full harnessing. |

## Benchmark smoke

Command:

```bash
external/adr/mamba_env/bin/python external/adr/run_adr.py --method horizon --limit 3 --force --verbose --timeout 120
```

Outputs:

- `external/rein/results/predictions_adr_horizon.jsonl` — 95 predicted cells.
- `external/rein/results/predictions_adr_horizon.jsonl.ran.json` — scope 3 (`adr_horizon`: 3 datasets).
- `external/rein/results/predictions_adr_horizon.jsonl.errors.json` — one table had no unary FDs and therefore emitted zero detections.

Scoring smoke:

```bash
venv/bin/python external/rein/score.py --predictions external/rein/results/predictions_adr_horizon.jsonl
```

Result over the 3-dataset limited scope: P=0.000, R=0.000, F1=0.000 (0 TP, 95 FP, 6 FN).  This is only a smoke test; the full 1,007-table run was intentionally not launched.

## Remaining work / blockers

1. Schedule (do not run concurrently with LLM jobs) a full `adr_horizon` corpus run with no `--limit`.
2. Tune or report the unsupervised FD miner threshold; current default prioritizes recall of plausible FDs and can over-repair.
3. Add method-specific adapters for Relative Trust/Unified (FD-compatible) and Holistic/BigDansing/Daisy (DC-compatible).
4. Reconstruct NADEEF and MLNClean Java builds; MLNClean's published script references an unavailable `/tmp/cp_*.jar`.
5. Treat SCARE and BoostClean separately because they consume/learn detections rather than authored DC/FD constraints.

## Validation results — 5 additional Python methods (2026-07-16)

Generalized `adapter.py::run_method` to drive all six Python FD/DC methods through
one code path (per-method: upstream script, `Repaired_res/<dir>` subdir, FD-vs-DC
rule file; repaired file located by glob to tolerate both upstream filename
conventions). Each method was smoke-validated on the bundled `hospital` table
(shipped rules) and through the adapter on real benchmark tables. Missing upstream
deps were installed into `mamba_env` faithfully (versions per the repo's own
`environment.yml`): `line_profiler==4.1.1`, `jaro-winkler==2.0.3`, and
`scikit-learn` (imported by Relative Trust; unpinned upstream).

| method | rule | hospital smoke | adapter on real tables | verdict |
|---|---|---|---|---|
| horizon | FD | OK (618 cells changed) | OK | wired + scored full-corpus (F1≈0.002) |
| relative_trust | FD | runs (0 changed; conservative) | OK rc=0 | RUNNABLE → full-corpus launched |
| unified | FD | runs (0 changed on hospital) | OK rc=0; **58 cells on a real table** | RUNNABLE → full-corpus launched |
| bigdansing | DC | OK (899 cells changed) | **crashes** rc=1 on ~all ≤100-row tables | upstream `detect()` bug → full-corpus launched for the record |
| holistic | DC | very slow (>900s/1000 rows) | **crashes** rc=1 (same bug) | upstream bug + slow → full-corpus launched (180s/table cap) |
| daisy | DC | **import fails** | — | **BLOCKED (upstream incomplete)** |

Key findings:
- **Unified / Relative Trust (FD)** run cleanly through the adapter and produce
  real detections on some tables (Unified flagged 58 cells on one). Full-corpus
  runs launched (CPU-only, detached, auto-scored).
- **BigDansing / Holistic (DC)** only run on the atypical 1000-row `hospital`
  table. On our typical ≤100-row tables the upstream `detect()` raises
  `TypeError: 'int' object is not subscriptable` (`BigDansing_Holistic/bigdansing.py:341`,
  `l = k[0]`) — a data-structure assumption that does not hold for our tables. Per
  the vendor-unmodified contract this is **not patched**; crashing tables record 0
  detections (honest floor), and the full-corpus run documents the crash rate.
- **Daisy** is unrunnable from the public repo: `Daisy/daisy/__init__.py`
  unconditionally does `from daisy_beer.basic import *` (also `daisy_hospital`,
  `daisy_flight`) via hardcoded Windows paths, and those packages are **absent
  from the repository**. Reconstructing unpublished author code would violate
  faithfulness → **BLOCKED**.
- NADEEF + MLNClean (Java) remain hard-build blockers; SCARE + BoostClean consume/
  learn detections rather than authored constraints (separate handling).
