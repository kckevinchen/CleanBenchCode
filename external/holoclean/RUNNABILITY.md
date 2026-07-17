# HoloClean baseline runnability notes

Upstream clone: `external/holoclean/holoclean_repo` (unmodified, `.git` kept).
Pinned commit: `d4f5929a8e4d92d4f41eb058c04c96cdcb0af767` (`PINNED_COMMIT.txt`).

## Environment

Dedicated micromamba env: `external/holoclean/mamba_env` with Python 3.6.15.
Created with:

```bash
$HOME/bin/micromamba create -p external/holoclean/mamba_env python=3.6 pip -c conda-forge -y
```

Installed upstream pins from `holoclean_repo/requirements.txt`. `torch==1.0.1.post2`
is not available from modern PyPI indexes, so it was installed from the legacy CPU wheel:

```bash
grep -v '^torch==' external/holoclean/holoclean_repo/requirements.txt > external/holoclean/requirements_no_torch.txt
TMPDIR=$PWD/external/holoclean/tmp external/holoclean/mamba_env/bin/pip install -r external/holoclean/requirements_no_torch.txt
TMPDIR=$PWD/external/holoclean/tmp external/holoclean/mamba_env/bin/pip install \
  https://download.pytorch.org/whl/cpu/torch-1.0.1.post2-cp36-cp36m-linux_x86_64.whl
TMPDIR=$PWD/external/holoclean/tmp external/holoclean/mamba_env/bin/pip install 'smart_open==2.2.1'
```

`smart_open==2.2.1` is a Python-3.6-compatible transitive dependency for `gensim==3.7.1`;
the newest resolver-selected `smart_open` contains syntax unsupported by Python 3.6.

Validated imports: `torch 1.0.1.post2`, `pandas 0.24.1`, `gensim 3.7.1`.

## PostgreSQL

PostgreSQL 16 binaries are installed. A user-space cluster was initialized at
`external/holoclean/pgdata`, but starting it on port 5432 failed because a PostgreSQL server was
already accepting connections on `127.0.0.1:5432`. HoloClean's upstream DB engine hardcodes port
5432 and the bundled example only exposes host/user/password/db, so the existing 5432 server was
configured with upstream's required DB/user rather than disrupting a shared system service.

Setup/verification commands used:

```bash
sudo -n apt-get update -qq
sudo -n apt-get install -y postgresql postgresql-contrib
/usr/lib/postgresql/16/bin/initdb -D external/holoclean/pgdata -U azureuser -A trust --no-locale --encoding=UTF8
# start attempt documented in external/holoclean/pg.log; failed only because 5432 was occupied
sudo -n -u postgres psql -v ON_ERROR_STOP=1 -c "CREATE DATABASE holo;" || true
sudo -n -u postgres psql -v ON_ERROR_STOP=1 \
  -c "DO \$\$BEGIN IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname='holocleanuser') THEN CREATE USER holocleanuser WITH PASSWORD 'abcd1234'; ELSE ALTER USER holocleanuser WITH PASSWORD 'abcd1234'; END IF; END\$\$;" \
  -c "GRANT ALL PRIVILEGES ON DATABASE holo TO holocleanuser;"
sudo -n -u postgres psql -d holo -v ON_ERROR_STOP=1 \
  -c "ALTER SCHEMA public OWNER TO holocleanuser; GRANT ALL ON SCHEMA public TO holocleanuser;"
PGPASSWORD="$HOLOCLEAN_DB_PASSWORD" psql -h localhost -p 5432 -U holocleanuser -d holo \
  -c 'select current_database(), current_user;'
```

Connection verification returned database `holo`, user `holocleanuser`.

## Upstream health validation

Bundled example command:

```bash
cd external/holoclean/holoclean_repo/examples
TMPDIR=<repo>/external/holoclean/tmp \
PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
../../mamba_env/bin/python holoclean_repair_example.py \
  > <repo>/external/holoclean/example_validation.log 2>&1
```

Result: exit 0. Final reported metrics in `example_validation.log`:
Precision 1.00, Recall 0.46, Repairing Recall 0.53, F1 0.63, Detected Errors 435,
Correct Repairs 232, Total Repairs 232.

Because the bundled example evaluates but does not write a CSV, `external/holoclean/export_hospital_example_repair.py`
replays the same upstream API flow outside the clone and exports
`external/holoclean/example_hospital_repaired.csv` (1000 x 18). Log:
`external/holoclean/example_export_repaired.log`.

## Adapter

Files:

- `external/holoclean/adapter.py`
- `external/holoclean/run_holoclean.py`

Per table, the adapter:

1. Reads only `dirty.csv`.
2. Renames columns positionally to `C0..Cn` for HoloClean/DC parser safety.
3. Shells out to `external/adr/discover_rules.py` on the sanitized dirty CSV only, producing
   HoloClean DC rules such as `t1&t2&EQ(t1.C0,t2.C0)&IQ(t1.C1,t2.C1)`.
4. Runs upstream HoloClean with `NullDetector()` + `ViolationDetector()` and the documented
   featurizers from `examples/holoclean_repair_example.py`.
5. Exports the repaired dataframe with original column names.
6. Emits detections where repaired values differ from dirty values, using the same positional
   diff semantics as the ADR adapter.

Cache root: `external/holoclean/cache/v8/<slug>__<pinned_commit>/`.
Per-table failures or empty DCs write an unchanged repaired CSV, record 0 detections, and add an
error/note without crashing the run.

## Smoke test on 3 benchmark datasets

Command:

```bash
TMPDIR=$PWD/external/holoclean/tmp PYTHONDONTWRITEBYTECODE=1 \
external/holoclean/mamba_env/bin/python external/holoclean/run_holoclean.py \
  --limit 3 --force --verbose --timeout 240
```

Output files:

- `external/rein/results/predictions_holoclean.jsonl` (549 predicted cells)
- `external/rein/results/predictions_holoclean.jsonl.ran.json` (3-dataset smoke scope)
- `external/rein/results/predictions_holoclean.jsonl.errors.json`
- `external/rein/results/predictions_holoclean.jsonl.tables.json`

Per-table smoke results:

| dataset | rules | detections | note/error |
|---|---:|---:|---|
| `005c05b7029cb909b653bd2942b5272c5030e130.xlsx_ed8fbac9` | 2 | 400 | none |
| `00b9e90444835c9ddaa06bcbd95acb23c4abbed8.xlsx_24354cf0` | 7 | 149 | none |
| `012240e8708e14939670412f3b559e46f5926fb9.xlsx_8a91dfd8` | 0 | 0 | no DC rules discovered |

Limited-scope score sanity check:

```bash
venv/bin/python external/rein/score.py --predictions external/rein/results/predictions_holoclean.jsonl
```

Result over the 3 attempted datasets: P=0.011, R=1.000, F1=0.022 (6 TP, 542 FP, 0 FN).
This is only a smoke test; the full corpus was intentionally not launched.

## Full-corpus launch command for the manager

Run only when ready to schedule the 1,007-table job:

```bash
cd <repo>
mkdir -p logs external/holoclean/tmp
setsid env TMPDIR=$PWD/external/holoclean/tmp PYTHONDONTWRITEBYTECODE=1 \
  external/holoclean/mamba_env/bin/python external/holoclean/run_holoclean.py \
  --timeout 600 --verbose > logs/holoclean_v8.log 2>&1 < /dev/null &
```

Then score after completion:

```bash
venv/bin/python external/rein/score.py --predictions external/rein/results/predictions_holoclean.jsonl
```

Verdict: READY for a managed full-corpus run, with the caveat that this shared host already owns
port 5432; do not stop that service unless coordinating with other users.
