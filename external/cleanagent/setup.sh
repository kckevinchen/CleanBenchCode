#!/usr/bin/env bash
# Idempotent setup for the CleanAgent baseline.
#   - vendors sfu-db/CleanAgent (prompts = source of truth; gitignored clone)
#   - provisions an isolated CPython 3.10 via uv (dataprep won't build on 3.12)
#   - installs dataprep into external/cleanagent/venv
# The LLM annotation phase (annotate.py) runs in the project's MAIN venv instead
# (it needs src.core.llm_client / Azure); only detect.py needs this 3.10 venv.
# Usage:  bash external/cleanagent/setup.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLONE="$HERE/CleanAgent"
TOOLVENV="$HERE/toolvenv"
VENV="$HERE/venv"

echo "== 1/3 vendor CleanAgent (prompts source of truth) =="
if [ ! -d "$CLONE/.git" ]; then
  git clone https://github.com/sfu-db/CleanAgent.git "$CLONE"
else
  echo "   already present: $CLONE"
fi
git -C "$CLONE" fetch --depth 1 origin 66c10c2c1d141e7d63946f80a3f660e072d5bbe0
git -C "$CLONE" checkout --detach 66c10c2c1d141e7d63946f80a3f660e072d5bbe0

echo "== 2/3 provision CPython 3.10 via uv =="
if [ ! -x "$VENV/bin/python" ]; then
  [ -d "$TOOLVENV" ] || python3 -m venv "$TOOLVENV"
  # shellcheck disable=SC1091
  source "$TOOLVENV/bin/activate"
  pip install -q uv
  uv python install 3.10
  uv venv --python 3.10 "$VENV"
  deactivate
else
  echo "   already present: $VENV ($("$VENV/bin/python" --version))"
fi

echo "== 3/3 install dataprep into the 3.10 venv =="
[ -d "$TOOLVENV" ] && source "$TOOLVENV/bin/activate"
uv pip install --python "$VENV/bin/python" -r "$HERE/requirements.txt"
"$VENV/bin/python" - <<'PY'
import dataprep, pandas, sys
from dataprep.clean import clean_date, clean_country, clean_phone, clean_email, clean_ip  # noqa
print(f"   ok: python={sys.version.split()[0]} dataprep={dataprep.__version__} pandas={pandas.__version__}")
PY

cat <<'EOF'

Done. Run the two phases (datasets materialized by external/rein/prepare_datasets.py):
  # phase 1 — LLM column annotation (MAIN venv, needs az login):
  source venv/bin/activate
  python external/cleanagent/annotate.py --model gpt-4o
  # phase 2 — dataprep standardize + diff (this 3.10 venv):
  external/cleanagent/venv/bin/python external/cleanagent/detect.py --mode changed
  # score:
  python external/rein/score.py --predictions external/rein/results/predictions_cleanagent_changed.jsonl
EOF
