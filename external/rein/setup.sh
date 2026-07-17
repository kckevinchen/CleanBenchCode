#!/usr/bin/env bash
# Idempotent setup for the REIN-on-benchmark baseline.
#   - clones the REIN repo (third-party, gitignored) into rein-benchmark/
#   - creates a dedicated venv (external/rein/venv) with the pinned detector deps
# Usage:  bash external/rein/setup.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$HERE/rein-benchmark"
VENV="$HERE/venv"

echo "== 1/3 clone REIN (if missing) =="
if [ ! -d "$REPO/.git" ]; then
  git clone --recurse-submodules \
    https://github.com/mohamedyd/rein-benchmark.git "$REPO"
else
  echo "   already present: $REPO"
fi
git -C "$REPO" fetch --depth 1 origin b8a0727fdeb6d295eb88c702a344093a1d9096d1
git -C "$REPO" checkout --detach b8a0727fdeb6d295eb88c702a344093a1d9096d1
git -C "$REPO" submodule update --init --recursive

echo "== 2/3 create venv (if missing) =="
if [ ! -d "$VENV" ]; then
  python3 -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --quiet --upgrade pip

echo "== 3/3 install pinned deps =="
pip install --quiet -r "$HERE/requirements.txt"
python - <<'PY'
import pandas, numpy, sklearn, raha  # noqa: F401
print(f"   ok: pandas={pandas.__version__} numpy={numpy.__version__} sklearn={sklearn.__version__}")
PY

cat <<'EOF'

Done. Next (from the repo root, with the venv active):
  source external/rein/venv/bin/activate
  python external/rein/prepare_datasets.py
  python external/rein/run_rein.py --detector raha
  python external/rein/score.py --predictions external/rein/results/predictions_raha.jsonl
EOF
