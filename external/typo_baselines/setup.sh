#!/usr/bin/env bash
# Idempotent setup for the typo-specialist baselines (SymSpell + NeuSpell).
#   - dedicated venv (separate from external/rein because neuspell needs torch+transformers)
#   - installs torch (CPU) first so neuspell does not pull a GPU build
#   - fetches NeuSpell's 2.2GB subwordbert checkpoint via gdown (NeuSpell's own
#     Google-Drive downloader returns an HTML "virus scan" page for large files)
# Usage:  bash external/typo_baselines/setup.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$HERE/venv"

echo "== 1/4 venv =="
[ -d "$VENV" ] || python3 -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install -q --upgrade pip

echo "== 2/4 torch (CPU) first =="
python -c "import torch" 2>/dev/null || \
  pip install -q torch --index-url https://download.pytorch.org/whl/cpu

echo "== 3/4 deps (symspellpy, neuspell, transformers==4.38.2, gdown, pandas) =="
pip install -q -r "$HERE/requirements.txt"

echo "== 4/4 NeuSpell subwordbert checkpoint (~2.2GB via gdown) =="
CKPT_DIR="$(python -c "import neuspell,os;print(os.path.join(os.path.dirname(neuspell.__file__),os.pardir,'data','checkpoints','subwordbert-probwordnoise'))")"
mkdir -p "$CKPT_DIR"
# model.pth.tar (large -> gdown handles the Drive confirm token)
if [ ! -f "$CKPT_DIR/model.pth.tar" ] || [ "$(stat -c%s "$CKPT_DIR/model.pth.tar")" -lt 1000000000 ]; then
  gdown 13FnCUPAG-P0-rFIRNewHYXZzwp4HBIjr -O "$CKPT_DIR/model.pth.tar"
fi
# vocab.pkl (small)
if [ ! -f "$CKPT_DIR/vocab.pkl" ] || [ "$(stat -c%s "$CKPT_DIR/vocab.pkl")" -lt 100000 ]; then
  gdown 11Bo86aI0MxAU1MHpF-eYfAHg3HqiT9me -O "$CKPT_DIR/vocab.pkl"
fi

python - <<'PY'
import os, neuspell
ck = os.path.join(os.path.dirname(neuspell.__file__), os.pardir, "data",
                  "checkpoints", "subwordbert-probwordnoise")
m = os.path.join(ck, "model.pth.tar")
ok = os.path.exists(m) and os.path.getsize(m) > 1e9
print(f"   model.pth.tar: {os.path.getsize(m)/1e9:.2f} GB {'OK' if ok else 'TOO SMALL/MISSING'}")
PY

cat <<'EOF'

Done. Next (datasets are materialized by external/rein/prepare_datasets.py):
  source external/typo_baselines/venv/bin/activate
  python external/typo_baselines/run_typo.py --tool symspell --tool neuspell
  python external/rein/score.py --predictions external/rein/results/predictions_union.jsonl
EOF
