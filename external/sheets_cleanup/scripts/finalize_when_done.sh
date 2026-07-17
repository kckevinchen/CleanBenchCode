#!/usr/bin/env bash
# Wait for the full apply_and_diff run to finish, then diff + score automatically.
# Fires when no apply_and_diff.py process has been seen for two consecutive checks
# (so a brief manual resume gap doesn't trigger it early). Writes the final score
# to outputs/final_score_v8.txt.
set -u
cd "$(dirname "$0")/.."                       # -> external/sheets_cleanup
ROOT="$(cd ../.. && pwd)"
VENV="$ROOT/venv/bin/activate"
REIN_PY="$ROOT/external/rein/venv/bin/python"
LOG="outputs/finalize_v8.log"

echo "watcher started $(date)" > "$LOG"
gone=0
while true; do
  sleep 300
  if pgrep -f "apply_and_diff.py --manifest" >/dev/null 2>&1; then
    gone=0
  else
    gone=$((gone+1))
    echo "$(date) apply process absent (streak=$gone)" >> "$LOG"
    [ "$gone" -ge 2 ] && break
  fi
done

echo "$(date) finalizing: diff + score" >> "$LOG"
# shellcheck disable=SC1090
source "$VENV" 2>/dev/null
python3 scripts/diff_cleanup.py >> "$LOG" 2>&1
"$REIN_PY" "$ROOT/external/rein/score.py" \
  --predictions "$ROOT/external/rein/results/predictions_gsheets.jsonl" \
  > outputs/final_score_v8.txt 2>> "$LOG"
echo "$(date) DONE -> outputs/final_score_v8.txt" >> "$LOG"
cat outputs/final_score_v8.txt >> "$LOG"
