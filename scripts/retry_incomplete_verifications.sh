#!/usr/bin/env bash
# Retry verification records that were spuriously errored with
# "unexpected agent run status: incomplete" (the bug we just fixed in
# src/verifier/agent_backend.py).
#
# Usage: scripts/retry_incomplete_verifications.sh <run-id>
#
# Strategy: drop only the affected records from web_verifications.json
# and re-run the verifier (not --force). The verifier auto-skips
# clusters whose IDs already appear in priors, so only the dropped
# IDs will be re-attempted. Cached web search payloads (web_cache/)
# are kept, so the retry is cheap.

set -euo pipefail
RUN_ID="${1:?usage: $0 <run-id>}"
RUN_DIR="outputs/${RUN_ID}"
PRIOR="${RUN_DIR}/web_verifications.json"

if [[ ! -f "${PRIOR}" ]]; then
  echo "no web_verifications.json under ${RUN_DIR}" >&2
  exit 2
fi

TS=$(date -u +%Y%m%dT%H%M%SZ)
BACKUP="${RUN_DIR}/web_verifications.before_incomplete_retry.${TS}.json"
cp "${PRIOR}" "${BACKUP}"
echo "backed up prior verifications -> ${BACKUP}"

python3 - "${PRIOR}" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
data = json.loads(p.read_text())
keep, drop = [], []
for v in data:
    err = (v.get("error") or "").lower()
    if v.get("verdict") == "errored" and "unexpected agent run status: incomplete" in err:
        drop.append(v["cluster_id"])
    else:
        keep.append(v)
p.write_text(json.dumps(keep, indent=2))
print(f"kept {len(keep)} verifications; dropped {len(drop)} affected by the incomplete-status bug")
if drop[:5]:
    print("  examples:", drop[:5])
PY

LOG="logs/jobs/web_verify_retry_incomplete_${RUN_ID}_${TS}.log"
echo "log: ${LOG}"
exec setsid nohup python -m experiments.run_web_verify \
  --run-id "${RUN_ID}" --error-types typo --concurrency 4 -v \
  > "${LOG}" 2>&1 < /dev/null &
disown
echo "retry pid=$!"
