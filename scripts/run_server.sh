#!/usr/bin/env bash
# ------------------------------------------------------------------
# Run experiments.workbench detached from the current shell so it
# survives SSH disconnects. Writes its PID to logs/workbench.pid
# and stdout/stderr to logs/workbench.log.
#
# Usage:
#   ./scripts/run_server.sh [--port 8000] [--host 127.0.0.1]
#   ./scripts/run_server.sh stop        # kill running instance
#   ./scripts/run_server.sh restart     # stop (if running) then start
#   ./scripts/run_server.sh status      # show pid + log tail
# ------------------------------------------------------------------
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$HERE/.." && pwd)"
cd "$ROOT"

PID_FILE="$ROOT/logs/workbench.pid"
LOG_FILE="$ROOT/logs/workbench.log"
mkdir -p "$ROOT/logs"

is_running() {
  [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null
}

case "${1:-start}" in
  stop)
    if is_running; then
      PID="$(cat "$PID_FILE")"
      echo "stopping pid $PID"
      kill -TERM "$PID" || true
      rm -f "$PID_FILE"
    else
      echo "no running server."
    fi
    exit 0
    ;;
  restart)
    shift
    if is_running; then
      PID="$(cat "$PID_FILE")"
      echo "stopping pid $PID"
      kill -TERM "$PID" || true
      # Wait up to 10s for graceful shutdown.
      attempts=0
      while [[ "$attempts" -lt 20 ]]; do
        kill -0 "$PID" 2>/dev/null || break
        sleep 0.5
        attempts=$((attempts + 1))
      done
      if kill -0 "$PID" 2>/dev/null; then
        echo "force-killing pid $PID"
        kill -KILL "$PID" || true
      fi
      rm -f "$PID_FILE"
    fi
    # Fall through to start below.
    ;;
  status)
    if is_running; then
      echo "running (pid $(cat "$PID_FILE"))"
      echo "--- last 10 log lines ---"
      tail -n 10 "$LOG_FILE" 2>/dev/null || true
    else
      echo "not running."
    fi
    exit 0
    ;;
  start|"")
    shift $(( $# > 0 ? 1 : 0 ))
    ;;
  *)
    # Anything else -> pass through to workbench as flags.
    ;;
esac

if is_running; then
  echo "server already running (pid $(cat "$PID_FILE")). Use 'stop' first."
  exit 1
fi

if [[ -f "$ROOT/venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "$ROOT/venv/bin/activate"
fi

# Load secrets / Foundry Agents config from .env so the spawned
# server inherits AZURE_AGENTS_ENDPOINT, AZURE_AGENTS_VERIFIER_AGENT_ID,
# etc. Without this the web-grounded verifier returns
# ``error: missing agent_id`` from the UI even when .env is populated.
if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "$ROOT/.env"
  set +a
fi

# nohup survives SSH hangup. Linux also gets a dedicated session via setsid;
# standard macOS does not ship setsid, so fall back to launching Python
# directly while retaining pidfile-based stop/restart behavior.
if command -v setsid >/dev/null 2>&1; then
  nohup setsid python -m experiments.workbench "$@" \
    >>"$LOG_FILE" 2>&1 < /dev/null &
else
  nohup python -m experiments.workbench "$@" \
    >>"$LOG_FILE" 2>&1 < /dev/null &
fi
PID=$!
echo "$PID" > "$PID_FILE"
sleep 0.4
if is_running; then
  echo "started pid $PID"
  echo "log: $LOG_FILE"
  tail -n 4 "$LOG_FILE" 2>/dev/null || true
else
  echo "failed to start - see $LOG_FILE"
  exit 1
fi
