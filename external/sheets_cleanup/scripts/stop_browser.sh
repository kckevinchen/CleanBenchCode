#!/usr/bin/env bash
# Stop the Chromium / Xvfb / x11vnc started by launch_browser.sh
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGDIR="$(cd "$HERE/.." && pwd)/outputs/logs"
for name in chrome xvfb; do
  if [[ -f "$LOGDIR/$name.pid" ]]; then
    pid="$(cat "$LOGDIR/$name.pid")"
    if kill -0 "$pid" 2>/dev/null; then echo "stopping $name ($pid)"; kill "$pid" 2>/dev/null || true; fi
    rm -f "$LOGDIR/$name.pid"
  fi
done
# x11vnc runs with -bg; stop by its rfbport match
VNC_PORT="${VNC_PORT:-5900}"
vpid="$(pgrep -f "x11vnc.*-rfbport $VNC_PORT" || true)"
[[ -n "$vpid" ]] && { echo "stopping x11vnc ($vpid)"; kill $vpid 2>/dev/null || true; }
echo "done"
