#!/usr/bin/env bash
# Launch a clean Chromium on a virtual display with remote-debugging + VNC.
# Playwright connects over CDP (http://127.0.0.1:9222). You log in to Google
# once through the VNC tunnel; the profile (auth) persists for later runs.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
PROFILE="$ROOT/profile"
LOGDIR="$ROOT/outputs/logs"
mkdir -p "$PROFILE" "$LOGDIR"

DISPLAY_NUM="${DISPLAY_NUM:-99}"
CDP_PORT="${CDP_PORT:-9222}"
VNC_PORT="${VNC_PORT:-5900}"
GEOM="${GEOM:-1600x1000x24}"
CHROME="${CHROME:-$HOME/.cache/ms-playwright/chromium-1223/chrome-linux64/chrome}"

echo "[launch] display=:$DISPLAY_NUM cdp=$CDP_PORT vnc=$VNC_PORT"

# 1) Xvfb
if ! xdpyinfo -display ":$DISPLAY_NUM" >/dev/null 2>&1; then
  echo "[launch] starting Xvfb :$DISPLAY_NUM ($GEOM)"
  Xvfb ":$DISPLAY_NUM" -screen 0 "$GEOM" -ac >"$LOGDIR/xvfb.log" 2>&1 &
  echo $! > "$LOGDIR/xvfb.pid"
  sleep 2
else
  echo "[launch] Xvfb :$DISPLAY_NUM already running"
fi

export DISPLAY=":$DISPLAY_NUM"

# 2) Chromium (manual launch => no navigator.webdriver automation flag)
if curl -s "http://127.0.0.1:$CDP_PORT/json/version" >/dev/null 2>&1; then
  echo "[launch] chromium already serving CDP on $CDP_PORT"
else
  echo "[launch] starting chromium"
  "$CHROME" \
    --remote-debugging-port="$CDP_PORT" \
    --remote-debugging-address=127.0.0.1 \
    --user-data-dir="$PROFILE" \
    --no-first-run --no-default-browser-check \
    --no-sandbox --disable-dev-shm-usage \
    --disable-blink-features=AutomationControlled \
    --window-position=0,0 --window-size=1600,1000 \
    --lang=en-US \
    "https://accounts.google.com/" \
    >"$LOGDIR/chrome.log" 2>&1 &
  echo $! > "$LOGDIR/chrome.pid"
  sleep 4
fi

# 3) x11vnc (localhost only; reach it via SSH tunnel)
if ! pgrep -f "x11vnc.*-rfbport $VNC_PORT" >/dev/null 2>&1; then
  echo "[launch] starting x11vnc on 127.0.0.1:$VNC_PORT"
  x11vnc -display ":$DISPLAY_NUM" -localhost -rfbport "$VNC_PORT" \
    -nopw -forever -shared -noxdamage -bg -o "$LOGDIR/x11vnc.log" >/dev/null 2>&1
else
  echo "[launch] x11vnc already running on $VNC_PORT"
fi

echo
echo "[launch] CDP version:"
curl -s "http://127.0.0.1:$CDP_PORT/json/version" || echo "  (CDP not reachable yet)"
echo
echo "[launch] READY."
echo "  From your laptop:  ssh -N -L $VNC_PORT:127.0.0.1:$VNC_PORT <user>@<this-host>"
echo "  Then open a VNC viewer at 127.0.0.1:$VNC_PORT and log in to Google."
