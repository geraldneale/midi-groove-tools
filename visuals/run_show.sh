#!/usr/bin/env bash
# Gig launcher: fullscreen on the projector, restarts within a second if the app crashes.
# Esc quits cleanly and ends the loop.
cd "$(dirname "$0")"
# no screen blanking or sleep mid set
if [ "$(uname)" = "Darwin" ]; then
  caffeinate -dis -w $$ &      # macOS: stay awake until this script ends
else
  xset s off -dpms 2>/dev/null
fi
while true; do
  python3 vis_render.py show.toml --fullscreen --display "${DISPLAY_INDEX:-1}" "$@"
  [ $? -eq 0 ] && break
  echo "visuals exited, restarting" >&2
  sleep 1
done
