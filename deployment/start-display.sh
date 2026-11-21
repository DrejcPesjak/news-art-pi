#!/bin/bash
set -euo pipefail
systemctl --user import-environment DISPLAY XAUTHORITY WAYLAND_DISPLAY XDG_SESSION_TYPE
# XWayland may ignore these; keep the command optional on Wayland desktops.
if command -v xset >/dev/null; then
    xset s off -dpms 2>/dev/null || true
fi
systemctl --user reset-failed news-art-display.service 2>/dev/null || true
systemctl --user restart news-art-display.service

