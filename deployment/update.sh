#!/bin/bash
# Installed locally at bootstrap; only the public deploy branch is fetched.
set -euo pipefail
source /etc/news-art-pi.env
exec 9>/run/lock/news-art-pi-update.lock
flock -n 9 || exit 0
base=/opt/news-art-pi
export GIT_TERMINAL_PROMPT=0
git -C "$base/repository" fetch --quiet --depth=1 origin deploy
revision=$(git -C "$base/repository" rev-parse FETCH_HEAD)
old=$(readlink -f "$base/current")
if [[ -f "$old/REVISION" && $(cat "$old/REVISION") == "$revision" ]]; then
    echo "Already deployed: $revision"
    exit 0
fi
release="$base/releases/$revision"
mkdir -p "$release"
git -C "$base/repository" archive "$revision" | tar -x -C "$release"
printf '%s\n' "$revision" > "$release/REVISION"
# Validate every candidate on the Pi before changing the active release.
cd "$release"
runuser -u "$PI_USER" -- /usr/bin/python3 -m unittest discover -s tests -v
runuser -u "$PI_USER" -- /usr/bin/python3 -m news_art.worker --check
switch_release() {
    ln -sfn "$1" "$base/current.next"
    mv -Tf "$base/current.next" "$base/current"
}
restart_display() {
    uid=$(id -u "$PI_USER")
    user_systemctl=(runuser -u "$PI_USER" -- env XDG_RUNTIME_DIR="/run/user/$uid"
        DBUS_SESSION_BUS_ADDRESS="unix:path=/run/user/$uid/bus" systemctl --user)
    if "${user_systemctl[@]}" is-active --quiet news-art-display.service; then
        "${user_systemctl[@]}" restart news-art-display.service
        sleep 3
        "${user_systemctl[@]}" is-active --quiet news-art-display.service
    fi
}
switch_release "$release"
systemctl restart news-art.service
sleep 4
if ! systemctl is-active --quiet news-art.service; then
    switch_release "$old"
    systemctl restart news-art.service
    echo "Worker startup failed; restored $old" >&2
    exit 1
fi
if ! restart_display; then
    switch_release "$old"
    systemctl restart news-art.service
    restart_display || true
    echo "Display restart failed; restored $old" >&2
    exit 1
fi
echo "Deployed $revision (previous: $old)"
# Keep releases for manual rollback; they are small and exclude model weights.

