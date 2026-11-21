#!/bin/bash
# First installation only: sudo bash deployment/install.sh USER REPOSITORY_HTTPS_URL
set -euo pipefail
[[ $EUID -eq 0 ]] || { echo 'Run with sudo'; exit 1; }
PI_USER=${1:?Supply the Pi login username}
REPO_URL=${2:?Supply the public HTTPS GitHub repository URL}
[[ $PI_USER =~ ^[a-z_][a-z0-9_-]*$ ]] || exit 1
[[ $REPO_URL =~ ^https://github.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(\.git)?$ ]] || exit 1
home_dir=$(getent passwd "$PI_USER" | cut -d: -f6)
base=/opt/news-art-pi
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y git python3-pil python3-pil.imagetk python3-tk fonts-dejavu-core
install -d "$base/releases"
if [[ ! -d "$base/repository" ]]; then
    git clone --bare --depth=1 --branch deploy "$REPO_URL" "$base/repository"
fi
git -C "$base/repository" fetch --quiet --depth=1 origin deploy
revision=$(git -C "$base/repository" rev-parse FETCH_HEAD)
release="$base/releases/$revision"
mkdir -p "$release"
git -C "$base/repository" archive "$revision" | tar -x -C "$release"
printf '%s\n' "$revision" > "$release/REVISION"
cd "$release"
runuser -u "$PI_USER" -- python3 -m unittest discover -s tests -v
runuser -u "$PI_USER" -- python3 -m news_art.worker --check
ln -sfn "$release" "$base/current"
printf 'PI_USER=%q\n' "$PI_USER" > /etc/news-art-pi.env
install -d -o "$PI_USER" -g "$(id -gn "$PI_USER")" /var/lib/news-art-pi
sed "s/@USER@/$PI_USER/g" deployment/news-art.service > /etc/systemd/system/news-art.service
install -m 644 deployment/news-art-update.{service,timer} /etc/systemd/system/
install -d -o "$PI_USER" -g "$(id -gn "$PI_USER")" "$home_dir/.config/systemd/user" "$home_dir/.config/autostart"
install -o "$PI_USER" -g "$(id -gn "$PI_USER")" -m 644 deployment/news-art-display.service "$home_dir/.config/systemd/user/"
cat > "$home_dir/.config/autostart/news-art.desktop" <<'EOF'
[Desktop Entry]
Type=Application
Name=News Art Pi
Exec=/bin/bash /opt/news-art-pi/current/deployment/start-display.sh
Terminal=false
EOF
chown "$PI_USER:$(id -gn "$PI_USER")" "$home_dir/.config/autostart/news-art.desktop"
systemctl daemon-reload
systemctl enable --now news-art.service news-art-update.timer
uid=$(id -u "$PI_USER")
runuser -u "$PI_USER" -- env XDG_RUNTIME_DIR="/run/user/$uid" \
    DBUS_SESSION_BUS_ADDRESS="unix:path=/run/user/$uid/bus" systemctl --user daemon-reload
echo 'Installed. The display starts on the next desktop login. See README for starting it now.'

