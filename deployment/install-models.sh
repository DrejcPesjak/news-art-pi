#!/bin/bash
# One-time model bootstrap. Routine code/config updates continue through deploy.
# sudo bash deployment/install-models.sh USER VERIFIED_BUNDLE NCNN_RUNTIME_DIRECTORY
set -euo pipefail
[[ $EUID -eq 0 ]] || { echo 'Run with sudo'; exit 1; }
PI_USER=${1:?Supply the Pi login username}
source_models=$(realpath "${2:?Supply a verified runtime bundle}")
source_runtime=$(realpath "${3:?Supply the NCNN package directory}")
[[ $PI_USER =~ ^[a-z_][a-z0-9_-]*$ ]] || exit 1
id "$PI_USER" >/dev/null
code_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$code_dir"
exec 9>/run/lock/news-art-pi-update.lock
flock 9
python3 -m research.supra.bundle verify "$source_models"
export DEBIAN_FRONTEND=noninteractive
apt-get install -y python3-numpy python3-sentencepiece
runuser -u "$PI_USER" -- env PYTHONPATH="$source_runtime" \
    python3 -c 'import ncnn, numpy, sentencepiece; print("Native runtime:", ncnn.__version__)'

was_active=0
if systemctl is-active --quiet news-art.service; then
    was_active=1
    systemctl stop news-art.service
fi
restore_worker() {
    if [[ $was_active == 1 ]]; then systemctl start news-art.service; fi
}
trap restore_worker EXIT

install -d /opt/news-art-pi/runtime /var/lib/news-art-pi/models
cp -a "$source_runtime/." /opt/news-art-pi/runtime/
chown -R root:root /opt/news-art-pi/runtime
chmod -R go+rX /opt/news-art-pi/runtime
fingerprint=$(sha256sum "$source_models/manifest.json" | cut -d ' ' -f1)
destination="/var/lib/news-art-pi/models/$fingerprint"
python3 - "$source_models" "$destination" <<'PY'
import json, shutil, sys
from pathlib import Path
from research.supra.bundle import verify
source, target = map(Path, sys.argv[1:])
target.mkdir(parents=True, exist_ok=True)
manifest = json.loads((source/'manifest.json').read_text())
for name in manifest:
    if Path(name).name != name:
        raise ValueError('Invalid manifest filename')
    if not (target/name).exists():
        shutil.copyfile(source/name, target/name)
shutil.copyfile(source/'manifest.json', target/'manifest.json')
verify(target)
PY
chmod -R go+rX "$destination"
ln -sfn "$fingerprint" /var/lib/news-art-pi/models/supra.next
mv -Tf /var/lib/news-art-pi/models/supra.next /var/lib/news-art-pi/models/supra
runuser -u "$PI_USER" -- python3 -m news_art.worker --check --check-runtime
echo "Installed verified model bundle: $fingerprint"
