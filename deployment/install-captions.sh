#!/bin/bash
# One-time bootstrap of local NLTK data; routine updates still use Git deploy.
# sudo bash deployment/install-captions.sh USER EXTRACTED_NLTK_DATA
set -euo pipefail
[[ $EUID -eq 0 ]] || { echo 'Run with sudo'; exit 1; }
pi_user=${1:?Supply the Pi login username}
source_data=$(realpath "${2:?Supply the extracted NLTK data directory}")
[[ $pi_user =~ ^[a-z_][a-z0-9_-]*$ ]] || exit 1
id "$pi_user" >/dev/null
export DEBIAN_FRONTEND=noninteractive
apt-get install -y --no-install-recommends python3-nltk python3-lxml
destination=/var/lib/news-art-pi/models/nltk
install -d "$destination"
for directory in chunkers corpora taggers; do
    [[ -d "$source_data/$directory" ]] || { echo "Missing $directory"; exit 1; }
    cp -a "$source_data/$directory" "$destination/"
done
chown -R root:root "$destination"
chmod -R a+rX "$destination"
code_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$code_dir"
runuser -u "$pi_user" -- /usr/bin/python3 -m news_art.captions
