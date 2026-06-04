#!/usr/bin/env bash
# push data to a vast.ai vm. run on your LOCAL machine, from the repo root.
# code travels via git; this only pushes the gitignored data/ tree (raw +
# processed). third_party is cloned by setup.sh on the vm, so it isn't pushed.
#
#   VAST_HOST=root@1.2.3.4 VAST_PORT=12345 bash scripts/vastai/sync_to_vast.sh
#
# (vast.ai gives you the ssh host + port on the instance "connect" page.)
set -euo pipefail

: "${VAST_HOST:?set VAST_HOST=root@<ip> (from the vast connect page)}"
VAST_PORT="${VAST_PORT:-22}"
VAST_DIR="${VAST_DIR:-/root/research-tud}"

echo "-> data/ -> ${VAST_HOST}:${VAST_DIR}/data/  (port ${VAST_PORT})"
rsync -avz --progress -e "ssh -p ${VAST_PORT}" data/ "${VAST_HOST}:${VAST_DIR}/data/"

echo "done. on the vm you can now run training."
