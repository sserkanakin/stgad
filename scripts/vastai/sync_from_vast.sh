#!/usr/bin/env bash
# pull results + logs back from a vast.ai vm. run on your LOCAL machine.
#
#   VAST_HOST=root@1.2.3.4 VAST_PORT=12345 bash scripts/vastai/sync_from_vast.sh
set -uo pipefail   # not -e: a missing remote dir (not produced yet) must not abort

: "${VAST_HOST:?set VAST_HOST=root@<ip> (from the vast connect page)}"
VAST_PORT="${VAST_PORT:-22}"
VAST_DIR="${VAST_DIR:-/root/research-tud}"

mkdir -p results logs
echo "-> results/"
rsync -avz --progress -e "ssh -p ${VAST_PORT}" "${VAST_HOST}:${VAST_DIR}/results/" results/
echo "-> logs/"
rsync -avz --progress -e "ssh -p ${VAST_PORT}" "${VAST_HOST}:${VAST_DIR}/logs/" logs/

echo "done."
