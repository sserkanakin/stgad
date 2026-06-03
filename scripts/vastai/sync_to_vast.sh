#!/bin/bash
# Push data files (NOT code) to the vast.ai box. Run from your local machine,
# repo root.
#
# Code travels via git (git push → GitHub → git pull on the box). This helper
# pushes the large binary artefacts that aren't in git: raw .h5 + adjacency,
# processed .npz + scalers, residual caches, fitted priors, backbone/diffusion
# checkpoints, and the vendored third_party/ tree (gitignored).
#
# Connection uses the `vastai` ssh alias from ~/.ssh/config — no host/port/key
# to pass. Edit VAST_HOST/VAST_DIR below if the alias or path changes.
#
#   bash scripts/vastai/sync_to_vast.sh                # processed + residuals + priors + third_party
#   bash scripts/vastai/sync_to_vast.sh --all          # also raw + checkpoints
#   bash scripts/vastai/sync_to_vast.sh --raw          # just raw .h5
#   bash scripts/vastai/sync_to_vast.sh --processed    # just processed .npz
#   bash scripts/vastai/sync_to_vast.sh --residuals    # just residual caches
#   bash scripts/vastai/sync_to_vast.sh --priors       # just fitted prior dirs
#   bash scripts/vastai/sync_to_vast.sh --checkpoints  # just checkpoints
#   bash scripts/vastai/sync_to_vast.sh --third-party  # just vendored third_party/

set -euo pipefail

VAST_HOST="${VAST_HOST:-vastai}"
VAST_DIR="${VAST_DIR:-/workspace/research-tud}"

# Defaults: the artefacts diffusion training needs to run. raw + checkpoints
# are large and usually already present, so they're off unless requested.
PUSH_RAW=false
PUSH_PROCESSED=true
PUSH_RESIDUALS=true
PUSH_PRIORS=true
PUSH_CKPTS=false
PUSH_THIRDPARTY=true

if [ "${1:-}" = "--all" ]; then
  PUSH_RAW=true
  PUSH_CKPTS=true
elif [ -n "${1:-}" ]; then
  PUSH_PROCESSED=false; PUSH_RESIDUALS=false; PUSH_PRIORS=false; PUSH_THIRDPARTY=false
  for arg in "$@"; do
    case "$arg" in
      --raw)         PUSH_RAW=true ;;
      --processed)   PUSH_PROCESSED=true ;;
      --residuals)   PUSH_RESIDUALS=true ;;
      --priors)      PUSH_PRIORS=true ;;
      --checkpoints) PUSH_CKPTS=true ;;
      --third-party) PUSH_THIRDPARTY=true ;;
      *) echo "Unknown flag: $arg" >&2
         echo "Use: --all --raw --processed --residuals --priors --checkpoints --third-party" >&2
         exit 2 ;;
    esac
  done
fi

RSYNC_BASE=(rsync --progress -avz --no-perms \
            -e 'ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=10')

_push() {
  local label="$1" src="$2" dst="$3"
  [ -d "$src" ] || { echo "→ ${label}: local ${src} absent, skipping"; return; }
  echo "→ ${label}: ${src} → ${VAST_HOST}:${dst}"
  ssh "$VAST_HOST" "mkdir -p '${dst}'"
  "${RSYNC_BASE[@]}" "${src}/" "${VAST_HOST}:${dst}/"
}

echo "→ Code: NOT pushed (use git push / git pull instead)."
echo

$PUSH_RAW        && _push "raw data"     data/raw                "${VAST_DIR}/data/raw"
$PUSH_PROCESSED  && _push "processed"     data/processed          "${VAST_DIR}/data/processed"
$PUSH_RESIDUALS  && _push "residuals"     results/residuals       "${VAST_DIR}/results/residuals"
$PUSH_PRIORS     && _push "prior params"  results/prior_params    "${VAST_DIR}/results/prior_params"
$PUSH_PRIORS     && _push "section3"      results/section3        "${VAST_DIR}/results/section3"
$PUSH_CKPTS      && _push "checkpoints"   results/checkpoints     "${VAST_DIR}/results/checkpoints"
$PUSH_THIRDPARTY && "${RSYNC_BASE[@]}" --exclude='.git' --exclude='__pycache__' \
                      third_party/ "${VAST_HOST}:${VAST_DIR}/third_party/" 2>/dev/null \
                   && echo "→ third_party pushed" || true

echo
echo "Push complete. Reminder: code goes via git — on the box:  cd ${VAST_DIR} && git pull"
