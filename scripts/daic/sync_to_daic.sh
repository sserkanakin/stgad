#!/bin/bash
# push data files (not code) to daic. run from your local machine, repo root.
#
# code travels via git. this only pushes the large gitignored artefacts:
# raw + processed data, trained backbone/diffusion checkpoints, and optionally
# residuals or the vendored third_party/.
#
#   source scripts/daic/env.sh
#   bash scripts/daic/sync_to_daic.sh                 # raw + processed + models
#   bash scripts/daic/sync_to_daic.sh --all           # also push residuals
#   bash scripts/daic/sync_to_daic.sh --raw           # just raw
#   bash scripts/daic/sync_to_daic.sh --processed     # just processed
#   bash scripts/daic/sync_to_daic.sh --checkpoints   # backbone + diffusion ckpts
#   bash scripts/daic/sync_to_daic.sh --residuals     # extracted residuals
#   bash scripts/daic/sync_to_daic.sh --third-party   # vendored third_party/
#
# rsync mirrors the local tree, so the per-dataset layout (data/processed/<ds>/)
# propagates as-is. it does NOT delete remote files, so if you reorganised the
# layout, clean up the old flat files on daic once:
#   ssh "$DAIC_NETID@login.daic.tudelft.nl" \
#     "find $DAIC_DATA_DIR/processed -maxdepth 1 -type f -delete"

set -euo pipefail

if [ -z "${DAIC_NETID:-}" ]; then
  echo "ERROR: env.sh not sourced. Run: source scripts/daic/env.sh" >&2
  exit 1
fi

PUSH_RAW=true
PUSH_PROCESSED=true
PUSH_CKPTS=true
PUSH_RESIDUALS=false
PUSH_THIRDPARTY=false

if [ "${1:-}" = "--all" ]; then
  PUSH_RESIDUALS=true
elif [ -n "${1:-}" ]; then
  # specific-target mode: turn everything off, enable only the named ones.
  PUSH_RAW=false; PUSH_PROCESSED=false; PUSH_CKPTS=false
  for arg in "$@"; do
    case "$arg" in
      --raw)         PUSH_RAW=true ;;
      --processed)   PUSH_PROCESSED=true ;;
      --checkpoints) PUSH_CKPTS=true ;;
      --residuals)   PUSH_RESIDUALS=true ;;
      --third-party) PUSH_THIRDPARTY=true ;;
      *) echo "Unknown flag: $arg" >&2
         echo "Use: --all --raw --processed --checkpoints --residuals --third-party" >&2
         exit 2 ;;
    esac
  done
fi

REMOTE="${DAIC_NETID}@${DAIC_LOGIN_HOST}"
RSYNC_BASE=(rsync --progress -avz --no-perms \
            -e 'ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=10')

echo "→ code: NOT pushed (use git push / git pull instead)."
echo

if $PUSH_RAW && [ -d data/raw ]; then
  echo "→ data/raw → ${REMOTE}:${DAIC_DATA_DIR}/raw"
  "${RSYNC_BASE[@]}" data/raw/ "${REMOTE}:${DAIC_DATA_DIR}/raw/"
fi
if $PUSH_PROCESSED && [ -d data/processed ]; then
  echo "→ data/processed → ${REMOTE}:${DAIC_DATA_DIR}/processed"
  "${RSYNC_BASE[@]}" data/processed/ "${REMOTE}:${DAIC_DATA_DIR}/processed/"
fi
if $PUSH_CKPTS; then
  for stage in backbone diffusion; do
    if [ -d "results/${stage}" ]; then
      echo "→ results/${stage} → ${REMOTE}:${DAIC_DATA_DIR}/${stage}"
      "${RSYNC_BASE[@]}" "results/${stage}/" "${REMOTE}:${DAIC_DATA_DIR}/${stage}/"
    fi
  done
fi
if $PUSH_RESIDUALS && [ -d results/residuals ]; then
  echo "→ results/residuals → ${REMOTE}:${DAIC_DATA_DIR}/residuals"
  "${RSYNC_BASE[@]}" results/residuals/ "${REMOTE}:${DAIC_DATA_DIR}/residuals/"
fi
if $PUSH_THIRDPARTY && [ -d third_party ]; then
  # third_party is code-like but gitignored, so it lives in the project dir.
  echo "→ third_party → ${REMOTE}:${DAIC_PROJECT_DIR}/third_party"
  "${RSYNC_BASE[@]}" --exclude='.git' --exclude='__pycache__' \
    third_party/ "${REMOTE}:${DAIC_PROJECT_DIR}/third_party/"
fi

echo
echo "data sync complete."
echo "reminder: code goes via git. on daic:  cd ~/research-tud && git pull"
