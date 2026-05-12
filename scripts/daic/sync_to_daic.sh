#!/bin/bash
# Push code + raw + processed data + checkpoints to DAIC. Run from your
# local machine, repo root.
#
# Defaults: code → $DAIC_PROJECT_DIR, large artefacts → $DAIC_DATA_DIR.
# Override via flags:
#   --code-only          # skip data (raw/processed/checkpoints/residuals)
#   --data-only          # skip code
#   --include-results    # also push results/ (figures, jsons) — usually pull, not push
#
# Requires env.sh sourced *locally* (it just supplies DAIC_NETID + paths):
#   source scripts/daic/env.sh
#   bash scripts/daic/sync_to_daic.sh

set -euo pipefail

if [ -z "${DAIC_NETID:-}" ]; then
  echo "ERROR: env.sh not sourced. Run: source scripts/daic/env.sh" >&2
  exit 1
fi

CODE=true
DATA=true
RESULTS=false
for arg in "$@"; do
  case "$arg" in
    --code-only)         DATA=false ;;
    --data-only)         CODE=false ;;
    --include-results)   RESULTS=true ;;
    *) echo "Unknown flag: $arg" >&2; exit 2 ;;
  esac
done

REMOTE="${DAIC_NETID}@login.daic.tudelft.nl"
RSYNC_BASE=(rsync --progress -avz --no-perms)

if $CODE; then
  echo "→ Pushing code to ${REMOTE}:${DAIC_PROJECT_DIR} ..."
  # Excludes match .gitignore; we explicitly DO want pyproject + scripts.
  "${RSYNC_BASE[@]}" \
    --exclude='.git/' \
    --exclude='.venv/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='wandb/' \
    --exclude='a-thesis-vault/' \
    --exclude='data/' \
    --exclude='results/' \
    --exclude='*.npz' --exclude='*.h5' --exclude='*.pt' --exclude='*.pkl' \
    ./ "${REMOTE}:${DAIC_PROJECT_DIR}/"
fi

if $DATA; then
  echo "→ Pushing data to ${REMOTE}:${DAIC_DATA_DIR} ..."
  # raw is small enough to push every time; processed npz is bigger.
  if [ -d data/raw ]; then
    "${RSYNC_BASE[@]}" data/raw/ "${REMOTE}:${DAIC_DATA_DIR}/raw/"
  fi
  if [ -d data/processed ]; then
    "${RSYNC_BASE[@]}" data/processed/ "${REMOTE}:${DAIC_DATA_DIR}/processed/"
  fi
  if [ -d results/checkpoints ]; then
    "${RSYNC_BASE[@]}" results/checkpoints/ "${REMOTE}:${DAIC_DATA_DIR}/checkpoints/"
  fi
  if [ -d results/residuals ]; then
    "${RSYNC_BASE[@]}" results/residuals/ "${REMOTE}:${DAIC_DATA_DIR}/residuals/"
  fi
  if [ -d results/prior_params ]; then
    "${RSYNC_BASE[@]}" results/prior_params/ "${REMOTE}:${DAIC_DATA_DIR}/prior_params/"
  fi
fi

if $RESULTS; then
  echo "→ Pushing results/ tree (figures + jsons) ..."
  "${RSYNC_BASE[@]}" \
    --exclude='checkpoints/' --exclude='residuals/' --exclude='prior_params/' \
    results/ "${REMOTE}:${DAIC_DATA_DIR}/results/"
fi

echo
echo "Sync complete."
echo
echo "On DAIC, symlink the data dir into the project tree so scripts find it:"
echo "    cd ${DAIC_PROJECT_DIR}"
echo "    ln -sfn ${DAIC_DATA_DIR}/raw          data/raw"
echo "    ln -sfn ${DAIC_DATA_DIR}/processed    data/processed"
echo "    ln -sfn ${DAIC_DATA_DIR}/checkpoints  results/checkpoints"
echo "    ln -sfn ${DAIC_DATA_DIR}/residuals    results/residuals"
echo "    ln -sfn ${DAIC_DATA_DIR}/prior_params results/prior_params"
