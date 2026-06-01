#!/bin/bash
# Pull results back from DAIC. Run from your local machine, repo root.
#
# Pulls back checkpoints, residuals, fitted prior YAMLs, figures, and JSON
# results. Does NOT touch code (use git for that).
#
#   source scripts/daic/env.sh
#   bash scripts/daic/sync_from_daic.sh
#
# Flags:
#   --no-checkpoints     # skip large checkpoint files
#   --no-residuals       # skip residual .npz files
#   --no-priors          # skip fitted prior YAMLs
#   --no-logs            # skip Slurm logs

set -uo pipefail   # not -e: rsync exit 23/24 (missing files) must not abort

if [ -z "${DAIC_NETID:-}" ]; then
  echo "ERROR: env.sh not sourced. Run: source scripts/daic/env.sh" >&2
  exit 1
fi

CKPTS=true
RESIDS=true
PRIORS=true
LOGS=true
for arg in "$@"; do
  case "$arg" in
    --no-checkpoints) CKPTS=false ;;
    --no-residuals)   RESIDS=false ;;
    --no-priors)      PRIORS=false ;;
    --no-logs)        LOGS=false ;;
    *) echo "Unknown flag: $arg" >&2; exit 2 ;;
  esac
done

REMOTE="${DAIC_NETID}@login.daic.tudelft.nl"
RSYNC_BASE=(rsync --progress -avz --no-perms)

# rsync exits 23 (some files vanished) or 24 (source dir doesn't exist) when
# we ask for an artefact dir that hasn't been produced yet on DAIC. Treat
# those as warnings, not fatal errors — the pipeline produces artefacts in
# stages and we want partial pulls to succeed.
_pull() {
  local label="$1" src="$2" dst="$3"
  echo "← Pulling ${label} ..."
  mkdir -p "$dst"
  "${RSYNC_BASE[@]}" "${REMOTE}:${src}" "$dst"
  local rc=$?
  if [ $rc -ne 0 ] && [ $rc -ne 23 ] && [ $rc -ne 24 ]; then
    echo "  rsync failed with exit code $rc — aborting" >&2
    exit $rc
  fi
  if [ $rc -ne 0 ]; then
    echo "  (skipped: remote ${src} not present yet)"
  fi
}

if $PRIORS; then
  _pull "prior params" "${DAIC_DATA_DIR}/prior_params/" results/prior_params
fi

if $CKPTS; then
  _pull "checkpoints" "${DAIC_DATA_DIR}/checkpoints/" results/checkpoints
fi

if $RESIDS; then
  _pull "residuals" "${DAIC_DATA_DIR}/residuals/" results/residuals
fi

# Evaluation outputs (metrics JSON, comparison tables, figures, small sample
# bank) live in the project dir, not the data dir. All small — pull wholesale.
_pull "eval results" "${DAIC_PROJECT_DIR}/results/eval/" results/eval

echo "← Pulling experiments tree (figures + JSON, no large arrays) ..."
mkdir -p experiments
"${RSYNC_BASE[@]}" \
  --include='*.png' --include='*.json' --include='*.md' --include='*.yaml' \
  --include='*/' --exclude='*' \
  "${REMOTE}:${DAIC_PROJECT_DIR}/experiments/" experiments/ || true

if $LOGS; then
  _pull "Slurm logs" "${DAIC_PROJECT_DIR}/logs/" logs
fi

echo
echo "Pull complete."
