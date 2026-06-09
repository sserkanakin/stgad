#!/bin/bash
# pull results back from daic. run from your local machine, repo root.
#
# pulls backbone + diffusion checkpoints, residuals, eval outputs, and slurm
# logs. does NOT touch code (use git for that).
#
#   source scripts/daic/env.sh
#   bash scripts/daic/sync_from_daic.sh
#
# flags:
#   --no-checkpoints     # skip backbone + diffusion checkpoints
#   --no-residuals       # skip residual .npz files
#   --no-logs            # skip slurm logs

set -uo pipefail   # not -e: rsync exit 23/24 (missing files) must not abort

if [ -z "${DAIC_NETID:-}" ]; then
  echo "ERROR: env.sh not sourced. Run: source scripts/daic/env.sh" >&2
  exit 1
fi

CKPTS=true
RESIDS=true
LOGS=true
for arg in "$@"; do
  case "$arg" in
    --no-checkpoints) CKPTS=false ;;
    --no-residuals)   RESIDS=false ;;
    --no-logs)        LOGS=false ;;
    *) echo "Unknown flag: $arg" >&2; exit 2 ;;
  esac
done

REMOTE="${DAIC_NETID}@${DAIC_LOGIN_HOST}"
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

if $CKPTS; then
  _pull "backbone checkpoints"  "${DAIC_DATA_DIR}/backbone/"  results/backbone
  _pull "diffusion checkpoints" "${DAIC_DATA_DIR}/diffusion/" results/diffusion
fi

if $RESIDS; then
  _pull "residuals" "${DAIC_DATA_DIR}/residuals/" results/residuals
fi

# eval outputs (metrics json, tables, figures, small sample bank).
_pull "eval results" "${DAIC_DATA_DIR}/eval/" results/eval

if $LOGS; then
  _pull "slurm logs" "${DAIC_PROJECT_DIR}/logs/" logs
fi

echo
echo "Pull complete."
