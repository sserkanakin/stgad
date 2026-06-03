#!/bin/bash
# Pull results back from the vast.ai box. Run from your local machine, repo root.
#
# Pulls diffusion checkpoints, generated samples, eval outputs (metrics JSON,
# tables, figures), and training logs. Does NOT touch code (use git for that).
#
# Connection uses the `vastai` ssh alias from ~/.ssh/config. Edit
# VAST_HOST/VAST_DIR below if the alias or path changes.
#
#   bash scripts/vastai/sync_from_vast.sh
#
# Flags:
#   --no-checkpoints   # skip diffusion checkpoint files (large)
#   --no-samples       # skip generated sample arrays (large)
#   --no-eval          # skip eval metrics/tables/figures
#   --no-logs          # skip training logs
#   --wandb            # also pull offline wandb run dirs

set -uo pipefail   # not -e: rsync exit 23/24 (missing files) must not abort

VAST_HOST="${VAST_HOST:-vastai}"
VAST_DIR="${VAST_DIR:-/workspace/research-tud}"

CKPTS=true; SAMPLES=true; EVAL=true; LOGS=true; WANDB=false
for arg in "$@"; do
  case "$arg" in
    --no-checkpoints) CKPTS=false ;;
    --no-samples)     SAMPLES=false ;;
    --no-eval)        EVAL=false ;;
    --no-logs)        LOGS=false ;;
    --wandb)          WANDB=true ;;
    *) echo "Unknown flag: $arg" >&2; exit 2 ;;
  esac
done

RSYNC_BASE=(rsync --progress -avz --no-perms)

# rsync exits 23/24 when a requested dir hasn't been produced on the box yet
# (e.g. eval before training finishes). Treat those as warnings, not errors.
_pull() {
  local label="$1" src="$2" dst="$3"
  echo "← Pulling ${label} ..."
  mkdir -p "$dst"
  "${RSYNC_BASE[@]}" "${VAST_HOST}:${src}" "$dst"
  local rc=$?
  if [ $rc -ne 0 ] && [ $rc -ne 23 ] && [ $rc -ne 24 ]; then
    echo "  rsync failed with exit code $rc — aborting" >&2
    exit $rc
  fi
  [ $rc -ne 0 ] && echo "  (skipped: ${src} not present on box yet)"
}

$CKPTS   && _pull "diffusion checkpoints" "${VAST_DIR}/results/checkpoints/diffusion/" results/checkpoints/diffusion
$SAMPLES && _pull "generated samples"     "${VAST_DIR}/results/diffusion/"             results/diffusion
$EVAL    && _pull "eval results"          "${VAST_DIR}/results/eval/"                  results/eval
$LOGS    && _pull "training logs"         "${VAST_DIR}/logs/"                          logs
$WANDB   && _pull "wandb runs"            "${VAST_DIR}/wandb/"                         wandb

echo
echo "Pull complete."
