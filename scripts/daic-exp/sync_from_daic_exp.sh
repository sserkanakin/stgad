#!/bin/bash
# Pull results back from experimental DAIC. Run from your local machine,
# repo root. This is how YOU (the project owner) get the friend's results.
#
# Wandb already gives you the loss curves; this puller is for the raw
# artefacts (checkpoints, sampled forecasts, fitted priors, figures).
#
#   source scripts/daic-exp/env.sh
#   bash scripts/daic-exp/sync_from_daic_exp.sh
#
# Flags:
#   --no-checkpoints     # skip large checkpoint files
#   --no-residuals       # skip residual .npz files
#   --no-logs            # skip Slurm logs

set -euo pipefail

if [ -z "${DAIC_EXP_NETID:-}" ]; then
  echo "ERROR: env.sh not sourced. Run: source scripts/daic-exp/env.sh" >&2
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

REMOTE="${DAIC_EXP_NETID}@${DAIC_EXP_LOGIN_HOST}"
RSYNC_BASE=(rsync --progress -avz --no-perms)

echo "← Pulling prior params ..."
mkdir -p results/prior_params
"${RSYNC_BASE[@]}" "${REMOTE}:${DAIC_EXP_DATA_DIR}/prior_params/" results/prior_params/

if $CKPTS; then
  echo "← Pulling checkpoints ..."
  mkdir -p results/checkpoints
  "${RSYNC_BASE[@]}" "${REMOTE}:${DAIC_EXP_DATA_DIR}/checkpoints/" results/checkpoints/
fi

if $RESIDS; then
  echo "← Pulling residuals ..."
  mkdir -p results/residuals
  "${RSYNC_BASE[@]}" "${REMOTE}:${DAIC_EXP_DATA_DIR}/residuals/" results/residuals/
fi

echo "← Pulling experiments tree (figures + JSON + SUMMARYs, no large arrays) ..."
"${RSYNC_BASE[@]}" \
  --include='*.png' --include='*.json' --include='*.md' --include='*.yaml' \
  --include='*/' --exclude='*' \
  "${REMOTE}:${DAIC_EXP_PROJECT_DIR}/experiments/" experiments/

if $LOGS; then
  echo "← Pulling Slurm logs ..."
  mkdir -p logs
  "${RSYNC_BASE[@]}" "${REMOTE}:${DAIC_EXP_PROJECT_DIR}/logs/" logs/
fi

echo
echo "Pull complete."
echo "Wandb runs are live at:"
echo "  https://wandb.ai/\${WANDB_ENTITY:-<your-entity>}/${WANDB_PROJECT:-thesis-diffusion}"
