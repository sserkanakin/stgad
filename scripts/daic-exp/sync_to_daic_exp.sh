#!/bin/bash
# Push data (NOT code) to experimental DAIC. Run from your local machine,
# repo root. Code travels via git push / git pull on the cluster.
#
#   source scripts/daic-exp/env.sh
#   bash scripts/daic-exp/sync_to_daic_exp.sh             # raw + processed + checkpoints
#   bash scripts/daic-exp/sync_to_daic_exp.sh --all       # also push residuals + priors
#   bash scripts/daic-exp/sync_to_daic_exp.sh --raw       # just raw .h5
#   bash scripts/daic-exp/sync_to_daic_exp.sh --processed
#   bash scripts/daic-exp/sync_to_daic_exp.sh --checkpoints
#   bash scripts/daic-exp/sync_to_daic_exp.sh --residuals
#   bash scripts/daic-exp/sync_to_daic_exp.sh --priors

set -euo pipefail

if [ -z "${DAIC_EXP_NETID:-}" ]; then
  echo "ERROR: env.sh not sourced. Run: source scripts/daic-exp/env.sh" >&2
  exit 1
fi
if [ "${DAIC_EXP_NETID}" = "CHANGE_ME" ]; then
  echo "ERROR: edit DAIC_EXP_NETID in scripts/daic-exp/env.sh" >&2
  exit 1
fi

PUSH_RAW=true
PUSH_PROCESSED=true
PUSH_CKPTS=true
PUSH_RESIDUALS=false
PUSH_PRIORS=false

if [ "${1:-}" = "--all" ]; then
  PUSH_RESIDUALS=true
  PUSH_PRIORS=true
elif [ -n "${1:-}" ]; then
  PUSH_RAW=false; PUSH_PROCESSED=false; PUSH_CKPTS=false
  for arg in "$@"; do
    case "$arg" in
      --raw)         PUSH_RAW=true ;;
      --processed)   PUSH_PROCESSED=true ;;
      --checkpoints) PUSH_CKPTS=true ;;
      --residuals)   PUSH_RESIDUALS=true ;;
      --priors)      PUSH_PRIORS=true ;;
      *) echo "Unknown flag: $arg" >&2
         echo "Use one of: --all --raw --processed --checkpoints --residuals --priors" >&2
         exit 2 ;;
    esac
  done
fi

REMOTE="${DAIC_EXP_NETID}@${DAIC_EXP_LOGIN_HOST}"
RSYNC_BASE=(rsync --progress -avz --no-perms \
            -e 'ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=10')

# Make sure remote subdirs exist (idempotent, harmless if they do).
ssh "${REMOTE}" "mkdir -p ${DAIC_EXP_DATA_DIR}/{raw,processed,checkpoints,residuals,prior_params}"

echo "→ Code: NOT pushed (use git push / git pull instead)."
echo "  Remote project root:  ${REMOTE}:${DAIC_EXP_PROJECT_DIR}"
echo "  Remote data root:     ${REMOTE}:${DAIC_EXP_DATA_DIR}"
echo

if $PUSH_RAW && [ -d data/raw ]; then
  echo "→ data/raw → ${REMOTE}:${DAIC_EXP_DATA_DIR}/raw"
  "${RSYNC_BASE[@]}" data/raw/ "${REMOTE}:${DAIC_EXP_DATA_DIR}/raw/"
fi
if $PUSH_PROCESSED && [ -d data/processed ]; then
  echo "→ data/processed → ${REMOTE}:${DAIC_EXP_DATA_DIR}/processed"
  "${RSYNC_BASE[@]}" data/processed/ "${REMOTE}:${DAIC_EXP_DATA_DIR}/processed/"
fi
if $PUSH_CKPTS && [ -d results/checkpoints ]; then
  echo "→ results/checkpoints → ${REMOTE}:${DAIC_EXP_DATA_DIR}/checkpoints"
  "${RSYNC_BASE[@]}" results/checkpoints/ "${REMOTE}:${DAIC_EXP_DATA_DIR}/checkpoints/"
fi
if $PUSH_RESIDUALS && [ -d results/residuals ]; then
  echo "→ results/residuals → ${REMOTE}:${DAIC_EXP_DATA_DIR}/residuals"
  "${RSYNC_BASE[@]}" results/residuals/ "${REMOTE}:${DAIC_EXP_DATA_DIR}/residuals/"
fi
if $PUSH_PRIORS && [ -d results/prior_params ]; then
  echo "→ results/prior_params → ${REMOTE}:${DAIC_EXP_DATA_DIR}/prior_params"
  "${RSYNC_BASE[@]}" results/prior_params/ "${REMOTE}:${DAIC_EXP_DATA_DIR}/prior_params/"
fi

echo
echo "Data sync complete."
echo "Reminder: code goes via git. On the cluster:"
echo "    cd ${DAIC_EXP_PROJECT_DIR} && git pull"
