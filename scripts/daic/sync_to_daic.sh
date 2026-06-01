#!/bin/bash
# Push data files (NOT code) to DAIC. Run from your local machine, repo root.
#
# Code travels via git push / git pull on DAIC. This helper only pushes the
# large binary artefacts that aren't in git: raw .h5 + adjacency pickles,
# processed .npz + scalers, trained backbone checkpoints, and (optionally)
# extracted residuals or fitted prior YAMLs.
#
# Defaults (most common workflow): pushes raw + processed + checkpoints.
# Residuals + prior_params are rebuilt on DAIC by sbatch jobs, so they're
# skipped by default. Use --all to push them too.
#
#   source scripts/daic/env.sh
#   bash scripts/daic/sync_to_daic.sh                 # raw + processed + checkpoints
#   bash scripts/daic/sync_to_daic.sh --all           # also push residuals + priors
#   bash scripts/daic/sync_to_daic.sh --raw           # just raw .h5
#   bash scripts/daic/sync_to_daic.sh --processed     # just processed .npz
#   bash scripts/daic/sync_to_daic.sh --checkpoints   # just backbone ckpts
#   bash scripts/daic/sync_to_daic.sh --residuals     # just extracted residuals
#   bash scripts/daic/sync_to_daic.sh --priors        # just fitted prior YAMLs
#   bash scripts/daic/sync_to_daic.sh --third-party    # vendored third_party/ (gitignored)
#
# third_party/ is gitignored (see THIRD_PARTY.md) — it does NOT travel via git,
# so it must be pushed here (or cloned on DAIC). Required for diffusion training
# (DiffSTG UGnet) and backbone (Graph-WaveNet). .git is excluded from the push.

set -euo pipefail

if [ -z "${DAIC_NETID:-}" ]; then
  echo "ERROR: env.sh not sourced. Run: source scripts/daic/env.sh" >&2
  exit 1
fi

# Default: push raw + processed + checkpoints. Residuals + priors off.
PUSH_RAW=true
PUSH_PROCESSED=true
PUSH_CKPTS=true
PUSH_RESIDUALS=false
PUSH_PRIORS=false
PUSH_THIRDPARTY=false

if [ "${1:-}" = "--all" ]; then
  PUSH_RESIDUALS=true
  PUSH_PRIORS=true
elif [ -n "${1:-}" ]; then
  # Specific-target mode: turn everything off, then enable only the named one(s).
  PUSH_RAW=false; PUSH_PROCESSED=false; PUSH_CKPTS=false
  for arg in "$@"; do
    case "$arg" in
      --raw)         PUSH_RAW=true ;;
      --processed)   PUSH_PROCESSED=true ;;
      --checkpoints) PUSH_CKPTS=true ;;
      --residuals)   PUSH_RESIDUALS=true ;;
      --priors)      PUSH_PRIORS=true ;;
      --third-party) PUSH_THIRDPARTY=true ;;
      *) echo "Unknown flag: $arg" >&2
         echo "Use one of: --all --raw --processed --checkpoints --residuals --priors --third-party" >&2
         exit 2 ;;
    esac
  done
fi

REMOTE="${DAIC_NETID}@login.daic.tudelft.nl"
RSYNC_BASE=(rsync --progress -avz --no-perms \
            -e 'ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=10')

echo "→ Code: NOT pushed (use git push / git pull instead)."
echo

if $PUSH_RAW && [ -d data/raw ]; then
  echo "→ data/raw → ${REMOTE}:${DAIC_DATA_DIR}/raw"
  "${RSYNC_BASE[@]}" data/raw/ "${REMOTE}:${DAIC_DATA_DIR}/raw/"
fi
if $PUSH_PROCESSED && [ -d data/processed ]; then
  echo "→ data/processed → ${REMOTE}:${DAIC_DATA_DIR}/processed"
  "${RSYNC_BASE[@]}" data/processed/ "${REMOTE}:${DAIC_DATA_DIR}/processed/"
fi
if $PUSH_CKPTS && [ -d results/checkpoints ]; then
  echo "→ results/checkpoints → ${REMOTE}:${DAIC_DATA_DIR}/checkpoints"
  "${RSYNC_BASE[@]}" results/checkpoints/ "${REMOTE}:${DAIC_DATA_DIR}/checkpoints/"
fi
if $PUSH_RESIDUALS && [ -d results/residuals ]; then
  echo "→ results/residuals → ${REMOTE}:${DAIC_DATA_DIR}/residuals"
  "${RSYNC_BASE[@]}" results/residuals/ "${REMOTE}:${DAIC_DATA_DIR}/residuals/"
fi
if $PUSH_PRIORS && [ -d results/prior_params ]; then
  echo "→ results/prior_params → ${REMOTE}:${DAIC_DATA_DIR}/prior_params"
  "${RSYNC_BASE[@]}" results/prior_params/ "${REMOTE}:${DAIC_DATA_DIR}/prior_params/"
fi
if $PUSH_THIRDPARTY && [ -d third_party ]; then
  # third_party lives in the PROJECT dir on DAIC (it's code), not the data dir.
  # Exclude .git / __pycache__ — DAIC only needs the source files.
  echo "→ third_party → ${REMOTE}:${DAIC_PROJECT_DIR}/third_party"
  "${RSYNC_BASE[@]}" --exclude='.git' --exclude='__pycache__' \
    third_party/ "${REMOTE}:${DAIC_PROJECT_DIR}/third_party/"
fi

echo
echo "Data sync complete."
echo "Reminder: code goes via git. On DAIC:  cd ~/research-tud && git pull"
