#!/usr/bin/env bash
# The matched-prior campaign, FROM RAW, on DAIC. No archived caches are ever uploaded:
# the only data that leaves the Mac is data/raw (the source files). Everything
# derived is rebuilt fresh: processed splits on the MAC (cpu), backbones / residuals /
# diffusion runs on the cluster (gpu), under exactly these windowshome folders:
#     research-tud-data/{raw,processed,residuals}     research-tud-out/<exp>, logs/
#
#   scripts/daic/launch_matched_campaign.sh raw        # 1. upload data/raw (once)
#   scripts/daic/launch_matched_campaign.sh build      # 2. build processed/ from raw ON THE MAC (cpu), upload it
#   scripts/daic/launch_matched_campaign.sh backbones  # 3. train 6 GWN backbones + extract residuals (slurm)
#   scripts/daic/launch_matched_campaign.sh diffusion  # 4. 6 x {iso,kron} x 4 seeds, once residuals exist (slurm)
#   scripts/daic/launch_matched_campaign.sh status     # queue + what exists on windowshome
# DRY=1 prints instead of doing, for the slurm stages.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/env.sh"
STAGE="${1:?stage: raw | build | backbones | diffusion | status}"
EXP="${EXP:-20260907_matched}"
BB_EXP="${BB_EXP:-20260907_backbones}"
SEEDS="${SEEDS:-42 1 2 3}"
DATASETS="${DATASETS:-pems08f pems08s pems04f pems04s metrla pemsbay}"
REMOTE="${DAIC_NETID}@${DAIC_LOGIN_HOST}"
OUT="${DAIC_OUT_DIR:-/home/${DAIC_NETID}/windowshome/research-tud-out}"
VENV_PY="$(dirname "${DAIC_PROJECT_DIR}")/.venvs/thesis-diffusion/bin/python"
DRY="${DRY:-0}"
SSH=(ssh -o ControlPath=none -o ConnectTimeout=20 "$REMOTE")

# the /tudelft.net shares need a live kerberos ticket; an expired one shows as
# 'Permission denied' and everything after fails in ways a filtered log hides.
"${SSH[@]}" "test -r '${DAIC_PROJECT_DIR}/.git/HEAD' && test -w '${DAIC_DATA_DIR}'" \
  || { echo "cannot access the shares on ${DAIC_LOGIN_HOST}: run  ssh -t ${REMOTE} kinit  and retry" >&2; exit 3; }

case "$STAGE" in
  raw)
    for src in metrla pemsbay pems04 pems08; do        # molene: built but never used
      rsync -rlpt --no-perms --stats --rsync-path="mkdir -p ${DAIC_DATA_DIR}/raw/$src && rsync" \
        "${HERE}/../../data/raw/$src/" "$REMOTE:${DAIC_DATA_DIR}/raw/$src/" | grep -E 'Total transferred'
    done ;;
  build)
    # cpu work stays on the mac; the cluster's gpu time is for training only
    ( cd "${HERE}/../.." && .venv/bin/python scripts/build_datasets.py --all && .venv/bin/python scripts/check_connectivity.py | tail -8 )
    rsync -rlpt --no-perms --stats --exclude='molene' --exclude='*/gwn_native' \
      "${HERE}/../../data/processed/" "$REMOTE:${DAIC_DATA_DIR}/processed/" | grep -E 'Total transferred'
    "${SSH[@]}" "du -sh '${DAIC_DATA_DIR}/processed'" ;;
  backbones)
    "${SSH[@]}" "cd '${DAIC_PROJECT_DIR}' && source scripts/daic/activate.sh >/dev/null 2>&1; \
      EXP='${BB_EXP}' BACKBONES=gwn VARIANTS='${DATASETS}' DRY='${DRY}' scripts/daic/submit_backbone.sh" ;;
  diffusion)
    "${SSH[@]}" "mkdir -p '${OUT}/logs/${EXP}' && cd '${DAIC_PROJECT_DIR}' && source scripts/daic/activate.sh >/dev/null 2>&1; \
      EXP='${EXP}' ARMS='iso kron' BACKBONES=gwn SEEDS='${SEEDS}' CONDS=hist DATASETS='${DATASETS}' DRY='${DRY}' \
      scripts/daic/histcond.sh" ;;
  status)
    "${SSH[@]}" "squeue -u ${DAIC_NETID}; echo; du -sh ${DAIC_DATA_DIR}/* ${OUT}/* 2>/dev/null" ;;
  *) echo "unknown stage $STAGE" >&2; exit 2 ;;
esac
