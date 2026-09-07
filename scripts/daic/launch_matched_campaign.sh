#!/usr/bin/env bash
# Launch the two-arm matched-prior campaign on DAIC from the Mac, end to end:
#   1. the cluster checkout pulls the pushed commit
#   2. the six GWN residual caches (+ backbone models) and data/processed are
#      rsync'd onto windowshome (wiped 2026-09-02; 3.0 GB + 167 MB, fits the 8 GB share)
#   3. the slurm log dir is created (must exist before the first sbatch)
#   4. histcond.sh submits 6 datasets x {iso, kron} x 4 seeds = 48 cells / 24 jobs
#
#   scripts/daic/launch_matched_campaign.sh            # do it
#   DRY=1 scripts/daic/launch_matched_campaign.sh      # print the plan, submit nothing
#
# idempotent: rsync only copies what is missing/changed; histcond.sh skips cells whose
# cache is absent instead of half-running them.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/env.sh"
EXP="${EXP:-20260907_matched}"
SEEDS="${SEEDS:-42 1 2 3}"
DATASETS="${DATASETS:-pems08f pems08s pems04f pems04s metrla pemsbay}"
ARCH="${ARCH:-$HERE/../../archive/20260902/results}"      # the Mac's only copy of the caches
REMOTE="${DAIC_NETID}@${DAIC_LOGIN_HOST}"
DRY="${DRY:-0}"
run() { if [ "$DRY" = 1 ]; then echo "[dry] $*"; else "$@"; fi; }

echo "== 0. preflight: the /tudelft.net shares need a live kerberos ticket =="
# an expired ticket makes the project dir 'Permission denied' and every later step
# fails in ways that are easy to miss. check it first and say what to do.
if ! ssh "$REMOTE" "test -r '${DAIC_PROJECT_DIR}/.git/HEAD' && test -w '${DAIC_DATA_DIR}'" 2>/dev/null; then
  echo "cannot read ${DAIC_PROJECT_DIR} / write ${DAIC_DATA_DIR} on ${DAIC_LOGIN_HOST}." >&2
  echo "the kerberos ticket is probably expired: run   ssh -t ${REMOTE} kinit   and retry." >&2
  exit 3
fi

echo "== 1. cluster checkout pulls =="
run ssh "$REMOTE" "cd '${DAIC_PROJECT_DIR}' && rm -f .git/index.lock && git pull --ff-only && git log --oneline -1"

echo "== 2. residual caches + processed data -> windowshome =="
for ds in $DATASETS; do
  [ -d "$ARCH/residuals/$ds/gwn/seed42" ] || { echo "missing local cache $ARCH/residuals/$ds/gwn/seed42" >&2; exit 2; }
  run rsync -rlpt --no-perms --info=progress2 "$ARCH/residuals/$ds/gwn/seed42/" \
      "$REMOTE:${DAIC_DATA_DIR}/residuals/$ds/gwn/seed42/"
done
run rsync -rlpt --no-perms --info=progress2 "$HERE/../../data/processed/" "$REMOTE:${DAIC_DATA_DIR}/processed/"

echo "== 3. log dir =="
run ssh "$REMOTE" "mkdir -p '${DAIC_OUT_DIR:-/home/${DAIC_NETID}/windowshome/research-tud-out}/logs/${EXP}'"

echo "== 4. submit =="
run ssh "$REMOTE" "cd '${DAIC_PROJECT_DIR}' && source scripts/daic/activate.sh >/dev/null 2>&1; \
  EXP='${EXP}' ARMS='iso kron' BACKBONES=gwn SEEDS='${SEEDS}' CONDS=hist DATASETS='${DATASETS}' DRY='${DRY}' \
  scripts/daic/histcond.sh"
echo "done. watch with: ssh $REMOTE squeue -u ${DAIC_NETID}"
