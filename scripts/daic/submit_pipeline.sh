#!/usr/bin/env bash
# submit the backbone -> residuals chain for one dataset, with a slurm
# dependency so residual extraction starts only after the backbone finishes
# *successfully* (afterok). optionally chain diffusion after that too.
#
#   scripts/daic/submit_pipeline.sh metrla 0
#   scripts/daic/submit_pipeline.sh pemsbay 0
#   scripts/daic/submit_pipeline.sh metrla 0 --with-diffusion
#
# walltimes default from the 2-epoch smoke timings; override via env if needed:
#   BB_TIME  RES_TIME  DIFF_TIME
#
# note: if the backbone job fails, the dependent jobs are never satisfied and
# slurm cancels them - exactly what we want (no residuals off a failed model).
set -euo pipefail

DATASET="${1:?usage: submit_pipeline.sh <dataset> <seed> [--with-diffusion]}"
SEED="${2:?seed}"
WITH_DIFF="${3:-}"

# backbone walltime per dataset (pemsbay is the slowest at ~76 s/epoch).
case "$DATASET" in
  pemsbay) BB_TIME="${BB_TIME:-03:00:00}" ;;
  metrla)  BB_TIME="${BB_TIME:-01:30:00}" ;;
  *)       BB_TIME="${BB_TIME:-00:45:00}" ;;
esac
RES_TIME="${RES_TIME:-00:30:00}"
DIFF_TIME="${DIFF_TIME:-12:00:00}"

jobid_of() { awk '/Submitted batch job/{print $NF}'; }

echo "== backbone (${BB_TIME}) =="
BB_JID=$(bash scripts/daic/submit.sh backbone "$DATASET" "$SEED" --time="$BB_TIME" | tee /dev/stderr | jobid_of)
: "${BB_JID:?failed to submit backbone}"

echo "== residuals (${RES_TIME}, after backbone ${BB_JID} succeeds) =="
RES_JID=$(bash scripts/daic/submit.sh residuals "$DATASET" "$SEED" \
            --time="$RES_TIME" --dependency=afterok:"$BB_JID" | tee /dev/stderr | jobid_of)
: "${RES_JID:?failed to submit residuals}"

if [ "$WITH_DIFF" = "--with-diffusion" ]; then
  echo "== diffusion (${DIFF_TIME}, after residuals ${RES_JID} succeeds) =="
  bash scripts/daic/submit.sh diffusion "$DATASET" "$SEED" \
       --time="$DIFF_TIME" --dependency=afterok:"$RES_JID" | tee /dev/stderr | jobid_of >/dev/null
fi

echo
echo "chain submitted for ${DATASET} seed ${SEED}. watch with: squeue --me"
echo "(the residual job sits in PENDING with reason (Dependency) until the backbone finishes)"
