#!/usr/bin/env bash
# re-run the 6-cell grid (3 datasets x {kronecker, isotropic}) with the
# NORMALIZED pipeline, and chain each test eval to afterok of its train job:
# eval only starts if its training finished cleanly, and reuses the same gpu.
#
#   bash scripts/daic/submit_norm_rerun.sh
#
# knobs (env or edit below): SEED, EPOCHS, PATIENCE.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${HERE}/env.sh"          # cluster partition / qos / gres

SEED="${SEED:-0}"
EPOCHS="${EPOCHS:-500}"
PATIENCE="${PATIENCE:-25}"
# lightweight test-eval config (the one we validated: no OOM, sane runtime)
EVAL_COMMON="--seed ${SEED} --k 30 --n-steps 50 --max-windows 2000 --gen-batch 1024 --w1-samples 4"

mkdir -p logs/diffusion logs/eval

for DS in metrla pemsbay pems08f; do
  for PR in kronecker isotropic; do
    echo "=== ${DS} / ${PR} ==="

    # train. submit.sh injects partition/qos/gres + names the job; --parsable
    # makes sbatch print just the jobid (set -x trace goes to stderr).
    TID=$(EPOCHS="${EPOCHS}" PRIOR="${PR}" EXTRA_FLAGS="--patience ${PATIENCE}" \
          bash "${HERE}/submit.sh" diffusion "${DS}" "${SEED}" --parsable 2>/dev/null)
    TID="${TID%%;*}"                       # strip ';cluster' if federated
    echo "  train job ${TID}  (epochs=${EPOCHS}, patience=${PATIENCE})"

    # eval. raw sbatch (doesn't go through submit.sh) so pass resources here;
    # afterok => waits for THIS training job to succeed, then runs on a gpu.
    EID=$(EVAL_ARGS="--dataset ${DS} --prior ${PR} ${EVAL_COMMON}" \
          sbatch --parsable --dependency="afterok:${TID}" \
                 --job-name="eval_${DS}_${PR}_seed${SEED}" \
                 --partition="${DAIC_PARTITION}" --gres="${DAIC_GRES_GPU}" \
                 ${DAIC_QOS_SHORT:+--qos="${DAIC_QOS_SHORT}"} \
                 "${HERE}/submit_eval.sbatch")
    EID="${EID%%;*}"
    echo "  eval  job ${EID}  (afterok:${TID})"
  done
done

echo
echo "submitted 6 train + 6 eval. each eval is pending until its train finishes."
echo "watch:  squeue -u \$USER --format='%.10i %.30j %.2t %.12M %R'"
