#!/bin/bash
# Full train→eval chain for one or more datasets, one seed each. Each eval is
# chained --dependency=afterok: its training job. Submits with the project's
# standard `sbatch --export=KEY=VAL,...` convention.
#
# Training runs on `short` QoS / 4h walltime by default (faster scheduling than
# medium). Training checkpoints every epoch and resumes, so if a run hits the 4h
# wall it can be resubmitted to continue — BUT a wall-killed job is not COMPLETED,
# so its afterok eval will NOT fire; resubmit that training (it resumes) and the
# eval once it finishes. Override per submission, e.g. for a slow dataset:
#   QOS=medium TIME=12:00:00 bash scripts/daic/submit_full_pipeline.sh "pemsbay"
#
# Run on the DAIC login node, AFTER the smoke pipeline has passed:
#   cd ~/research-tud && git pull
#   bash scripts/daic/submit_full_pipeline.sh "metrla pems08f pemsbay"
#
# Eval auto-discovers the real checkpoint and ignores any leftover smoke one
# (see submit_generate_samples.sbatch), so no manual cleanup is needed.
set -euo pipefail

DATASETS="${1:?usage: submit_full_pipeline.sh \"ds1 ds2 ...\" [seed]}"
SEED="${2:-0}"
QOS="${QOS:-short}"            # training QoS
TIME="${TIME:-04:00:00}"      # training walltime cap
EVAL="${EVAL:-1}"             # set EVAL=0 to submit training only (size eval later)
EVAL_QOS="${EVAL_QOS:-short}"
EVAL_TIME="${EVAL_TIME:-04:00:00}"
K="${K:-100}"                 # eval ensemble size
N_STEPS="${N_STEPS:-100}"     # eval reverse steps  (biggest eval-time lever)

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"

# Slurm opens each job's --output file BEFORE the job script runs, so the log
# dirs must already exist at submission time (the mkdir inside the .sbatch body
# is too late for the first-ever run). Create them here on the login node.
mkdir -p logs/train-diffusion logs/eval-samples

for ds in ${DATASETS}; do
  train_id=$(sbatch --parsable \
    --job-name="diff-${ds}-s${SEED}" --qos="${QOS}" --time="${TIME}" \
    --export="DATASET=${ds},SEED=${SEED}" \
    scripts/daic/submit_train_diffusion.sbatch)
  echo "[${ds}] train : ${train_id}  (qos=${QOS}, time=${TIME})"

  if [ "${EVAL}" != "1" ]; then
    echo "[${ds}] eval  : skipped (EVAL=${EVAL}); submit it sized later"
    continue
  fi
  eval_id=$(sbatch --parsable --dependency=afterok:"${train_id}" \
    --job-name="eval-${ds}-s${SEED}" --qos="${EVAL_QOS}" --time="${EVAL_TIME}" \
    --export="DATASET=${ds},SEED=${SEED},SPLIT=test,K=${K},N_STEPS=${N_STEPS}" \
    scripts/daic/submit_generate_samples.sbatch)
  echo "[${ds}] eval  : ${eval_id}  (qos=${EVAL_QOS}, time=${EVAL_TIME}, K=${K}, n_steps=${N_STEPS})"
done

echo
echo "Submitted. Results land in results/eval/<dataset>/ as each eval finishes."
echo "Pull them to your Mac with:  bash scripts/daic/sync_from_daic.sh"
