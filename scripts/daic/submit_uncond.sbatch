#!/bin/bash
#SBATCH --job-name=uncond_eval
#SBATCH --partition=all
#SBATCH --gres=gpu:nvidia_rtx_pro_6000:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/uncond_eval_%j.out
#
# GAD-metric evaluation of the UNCONDITIONAL runs (aMMD over TV_S/SC_S/TV_T at
# ancestral steps 2..50, kron_u vs iso_u). submit AFTER the training array:
#   DATASETS="metrla pems08f pems08s" sbatch --dependency=afterok:<TRAIN_ID> \
#     scripts/daic/submit_uncond.sbatch
set -uo pipefail
export PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT_DIR"
export STAGE=eval
source "${PROJECT_DIR}/scripts/daic/_common.sh"
export DATA_ROOT="${DAIC_DATA_DIR}" RESULTS_ROOT="${DAIC_DATA_DIR}"
export OUTPUT_ROOT="${HOME}/windowshome/research-tud-out"
DATASETS="${DATASETS:-metrla}"
EXP_ROOT="${EXP_ROOT:-20260716_uncond}"
SAMPLER="${SAMPLER:-ancestral}"
STEPS="${STEPS:-2 5 10 25 50}"
for ds in ${DATASETS}; do
  echo "##### evaluating ${ds} #####"
  srun --unbuffered "${DAIC_PYTHON}" -u scripts/eval_uncond_gadst.py \
    --dataset "${ds}" --exp "${EXP_ROOT}" \
    --arms kron_u:kronecker iso_u:isotropic --sampler "${SAMPLER}" --steps ${STEPS} --n-samples 512 \
    || echo "WARN: eval failed (${ds})"
done
echo "##### UNCOND EVAL COMPLETE #####"
