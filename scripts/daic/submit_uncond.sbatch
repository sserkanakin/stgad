#!/bin/bash
#SBATCH --job-name=uncond
#SBATCH --partition=all
#SBATCH --gres=gpu:nvidia_rtx_pro_6000:1
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/uncond_%j.out
#
# UNCONDITIONAL GAD-ST DEMONSTRATION (GAD's published winning setting, one axis up):
# generate centered raw spatio-temporal windows with a low-capacity GCNN, kron
# (cartesian product-graph prior) vs iso, then score with GAD's own metric suite
# (aMMD over TV_S/SC_S/TV_T) at ancestral steps 2/5/10/25/50.
#
#   sbatch scripts/daic/submit_uncond.sbatch                      # metrla
#   DATASETS="metrla pems08s pems08f" sbatch scripts/daic/submit_uncond.sbatch
set -uo pipefail
export PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT_DIR"
export STAGE=diffusion
source "${PROJECT_DIR}/scripts/daic/_common.sh"
export DATA_ROOT="${DAIC_DATA_DIR}" RESULTS_ROOT="${DAIC_DATA_DIR}"
export OUTPUT_ROOT="${HOME}/windowshome/research-tud-out"
export WANDB_MODE=disabled

DATASETS="${DATASETS:-metrla}"
EPOCHS="${EPOCHS:-30}"
EXP_ROOT="${EXP_ROOT:-20260716_uncond}"

for ds in ${DATASETS}; do
  for arm in kron_u:kronecker iso_u:isotropic; do
    pr=${arm#*:}; a=${arm%:*}
    echo "##### training ${ds} ${a} (${pr}) #####"
    srun --unbuffered "${DAIC_PYTHON}" -u scripts/train_diffusion.py \
      --dataset "${ds}" --prior "${pr}" --backbone raw \
      --temporal cartesian --denoiser gcnn --uncond \
      --exp "${EXP_ROOT}/${a}" --seed 42 --epochs "${EPOCHS}" \
      --fast-val-every 0 --val-every 1000 --patience 999 \
      || echo "WARN: training failed (${ds} ${a})"
  done
  echo "##### evaluating ${ds} #####"
  srun --unbuffered "${DAIC_PYTHON}" -u scripts/eval_uncond_gadst.py \
    --dataset "${ds}" --exp "${EXP_ROOT}" \
    --arms kron_u:kronecker iso_u:isotropic --steps 2 5 10 25 50 --n-samples 512 \
    || echo "WARN: eval failed (${ds})"
done
echo "##### UNCOND EXPERIMENT COMPLETE #####"
