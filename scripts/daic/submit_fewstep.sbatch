#!/bin/bash
#SBATCH --job-name=fewstep
#SBATCH --partition=all
#SBATCH --gres=gpu:nvidia_rtx_pro_6000:1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/fewstep_%j.out
#
# FEW-STEP PROBE: ancestral sampling at 2/5/10/25 steps, kron vs iso, on the saved
# best_crps.pt checkpoints — the regime where the structured prior MUST show if it
# helps at all (GAD Fig.3 / GiFlow Thm.1: the gap opens as steps shrink).
#
#   sbatch scripts/daic/submit_fewstep.sbatch                                   # sakin's 3 ds
#   DATASETS="pemsbay pems04f pems04s" sbatch scripts/daic/submit_fewstep.sbatch  # denis's 3
#
# writes val_metrics_bestcrps_ancestral<steps>.json into each arm's eval/ dir.
set -uo pipefail
export PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT_DIR"
export STAGE=eval
source "${PROJECT_DIR}/scripts/daic/_common.sh"
export DATA_ROOT="${DAIC_DATA_DIR}" RESULTS_ROOT="${DAIC_DATA_DIR}"
export OUTPUT_ROOT="${HOME}/windowshome/research-tud-out"

DATASETS="${DATASETS:-pems08s pems08f metrla}"
STEPS="${STEPS:-2 5 10 25}"
EXP_ROOT="${EXP_ROOT:-20260715_gadst_res}"
BACKBONE="${BACKBONE:-gwn}"
ARMS="${ARMS:-kron_res:kronecker iso_res:isotropic}"

for ds in ${DATASETS}; do
  for arm in ${ARMS}; do
    pr=${arm#*:}; a=${arm%:*}
    for st in ${STEPS}; do
      echo "=== fewstep: ${ds} ${a} ancestral@${st} ==="
      srun --unbuffered "${DAIC_PYTHON}" -u scripts/eval_diffusion.py \
        --dataset "${ds}" --prior "${pr}" --backbone "${BACKBONE}" --exp "${EXP_ROOT}/${a}" --seed 42 \
        --split val --k 8 --sampler ancestral --n-steps "${st}" --ckpt best_crps.pt \
        --max-windows 256 --gen-batch 256 --w1-samples 4 \
        || echo "WARN: cell failed (${ds} ${a} ${st}) — continuing"
    done
  done
done
echo "fewstep sweep complete"
