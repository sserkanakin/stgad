#!/bin/bash
#SBATCH --job-name=residuals
#SBATCH --time=00:30:00
# partition / qos / gres are injected per-cluster by submit.sh.
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
# residual extraction holds a whole split's predictions in memory before saving;
# pems-bay peaks ~9g, so 16g leaves headroom. any gpu is fine — short inference.
#SBATCH --output=logs/residuals/%x-%j.out
#SBATCH --error=logs/residuals/%x-%j.err
#SBATCH --mail-user=sserkanakin@icloud.com
#SBATCH --mail-type=END,FAIL
#
# Dump train/val/test residuals from a trained backbone checkpoint via the repo
# model (scripts/gwn_extract_residuals.py). Flags MUST match the backbone.
#   sbatch --export=DATASET=metrla,SEED=0,BACKBONE=gwn \
#          scripts/daic/submit_extract_residuals.sbatch

set -euo pipefail
mkdir -p logs/residuals
export PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
source "${PROJECT_DIR}/scripts/daic/_common.sh"

: "${DATASET:?Set DATASET=metrla|pemsbay|pems04f|pems04s|pems08f|pems08s}"
: "${SEED:?Set SEED=<int>}"
BACKBONE="${BACKBONE:-gwn}"

# null_val + backbone flags must match the checkpoint's training config.
case "${DATASET}" in metrla|pemsbay) NULL_VAL=0.0 ;; *) NULL_VAL=nan ;; esac
case "${BACKBONE}" in
  gwn) RES_FLAGS=(--adjtype doubletransition --gcn_order 2) ;;      # gcn_bool+addaptadj default
  gcn) RES_FLAGS=(--adjtype symnadj --gcn_order 1 --no_addaptadj) ;;
  mlp) RES_FLAGS=(--adjtype identity --no_gcn --no_addaptadj) ;;
  *) echo "unknown BACKBONE=${BACKBONE}" >&2; exit 2 ;;
esac

CKPT="results/backbone/${DATASET}/${BACKBONE}/seed${SEED}/best.pth"
if [ ! -f "${CKPT}" ]; then
  echo "ERROR: checkpoint not found: ${CKPT} (run the backbone stage first)" >&2
  exit 1
fi

export STAGE="residuals"
RUN_NAME="residuals_${BACKBONE}_${DATASET}_seed${SEED}"
scontrol update job="${SLURM_JOB_ID}" JobName="${RUN_NAME}" 2>/dev/null || true

log_run_header "${DAIC_PYTHON} scripts/gwn_extract_residuals.py --variant ${DATASET} --seed ${SEED} --checkpoint ${CKPT} --null_val ${NULL_VAL} ${RES_FLAGS[*]} (backbone=${BACKBONE})"

srun --export=ALL "${DAIC_PYTHON}" scripts/gwn_extract_residuals.py \
    --variant "${DATASET}" \
    --seed "${SEED}" \
    --checkpoint "${CKPT}" \
    --null_val "${NULL_VAL}" \
    "${RES_FLAGS[@]}"
