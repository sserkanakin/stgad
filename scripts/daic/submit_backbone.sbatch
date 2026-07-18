#!/bin/bash
#SBATCH --job-name=backbone
# one array TASK = one (variant, backbone) cell: train the point-forecast backbone
# to convergence, then extract residuals over train/val/test. GWN (~309k) is the
# heavy case; mlp/gcn are minutes. 12h is generous headroom on an A40.
#SBATCH --time=12:00:00
# partition / qos / gres + --array + --output are injected by submit_backbone.sh.
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --mail-user=sserkanakin@icloud.com
#SBATCH --mail-type=END,FAIL
#
# reads the SLURM_ARRAY_TASK_ID-th row of GRID_FILE (idx<TAB>variant<TAB>backbone).
# submit via scripts/daic/submit_backbone.sh (never sbatch this file directly).

set -euo pipefail
export PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
source "${PROJECT_DIR}/scripts/daic/_common.sh"

: "${GRID_FILE:?Set GRID_FILE via --export}"
: "${SEED:?Set SEED via --export}"
: "${EXP:?Set EXP via --export}"
EPOCHS="${EPOCHS:-100}"
# DATA_ROOT/RESULTS_ROOT/OUTPUT_ROOT come through --export=ALL and are read by src/utils/paths.py.

row="$(awk -F'\t' -v i="${SLURM_ARRAY_TASK_ID}" '$1==i {print; exit}' "${GRID_FILE}")"
VARIANT="$(printf '%s' "${row}" | cut -f2)"
BACKBONE="$(printf '%s' "${row}" | cut -f3)"
: "${VARIANT:?empty variant for task ${SLURM_ARRAY_TASK_ID}}"

CELL="${RESULTS_ROOT:-<repo>}/residuals/${VARIANT}/${BACKBONE}/seed${SEED}"
echo "===== ${VARIANT} / ${BACKBONE} / seed${SEED}  EXP=${EXP}  (task ${SLURM_ARRAY_TASK_ID}) ====="
echo "  DATA_ROOT   = ${DATA_ROOT:-<repo>/data}"
echo "  RESULTS_ROOT= ${RESULTS_ROOT:-<repo>/results}   (residuals + model here)"
echo "  OUTPUT_ROOT = ${OUTPUT_ROOT:-<repo>/results}    (logs here)"
echo "  cell        = ${CELL}/"
echo "                  {train,val,test}.npz + model/{best.pt,config.json,history.json}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

# 1) train the backbone (best.pt by val MAE)
"${DAIC_PYTHON}" scripts/train_backbone.py \
    --variant "${VARIANT}" --backbone "${BACKBONE}" --seed "${SEED}" \
    --epochs "${EPOCHS}" --device cuda

# 2) extract residuals -> RESULTS_ROOT/residuals/<variant>/<backbone>/seed<seed>/
"${DAIC_PYTHON}" scripts/extract_residuals.py \
    --variant "${VARIANT}" --backbone "${BACKBONE}" --seed "${SEED}" --device cuda

echo "===== done ${VARIANT} / ${BACKBONE} / seed${SEED} ====="
