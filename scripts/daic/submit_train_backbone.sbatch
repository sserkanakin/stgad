#!/bin/bash
#SBATCH --job-name=backbone
# sized for a full ~100-epoch run; override with --time for a quick test.
#SBATCH --time=02:00:00
# partition / qos / gres are injected per-cluster by submit.sh.
#SBATCH --cpus-per-task=2
# util.load_dataset holds the WHOLE dataset in host RAM; pemsbay (N=325, 36k
# windows) needs ~16G+, so 8G OOM-kills it. 32G is safe for all variants.
#SBATCH --mem=32G
#SBATCH --output=logs/backbone/%x-%j.out
#SBATCH --error=logs/backbone/%x-%j.err
#SBATCH --mail-user=sserkanakin@icloud.com
#SBATCH --mail-type=END,FAIL
#
# Train a backbone (GWN / GCN / MLP) on one dataset/seed, DIRECTLY via the
# vendored repo's train.py on the gwn_native view (no project wrapper).
#   sbatch --export=DATASET=metrla,SEED=0,BACKBONE=gwn \
#          scripts/daic/submit_train_backbone.sbatch
# BACKBONE: gwn (default) | gcn | mlp. The best checkpoint is copied to
# results/backbone/<dataset>/<backbone>/seed<seed>/best.pth for the residual step.

set -euo pipefail
mkdir -p logs/backbone
export PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
source "${PROJECT_DIR}/scripts/daic/_common.sh"
export MPLBACKEND=Agg   # train.py imports matplotlib; headless node has no display

: "${DATASET:?Set DATASET=metrla|pemsbay|pems04f|pems04s|pems08f|pems08s}"
: "${SEED:?Set SEED=<int> via --export}"
BACKBONE="${BACKBONE:-gwn}"
EPOCHS="${EPOCHS:-100}"
EXP="${EXP:-dev}"

case "${DATASET}" in
  metrla) N=207 ;; pemsbay) N=325 ;;
  pems04f|pems04s) N=307 ;; pems08f|pems08s) N=170 ;;
  *) echo "unknown DATASET=${DATASET}" >&2; exit 2 ;;
esac
# null_val: speed datasets mask raw 0; PEMS use nan (real flow zeros kept).
case "${DATASET}" in metrla|pemsbay) NULL_VAL=0.0 ;; *) NULL_VAL=nan ;; esac
# backbone flags (see THIRD_PARTY_PATCHES.md cheatsheet).
case "${BACKBONE}" in
  gwn) BB_FLAGS=(--gcn_bool --addaptadj --adjtype doubletransition --gcn_order 2) ;;
  gcn) BB_FLAGS=(--gcn_bool --adjtype symnadj --gcn_order 1) ;;
  mlp) BB_FLAGS=(--adjtype identity) ;;
  *) echo "unknown BACKBONE=${BACKBONE}" >&2; exit 2 ;;
esac

NAT="data/processed/${DATASET}/gwn_native"
OUT="results/backbone/${DATASET}/${BACKBONE}/seed${SEED}"
mkdir -p "${OUT}"
SAVE="${OUT}/ckpt"

export STAGE="backbone"
RUN_NAME="backbone_${BACKBONE}_${DATASET}_seed${SEED}"
scontrol update job="${SLURM_JOB_ID}" JobName="${RUN_NAME}" 2>/dev/null || true

log_run_header "${DAIC_PYTHON} third_party/Graph-WaveNet/train.py --data ${NAT} --num_nodes ${N} --null_val ${NULL_VAL} ${BB_FLAGS[*]} (backbone=${BACKBONE})"

srun --export=ALL "${DAIC_PYTHON}" third_party/Graph-WaveNet/train.py \
    --data "${NAT}" \
    --adjdata "${NAT}/adj_mx.pkl" \
    --num_nodes "${N}" \
    --in_dim 2 --seq_length 12 \
    --device cuda:0 \
    --epochs "${EPOCHS}" \
    --save "${SAVE}" \
    --expid "${SEED}" \
    --null_val "${NULL_VAL}" \
    "${BB_FLAGS[@]}"

# repo names the best ckpt "${SAVE}_exp<seed>_best_<loss>.pth"; copy to a stable
# best.pth so submit_extract_residuals.sbatch can find it deterministically.
BEST="$(ls -t "${SAVE}"_exp"${SEED}"_best_*.pth 2>/dev/null | head -1 || true)"
if [ -n "${BEST}" ]; then
  cp -f "${BEST}" "${OUT}/best.pth"
  rm -f "${OUT}"/ckpt_*.pth     # drop the raw repo checkpoints; keep best.pth (quota)
  echo "best checkpoint -> ${OUT}/best.pth  (from ${BEST})"
else
  echo "ERROR: no best checkpoint at ${SAVE}_exp${SEED}_best_*.pth" >&2
  exit 1
fi
