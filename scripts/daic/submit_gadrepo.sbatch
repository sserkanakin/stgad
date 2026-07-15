#!/bin/bash
#SBATCH --job-name=gadrepo
#SBATCH --partition=all
#SBATCH --gres=gpu:nvidia_rtx_pro_6000:1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=logs/gadrepo_%j.out
#
# GAD'S OWN CODE, UNMODIFIED LOGIC, ON OUR SPATIO-TEMPORAL WINDOWS:
# runs exp_gasde / exp_vpsde / exp_vesde (train) + their *_test_only_EM scripts
# on the exported product-graph data (scripts/export_gad_repo_data.py output).
# only the hardcoded PATHS are rewritten, on COPIES (originals untouched).
#
#   DATASET=pems08f sbatch scripts/daic/submit_gadrepo.sbatch
#
# prerequisites (once, on the login node):
#   git clone https://github.com/vimalkb7/gad third_party/gad
#   source scripts/daic/activate.sh
#   python scripts/export_gad_repo_data.py --dataset pems08f --out third_party/gadst_data/pems08f
set -uo pipefail
export PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT_DIR"
export STAGE=eval
source "${PROJECT_DIR}/scripts/daic/_common.sh"
export OUTPUT_ROOT="${HOME}/windowshome/research-tud-out"

DATASET="${DATASET:-pems08f}"
GADDIR="${PROJECT_DIR}/third_party/gad"
DATA="${PROJECT_DIR}/third_party/gadst_data/${DATASET}"
RES="${OUTPUT_ROOT}/20260716_gadrepo/${DATASET}"
[ -d "$GADDIR" ] || { echo "ERROR: clone https://github.com/vimalkb7/gad to third_party/gad first"; exit 1; }
[ -f "$DATA/x_train.npy" ] || { echo "ERROR: export data first (scripts/export_gad_repo_data.py --dataset ${DATASET})"; exit 1; }
mkdir -p "$RES"/{GASDE,VPSDE,VESDE}/Exp{1,2,3}

# patched COPIES inside the repo dir (so their models/sde/signal_metrics imports work).
for f in exp_gasde exp_vpsde exp_vesde exp_gasde_test_only_Em exp_vpsde_test_only_EM exp_vesde_test_only_EM; do
  src="$GADDIR/$f.py"
  [ -f "$src" ] || { echo "WARN: $f.py not in repo, skipping"; continue; }
  sed -e "s|/home/nfs/vkumarasamybal/Code3/Dataset_Processed/Exp[0-9]*|${DATA}|g" \
      -e "s|/home/nfs/vkumarasamybal/Code3/Dataset|${DATA}|g" \
      -e "s|/tudelft.net/staff-bulk/ewi/insy/MMC/vimal/Results2|${RES}|g" \
      "$src" > "$GADDIR/${f}_ours_${DATASET}.py"
done
echo "path check (should all point at ${DATA} / ${RES}):"
grep -hE "np\.load|RESULTS_DIR *=" "$GADDIR"/exp_*_ours_${DATASET}.py | sort -u | head -12

cd "$GADDIR"
for m in gasde vpsde vesde; do
  echo "##### TRAIN ${m} on ${DATASET} #####"
  srun --unbuffered "${DAIC_PYTHON}" -u "exp_${m}_ours_${DATASET}.py" || { echo "WARN: train ${m} failed"; continue; }
  t="exp_${m}_test_only_Em_ours_${DATASET}.py"; [ -f "$t" ] || t="exp_${m}_test_only_EM_ours_${DATASET}.py"
  if [ -f "$t" ]; then
    echo "##### TEST ${m} on ${DATASET} #####"
    srun --unbuffered "${DAIC_PYTHON}" -u "$t" || echo "WARN: test ${m} failed"
  fi
done
echo "##### GADREPO COMPLETE (${DATASET}) #####"
