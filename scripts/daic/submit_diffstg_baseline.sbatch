#!/bin/bash
#SBATCH --job-name=diffstg
# faithful DiffSTG (bs=8 + 200-step DDPM val eval every epoch) is slow; 20h is
# generous. best.pt is checkpointed on every val improvement, but the TEST-set
# metrics (what we harvest) are only computed AFTER training finishes / early-stops,
# so the run must complete within walltime. pemsbay is the heavy one -- override
# with TIME=... in submit_diffstg.sh or --time=HH:MM:SS on sbatch if the qos allows.
#SBATCH --time=20:00:00
# partition / qos / gres are injected per-cluster by submit_diffstg.sh.
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/diffstg/%x-%j.out
#SBATCH --error=logs/diffstg/%x-%j.err
#SBATCH --mail-user=sserkanakin@icloud.com
#SBATCH --mail-type=END,FAIL
#
# Train the ORIGINAL DiffSTG (third_party/DiffSTG/train.py) on one of our datasets,
# faithful to their paper/code -- their loader, 60/20/20 split, global z-score,
# DDPM-200 training, DDIM-multi 40-step test sampling, their MAE/RMSE/MAPE/CRPS/MIS.
# Our only adaptations are the data files + four runtime patches (see THIRD_PARTY_PATCHES.md).
#
#   sbatch --export=ALL,DATASET=pems08f,SEED=43 scripts/daic/submit_diffstg_baseline.sbatch
#
# required: DATASET. optional: SEED (43), EXP (dev), and hyperparam overrides
# N / SAMPLE_STEPS / HIDDEN / NSAMPLES / BATCH_SIZE / LR / BETA_END
# (defaults = DiffSTG's released CLI defaults -- "the same way they run it").

set -euo pipefail
mkdir -p logs/diffstg
export PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
source "${PROJECT_DIR}/scripts/daic/_common.sh"

: "${DATASET:?Set DATASET=metrla|pemsbay|pems04f|pems04s|pems08f|pems08s via --export}"
SEED="${SEED:-43}"
EXP="${EXP:-dev}"
# DiffSTG released-code (argparse) defaults -- keep faithful; override only to
# trade fidelity for walltime (e.g. BATCH_SIZE=32 speeds training, not val eval).
N="${N:-200}"
SAMPLE_STEPS="${SAMPLE_STEPS:-200}"
HIDDEN="${HIDDEN:-32}"
NSAMPLES="${NSAMPLES:-8}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LR="${LR:-0.002}"
BETA_END="${BETA_END:-0.1}"

export STAGE=diffstg
RUN_NAME="diffstg_${DATASET}_seed${SEED}"
scontrol update job="${SLURM_JOB_ID}" JobName="${RUN_NAME}" 2>/dev/null || true

# Keep DiffSTG's data files AND its output/ (checkpoints, forecasts, metrics) OFF the
# tight linuxhome/code quota (~8G, mostly the torch venv): symlink both onto the roomy
# data share (windowshome), same idea as _common.sh's _relink. The build writes here and
# train.py reads/writes here, so nothing large ever lands on linuxhome. First run also
# clears any real dir left there (e.g. a partial/aborted rsync of the data files).
_relink_to_share() {  # $1 = repo path under third_party/DiffSTG, $2 = share subdir
  local repo_path="${PROJECT_DIR}/third_party/DiffSTG/$1" share="${DAIC_DATA_DIR}/diffstg/$2"
  mkdir -p "${share}" "$(dirname "${repo_path}")"
  if [ ! -L "${repo_path}" ]; then
    rm -rf "${repo_path}"
    ln -sfn "${share}" "${repo_path}"
  fi
}
_relink_to_share data/dataset data/dataset
_relink_to_share output       output

# deps: easydict is required by DiffSTG (tiny, pure-python); nni is stubbed in train.py.
python -c "import easydict" 2>/dev/null || pip install --quiet easydict

# build the raw-series files DiffSTG's loader expects, if not already present on the share.
DS_DIR="${PROJECT_DIR}/third_party/DiffSTG/data/dataset/${DATASET}"
if [ ! -f "${DS_DIR}/flow.npy" ] || [ ! -f "${DS_DIR}/adj.npy" ]; then
  python -c "import h5py" 2>/dev/null || pip install --quiet h5py
  python scripts/baselines/build_diffstg_data.py "${DATASET}"
fi

echo "─── diffstg baseline ${DATASET} seed ${SEED} "\
"(N=${N} steps=${SAMPLE_STEPS} hidden=${HIDDEN} n_samples=${NSAMPLES} bs=${BATCH_SIZE} lr=${LR}) ───"

# resource profiling: sample GPU every 10s so we can right-size future runs.
prof_csv="logs/diffstg/profile_${RUN_NAME}_${SLURM_JOB_ID}.csv"
nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader -l 10 > "${prof_csv}" 2>/dev/null &
PROF_PID=$!
trap 'kill ${PROF_PID} 2>/dev/null || true' EXIT

# run their training/eval from inside their repo so `utils.*` imports and the
# output/ paths (ws = the DiffSTG repo root) resolve correctly.
cd "${PROJECT_DIR}/third_party/DiffSTG"
set +e
"${DAIC_PYTHON}" train.py \
    --data "${DATASET}" \
    --seed "${SEED}" \
    --N "${N}" \
    --sample_steps "${SAMPLE_STEPS}" \
    --hidden_size "${HIDDEN}" \
    --n_samples "${NSAMPLES}" \
    --batch_size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --beta_end "${BETA_END}"
RC=$?
set -e
cd "${PROJECT_DIR}"

# harvest into the data share (DiffSTG's own output/ lives on the tight repo quota).
# DiffSTG.csv accumulates one row per run; the forecast pkl holds samples+targets
# for the first 50 test windows (for CRPS/calibration reuse if we want it later).
OUT="${DAIC_DATA_DIR}/diffstg/${EXP}"
mkdir -p "${OUT}"
SRC_CSV="third_party/DiffSTG/output/metrics/DiffSTG.csv"
if [ -f "${SRC_CSV}" ]; then
  # full accumulator (all datasets append here; jobs finish hours apart so appends
  # don't collide). last writer wins on the copy -- all copies hold every row.
  cp -f "${SRC_CSV}" "${OUT}/DiffSTG.csv"
  # race-proof per-run file: header + only this dataset's latest row (grep by name,
  # skips any stray header line, immune to concurrent appends).
  head -1 "${SRC_CSV}" > "${OUT}/${DATASET}_seed${SEED}.csv"
  awk -F, -v d="${DATASET}" '$1==d' "${SRC_CSV}" | tail -1 >> "${OUT}/${DATASET}_seed${SEED}.csv"
fi
cp -f third_party/DiffSTG/output/forecast/*"${DATASET}"*"${SEED}".pkl "${OUT}/" 2>/dev/null || true
echo "harvested -> ${OUT}/  (${DATASET}_seed${SEED}.csv + forecast pkl)"

kill ${PROF_PID} 2>/dev/null || true
echo "===================== resource profile ====================="
awk -F',' '{gsub(/ ?MiB/,"",$2); gsub(/ ?%/,"",$4); if($2+0>m)m=$2+0; if($4+0>u)u=$4+0}
    END{printf "peak GPU mem used: %d MiB | peak GPU util: %d%%\n", m, u}' "${prof_csv}" 2>/dev/null || true
echo "  seff ${SLURM_JOB_ID}   # CPU/RAM/walltime after the job ends"
echo "============================================================"
exit ${RC}
