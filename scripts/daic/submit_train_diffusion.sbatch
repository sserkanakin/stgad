#!/bin/bash
#SBATCH --job-name=diffusion
# 200 epochs / patience 15 finishes in ~1-2h; best.pt is saved on every
# improvement, so even a walltime kill keeps the best model. 6h is a safe cap
# (covers a slow no-early-stop pemsbay run). override with --time=HH:MM:SS.
#SBATCH --time=06:00:00
# partition / qos / gres are injected per-cluster by submit.sh.
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/diffusion/%x-%j.out
#SBATCH --error=logs/diffusion/%x-%j.err
#SBATCH --mail-user=sserkanakin@icloud.com
#SBATCH --mail-type=END,FAIL
#
# train the diffusion denoiser on one dataset / seed.
#   sbatch --export=DATASET=metrla,SEED=0 scripts/daic/submit_train_diffusion.sbatch
#
# required: DATASET, SEED. optional: PRIOR (default kronecker, used for naming),
# EPOCHS, BATCH_SIZE, LR, D_H, C_0, ALPHA, EXTRA_FLAGS (passed verbatim).
# for a run-named log file also pass --job-name=diffusion_<dataset>_<prior>_seed<n>.

set -euo pipefail
mkdir -p logs/diffusion
export PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
source "${PROJECT_DIR}/scripts/daic/_common.sh"

: "${DATASET:?Set DATASET=metrla|pemsbay|pems08f via --export}"
: "${SEED:?Set SEED=<int> via --export}"
export PRIOR="${PRIOR:-kronecker}"
EXP="${EXP:-dev}"
EPOCHS="${EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-1e-3}"
D_H="${D_H:-32}"
C_0="${C_0:-20000}"
ALPHA="${ALPHA:-7.0}"
EXTRA_FLAGS="${EXTRA_FLAGS:-}"

export STAGE="diffusion"
RUN_NAME="diffusion_${DATASET}_${PRIOR}_seed${SEED}"
scontrol update job="${SLURM_JOB_ID}" JobName="${RUN_NAME}" 2>/dev/null || true

log_run_header "${DAIC_PYTHON} scripts/train_diffusion.py --dataset ${DATASET} --prior ${PRIOR} --exp ${EXP} --seed ${SEED} \
--epochs ${EPOCHS} --batch-size ${BATCH_SIZE} --lr ${LR} --d-h ${D_H} --c-0 ${C_0} --alpha ${ALPHA} ${EXTRA_FLAGS}"

# resource profiling: sample GPU memory + utilisation every 10s in the background
# so we can size production resources from a smoke run. cleaned up on exit.
prof_csv="logs/diffusion/profile_${RUN_NAME}_${SLURM_JOB_ID}.csv"
nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu,utilization.memory \
    --format=csv,noheader -l 10 > "${prof_csv}" 2>/dev/null &
PROF_PID=$!
trap 'kill ${PROF_PID} 2>/dev/null || true' EXIT

# shellcheck disable=SC2086
set +e
srun --export=ALL "${DAIC_PYTHON}" scripts/train_diffusion.py \
    --dataset "${DATASET}" \
    --prior "${PRIOR}" \
    --exp "${EXP}" \
    --seed "${SEED}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --d-h "${D_H}" \
    --c-0 "${C_0}" \
    --alpha "${ALPHA}" \
    ${EXTRA_FLAGS}
RC=$?
set -e

# resource report — sizes --gres (GPU mem), --mem (MaxRSS), --time (Elapsed), --cpus.
kill ${PROF_PID} 2>/dev/null || true
echo "===================== resource profile ====================="
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)"
awk -F',' '{gsub(/ ?MiB/,"",$2); gsub(/ ?%/,"",$4); if($2+0>m)m=$2+0; if($4+0>u)u=$4+0}
    END{printf "peak GPU mem used: %d MiB | peak GPU util: %d%%\n", m, u}' "${prof_csv}" 2>/dev/null || true
echo "CPU/RAM/walltime (run AFTER the job ends):"
echo "  seff ${SLURM_JOB_ID}"
echo "  sacct -j ${SLURM_JOB_ID} --format=JobID,Elapsed,MaxRSS,MaxVMSize,ReqMem,AllocCPUS,State%20"
echo "============================================================"
exit ${RC}
