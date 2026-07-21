#!/usr/bin/env bash
# LADDER TEST-EVAL as slurm arrays: ONE array PER DATASET (datasets run in parallel), one
# TASK per (backbone × arm) that evaluates ALL seeds sequentially. batching seeds keeps the
# task count under the account's MaxSubmit=100 limit: 12 arms × 3 bb = 36 tasks/dataset
# (72 total for 08f+08s) instead of 360. MaxJobs=8 caps how many actually RUN at once, so the
# %THROTTLE is set at 7 (leaves a slot). ancestral k=100 on best_crps.pt over the test split.
# eval-only — reuses the trained checkpoints, no retraining.
#
#   EXP=20260721_ladder scripts/daic/submit_ladder_eval_array.sh            # full grid (08f+08s)
#   scripts/daic/submit_ladder_eval_array.sh --dry                          # preview, submit nothing
#   DATASETS=pems08f scripts/daic/submit_ladder_eval_array.sh               # one dataset (36 tasks)
#   ARMS="emp:kronecker rsh:kronecker" SEEDS=42 scripts/daic/submit_ladder_eval_array.sh
#   THROTTLE=5 scripts/daic/submit_ladder_eval_array.sh                     # fewer concurrent
# anything after --dry passes straight through to sbatch (e.g. --time=6:00:00).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${HERE}/env.sh"

EXP="${EXP:-20260721_ladder}"
DATASETS="${DATASETS:-pems08f pems08s}"          # 08f + 08s only, for now
BACKBONES="${BACKBONES:-mlp gcn gwn}"            # the backbone-strength ladder
SEEDS="${SEEDS:-42 1 126 2022 3407}"            # stage-2 diffusion seeds (all looped inside a task)
# arms as name:prior, BOTH conditioning blocks (fo + -past). the run tree is namespaced by arm
# name, and the prior column selects the Σ family (iso→isotropic, ddpm→vp, rest→kronecker).
ARMS_DEFAULT="iso:isotropic emp:kronecker rsh:kronecker emp_w:kronecker rsh_w:kronecker ddpm:vp \
iso-past:isotropic emp-past:kronecker rsh-past:kronecker emp_w-past:kronecker rsh_w-past:kronecker ddpm-past:vp"
ARMS="${ARMS:-${ARMS_DEFAULT}}"
# headline eval config (mirrored into the array sbatch via --export).
SPLIT="${SPLIT:-test}"; SAMPLER="${SAMPLER:-ancestral}"; NSTEPS="${NSTEPS:-50}"
K="${K:-100}"; CKPT="${CKPT:-best_crps.pt}"
RESIDUAL_SEED="${RESIDUAL_SEED:-42}"             # which residuals every cell reads (stage-1 seed)
# per-array running-concurrency: MaxJobs=8 is the hard cap, so 7 leaves one free slot.
THROTTLE="${THROTTLE:-7}"

DRY=""; PASSTHRU=()
for a in "$@"; do case "$a" in --dry) DRY=1 ;; *) PASSTHRU+=("$a") ;; esac; done

export DATA_ROOT="${DATA_ROOT:-${DAIC_DATA_DIR}}"
export RESULTS_ROOT="${RESULTS_ROOT:-${DAIC_DATA_DIR}}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${DAIC_OUT_DIR:-$HOME/windowshome/research-tud-out}}"
LOG_DIR="${OUTPUT_ROOT}/logs/${EXP}"; mkdir -p "${LOG_DIR}"

# PREFLIGHT (skip on --dry): OUTPUT_ROOT must have room (a 0-byte touch slips past a full
# quota, so write real blocks) and the residual INPUTS must exist under RESULTS_ROOT
# (train.npz for sigma+prior fit, test.npz for the eval split).
if [ -z "${DRY}" ]; then
  if ! dd if=/dev/zero of="${LOG_DIR}/.spacetest" bs=1M count=32 status=none 2>/dev/null; then
    rm -f "${LOG_DIR}/.spacetest"
    echo "ABORT: cannot write 32MB to OUTPUT_ROOT=${OUTPUT_ROOT} (quota full)." >&2; exit 1
  fi
  rm -f "${LOG_DIR}/.spacetest"
  missing=""
  for DS in ${DATASETS}; do for BB in ${BACKBONES}; do for SP in train test; do
    f="${RESULTS_ROOT}/residuals/${DS}/${BB}/seed${RESIDUAL_SEED}/${SP}.npz"
    [ -s "${f}" ] || missing="${missing}  ${f}"$'\n'
  done; done; done
  if [ -n "${missing}" ]; then
    echo "ABORT: missing residual inputs under RESULTS_ROOT=${RESULTS_ROOT}" >&2
    printf '%s' "${missing}" >&2; exit 1
  fi
fi

source "${HERE}/_gpu.sh"
GRES="$(gpu_blackwell_gres || true)"; GRES="${GRES:-${DAIC_GRES_GPU}}"
RES=(--partition="${DAIC_PARTITION}" --gres="${GRES}")
[ -n "${DAIC_QOS_MEDIUM:-}" ] && RES+=(--qos="${DAIC_QOS_MEDIUM}")
[ -n "${DAIC_GPU_EXCLUDE:-}" ] && RES+=(--exclude="${DAIC_GPU_EXCLUDE}")

ln -sfn "${OUTPUT_ROOT}" "$(cd "${HERE}/../.." && pwd)/out" 2>/dev/null || true
echo "===== ladder test-eval arrays  exp=${EXP}  ${SAMPLER}@${NSTEPS}  k=${K}  ckpt=${CKPT} ====="
echo "datasets=[${DATASETS}]  backbones=[${BACKBONES}]  seeds=[${SEEDS}]  throttle=%${THROTTLE}"
echo "(MaxSubmit=100, MaxJobs=8 on this account; check 'squeue --me' usage before a big submit)"

total_tasks=0; total_skip=0
for DS in ${DATASETS}; do
  GRID_FILE="${LOG_DIR}/eval_grid_${DS}.tsv"; : > "${GRID_FILE}"
  idx=0
  for BB in ${BACKBONES}; do
    for ARM in ${ARMS}; do
      NAME="${ARM%%:*}"; PR="${ARM##*:}"
      # enqueue the (arm,bb) row if AT LEAST ONE seed has a checkpoint; the sbatch loops seeds
      # and skips any individually-missing one. skip (and report) a row with no seeds at all.
      have=0
      for S in ${SEEDS}; do
        [ -f "${OUTPUT_ROOT}/${EXP}/${NAME}/${DS}/${BB}/${PR}/seed${S}/train/${CKPT}" ] && { have=1; break; }
      done
      # REQUIRE_CKPT=0 bypasses this gate — for submissions queued WITH A DEPENDENCY on a
      # still-running training array, whose checkpoints don't exist yet at submit time
      # (the sbatch still skips any seed that is missing when the eval actually runs).
      if [ "${REQUIRE_CKPT:-1}" -eq 1 ] && [ "${have}" -eq 0 ]; then
        echo "  skip (no ckpt any seed): ${EXP}/${NAME}/${DS}/${BB}/${PR}"
        total_skip=$((total_skip + 1)); continue
      fi
      printf '%d\t%s\t%s\t%s\t%s\n' "${idx}" "${DS}" "${BB}" "${NAME}" "${PR}" >> "${GRID_FILE}"
      idx=$((idx + 1))
    done
  done
  N="${idx}"; total_tasks=$((total_tasks + N))
  if [ "${N}" -eq 0 ]; then echo "  ${DS}: 0 tasks — nothing to submit"; continue; fi
  if [ -n "${DRY}" ]; then
    echo "  ${DS}: ${N} tasks (array 0-$((N - 1))%${THROTTLE}), ${N} × ${SEEDS// /,} seeds  grid=${GRID_FILE}"; continue
  fi
  EXPORTS="ALL,EXP=${EXP},GRID_FILE=${GRID_FILE},SEEDS=${SEEDS// /_}"
  EXPORTS="${EXPORTS},SPLIT=${SPLIT},SAMPLER=${SAMPLER},NSTEPS=${NSTEPS// /_},K=${K},CKPT=${CKPT}"
  EXPORTS="${EXPORTS},DATA_ROOT=${DATA_ROOT},RESULTS_ROOT=${RESULTS_ROOT},OUTPUT_ROOT=${OUTPUT_ROOT},WANDB_MODE=disabled"
  # don't let one rejected sbatch (queue limit) kill the loop under set -e.
  if ! JOBID="$(sbatch --parsable \
         --array="0-$((N - 1))%${THROTTLE}" \
         --export="${EXPORTS}" \
         --job-name="eval-${DS}-${EXP##*/}" \
         --output="${LOG_DIR}/%A_%a.out" \
         --error="${LOG_DIR}/%A_%a.err" \
         "${RES[@]}" "${PASSTHRU[@]}" "${HERE}/eval_ladder_array.sbatch")"; then
    echo "  ${DS}: SUBMIT FAILED (queue limit?) — resubmit later with DATASETS=${DS}" >&2
    continue
  fi
  echo "  ${DS}: array ${JOBID} (${N} tasks × ${SEEDS// /,})  grid=${GRID_FILE}"
done

echo "===== ${total_tasks} array task(s)${DRY:+ PREVIEWED (dry-run, nothing submitted)}; ${total_skip} arm-rows skipped (no ckpt) ====="
if [ -z "${DRY}" ]; then
  echo "watch:   squeue --me"
  echo "collect: find ${OUTPUT_ROOT}/${EXP} -name 'test_metrics_bestcrps_${SAMPLER}${NSTEPS}.json'"
fi
