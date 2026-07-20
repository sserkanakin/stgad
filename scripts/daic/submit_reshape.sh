#!/usr/bin/env bash
# PRIOR-RESHAPE experiment: submit ONE slurm array PER DATASET (datasets run in parallel).
# each array's tasks are the arms below — a prior × conditioning grid plus a sampler ablation,
# all on the small GCNN denoiser + per-node norm + FCPS (no schedule confound), ancestral eval.
#
#   EXP=20260719_reshape scripts/daic/submit_reshape.sh 42
#   DATASETS="pems08f pems08s pems04s" EXP=... scripts/daic/submit_reshape.sh 42   # subset
#   anything after the seed passes straight through to sbatch (e.g. --time=12:00:00)
#
# arms answer: (a) does the prior help? iso vs empirical vs reshaped; (b) does it interact with
# conditioning? forecast-only (fo) vs +residual-history (past); (c) sampler? loguniform vs powerlaw
# on the two stiff priors. metric = best per-epoch sampled ancestral CRPS (best_crps.pt).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${HERE}/env.sh"

# seeds: all of them run SEQUENTIALLY inside each array task, so a 5-seed grid is 18 jobs of
# ~2h rather than 90 jobs of ~25min. accepts either positional numbers or SEEDS=".. .. ..".
SEEDS="${SEEDS:-}"
while [[ "${1:-}" =~ ^[0-9]+$ ]]; do SEEDS="${SEEDS:+${SEEDS} }$1"; shift; done
SEEDS="${SEEDS:-42}"
: "${EXP:?set EXP=<dated tag>, e.g. EXP=20260719_reshape}"
# conditioning arm: fo = C is the forecast Yhat only; past = C is [R_past | Yhat].
COND="${COND:-fo}"
case "${COND}" in fo|past) ;; *) echo "COND must be fo|past, got '${COND}'" >&2; exit 1;; esac
# stage-1 seed: which residuals/<ds>/<bb>/seed<N>/ every arm trains on. PINNED at 42 so that
# varying SEED (stage 2) measures diffusion-training stochasticity with the backbone held
# fixed — the same design as the competitor's frozen-prior phase-2 runs. all arms then read
# byte-identical residuals, so backbone variance cancels in the paired per-seed difference.
# set RESIDUAL_SEED=$SEED to restore the old fully-coupled behaviour (needs per-seed
# backbones trained + residuals extracted first; only seed42 exists today).
RESIDUAL_SEED="${RESIDUAL_SEED:-42}"
DATASETS="${DATASETS:-metrla pemsbay pems04f pems04s pems08f pems08s}"
BACKBONES="${BACKBONES:-mlp gcn gwn}"    # backbone-strength ladder (all 3 residual sets)
EPOCHS="${EPOCHS:-200}"
# per-array throttle: with 6 dataset-arrays competing for the account's job slots, keep this
# small so ALL datasets progress in parallel instead of one array hogging every slot.
THROTTLE="${THROTTLE:-8}"

# arm := name:prior:temporal:cap:loss:schedule   ('-' = omitted). sampler=loguniform and
# --cond-past are FIXED in the sbatch (validated locally). tests: (a) prior iso/emp/reshaped,
# (b) loss snr vs whitened — does whitening make the structured prior actually help?, and
# (c) a vanilla DDPM baseline (plain white noise + standard schedule — the textbook control).
# NB no iso_w arm: with V_S=V_T=I and constant θ the whitening cancels algebraically, so
# whitened_mse IS gad_snr_weighted_mse for the isotropic prior (verified: loss diff 2.3e-12,
# grad diff 3.3e-15; the 20260720b run confirmed it empirically, iso 0.05799 == iso_w 0.05799).
# it is a duplicate of `iso`, not a control.
#
# NB no stiffness cap on the reshaped arms. the cap floors θ from below, i.e. truncates the
# LOUD modes — precisely the ones the reshaped prior exists to match. at cap=50 the clipped
# modes hold 2.8% of residual energy on pems08f but 35/61/65/91% on metrla/pems04s/pems08s/
# pemsbay, so `rsh` would be a different treatment on every dataset and "Σ_stat = v_emp"
# would be false. the cap only ever existed for EM stability; the ancestral sampler has no
# such floor (reverse.py:11-13) and this experiment is ancestral-only.
ARMS=(
  "ddpm:vp:-:-:snr:auto"                        # vanilla DDPM reference (NOT a controlled ablation)
  "iso:isotropic:-:-:snr:fcps"                  # iso control (white noise, our machinery)
  "emp:kronecker:empirical:-:snr:fcps"          # Kronecker prior, GAD snr loss   (KEY)
  "rsh:kronecker:reshaped:-:snr:fcps"           # reshaped prior, GAD snr loss
  "emp_w:kronecker:empirical:-:whitened:fcps"   # Kronecker + whitened (wins on gwn)
  "rsh_w:kronecker:reshaped:-:whitened:fcps"    # reshaped + whitened  (best arm on gwn)
)

export DATA_ROOT="${DATA_ROOT:-${DAIC_DATA_DIR}}"
export RESULTS_ROOT="${RESULTS_ROOT:-${DAIC_DATA_DIR}}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${DAIC_OUT_DIR:-$HOME/windowshome/research-tud-out}}"
LOG_DIR="${OUTPUT_ROOT}/logs/${EXP}"; mkdir -p "${LOG_DIR}"

# PREFLIGHT: these homes are quota'd (8G each) and the residuals alone nearly fill
# windowshome. an over-quota OUTPUT_ROOT lets every task train for minutes and then die
# with "OSError: [Errno 122] Disk quota exceeded" when it writes validation_history.json.
# a `touch` is NOT enough to detect it (0-byte files slip through) — write real blocks.
if ! dd if=/dev/zero of="${LOG_DIR}/.spacetest" bs=1M count=32 status=none 2>/dev/null; then
  rm -f "${LOG_DIR}/.spacetest"
  echo "ABORT: cannot write 32MB to OUTPUT_ROOT=${OUTPUT_ROOT} (quota full)." >&2
  echo "  quota -s | free up space, or point OUTPUT_ROOT somewhere with room." >&2
  exit 1
fi
rm -f "${LOG_DIR}/.spacetest"

# PREFLIGHT 2: the residuals are INPUTS and are read from RESULTS_ROOT (see paths.py:
# "RESULTS_ROOT inputs: residuals, backbone"). pointing RESULTS_ROOT at the *output* dir
# is a silent killer — every task starts, then dies in its own preflight. check here once.
missing=""
for DS in ${DATASETS}; do
  for BB in ${BACKBONES}; do
    for SP in train val; do
      f="${RESULTS_ROOT}/residuals/${DS}/${BB}/seed${RESIDUAL_SEED}/${SP}.npz"
      [ -s "${f}" ] || missing="${missing}  ${f}"$'\n'
    done
  done
done
if [ -n "${missing}" ]; then
  echo "ABORT: missing residual inputs under RESULTS_ROOT=${RESULTS_ROOT}" >&2
  printf '%s' "${missing}" >&2
  echo "  (residuals are read from RESULTS_ROOT, which must be the DATA share, not the output dir.)" >&2
  exit 1
fi

source "${HERE}/_gpu.sh"
GRES="$(gpu_blackwell_gres || true)"; GRES="${GRES:-${DAIC_GRES_GPU}}"
RES=(--partition="${DAIC_PARTITION}" --gres="${GRES}")
[ -n "${DAIC_QOS_MEDIUM:-}" ] && RES+=(--qos="${DAIC_QOS_MEDIUM}")
[ -n "${DAIC_GPU_EXCLUDE:-}" ] && RES+=(--exclude="${DAIC_GPU_EXCLUDE}")

# the summary job walks validation_history.json with stdlib python — it needs NO GPU.
# it must NOT inherit RES: sbatch command-line args override the script's #SBATCH lines, so
# passing --gres here would hand a GPU to a job that reads JSON, burn one of the 8 concurrent
# slots, and make it queue behind the very training jobs it is waiting for.
RES_CPU=(--partition="${DAIC_PARTITION}")
[ -n "${DAIC_QOS_MEDIUM:-}" ] && RES_CPU+=(--qos="${DAIC_QOS_MEDIUM}")

ln -sfn "${OUTPUT_ROOT}" "$(cd "${HERE}/../.." && pwd)/out" 2>/dev/null || true
echo "===== reshape experiment ${EXP}  seeds=[${SEEDS}]  cond=${COND}  backbones=[${BACKBONES}]  ${#ARMS[@]} arms × $(set -- ${BACKBONES}; echo $#) backbones /dataset ====="
echo "datasets: ${DATASETS}   (one array each — parallel)"

for DS in ${DATASETS}; do
  GRID_FILE="${LOG_DIR}/grid_${DS}_${COND}.tsv"; : > "${GRID_FILE}"
  idx=0
  for BB in ${BACKBONES}; do
    for ARM in "${ARMS[@]}"; do
      IFS=':' read -r NAME PR TMP CAP LOSS SCHED <<< "${ARM}"
      RUN_DIR="${OUTPUT_ROOT}/${EXP}/${NAME}/${DS}/${BB}/${PR}"
      printf '%d\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${idx}" "${DS}" "${BB}" "${NAME}" "${PR}" "${TMP}" "${CAP}" "${LOSS}" "${SCHED}" "${RUN_DIR}" >> "${GRID_FILE}"
      idx=$((idx + 1))
    done
  done
  N="${idx}"
  EXPORTS="ALL,EXP=${EXP},SEEDS=${SEEDS// /_},RESIDUAL_SEED=${RESIDUAL_SEED},COND=${COND},EPOCHS=${EPOCHS},GRID_FILE=${GRID_FILE}"
  EXPORTS="${EXPORTS},DATA_ROOT=${DATA_ROOT},RESULTS_ROOT=${RESULTS_ROOT},OUTPUT_ROOT=${OUTPUT_ROOT}"
  EXPORTS="${EXPORTS},WANDB_MODE=disabled"   # storage: all needed info is in validation_history.json
  # don't let one rejected sbatch (e.g. AssocMaxSubmitJobLimit) kill the loop under `set -e`:
  # that silently skipped every remaining dataset AND the summary job.
  if ! JOBID="$(sbatch --parsable \
         --array="0-$((N - 1))%${THROTTLE}" \
         --export="${EXPORTS}" \
         --job-name="${DS}-${COND}-${EXP}" \
         --output="${LOG_DIR}/%A_%a.out" \
         --error="${LOG_DIR}/%A_%a.err" \
         "${RES[@]}" "$@" "${HERE}/submit_reshape.sbatch")"; then
    echo "  ${DS}: SUBMIT FAILED (queue limit?) — skipping, resubmit later with DATASETS=${DS}" >&2
    FAILED_DS="${FAILED_DS:+${FAILED_DS} }${DS}"
    continue
  fi
  echo "  ${DS}: array ${JOBID} (${N} arms)  grid=${GRID_FILE}"
  ARRAY_IDS="${ARRAY_IDS:+${ARRAY_IDS}:}${JOBID}"
done

if [ -z "${ARRAY_IDS:-}" ]; then
  echo "no arrays submitted — nothing to summarize." >&2; exit 1
fi
[ -n "${FAILED_DS:-}" ] && echo "  !! not submitted: ${FAILED_DS}  (rerun: DATASETS=\"${FAILED_DS}\" EXP=${EXP} SEEDS=\"${SEEDS}\" COND=${COND} $0)"

# automatic end-of-run summary: a tiny job that waits for ALL dataset arrays (afterany),
# then writes OUTPUT_ROOT/EXP/SUMMARY.md.
SUM_ID="$(sbatch --parsable \
       --dependency="afterany:${ARRAY_IDS}" \
       --export="ALL,EXP=${EXP},OUTPUT_ROOT=${OUTPUT_ROOT}" \
       --job-name="sum-${EXP}" \
       --output="${LOG_DIR}/summary_%j.out" --error="${LOG_DIR}/summary_%j.err" \
       "${RES_CPU[@]}" "${HERE}/summarize_reshape.sbatch")"
echo "  summary job ${SUM_ID} (runs after all arrays → ${OUTPUT_ROOT}/${EXP}/SUMMARY.md)"
echo
echo "watch:   squeue --me   |   logs: ${LOG_DIR}/"
echo "summary: cat ${OUTPUT_ROOT}/${EXP}/SUMMARY.md   (auto-written when everything finishes)"
