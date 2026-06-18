#!/usr/bin/env bash
# ONE entry point for the diffusion pipeline on DAIC — hides the env-var soup
# (PRIOR / SPATIAL / EXTRA_FLAGS / ...) behind a friendly verb + arm name. it
# just wraps the tested scripts/daic/submit.sh + resume_pipeline.sh underneath.
#
#   scripts/run.sh profile <dataset> [arm]        # short PRODUCTION-config run, profiled
#   scripts/run.sh train   <dataset> <seed> [arm] # one full diffusion run
#   scripts/run.sh all                            # idempotent: submit every missing stage
#
# arm  (default kron):
#   kron = structured kronecker  Σ_S=(L+γI)^-1   (spatio-temporal, the contribution)
#   flat = temporal-only         Σ_S=I
#   iso  = isotropic             σ²I             (the baseline)
#
# env knobs (all optional): EXP=<tag> (default dev/profile), LOSS=<snr|mse|mahalanobis>.
# anything after the positional args is passed straight through to sbatch (e.g. --time).
#
# NOTE: needs residuals on DAIC first (run the backbone+extract stages, or sync the
# local ones). `profile` uses the production batch + the val diagnostic — the memory
# peak — but only ~40 train batches, so the GPU/CPU/RAM numbers reflect a real run.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMIT="${HERE}/daic/submit.sh"
LOSS="${LOSS:-snr}"

# friendly arm name -> the PRIOR value + the train_diffusion spatial flag it needs.
arm_flags() {
  case "${1:-kron}" in
    kron) ARM_PRIOR=kronecker; ARM_SPATIAL="--spatial structured" ;;
    flat) ARM_PRIOR=kronecker; ARM_SPATIAL="--spatial flat" ;;
    iso)  ARM_PRIOR=isotropic; ARM_SPATIAL="" ;;
    *) echo "arm must be: kron | flat | iso" >&2; exit 2 ;;
  esac
}

CMD="${1:?usage: run.sh <profile|train|all> ...}"; shift
case "$CMD" in
  profile)
    DS="${1:?dataset}"; ARM="${2:-kron}"; arm_flags "$ARM"
    # production batch + val diagnostic, but 1 epoch / 40 batches -> real memory, short.
    EXP="${EXP:-profile}" PROFILE=1 PRIOR="$ARM_PRIOR" \
      EXTRA_FLAGS="--epochs 1 --max-train-batches 40 --val-every 1 ${ARM_SPATIAL} --loss ${LOSS}" \
      bash "$SUBMIT" diffusion "$DS" 0 --time=00:30:00
    ;;
  train)
    DS="${1:?dataset}"; SEED="${2:?seed}"; ARM="${3:-kron}"; arm_flags "$ARM"
    [ $# -ge 3 ] && shift 3 || shift $#
    PRIOR="$ARM_PRIOR" PROFILE=1 EXTRA_FLAGS="${ARM_SPATIAL} --loss ${LOSS}" \
      bash "$SUBMIT" diffusion "$DS" "$SEED" "$@"
    ;;
  all)
    EXP="${EXP:-dev}" bash "${HERE}/daic/resume_pipeline.sh" "$@"
    ;;
  *) echo "command must be: profile | train | all" >&2; exit 2 ;;
esac
