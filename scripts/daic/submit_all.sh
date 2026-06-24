#!/usr/bin/env bash
# fan out the combined train+eval job (best.pt by val-CRPS, then locked test eval)
# for every (dataset, prior) pair at one seed. each job trains the denoiser and runs
# the test eval in-process, writing best.pt + validation_history.json + test_results.json
# under results/diffusion/<EXP>/... and results/eval/<EXP>/...
#
# backbone + residuals must already exist for (dataset, seed) — this launches only the
# diffusion stage. aggregate afterwards with: python scripts/summarize_runs.py --exp <EXP>
#
#   EXP=20260624 scripts/daic/submit_all.sh 42
#   EXP=20260624 DATASETS="metrla pems08f" scripts/daic/submit_all.sh 42
#
# override walltime per dataset via DIFF_TIME; pemsbay eval is the slow one (N=325).
set -euo pipefail

SEED="${1:?usage: submit_all.sh <seed>}"
export EXP="${EXP:-dev}"
DATASETS="${DATASETS:-metrla pemsbay pems08f pems04f pems04s pems08s}"
PRIORS="${PRIORS:-kronecker isotropic}"
EVAL_STEPS="${EVAL_STEPS:-50}"

echo "===== submit_all  EXP=${EXP}  seed=${SEED} ====="
echo "datasets: ${DATASETS}"
echo "priors:   ${PRIORS}"
echo

for DS in ${DATASETS}; do
  # pemsbay is the slowest (N=325): more train + much slower eval.
  case "$DS" in
    pemsbay) TIME="${DIFF_TIME:-16:00:00}" ;;
    *)       TIME="${DIFF_TIME:-12:00:00}" ;;
  esac
  for PR in ${PRIORS}; do
    echo "== ${DS} / ${PR} (${TIME}) =="
    PRIOR="$PR" EVAL_STEPS="$EVAL_STEPS" \
      bash scripts/daic/submit.sh traineval "$DS" "$SEED" --time="$TIME"
  done
done

echo
echo "submitted $(echo ${DATASETS} | wc -w | tr -d ' ')×$(echo ${PRIORS} | wc -w | tr -d ' ') combined jobs under EXP=${EXP}."
echo "watch: squeue --me   |   aggregate: python scripts/summarize_runs.py --exp ${EXP}"
