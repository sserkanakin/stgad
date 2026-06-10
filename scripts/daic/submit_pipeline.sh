#!/usr/bin/env bash
# submit the FULL per-seed pipeline for one dataset, chained with slurm afterok
# dependencies so each stage starts only after the previous one *succeeds*:
#
#   backbone -> residuals -> { kron diffusion -> kron eval }
#                         \-> { iso  diffusion -> iso  eval }
#
# everything namespaces under EXP (wandb group <EXP>-<dataset>, results/logs under
# <EXP>). backbone + residuals are shared per (dataset, seed), so re-running the
# pipeline for the same seed reuses them unless you clear them first.
#
#   EXP=20260610 scripts/daic/submit_pipeline.sh metrla 42
#
# walltimes default sensibly; override via env: BB_TIME RES_TIME DIFF_TIME EVAL_TIME.
# if any stage fails, its dependents are auto-cancelled (no eval off a failed train).
set -euo pipefail

DATASET="${1:?usage: submit_pipeline.sh <dataset> <seed>}"
SEED="${2:?seed}"
export EXP="${EXP:-dev}"

# backbone walltime per dataset (pemsbay is the slowest at ~76 s/epoch).
case "$DATASET" in
  pemsbay) BB_TIME="${BB_TIME:-03:00:00}" ;;
  metrla)  BB_TIME="${BB_TIME:-01:30:00}" ;;
  *)       BB_TIME="${BB_TIME:-01:00:00}" ;;
esac
RES_TIME="${RES_TIME:-00:30:00}"
DIFF_TIME="${DIFF_TIME:-06:00:00}"
EVAL_TIME="${EVAL_TIME:-03:00:00}"
# full test set (no --max-windows), CRPS auto. k/steps kept light enough to fit.
EVAL_CFG="--exp ${EXP} --seed ${SEED} --k 30 --n-steps 50 --gen-batch 1024 --w1-samples 4"

jobid_of() { awk '/Submitted batch job/{print $NF}'; }

echo "===== pipeline ${EXP} / ${DATASET} / seed ${SEED} ====="

echo "== backbone (${BB_TIME}) =="
BB=$(bash scripts/daic/submit.sh backbone "$DATASET" "$SEED" --time="$BB_TIME" | tee /dev/stderr | jobid_of)
: "${BB:?failed to submit backbone}"

echo "== residuals (${RES_TIME}, after backbone ${BB}) =="
RS=$(bash scripts/daic/submit.sh residuals "$DATASET" "$SEED" \
       --time="$RES_TIME" --dependency=afterok:"$BB" | tee /dev/stderr | jobid_of)
: "${RS:?failed to submit residuals}"

for PR in kronecker isotropic; do
  echo "== diffusion ${PR} (${DIFF_TIME}, after residuals ${RS}) =="
  DJ=$(PRIOR="$PR" bash scripts/daic/submit.sh diffusion "$DATASET" "$SEED" \
         --time="$DIFF_TIME" --dependency=afterok:"$RS" | tee /dev/stderr | jobid_of)
  : "${DJ:?failed to submit diffusion ${PR}}"

  echo "== eval ${PR} (${EVAL_TIME}, after diffusion ${DJ}) =="
  PRIOR="$PR" EVAL_ARGS="--dataset ${DATASET} --prior ${PR} ${EVAL_CFG}" \
    bash scripts/daic/submit.sh eval "$DATASET" "$SEED" \
      --time="$EVAL_TIME" --dependency=afterok:"$DJ" | tee /dev/stderr | jobid_of >/dev/null
done

echo
echo "submitted full chain for ${EXP}/${DATASET}/seed${SEED}. watch: squeue --me"
echo "(dependent jobs sit PENDING with reason (Dependency) until their parent finishes)"
