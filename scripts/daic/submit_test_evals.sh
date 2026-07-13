#!/usr/bin/env bash
# phase-1 TEST-split evals for the base-arm comparison (docs/evaluation_protocol.md).
# HEADLINE RQ2: kron_base vs iso_base — matched denoiser, matched conditioning (base),
# matched recipe (loss=snr, lr=1e-4); ONLY the prior differs. iso_base must be trained
# first (it was the missing r2 control):
#   ARMS="iso_base:isotropic:base:global" EXP=20260708_r2 \
#     scripts/daic/submit_experiment.sh 42 --nodelist=gpu[36-53]
#
# protocol: S=100 posterior samples (SpecSTG/CSDI/HySRD Table 1 convention), best_crps.pt
# (val-CRPS-selected -> no test leakage), BOTH samplers in parallel:
#   ancestral @ 25 steps  (headline; exact OU integrator, val-selected everywhere)
#   EM        @ val-selected steps  (GAD-parity/robustness; read from val_results.json)
# eval-only: no retraining. same evaluate() as training -> schema-identical payload
# (rmse/crps/crps_energy/winkler/coverage/w1/energy + per-horizon) for summarize_valsweep.
#
# GPU: EM (the long pole, 1000-2000 steps) is pinned to the Blackwell nodes (fastest,
# 96GB); ancestral-25 is cheap and runs on any pool GPU (fills L40/A40) so the whole
# pool stays busy. override with EM_GPU / ANC_GPU.
#
#   scripts/daic/submit_test_evals.sh --dry            # preview every job
#   scripts/daic/submit_test_evals.sh                  # submit (run ON DAIC)
#   scripts/daic/submit_test_evals.sh --dry pems08f    # one dataset
#   SAMPLERS=ancestral scripts/daic/submit_test_evals.sh   # headline sampler only
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

EXP_ROOT="${EXP_ROOT:-20260708_r2}"
SEED="${SEED:-42}"                    # phase 1 = seed 42 (add 43/44 in phase 2)
CKPT="${CKPT:-best_crps.pt}"          # both samplers on the SAME weights -> clean sampler contrast
ARMS="${ARMS:-kron_base iso_base}"    # the matched base pair (RQ2). others = ablations.
SAMPLERS="${SAMPLERS:-em ancestral}"  # EM first (current priority), then ancestral
DEFAULT_EM="${DEFAULT_EM:-1500}"      # EM steps fallback if val_results.json is unreadable
# GPU targeting. By default SOFT-prefer Blackwell if the cluster exposes it (auto-
# detected), else any gpu — jobs never block (they fall back to a40/l40). Set
# GPU_PREFER=any to force any-gpu; override EM_GPU/ANC_GPU to pin something specific.
source "${HERE}/_gpu.sh"
PREFER="$(gpu_prefer_flag || true)"
EM_GPU="${EM_GPU:-${PREFER}}"
ANC_GPU="${ANC_GPU:-${PREFER}}"

# --sampler/--n-steps/--ckpt added per job. --save-diagnostics dumps the companion
# <split>_diag_*.npz (PIT + reliability + example fan charts) in the SAME run -> no rerun.
EVAL_CFG="--split test --k 100 --w1-samples 4 --save-diagnostics"

DRY=""; ONLY_DS=""
for a in "$@"; do case "$a" in
  --dry) DRY="--dry" ;;
  metrla|pemsbay|pems04f|pems04s|pems08f|pems08s) ONLY_DS="$a" ;;
  *) echo "unknown arg: $a" >&2; exit 2 ;;
esac; done

# output tree (where training wrote checkpoints + val_results.json). prefer the ./out
# symlink submit_experiment.sh drops; fall back to the env/default windowshome tree.
OUT="${OUTPUT_ROOT:-}"
[ -z "$OUT" ] && [ -L "$(cd "${HERE}/../.." && pwd)/out" ] && OUT="$(cd "${HERE}/../.." && pwd)/out"
[ -z "$OUT" ] && OUT="${HOME}/windowshome/research-tud-out"

prior_of() { case "$1" in iso_*) echo isotropic;; *) echo kronecker;; esac; }
_gen_batch() { case "$1" in pemsbay) echo 2048;; metrla) echo 3072;; *) echo 4096;; esac; }
_eval_time() { case "$1" in pemsbay) echo 06:00:00;; *) echo 04:00:00;; esac; }

# val-selected EM steps for this cell, read from its val_results.json (best_crps track).
# ancestral is uniformly 25, so no lookup needed there.
em_steps() {  # arm ds prior
  local arm="$1" ds="$2" pr="$3"
  local vr="${OUT}/${EXP_ROOT}/${arm}/${ds}/gwn/${pr}/seed${SEED}/eval/val_results.json"
  local n=""
  [ -f "$vr" ] && n="$(python3 -c "import json,sys; print(json.load(open('$vr'))['checkpoints']['best_crps']['chosen_steps'])" 2>/dev/null || true)"
  echo "${n:-$DEFAULT_EM}"
}

echo "=== base-arm test evals: exp_root=${EXP_ROOT} seed=${SEED} ckpt=${CKPT} ==="
echo "=== arms=[${ARMS}] samplers=[${SAMPLERS}] out=${OUT} ==="
# datasets: DATASETS env (space-separated) scopes which run here — use it to split
# work across clusters without double-submitting (e.g. DATASETS="pems08f" on the
# cluster with spare slots). a positional dataset arg overrides both.
DATASETS="${DATASETS:-metrla pemsbay pems04f pems04s pems08f pems08s}"
[ -n "$ONLY_DS" ] && DATASETS="$ONLY_DS"
n=0
for arm in $ARMS; do
  pr="$(prior_of "$arm")"; exp="${EXP_ROOT}/${arm}"
  for ds in $DATASETS; do
    gb="$(_gen_batch "$ds")"; tl="$(_eval_time "$ds")"
    for smp in $SAMPLERS; do
      if [ "$smp" = "ancestral" ]; then steps=25; gpu="$ANC_GPU"; else steps="$(em_steps "$arm" "$ds" "$pr")"; gpu="$EM_GPU"; fi
      ea="--dataset ${ds} --prior ${pr} --exp ${exp} --seed ${SEED} ${EVAL_CFG} --sampler ${smp} --n-steps ${steps} --ckpt ${CKPT} --gen-batch ${gb}"
      n=$((n+1))
      if [ "$DRY" = "--dry" ]; then
        printf '  [%02d] %-10s %-9s %-9s steps=%-4s gpu=%-20s time=%s\n' \
          "$n" "$arm" "$ds" "$smp" "$steps" "${gpu:-(any)}" "$tl"
        continue
      fi
      # shellcheck disable=SC2086
      EXP="$exp" PRIOR="$pr" EVAL_ARGS="$ea" \
        bash "${HERE}/submit.sh" eval "$ds" "$SEED" --time="$tl" ${gpu}
    done
  done
done
echo "=== ${n} eval job(s)${DRY:+ PREVIEWED (dry-run, nothing submitted)} ==="
if [ -z "$DRY" ]; then
  echo "watch: squeue --me   |   aggregate: python scripts/summarize_valsweep.py --exp ${EXP_ROOT}"
fi
