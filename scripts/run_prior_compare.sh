#!/usr/bin/env bash
# fair diffusion PRIOR × SAMPLER comparison on one residual cell:
#   priors:   iso (white) · empirical (data-fit temporal) · reshaped (measured per-mode var)
#   samplers: powerlaw (GAD c̄=c_0·u^0.65) · loguniform (even across noise decades)
# everything else identical: small GCNN denoiser, per-node norm, FCPS schedule (forced -> no
# schedule confound), same epochs/budget, ancestral eval. metric = best per-epoch SAMPLED
# CRPS (EMA weights, fixed val windows, common random noise).
# --val-every 0  -> NO slow EM step-sweep;  no --final-val -> no final EM sweep.
set -uo pipefail
cd "$(dirname "$0")/.."
DS=${DS:-pems08s}; BB=${BB:-gwn}
EP=${EPOCHS:-45}; MTB=${MTB:-120}
LOSS=${LOSS:-snr}                 # snr (GAD) or whitened (PriorGrad precision-weighted)
SAMPLERS=${SAMPLERS:-powerlaw loguniform}
TAGSFX=${TAGSFX:-}               # extra tag suffix to keep runs/logs separate (e.g. _w)
COMMON="--dataset $DS --backbone $BB --seed 42 --denoiser gcnn --norm pernode \
 --schedule fcps --loss $LOSS --epochs $EP --max-train-batches $MTB --batch-size 32 \
 --val-every 0 --fast-val-every 1 --fast-val-steps 25 --val-anc-steps 25 --val-windows 128"

# name : train_diffusion flags
PRIORS=(
  "iso:--prior isotropic"
  "empirical:--prior kronecker --temporal empirical"
  "reshaped:--prior kronecker --temporal reshaped --stiffness-cap 50"
)

run () {  # name sampler flags...
  local name=$1 sampler=$2; shift 2
  local tag="${name}_${sampler}${TAGSFX}"
  echo "===== ARM $tag  ($DS/$BB, $EP ep, loss=$LOSS) ====="
  .venv/bin/python -u scripts/train_diffusion.py $COMMON --noise-sampler "$sampler" "$@" \
    --exp cmp_$tag > /tmp/cmp_${DS}_${tag}.log 2>&1
  grep "saved best_crps" /tmp/cmp_${DS}_${tag}.log | tail -1 || echo "  (no CRPS logged — check /tmp/cmp_${DS}_${tag}.log)"
}

for sampler in $SAMPLERS; do
  for p in "${PRIORS[@]}"; do
    name="${p%%:*}"; flags="${p#*:}"
    run "$name" "$sampler" $flags
  done
done

echo; echo "===== SUMMARY  $DS/$BB  ($EP ep, loss=$LOSS, GCNN, per-node norm, FCPS, ancestral) ====="
printf "  %-24s %-12s\n" "prior / sampler" "best_CRPS"
for sampler in $SAMPLERS; do
  for name in iso empirical reshaped; do
    L="/tmp/cmp_${DS}_${name}_${sampler}${TAGSFX}.log"
    c=$(grep -oE "sampled_crps=[0-9.]+" "$L" 2>/dev/null | sed 's/.*=//' | sort -n | head -1)
    printf "  %-24s %-12s\n" "${name} / ${sampler}" "${c:-NA}"
  done
done
