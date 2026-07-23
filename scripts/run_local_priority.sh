#!/usr/bin/env bash
# PRIORITY-ORDERED local sweep (Mac/MPS), full runs (not smoke). Highest-value cells first
# so the headline results land early; it keeps going down the list until you stop it.
#
#   scripts/run_local_priority.sh                 # defaults: 60 ep, 150 batches/ep
#   EPOCHS=80 MTB=250 scripts/run_local_priority.sh
#
# per cell it runs all 7 arms (ddpm baseline + iso/emp/rsh × snr/whitened) with the validated
# pipeline: small GCNN denoiser, per-node norm, loguniform sampler, ancestral eval, FCPS (VP
# for the ddpm baseline). metric = best per-epoch sampled CRPS. summary → results/analysis/
# local_summary.md (updated after every cell).
set -uo pipefail
cd "$(dirname "$0")/.."
EP=${EPOCHS:-60}; MTB=${MTB:-150}; EXP=${EXP:-local_reshape}
SUM=results/analysis/local_summary.md; mkdir -p results/analysis

# priority: best-headroom PEMS-flow + strong GWN first, then the weak-backbone (mlp) ladder
# on the same cells (where the prior should help MOST), then broaden to the rest.
CELLS=(
  "pems08f:gwn" "pems08s:gwn"
  "pems08f:mlp" "pems08s:mlp"
  "pems04s:gwn" "pems04f:gwn"
  "metrla:gwn"  "pemsbay:gwn"
  "pems04s:mlp" "pems04f:mlp"
  "pems08f:gcn" "pems08s:gcn"
  "metrla:mlp"  "pemsbay:mlp"
  "pems04s:gcn" "pems04f:gcn"
  "metrla:gcn"  "pemsbay:gcn"
)
# arm := name:prior:temporal:cap:loss:sched:sampler
ARMS=(
  "ddpm:vp:-:-:snr:auto:powerlaw"
  "iso:isotropic:-:-:snr:fcps:loguniform"
  "emp:kronecker:empirical:-:snr:fcps:loguniform"
  "rsh:kronecker:reshaped:50:snr:fcps:loguniform"
  "iso_w:isotropic:-:-:whitened:fcps:loguniform"
  "emp_w:kronecker:empirical:-:whitened:fcps:loguniform"
  "rsh_w:kronecker:reshaped:50:whitened:fcps:loguniform"
)
COMMON="--seed 42 --denoiser gcnn --norm pernode --epochs $EP --max-train-batches $MTB \
 --batch-size 32 --val-every 0 --fast-val-every 1 --fast-val-steps 25 --val-anc-steps 25 --val-windows 128"

crps_of(){ grep -oE "sampled_crps=[0-9.]+" "$1" 2>/dev/null | sed 's/.*=//' | sort -n | head -1; }

run(){  # ds bb armstr
  local ds=$1 bb=$2 arm=$3
  IFS=':' read -r name pr tmp cap loss sch smp <<< "$arm"
  local tag="${ds}_${bb}_${name}" log="/tmp/loc_${ds}_${bb}_${name}.log"
  local fl="--dataset $ds --backbone $bb --prior $pr --loss $loss --schedule $sch --noise-sampler $smp"
  [ "$tmp" != "-" ] && fl="$fl --temporal $tmp"
  [ "$cap" != "-" ] && fl="$fl --stiffness-cap $cap"
  echo "===== $tag  ($EP ep, $MTB b/ep, loss=$loss) ====="
  .venv/bin/python -u scripts/train_diffusion.py $COMMON $fl --exp "${EXP}_${name}" > "$log" 2>&1
  echo "  $tag -> best_crps=$(crps_of "$log")"
}

write_summary(){
  { echo "# Local priority sweep — best sampled CRPS ($EP ep, $MTB batches/ep)"; echo
    echo "| dataset | bb | ddpm | iso | emp | rsh | iso_w | emp_w | rsh_w | flip? |"
    echo "|---|---|--:|--:|--:|--:|--:|--:|--:|:--|"
    for cell in "${CELLS[@]}"; do
      local ds="${cell%%:*}" bb="${cell#*:}" row="" iso="" empw="" rshw=""
      for arm in "${ARMS[@]}"; do
        local n="${arm%%:*}" c; c=$(crps_of "/tmp/loc_${ds}_${bb}_${n}.log"); c="${c:-—}"
        row="$row ${c} |"; [ "$n" = iso ] && iso="$c"; [ "$n" = emp_w ] && empw="$c"; [ "$n" = rsh_w ] && rshw="$c"
      done
      # only show rows we've started
      [ -f "/tmp/loc_${ds}_${bb}_ddpm.log" ] || continue
      local verdict="—"
      if [ "$iso" != "—" ] && [ "$empw" != "—" ]; then
        local best; best=$(printf "%s\n%s\n" "$empw" "$rshw" | grep -E "^[0-9.]+$" | sort -n | head -1)
        [ -n "$best" ] && verdict=$(awk -v i="$iso" -v b="$best" 'BEGIN{print (b<i)?"YES":"no"}')
      fi
      echo "| $ds | $bb |$row $verdict |"
    done
  } > "$SUM"
}

echo "priority local sweep: ${#CELLS[@]} cells × ${#ARMS[@]} arms, $EP ep. summary → $SUM"
for cell in "${CELLS[@]}"; do
  ds="${cell%%:*}"; bb="${cell#*:}"
  for arm in "${ARMS[@]}"; do run "$ds" "$bb" "$arm"; done
  write_summary
  echo "===== done $ds/$bb — summary updated ($SUM) ====="
done
echo "ALL DONE."
