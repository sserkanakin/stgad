#!/bin/bash
#SBATCH --job-name=ancsweep
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
# Ancestral-sampler RESOLUTION sweep on SAVED checkpoints -- no training, eval only.
#
# WHY. reverse_sample_ancestral walks a geometric grid from c_0 down to a terminal
# floor c_lo and returns the denoiser's R0_hat there, discarding whatever noise is
# still present. How much that is, per arm, is the variance-weighted
#
#     f(c_lo) = sum_m var_m (1 - e^{-2 theta_m c_lo}) / sum_m var_m
#
# At the locked protocol's c_lo = 0.02 this is 3.15% for the isotropic control in
# EVERY cell (theta = 0.8 flat) and 19.0-28.2% for the structured prior, also in
# every cell. Training draws go down to 8e-4, where the figures are 0.13% and
# 0.9-2.4%. So the locked protocol resolves the two arms to fidelities differing by
# 6-9x, and it does so grid-wide -- not only on the stiffest cells.
#
# That asymmetry manufactures the "structure buys under-dispersion in every paired
# seed" result. Measured on pems08f (256 windows, k=24, seed 42): lowering c_lo to
# 8e-4 moves the isotropic arm's coverage by under a point on both gcn and gwn,
# and moves the structured arm's by +4.8 and +4.9, taking it PAST the control on
# coverage, Winkler and energy while it keeps its CRPS advantage. The one matched
# EM sweep on disk agrees by a much more expensive route: EM integrates to
# c_bar = 0 rather than stopping at a floor, and at 1500 steps it closes the CRPS
# gap (+0.9% -> +0.1%) and the coverage gap (emp 0.764 -> 0.855) on pems08f/gwn.
#
# This sweep separates prior from sampler at 1/30th of EM's cost: ancestral has no
# stability floor, so lowering c_lo and raising the step count is cheap.
#
# CELLS = semicolon-separated "arm|dataset|backbone|seed|cond", as in histcond.sh.
#   arm  : iso | kron
#   cond : hist -> 20260728_histcond/<arm>-hist ; fo -> 20260721_ladder/<arm>
#
#   sbatch --export=ALL,CELLS='emp|pems08f|gcn|42|hist;iso|pems08f|gcn|42|hist' \
#          scripts/daic/anc_sweep.sbatch
#
# Grid: C_LOS x STEPS. Results land in
#   <exp>/<ds>/<bb>/<prior>/seed<s>/eval/test_metrics_bestcrps_ancestral<n>[_clo<c>].json
# The default floor writes the LOCKED-PROTOCOL name and is skipped when it already
# exists, so this can never overwrite a headline number.
#
# Cost: ancestral@50 is ~10 min/cell at k=100; the grid below is 3 non-default
# combinations, so ~1 h per (arm, cell) plus whatever reruns are missing.
set -uo pipefail
export PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT_DIR"
export STAGE=eval
source "${PROJECT_DIR}/scripts/daic/_common.sh"
export DATA_ROOT="${DATA_ROOT:-${DAIC_DATA_DIR}}"
export RESULTS_ROOT="${RESULTS_ROOT:-${DAIC_DATA_DIR}}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${DAIC_OUT_DIR:-$HOME/windowshome/research-tud-out}}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

# the sweep. "default" means: omit --anc-c-lo entirely (sampler default 0.02), which
# reproduces the locked protocol byte-for-byte at 50 steps.
C_LOS="${C_LOS:-default 8e-4}"
STEPS="${STEPS:-50 200}"
K="${K:-100}"                   # locked-protocol k, so the numbers stay comparable
HIST_EXP="${HIST_EXP:-20260728_histcond}"
FO_EXP="${FO_EXP:-20260721_ladder}"

fail=0; ran=0; skipped=0
IFS=';' read -r -a CELL_ARR <<< "${CELLS:?set CELLS}"
for cell in "${CELL_ARR[@]}"; do
  IFS='|' read -r arm ds bb seed cond <<< "${cell}"
  case "${arm}" in
    iso) prior="isotropic" ;;
    kron) prior="kronecker" ;;
    *) echo "unknown arm '${arm}' in cell '${cell}'"; fail=$((fail+1)); continue ;;
  esac
  if [ "${cond}" = "hist" ]; then exp="${HIST_EXP}/${arm}-hist"; else exp="${FO_EXP}/${arm}"; fi

  ckdir="${OUTPUT_ROOT}/${exp}/${ds}/${bb}/${prior}/seed${seed}"
  if [ ! -s "${ckdir}/best_crps.pt" ] && [ ! -s "${ckdir}/train/best_crps.pt" ]; then
    echo "MISSING CHECKPOINT for ${cell}: no best_crps.pt under ${ckdir}"
    fail=$((fail+1)); continue
  fi

  for n in ${STEPS}; do
    for clo in ${C_LOS}; do
      if [ "${clo}" = "default" ]; then
        tag=""; flag=""
      else
        # %g in python's f-string; printf here must match it for the skip check
        tag="_clo$(printf '%g' "${clo}")"; flag="--anc-c-lo ${clo}"
      fi
      out="${ckdir}/eval/test_metrics_bestcrps_ancestral${n}${tag}.json"
      if [ -s "${out}" ]; then
        echo "skip (exists): ${cell} anc${n}${tag}"; skipped=$((skipped+1)); continue
      fi
      echo "═════ ANC@${n} c_lo=${clo}  ${arm} · ${ds} · ${bb} · seed${seed} · ${cond} ═════"
      # shellcheck disable=SC2086 -- flag is intentionally word-split (empty or two words)
      "${DAIC_PYTHON}" scripts/eval_diffusion.py \
          --dataset "${ds}" --backbone "${bb}" --prior "${prior}" \
          --exp "${exp}" --seed "${seed}" --split test \
          --sampler ancestral --n-steps "${n}" --k "${K}" \
          --ckpt best_crps.pt ${flag} \
        && ran=$((ran+1)) \
        || { echo "ANC EVAL FAILED ${cell} n=${n} c_lo=${clo}"; fail=$((fail+1)); }
    done
  done
done
echo "done: ${ran} ran, ${skipped} skipped, ${fail} failed"
[ "${fail}" -eq 0 ]
