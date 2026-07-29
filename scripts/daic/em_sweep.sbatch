#!/bin/bash
#SBATCH --job-name=emsweep
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
# EM step sweep on SAVED checkpoints -- no training, eval only.
#
# WHY. EM's stability floor scales as n >~ theta_max * c_0. The structured prior is
# stiff (theta_max 16 on pems08f/gwn, 94-152 on the gcn cells) while iso is flat
# (theta_max = theta_min = 0.8), so at any fixed step count EM integrates the
# structured arm far less accurately than the isotropic one. Ancestral has no such
# floor. That makes part of the structured arm's ancestral deficit INTEGRATION
# ERROR rather than a property of the prior, and the only way to separate the two
# is to push EM until it converges.
#
# What the one existing matched sweep (pems08f/gwn/seed42) shows: the emp-vs-iso
# CRPS gap closes +0.9% -> +0.1% between ancestral@50 and EM@1000 and then stops
# moving, while iso itself barely shifts (0.04966 -> 0.04953). Coverage moves much
# more: emp 0.764 -> 0.855, rsh 0.646 -> 0.878 (nominal 0.90). So on that cell more
# EM steps buy a TIE on accuracy and a real win on calibration.
#
# That cell is the strongest backbone, where both priors nearly tie anyway. The
# untested regime is mlp/gcn -- stiffer, and where the structured prior actually
# wins under ancestral. This sweep covers those.
#
# CELLS = semicolon-separated "arm|dataset|backbone|seed|cond", as in histcond.sh.
# The exp tree is derived from cond: hist -> 20260728_histcond/<arm>-hist,
# fo -> 20260721_ladder/<arm> (that campaign's forecast-only cells).
#
#   sbatch --export=ALL,CELLS='emp|pems08f|gcn|1|hist',STEPS=1000 scripts/daic/em_sweep.sbatch
#
# Cost: ~3.3 h per cell at 1000 steps, k=100 (vs 10 min for ancestral@50) -- 29x.
# One cell per job so an 8 h limit is never at risk.
set -uo pipefail
export PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT_DIR"
export STAGE=eval
source "${PROJECT_DIR}/scripts/daic/_common.sh"
export DATA_ROOT="${DATA_ROOT:-${DAIC_DATA_DIR}}"
export RESULTS_ROOT="${RESULTS_ROOT:-${DAIC_DATA_DIR}}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${DAIC_OUT_DIR:-$HOME/windowshome/research-tud-out}}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

STEPS="${STEPS:-1000}"          # space-separated list is allowed
K="${K:-100}"                   # locked-protocol k, so the numbers stay comparable
HIST_EXP="${HIST_EXP:-20260728_histcond}"
FO_EXP="${FO_EXP:-20260721_ladder}"

fail=0
IFS=';' read -r -a CELL_ARR <<< "${CELLS:?set CELLS}"
for cell in "${CELL_ARR[@]}"; do
  IFS='|' read -r arm ds bb seed cond <<< "${cell}"
  [ "${arm}" = "iso" ] && prior="isotropic" || prior="kronecker"
  if [ "${cond}" = "hist" ]; then exp="${HIST_EXP}/${arm}-hist"; else exp="${FO_EXP}/${arm}"; fi

  for n in ${STEPS}; do
    out="${OUTPUT_ROOT}/${exp}/${ds}/${bb}/${prior}/seed${seed}/eval/test_metrics_bestcrps_em${n}.json"
    if [ -s "${out}" ]; then echo "skip (exists): ${cell} em${n}"; continue; fi
    echo "═════ EM@${n}  ${arm} · ${ds} · ${bb} · seed${seed} · ${cond} (${exp}) ═════"
    "${DAIC_PYTHON}" scripts/eval_diffusion.py \
        --dataset "${ds}" --backbone "${bb}" --prior "${prior}" \
        --exp "${exp}" --seed "${seed}" --split test \
        --sampler em --n-steps "${n}" --k "${K}" \
        --ckpt best_crps.pt \
      || { echo "EM EVAL FAILED ${cell} n=${n}"; fail=$((fail+1)); }
  done
done
echo "done, ${fail} failed"; [ "${fail}" -eq 0 ]
