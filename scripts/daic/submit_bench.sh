#!/usr/bin/env bash
# fire the Blackwell-vs-A40 benchmark grid: each cell is (gpu type x torch stack),
# run for each mode (eval + train). scrape the slurm logs for "BENCH_RESULT" to
# build the side-by-side table.
#
#   ssh daic-exp && cd ~/linuxhome/research-tud && source scripts/daic/env.sh
#   bash scripts/daic/setup_bench_venv.sh          # one-time, builds cu128 venv
#   EXP=20260610 bash scripts/daic/submit_bench.sh  # eval needs a trained --exp
#   EXP=20260610 bash scripts/daic/submit_bench.sh --dry   # print, don't submit
#
# the three cells isolate the two effects:
#   a40-module      A40 + production torch 2.5.1   -> today's baseline
#   a40-cu128       A40 + cu128 torch              -> how much is just the torch
#   blackwell-cu128 RTX PRO 6000 + cu128 torch     -> the candidate hardware
# so (blackwell-cu128 / a40-module) is the real-world speedup, and a40-cu128
# tells you how much of it is the newer torch rather than the card.
#
# knobs (env): EXP (trained tag for eval), BENCH_DS, BENCH_PRIOR, BENCH_SEED,
#   BENCH_GEN_BATCH, BENCH_NSTEPS, BENCH_BATCH_SIZE, BENCH_MODES ("eval train"),
#   BENCH_CELLS (subset of the labels above), A40_GRES, BLACKWELL_GRES.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${HERE}/env.sh"

DRY=""
[ "${1:-}" = "--dry" ] && DRY=1

if [ "${DAIC_CLUSTER}" != "exp" ]; then
  echo "ERROR: Blackwell cards are only on exp. ssh daic-exp and retry." >&2
  exit 1
fi

EXP="${EXP:-dev}"
BENCH_DS="${BENCH_DS:-metrla}"
BENCH_PRIOR="${BENCH_PRIOR:-kronecker}"
BENCH_SEED="${BENCH_SEED:-42}"
BENCH_GEN_BATCH="${BENCH_GEN_BATCH:-2048}"   # fits a 48G a40; bump on Blackwell separately
BENCH_NSTEPS="${BENCH_NSTEPS:-50}"
BENCH_BATCH_SIZE="${BENCH_BATCH_SIZE:-32}"
BENCH_WARMUP="${BENCH_WARMUP:-2}"
BENCH_ITERS="${BENCH_ITERS:-5}"
BENCH_MODES="${BENCH_MODES:-eval train}"

# gres type names per cluster docs; override if `sinfo -o '%G'` shows different.
A40_GRES="${A40_GRES:-gpu:nvidia_a40:1}"
BLACKWELL_GRES="${BLACKWELL_GRES:-gpu:nvidia_rtx_pro_6000:1}"

# cell = label|gres|stack  (gres has colons, so use | as the field separator)
ALL_CELLS=(
  "a40-module|${A40_GRES}|module"
  "a40-cu128|${A40_GRES}|cu128"
  "blackwell-cu128|${BLACKWELL_GRES}|cu128"
)
# optional subset via BENCH_CELLS="a40-module blackwell-cu128"
SELECT="${BENCH_CELLS:-}"

mkdir -p logs/bench

echo "─── bench grid ──────────────────────────────────────────────────"
echo "exp:      ${EXP}   dataset: ${BENCH_DS}/${BENCH_PRIOR} seed ${BENCH_SEED}"
echo "eval:     gen_batch=${BENCH_GEN_BATCH} n_steps=${BENCH_NSTEPS}"
echo "train:    batch_size=${BENCH_BATCH_SIZE}"
echo "modes:    ${BENCH_MODES}   warmup=${BENCH_WARMUP} iters=${BENCH_ITERS}"
echo "─────────────────────────────────────────────────────────────────"

for cell in "${ALL_CELLS[@]}"; do
  label="${cell%%|*}"; rest="${cell#*|}"
  gres="${rest%%|*}"; stack="${rest##*|}"
  if [ -n "${SELECT}" ] && ! grep -qw "${label}" <<<"${SELECT}"; then
    continue
  fi
  for mode in ${BENCH_MODES}; do
    job="bench_${label}_${mode}"
    EXPORTS="ALL,BENCH_STACK=${stack},BENCH_MODE=${mode},BENCH_DS=${BENCH_DS}"
    EXPORTS="${EXPORTS},BENCH_PRIOR=${BENCH_PRIOR},BENCH_SEED=${BENCH_SEED},BENCH_EXP=${EXP}"
    EXPORTS="${EXPORTS},BENCH_GEN_BATCH=${BENCH_GEN_BATCH},BENCH_NSTEPS=${BENCH_NSTEPS}"
    EXPORTS="${EXPORTS},BENCH_BATCH_SIZE=${BENCH_BATCH_SIZE}"
    EXPORTS="${EXPORTS},BENCH_WARMUP=${BENCH_WARMUP},BENCH_ITERS=${BENCH_ITERS}"
    [ "${stack}" = "cu128" ] && EXPORTS="${EXPORTS},BENCH_VENV_DIR=${DAIC_VENV_DIR}-cu128"

    cmd=(sbatch --job-name="${job}"
         --partition="${DAIC_PARTITION}" --gres="${gres}"
         --export="${EXPORTS}"
         --output="logs/bench/${job}-%j.out"
         --error="logs/bench/${job}-%j.err"
         scripts/daic/bench_gpu.sbatch)
    if [ -n "${DRY}" ]; then
      printf '  would: %s\n' "${cmd[*]}"
    else
      "${cmd[@]}"
    fi
  done
done

if [ -z "${DRY}" ]; then
  echo
  echo "submitted. collect results once they finish with:"
  echo "  grep -h BENCH_RESULT logs/bench/*.out"
fi
