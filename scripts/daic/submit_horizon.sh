#!/bin/bash
# per-lead-time network-total eval pass (--tag horizon; same code path as the pooled pass, adds joint_tot_*_h): re-evaluate the 48 gwn checkpoints of 20260907_matched with the
# pooled joint metrics (sums over spectral clusters 64/16/4 + network), eval only, on the
# test split; metrics files get the _pooled suffix (--tag pooled). five cells per job for
# the five smaller datasets, two per job for pems-bay (its eval is ~4x slower).
#
#   bash scripts/daic/submit_pooled.sh          # submit 12 jobs
#   DRY=1 bash scripts/daic/submit_pooled.sh    # print the cell lists only
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "${HERE}/env.sh"
source "${HERE}/_gpu.sh" 2>/dev/null || true
GRES="$(gpu_blackwell_gres 2>/dev/null || true)"; GRES="${GRES:-${DAIC_GRES_GPU}}"
QOS=""; [ -n "${DAIC_QOS_MEDIUM:-}" ] && QOS="--qos=${DAIC_QOS_MEDIUM}"
EXP="20260907_matched"
LOG="${DAIC_OUT_DIR:-$HOME/windowshome/research-tud-out}/logs/${EXP}"; mkdir -p "${LOG}"
DRY="${DRY:-0}"

# cell spec = "exp/arm|dataset|prior|seed", the format eval_cells.sbatch reads
cells=(); bay=()
for ds in pems08f pems08s pems04f pems04s metrla; do
  for arm in "iso-hist|isotropic" "kron-hist|kronecker"; do
    IFS='|' read -r a p <<< "${arm}"
    for s in 42 1 2 3; do cells+=("${EXP}/${a}|${ds}|${p}|${s}"); done
  done
done
for arm in "iso-hist|isotropic" "kron-hist|kronecker"; do
  IFS='|' read -r a p <<< "${arm}"
  for s in 42 1 2 3; do bay+=("${EXP}/${a}|pemsbay|${p}|${s}"); done
done

submit() {  # $1 = job name, $2 = cells joined by ';'
  if [ "${DRY}" = "1" ]; then echo "[dry] $1: $2"; return; fi
  jid=$(sbatch --parsable --partition="${DAIC_PARTITION}" ${QOS} --gres="${GRES}" \
        ${DAIC_GPU_EXCLUDE:+--exclude=$DAIC_GPU_EXCLUDE} \
        --job-name="$1" --output="${LOG}/%j_$1.out" \
        --export=ALL,CELLS="$2",EVAL_EXTRA="--tag horizon" \
        "${HERE}/eval_cells.sbatch")
  echo "$1 (${jid}): $2"
}
for ((i = 0; i < ${#cells[@]}; i += 5)); do submit horizon "$(IFS=';'; echo "${cells[*]:i:5}")"; done
for ((i = 0; i < ${#bay[@]};   i += 2)); do submit horizon-bay "$(IFS=';'; echo "${bay[*]:i:2}")"; done
