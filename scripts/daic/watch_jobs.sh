#!/bin/bash
# Live monitoring helper. Run from local Mac:
#
#   source scripts/daic/env.sh
#   bash scripts/daic/watch_jobs.sh           # one snapshot
#   bash scripts/daic/watch_jobs.sh --follow  # refresh every 10s
#   bash scripts/daic/watch_jobs.sh --tail    # tail the newest slurm log
#   bash scripts/daic/watch_jobs.sh --tail <job_id>   # tail a specific job
#
# Combines `squeue --me`, the newest slurm log, and wandb project link
# so you can see everything happening on DAIC without three terminals.

set -euo pipefail

if [ -z "${DAIC_NETID:-}" ]; then
  echo "ERROR: env.sh not sourced. Run: source scripts/daic/env.sh" >&2
  exit 1
fi

REMOTE="${DAIC_NETID}@login.daic.tudelft.nl"
LOGS_DIR="${DAIC_PROJECT_DIR}/logs"

FOLLOW=false
TAIL=false
JOB_ID=""
for arg in "$@"; do
  case "$arg" in
    --follow)  FOLLOW=true ;;
    --tail)    TAIL=true ;;
    [0-9]*)    JOB_ID="$arg"; TAIL=true ;;
    *) echo "Unknown flag: $arg" >&2; exit 2 ;;
  esac
done

print_snapshot() {
  clear 2>/dev/null || true
  echo "── $(date '+%H:%M:%S')  DAIC jobs for ${DAIC_NETID} ─────────────"
  echo
  ssh -o ConnectTimeout=5 "${REMOTE}" \
      "squeue --me -o '%.10i %.9P %.20j %.8T %.10M %.10l %.6D %R'" \
    || echo "  (squeue failed — VPN connected? kerberos valid?)"
  echo
  echo "── wandb ───────────────────────────────────────────────────────"
  echo "  project: https://wandb.ai/${WANDB_ENTITY:-tud-thesis}/${WANDB_PROJECT:-thesis-diffusion}"
  echo
  echo "── newest slurm log ────────────────────────────────────────────"
  ssh -o ConnectTimeout=5 "${REMOTE}" \
      "ls -1t ${LOGS_DIR}/*/*.out 2>/dev/null | head -3" \
    || echo "  (no logs found)"
}

tail_log() {
  local newest_arg=""
  if [ -n "${JOB_ID}" ]; then
    newest_arg="${LOGS_DIR}/*/*-${JOB_ID}.out"
  else
    newest_arg="\$(ls -1t ${LOGS_DIR}/*/*.out | head -1)"
  fi
  # shellcheck disable=SC2029
  ssh "${REMOTE}" "tail -F ${newest_arg}"
}

if $TAIL; then
  tail_log
  exit 0
fi

if $FOLLOW; then
  while true; do
    print_snapshot
    sleep 10
  done
else
  print_snapshot
fi
