#!/bin/bash
# Live monitor for experimental DAIC jobs. Run from your local machine.
#
#   source scripts/daic-exp/env.sh
#   bash scripts/daic-exp/watch_jobs.sh           # one-shot status
#   bash scripts/daic-exp/watch_jobs.sh --follow  # refresh every 10s
#   bash scripts/daic-exp/watch_jobs.sh --tail    # tail -F the newest log
#   bash scripts/daic-exp/watch_jobs.sh --tail 12345  # tail a specific job

set -euo pipefail

if [ -z "${DAIC_EXP_NETID:-}" ]; then
  echo "ERROR: env.sh not sourced." >&2
  exit 1
fi
REMOTE="${DAIC_EXP_NETID}@${DAIC_EXP_LOGIN_HOST}"

case "${1:-}" in
  --follow)
    ssh "${REMOTE}" "watch -n 10 'squeue --me -o \"%.10i %.9P %.20j %.8T %.10M %.6D %R\"'"
    ;;
  --tail)
    JOB="${2:-}"
    if [ -n "${JOB}" ]; then
      ssh "${REMOTE}" "tail -F ${DAIC_EXP_PROJECT_DIR}/logs/*/*-${JOB}.out"
    else
      ssh "${REMOTE}" "tail -F \$(ls -t ${DAIC_EXP_PROJECT_DIR}/logs/*/*.out | head -1)"
    fi
    ;;
  *)
    ssh "${REMOTE}" "squeue --me -o '%.10i %.9P %.20j %.8T %.10M %.6D %R' && echo && ls -t ${DAIC_EXP_PROJECT_DIR}/logs/*/*.out 2>/dev/null | head -3"
    ;;
esac
