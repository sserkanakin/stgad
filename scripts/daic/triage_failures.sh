#!/usr/bin/env bash
# triage failed/cancelled jobs in an experiment: dump every non-empty .err with
# its tail, plus a sacct state summary. run on DAIC from the repo root.
#   EXP=20260610 bash scripts/daic/triage_failures.sh
set -uo pipefail
EXP="${EXP:-20260610}"

echo "================ non-empty error logs under logs/${EXP} ================"
found=0
for f in logs/${EXP}/*/*/*.err; do
  [ -s "$f" ] || continue
  found=1
  run="$(basename "$(dirname "$f")")"
  echo "----- ${run}  ($(basename "$f")) -----"
  tail -8 "$f"
  echo
done
[ "$found" = 0 ] && echo "(no non-empty .err logs — nothing has errored)"

echo "================ sacct states since 2026-06-10 (non-OK only) ================"
sacct -u "$USER" -X -S 2026-06-10 \
  --format=JobID,JobName%34,State,Elapsed,ExitCode 2>/dev/null \
  | grep -viE 'COMPLETED|RUNNING|PENDING| CG ' \
  || echo "(none / sacct unavailable)"

echo "================ which cells have a result vs not ================"
for ds in metrla pemsbay pems08f; do
  for pr in kronecker isotropic; do
    for s in 42 43 44; do
      j="results/eval/${EXP}/${ds}/${pr}/seed${s}/test_metrics.json"
      [ -f "$j" ] && echo "  ok    ${ds}/${pr}/seed${s}" || echo "  MISS  ${ds}/${pr}/seed${s}"
    done
  done
done
