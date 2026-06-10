#!/usr/bin/env bash
# idempotent resume for one experiment: inspect every (dataset, seed, prior) and
# submit ONLY the missing stage. safe to re-run — already-done cells are skipped.
#
#   per cell:
#     residual npz missing/corrupt  -> full chain (backbone resumes -> residuals -> diff -> eval)
#     eval json exists              -> skip (done)
#     diffusion best.pt exists      -> eval only
#     else                          -> diffusion + eval
#
#   EXP=20260610 bash scripts/daic/resume_pipeline.sh
#   EXP=20260610 bash scripts/daic/resume_pipeline.sh --dry   # just print the plan
set -uo pipefail
EXP="${EXP:-dev}"; export EXP
DRY="${1:-}"
DATA="${DAIC_DATA_DIR:-$HOME/windowshome/research-tud-data}"
DIFF_TIME="${DIFF_TIME:-06:00:00}"; EVAL_TIME="${EVAL_TIME:-03:00:00}"
EVAL_CFG="--k 30 --n-steps 50 --gen-batch 1024 --w1-samples 4"
jid() { awk '/Submitted batch job/{print $NF}'; }

# safety: resume reasons about FINISHED on-disk state, so it must run against an
# EMPTY queue — a still-running residual/diffusion would be misread (partial file
# = "corrupt", no best.pt yet = "missing") and double-submitted. refuse otherwise.
if [ "$DRY" != "--dry" ]; then
  active=$(squeue -u "$USER" -h -o '%j' 2>/dev/null | grep -cE 'backbone_|residuals_|diffusion_|eval_' || true)
  if [ "${active:-0}" -gt 0 ]; then
    echo "ERROR: ${active} pipeline jobs still in the queue — resume needs an empty queue." >&2
    echo "       run 'scancel -u $USER' (or wait for it to drain), then retry." >&2
    echo "       (use --dry anytime to preview the plan.)" >&2
    exit 1
  fi
fi

residuals_ok() {  # ds seed -> 0/1, valid only if all 3 split npz pass zip integrity
  local ds=$1 s=$2 f
  for split in train val test; do
    f="$DATA/residuals/$ds/seed$s/$split.npz"
    [ -f "$f" ] && unzip -tq "$f" >/dev/null 2>&1 || return 1
  done
  return 0
}

submit_eval() {  # ds seed prior [afterok-jobid]
  local ds=$1 s=$2 pr=$3 dep="${4:-}"
  local extra=(--time="$EVAL_TIME"); [ -n "$dep" ] && extra+=(--dependency=afterok:"$dep")
  [ "$DRY" = "--dry" ] && { echo "      would: eval $ds/$pr/seed$s ${dep:+(after $dep)}"; return; }
  PRIOR="$pr" EVAL_ARGS="--dataset $ds --prior $pr --exp $EXP --seed $s $EVAL_CFG" \
    bash scripts/daic/submit.sh eval "$ds" "$s" "${extra[@]}" | tee /dev/stderr | jid >/dev/null
}

echo "===== resume ${EXP} (DATA=$DATA) ====="
for ds in metrla pemsbay pems08f; do
  for s in 42 43 44; do
    if ! residuals_ok "$ds" "$s"; then
      echo "[$ds/seed$s] residuals missing/corrupt -> FULL CHAIN"
      [ "$DRY" = "--dry" ] && { echo "      would: submit_pipeline $ds $s"; continue; }
      bash scripts/daic/submit_pipeline.sh "$ds" "$s"
      continue
    fi
    for pr in kronecker isotropic; do
      evalj="$DATA/eval/$EXP/$ds/$pr/seed$s/test_metrics.json"
      best="$DATA/diffusion/$EXP/$ds/$pr/seed$s/best.pt"
      if [ -f "$evalj" ]; then
        echo "[$ds/$pr/seed$s] DONE (eval exists) -> skip"
      elif [ -f "$best" ] && unzip -tq "$best" >/dev/null 2>&1; then  # .pt is a zip; reject truncated
        echo "[$ds/$pr/seed$s] diffusion done, eval missing -> EVAL ONLY"
        submit_eval "$ds" "$s" "$pr"
      else
        echo "[$ds/$pr/seed$s] missing -> DIFFUSION + EVAL"
        if [ "$DRY" = "--dry" ]; then echo "      would: diffusion $ds/$pr/seed$s then eval"; continue; fi
        DJ=$(PRIOR="$pr" bash scripts/daic/submit.sh diffusion "$ds" "$s" --time="$DIFF_TIME" | tee /dev/stderr | jid)
        submit_eval "$ds" "$s" "$pr" "$DJ"
      fi
    done
  done
done
echo "===== done. watch: squeue --me ====="
