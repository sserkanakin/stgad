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
# args (any order): a dataset name to scope the run to, and/or --dry to preview.
DRY=""; ONLY_DS=""
for a in "$@"; do
  case "$a" in
    --dry) DRY="--dry" ;;
    metrla|pemsbay|pems04f|pems04s|pems08f|pems08s) ONLY_DS="$a" ;;
  esac
done
DATA="${DAIC_DATA_DIR:-$HOME/windowshome/research-tud-data}"
# 10h walltime for every diffusion run — generous, and kill-safe (best.pt saves on every
# improvement). profiled 2026-06-19: GPU mem only 6-12G/46G, util 100% (compute-bound).
DIFF_TIME="${DIFF_TIME:-10:00:00}"
EVAL_CFG="--k 30 --n-steps 50 --w1-samples 4"

# gen-batch is sized PER DATASET: the ugnet's spatial graph-conv activation scales
# with batch*N, so a flat batch over-allocates for high-N datasets. pemsbay (N=325)
# OOM'd at 4096 on the 44G l40; keep batch*N ~constant (~0.65M) so peak memory fits
# the smallest gpu in the pool while still using ~3x more of it than the old 1024.
_gen_batch() {
  case "$1" in
    pemsbay) echo 2048 ;;   # N=325, heaviest
    metrla)  echo 3072 ;;   # N=207
    *)       echo 4096 ;;   # pems08f N=170
  esac
}

# eval walltime is also PER DATASET: the point phase runs k samples over ALL test
# windows, and pemsbay has both the most windows (~10k) and most nodes, so its full
# eval needs ~8h where metrla/pems08f finish well under 6h. EVAL_TIME env overrides.
_eval_time() {
  [ -n "${EVAL_TIME:-}" ] && { echo "$EVAL_TIME"; return; }
  case "$1" in
    pemsbay) echo 10:00:00 ;;
    *)       echo 06:00:00 ;;
  esac
}
jid() { awk '/Submitted batch job/{print $NF}'; }

# safety: resume reasons about FINISHED on-disk state, so a still-running job for
# a cell it's about to submit would be misread (partial file = "corrupt", no
# best.pt yet = "missing") and double-submitted. only the cells we're about to
# touch matter, so when scoped to one dataset we only block on THAT dataset's
# jobs — letting you launch the next dataset while the previous one still runs.
if [ "$DRY" != "--dry" ]; then
  pat='(backbone|residuals|diffusion|eval)_'
  [ -n "$ONLY_DS" ] && pat="(backbone|residuals|diffusion|eval)_${ONLY_DS}_"
  active=$(squeue -u "$USER" -h -o '%j' 2>/dev/null | grep -cE "$pat" || true)
  if [ "${active:-0}" -gt 0 ]; then
    echo "ERROR: ${active} ${ONLY_DS:+${ONLY_DS} }jobs still in the queue — let them finish first." >&2
    echo "       wait for them to drain (or scancel), then retry. (--dry previews anytime.)" >&2
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
  local gb; gb=$(_gen_batch "$ds")
  local tl; tl=$(_eval_time "$ds")
  local extra=(--time="$tl"); [ -n "$dep" ] && extra+=(--dependency=afterok:"$dep")
  [ "$DRY" = "--dry" ] && { echo "      would: eval $ds/$pr/seed$s gen-batch=$gb time=$tl ${dep:+(after $dep)}"; return; }
  PRIOR="$pr" EVAL_ARGS="--dataset $ds --prior $pr --exp $EXP --seed $s $EVAL_CFG --gen-batch $gb" \
    bash scripts/daic/submit.sh eval "$ds" "$s" "${extra[@]}" | tee /dev/stderr | jid >/dev/null
}

echo "===== resume ${EXP} (DATA=$DATA) ====="
for ds in metrla pemsbay pems04f pems04s pems08f pems08s; do
  [ -n "$ONLY_DS" ] && [ "$ds" != "$ONLY_DS" ] && continue
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
