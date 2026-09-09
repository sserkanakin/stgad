#!/bin/bash
#SBATCH --job-name=run-after
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
# Tiny CPU job used with --dependency=afterok:<jobid> to submit the next stage of a campaign
# once the previous one has finished (e.g. the diffusion cells after the backbone array),
# without a watcher on the Mac. CMD = the command to run from the project root.
#   sbatch --dependency=afterok:285739 --export=ALL,CMD="EXP=... scripts/daic/histcond.sh" scripts/daic/run_after.sbatch
set -uo pipefail
export PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "${PROJECT_DIR}"
source scripts/daic/activate.sh >/dev/null 2>&1 || true
echo "[$(date)] running: ${CMD:?set CMD}"
bash -c "${CMD}"
