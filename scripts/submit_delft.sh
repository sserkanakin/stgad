#!/usr/bin/env bash
# SLURM array job template for Delft HPC (Phase 7).
# Submit with: sbatch scripts/submit_delft.sh
# Filled in during Phase 7 — placeholder only.

#SBATCH --job-name=struct-diff
#SBATCH --array=0-59           # 6 priors × 2 datasets × 5 seeds
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=results/slurm/%A_%a.out

echo "SLURM submission script — implement in Phase 7."
exit 1
