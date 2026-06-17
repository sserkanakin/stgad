# DAIC environment template. Copy to scripts/daic/env.sh and edit the few
# values below; env.sh is gitignored, so keep one per machine (your Mac and
# each cluster). All the per-cluster paths/modules/slurm settings live in the
# version-controlled clusters.sh — you normally only set your NetID here.
#
#   cp scripts/daic/env.example.sh scripts/daic/env.sh
#   $EDITOR scripts/daic/env.sh
#   source scripts/daic/env.sh

# Your TU Delft NetID — the SSH user and the owner of the remote paths.
export DAIC_NETID="CHANGE_ME"

# Which cluster to target. Leave unset to auto-detect (on a cluster it picks
# itself; off-cluster — your Mac — it defaults to the experimental cluster).
# Force it when syncing from the Mac to a specific cluster:
#   export DAIC_CLUSTER="exp"      # daic02.hpc.tudelft.nl (new GPUs)
#   export DAIC_CLUSTER="stable"   # login.daic.tudelft.nl (old setup)
# export DAIC_CLUSTER="exp"

# OFF-CLUSTER ONLY: when running sync/watch from your Mac against the
# experimental cluster, the default exp paths use $HOME (which is your Mac
# home, not the cluster's). Set the absolute remote paths so rsync targets
# the right place. On the cluster itself these auto-resolve — leave unset.
#   export DAIC_PROJECT_DIR="/tudelft.net/student-homes-linux/<x>/<netid>/research-tud"
#   export DAIC_DATA_DIR="/tudelft.net/student-homes/<x>/<netid>/research-tud-data"

# wandb. WANDB_API_KEY can live in .env at the repo root (gitignored) or in
# ~/.netrc via `wandb login` (setup_env.sh prompts for this).
export WANDB_MODE="${WANDB_MODE:-online}"   # override per-run: WANDB_MODE=offline ...
export WANDB_ENTITY="tud-thesis"    # your wandb team or username

# pull in all the per-cluster paths, modules, partition/qos/gres.
source "$(dirname "${BASH_SOURCE[0]}")/clusters.sh"
