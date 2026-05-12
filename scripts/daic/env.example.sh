# DAIC environment template. Copy to scripts/daic/env.sh and edit the
# placeholders below; env.sh is gitignored. Source this file at the top
# of every sbatch script and on the login node before running sync helpers.
#
#   cp scripts/daic/env.example.sh scripts/daic/env.sh
#   $EDITOR scripts/daic/env.sh
#   source scripts/daic/env.sh

# Your TU Delft NetID — used by the rsync helpers as the SSH user.
export DAIC_NETID="CHANGE_ME"

# Project root on DAIC. Code lives here. Default: $HOME/research-tud.
# If your home quota is tight, point this at staff-umbrella instead, e.g.
# /tudelft.net/staff-umbrella/<group>/<netid>/research-tud.
export DAIC_PROJECT_DIR="$HOME/research-tud"

# Where large artefacts live (raw + processed data, residuals, checkpoints).
# DAIC recommends staff-umbrella for research data. Adjust the group folder.
# If you want everything in one place, set it equal to DAIC_PROJECT_DIR.
export DAIC_DATA_DIR="/tudelft.net/staff-umbrella/CHANGE_ME_GROUP/${DAIC_NETID:-USER}/research-tud-data"

# Python venv location. Conda envs in $HOME can blow quotas; we use a
# plain venv on the project filesystem. The setup script creates this.
export DAIC_VENV="${DAIC_PROJECT_DIR}/.venv"

# CUDA / cuDNN module versions. CUDA 12.1 is the most recent on DAIC and
# pairs with PyTorch's cu121 wheels.
export DAIC_CUDA_MODULE="cuda/12.1"
export DAIC_CUDNN_MODULE="cudnn/12-8.9.1.23"

# Default Slurm partition + QoS. `general` is open to everyone; add your
# group partition first if you have one (e.g. "insy,general").
export DAIC_PARTITION="general"
export DAIC_QOS_SHORT="short"     # ≤ 4h jobs
export DAIC_QOS_MEDIUM="medium"   # ≤ 1.5d jobs
export DAIC_QOS_LONG="long"       # ≤ 7d jobs

# Default GPU type. A40 (46 GB Ampere, 84 nodes) is the most plentiful;
# L40 (49 GB Ada Lovelace, 18 nodes) is newer. RTX 2080 Ti has only 11 GB
# which is fine for our score net but not for the backbone batch sizes.
export DAIC_GPU_TYPE="a40"

# wandb. WANDB_API_KEY can live in .env at the repo root (gitignored) or in
# ~/.netrc via `wandb login` (setup_env.sh prompts for this). Online mode
# gives live loss curves / GPU util in the wandb dashboard during the run.
export WANDB_DIR="${DAIC_DATA_DIR}/wandb"
export WANDB_MODE="online"      # "offline" if compute nodes are firewalled
export WANDB_PROJECT="thesis-diffusion"
