# Experimental DAIC environment template. Copy to scripts/daic-exp/env.sh
# and edit the placeholders below; env.sh is gitignored.
#
#   cp scripts/daic-exp/env.example.sh scripts/daic-exp/env.sh
#   $EDITOR scripts/daic-exp/env.sh
#   source scripts/daic-exp/env.sh
#
# Differences vs the regular DAIC (scripts/daic/env.example.sh):
#   - login host:      daic01.hpc.tudelft.nl
#   - SSH:             public keys via ssh-copy-id are accepted (kinit still
#                      required on first session to mount network storage)
#   - module tree:     `module load 2025/gpu` first, then cuda/12.9 + cudnn
#   - no miniconda module: setup_env.sh installs Miniforge into ~/linuxhome
#   - account flag:    --account=<your account> is MANDATORY on every job
#   - partition:       `all` (open to everyone), `test` (CPU only)
#   - GPU gres prefix: `nvidia_a40` / `nvidia_l40` / `nvidia_rtx_pro_6000`
#   - storage:         /trinity/home is 5 MB; put everything under ~/linuxhome
#                      (30 GB) or under a staff-umbrella project storage.

# ── Identity ───────────────────────────────────────────────────────────────
# Your TU Delft NetID — used as the SSH user by the rsync helpers.
export DAIC_EXP_NETID="CHANGE_ME"

# Mandatory Slurm account. Find yours on the login node with:
#   sacctmgr show user $USER withassoc format=account%30
# If you see multiple, pick the one for the group/project that should be
# charged for this work. Without this, every sbatch is rejected.
export DAIC_EXP_ACCOUNT="CHANGE_ME"

# ── Hosts ──────────────────────────────────────────────────────────────────
# Either daic01 or daic02. They're equivalent; daic02 is the backup.
export DAIC_EXP_LOGIN_HOST="daic01.hpc.tudelft.nl"

# ── Paths ──────────────────────────────────────────────────────────────────
# Project root on the cluster. ~/linuxhome (30 GB) is the right default;
# /trinity/home is only 5 MB and is for shell config only.
export DAIC_EXP_PROJECT_DIR="${HOME}/linuxhome/research-tud"

# Where large artefacts live (raw + processed data, residuals, checkpoints,
# wandb run dirs). Use staff-umbrella once your supervisor grants project
# storage; otherwise ~/linuxhome works for a single-user setup.
#   staff-umbrella:  /tudelft.net/staff-umbrella/<project>/<netid>/research-tud-data
export DAIC_EXP_DATA_DIR="${HOME}/linuxhome/research-tud-data"

# Conda env lives on the Linux home for POSIX compatibility (cluster home is
# too small; some NFS / SMB mounts can't satisfy pip's rename ops).
export DAIC_EXP_CONDA_ENV="thesis-diffusion"
export DAIC_EXP_PYTHON_VERSION="3.11"
# Root for the Miniforge installation that setup_env.sh creates.
export DAIC_EXP_MINIFORGE_ROOT="${HOME}/linuxhome/miniforge3"

# ── Modules (Lmod) ─────────────────────────────────────────────────────────
# Base env: 2025/gpu unlocks GPU-related software stacks.
export DAIC_EXP_BASE_MODULE="2025/gpu"
# Project-vendored conda env uses cu121 torch wheels, so cuda/12.1 would be
# the natural match, but experimental DAIC only ships cuda/12.9. PyTorch's
# cu121 wheels run fine against cuda/12.9's driver because the CUDA runtime
# is bundled. We load cuda/12.9 anyway so nvcc / cuobjdump are on PATH for
# any C-extension build that needs them.
export DAIC_EXP_CUDA_MODULE="cuda/12.9"
# cuDNN is bundled in PyTorch wheels; the module load is best-effort.
export DAIC_EXP_CUDNN_MODULE="cudnn/8.9.7.29-12"

# ── Slurm ──────────────────────────────────────────────────────────────────
# `all` is the default open partition. If your account has access to a
# higher-priority dept partition (ewi-insy, ewi-me, ewi-st, me-cor),
# put it FIRST so Slurm prefers it: "ewi-insy,all".
export DAIC_EXP_PARTITION="all"
# QoS defaults to `normal` on experimental DAIC — leave unset to use the
# default, or override for time-limited reservations.
export DAIC_EXP_QOS=""

# ── GPU ────────────────────────────────────────────────────────────────────
# nvidia_a40 (48 GB, 4 nodes × 3) is the project default. Alternatives:
#   nvidia_l40            — 48 GB, 2 nodes × 3 (Ada Lovelace, newer)
#   nvidia_rtx_pro_6000   — 96 GB, 10 nodes × 2 (largest mem, plentiful)
# Set to "" to let Slurm pick any free GPU.
export DAIC_EXP_GPU_TYPE="nvidia_a40"

# ── Slurm defaults (picked up by sbatch automatically) ────────────────────
# sbatch honours SBATCH_* env vars without needing -A / -p on every command.
# Precedence: command-line > SBATCH_* env > #SBATCH directives in the script.
# So sourcing env.sh is enough; no per-submission flags required.
export SBATCH_ACCOUNT="${DAIC_EXP_ACCOUNT}"
[ -n "${DAIC_EXP_PARTITION:-}" ] && export SBATCH_PARTITION="${DAIC_EXP_PARTITION}"
[ -n "${DAIC_EXP_QOS:-}" ]       && export SBATCH_QOS="${DAIC_EXP_QOS}"
# GPU gres can't be set via a single SBATCH_* var (gres has multiple types),
# so the per-job templates encode --gres=gpu:${DAIC_EXP_GPU_TYPE}:1 directly.

# ── wandb ──────────────────────────────────────────────────────────────────
# Wandb stays online so the user can monitor live from anywhere. If
# compute nodes can't reach the public internet on this cluster, flip to
# `offline` and the friend runs `wandb sync` from the login node afterwards.
export WANDB_DIR="${DAIC_EXP_DATA_DIR}/wandb"
export WANDB_MODE="online"
export WANDB_PROJECT="thesis-diffusion"
