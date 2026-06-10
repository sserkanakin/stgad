# per-cluster configuration for DAIC. version-controlled (unlike env.sh), so
# the same logic travels via git to the Mac and both clusters. env.sh sets the
# few per-user / per-machine bits (NetID, secrets, off-cluster path overrides)
# and then sources this file.
#
# we run on two different clusters now:
#   stable  -> login.daic.tudelft.nl   (insy modulefiles + conda, the old setup)
#   exp     -> daic01.hpc.tudelft.nl   (2025 Lmod tree + venv overlay, new hw)
# everything that differs between them lives here, keyed off $DAIC_CLUSTER.

: "${DAIC_NETID:?set DAIC_NETID in scripts/daic/env.sh before sourcing}"

# --- pick the target cluster ------------------------------------------------
# auto-detect when sourced ON a cluster (by its module tree). off-cluster
# (the Mac, for sync/watch) defaults to exp; force either with
#   export DAIC_CLUSTER=stable   # or exp
if [ "${DAIC_CLUSTER:-auto}" = auto ]; then
  if   [ -d /apps/generic/modulefiles ]; then DAIC_CLUSTER=exp
  elif [ -d /opt/insy/modulefiles     ]; then DAIC_CLUSTER=stable
  else                                        DAIC_CLUSTER=exp   # off-cluster default
  fi
fi
export DAIC_CLUSTER

case "$DAIC_CLUSTER" in
  # ── stable DAIC (login.daic.tudelft.nl) ─────────────────────────────────
  stable)
    export DAIC_LOGIN_HOST="login.daic.tudelft.nl"
    # home is plain NFS here; data lives on /winhome (1.1 TB).
    : "${DAIC_PROJECT_DIR:=/home/nfs/${DAIC_NETID}/research-tud}"
    : "${DAIC_DATA_DIR:=/winhome/${DAIC_NETID}/research-tud-data}"

    # software: insy modulefiles + a conda env on NFS home.
    export DAIC_PYTHON_KIND="conda"
    export DAIC_MODULE_USE="/opt/insy/modulefiles"
    export DAIC_CUDA_MODULE="cuda/12.1"
    export DAIC_CUDNN_MODULE="cudnn/12-8.9.1.23"
    export DAIC_CONDA_MODULE="miniconda/3"
    export DAIC_CONDA_ENV="thesis-diffusion"
    export DAIC_CONDA_PREFIX="${HOME}/.conda/envs/${DAIC_CONDA_ENV}"

    # slurm: 'general' partition with short/medium/long QoS.
    export DAIC_PARTITION="general"
    export DAIC_QOS_SHORT="short"
    export DAIC_QOS_MEDIUM="medium"
    export DAIC_QOS_LONG="long"
    export DAIC_GRES_GPU="gpu:1"          # any gpu
    export DAIC_GRES_GPU_TYPED="gpu:a40:1" # pin an a40 when needed
    ;;

  # ── experimental DAIC (daic01.hpc.tudelft.nl) ───────────────────────────
  exp)
    export DAIC_LOGIN_HOST="daic01.hpc.tudelft.nl"
    # $HOME (/trinity/home, ~5 MB) is config-only. real space is under the two
    # auto-mounted home shares: linuxhome (8 GB, code) and windowshome (data).
    : "${DAIC_PROJECT_DIR:=${HOME}/linuxhome/research-tud}"
    : "${DAIC_DATA_DIR:=${HOME}/windowshome/research-tud-data}"

    # software: the 2025 Lmod tree provides python 3.11 + a cuda torch build;
    # we layer a venv (--system-site-packages) on top for our pure-python deps.
    export DAIC_PYTHON_KIND="modules"
    export DAIC_MODULE_TREE="2025/gpu"
    export DAIC_PYTHON_MODULE="python/3.11.9"
    export DAIC_TORCH_MODULE="py-torch/2.5.1"
    export DAIC_CUDA_MODULE="cuda/12.9"
    export DAIC_CUDNN_MODULE="cudnn/8.9.7.29-12"
    # venv lives on linuxhome (POSIX share, sibling of the project dir, outside
    # the git tree). NOT on windowshome: that share's odd permissions break
    # `python -m venv` ensurepip. linuxhome is small (8 GB) but big enough once
    # the orphaned old-cluster conda env is removed.
    export DAIC_VENV_DIR="$(dirname "${DAIC_PROJECT_DIR}")/.venvs/thesis-diffusion"

    # slurm: 'all' is the default partition; the only QoS is 'normal', so we
    # leave QoS empty (passing --qos=short would be rejected here). gres GPU
    # names are the full nvidia_* form on this cluster.
    export DAIC_PARTITION="all"
    export DAIC_QOS_SHORT=""
    export DAIC_QOS_MEDIUM=""
    export DAIC_QOS_LONG=""
    # request ANY working gpu, not just a40: ask for a generic gpu:1 and exclude
    # the rtx pro 6000 nodes (gpu36-53), which are Blackwell (sm_120) and crash
    # py-torch/2.5.1 (built for sm_80/sm_90). this opens both supported pools —
    # a40 (gpu12,23,24,29 = 12 cards) + l40 (gpu30,31 = 6 cards) — so jobs queue
    # across 18 gpus and schedule faster. see [[daic-cluster-reference]].
    export DAIC_GRES_GPU="gpu:1"
    export DAIC_GPU_EXCLUDE="gpu[36-53]"
    export DAIC_GRES_GPU_TYPED="gpu:nvidia_a40:1"   # for pinning one type if ever needed
    ;;

  *)
    echo "ERROR: unknown DAIC_CLUSTER='${DAIC_CLUSTER}' (use stable|exp)" >&2
    return 1 2>/dev/null || exit 1
    ;;
esac

# --- common (cluster-independent) -------------------------------------------
export DAIC_PYTHON_VERSION="3.11"
# wandb local run cache lives OFF the tight data quota: the data share (windowshome,
# 8G) holds the ~3.6G residuals, and wandb grows ~30M/run, so it used to fill the
# quota and crash residual/checkpoint writes. node-local /tmp is disposable (wandb
# syncs to the cloud live), so it never touches any persistent quota.
export WANDB_DIR="${WANDB_DIR:-/tmp/wandb-${USER:-sakin}}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-thesis-diffusion}"
