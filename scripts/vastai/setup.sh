#!/usr/bin/env bash
# one-time setup on a vast.ai vm. run from the repo root after cloning:
#   git clone https://github.com/sserkanakin/research-tud.git
#   cd research-tud && bash scripts/vastai/setup.sh
#
# clones the vendored backbones (gitignored), installs the project, and checks
# that torch sees the gpu and the project imports.
set -euo pipefail

mkdir -p third_party logs

# graph wavenet (pinned) — the deterministic backbone
if [ ! -d third_party/Graph-WaveNet ]; then
  echo "-> cloning Graph-WaveNet"
  git clone https://github.com/nnzhan/Graph-WaveNet.git third_party/Graph-WaveNet
  ( cd third_party/Graph-WaveNet && git checkout 6b162e8 )
fi

# diffstg — provides the ugnet denoiser used by the diffusion model
if [ ! -d third_party/DiffSTG ]; then
  echo "-> cloning DiffSTG"
  git clone https://github.com/wenhaomin/DiffSTG.git third_party/DiffSTG
fi

# project deps. the vast pytorch image already ships torch+cuda; pip sees it as
# satisfied (torch>=2.2) and won't reinstall it. add tables if you preprocess
# .h5 on the vm (not needed if you rsync the processed data over).
echo "-> installing project"
python -m pip install --upgrade pip
pip install -e ".[dev]" || pip install -e .

echo
echo "-> sanity check"
python - <<'PY'
import torch
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no gpu")
import src.diffusion.forward_fcps, src.diffusion.denoiser, src.diffusion.prior  # noqa: F401
import src.backbones, src.data.adjacency  # noqa: F401
from src.utils import paths  # noqa: F401
print("imports ok")
PY

echo
echo "next:"
echo "  1. on your mac, push data:  VAST_HOST=root@<ip> VAST_PORT=<port> bash scripts/vastai/sync_to_vast.sh"
echo "  2. set wandb:               wandb login   (or export WANDB_API_KEY=...)"
echo "  3. train:                   python scripts/train_backbone.py --dataset metrla --seed 0"
