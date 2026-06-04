# all file and folder locations for the project live here.
# import these helpers instead of hardcoding paths, so the layout is
# defined in one place and easy to change.
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RESULTS = ROOT / "results"

# raw source files keep their original names, grouped under data/raw/<dataset>/.
_RAW_PRIMARY = {
    "metrla": "metr-la.h5",
    "pemsbay": "pems-bay.h5",
    "pems08f": "PEMS08.npz",
}
_RAW_ADJ = {
    "metrla": "adj_mx_metrla.pkl",
    "pemsbay": "adj_mx_pemsbay.pkl",
    "pems08f": "PEMS08.csv",
}


# --- raw + processed data, one folder per dataset ---

def raw_dir(dataset):
    return DATA / "raw" / dataset


def raw_primary(dataset):
    # the speed/flow source file (.h5 for metr-la/pems-bay, .npz for pems08)
    return raw_dir(dataset) / _RAW_PRIMARY[dataset]


def raw_adj(dataset):
    # the adjacency source file (.pkl for metr-la/pems-bay, .csv for pems08)
    return raw_dir(dataset) / _RAW_ADJ[dataset]


def processed_dir(dataset):
    return DATA / "processed" / dataset


def split_npz(dataset, split):
    # split is "train", "val" or "test"
    return processed_dir(dataset) / f"{split}.npz"


def scaler_yaml(dataset):
    return processed_dir(dataset) / "scaler.yaml"


def laplacian_eigs(dataset):
    return processed_dir(dataset) / "laplacian_eigs.npz"


# --- results, grouped by stage then dataset ---

def backbone_dir(dataset, seed):
    return RESULTS / "backbone" / dataset / f"seed{seed}"


def residual_dir(dataset, seed):
    return RESULTS / "residuals" / dataset / f"seed{seed}"


def residual_npz(dataset, split, seed):
    return residual_dir(dataset, seed) / f"{split}.npz"


def diffusion_dir(dataset, prior, seed):
    return RESULTS / "diffusion" / dataset / prior / f"seed{seed}"


def eval_dir(dataset, prior, seed):
    return RESULTS / "eval" / dataset / prior / f"seed{seed}"


# --- run naming ---
# the wandb run name, the sbatch job name, and the slurm log file all share
# this string, so a run is easy to find across every system.

def run_name(stage, dataset, seed, prior=None):
    parts = [stage, dataset]
    if prior is not None:
        parts.append(prior)
    parts.append(f"seed{seed}")
    return "_".join(parts)
