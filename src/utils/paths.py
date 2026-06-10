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


def diffusion_dir(dataset, prior, seed, exp=None):
    # exp namespaces a whole experiment run (results/diffusion/<exp>/...), so a
    # fresh launch is fully separate and a re-run never clobbers an old one.
    base = RESULTS / "diffusion" / exp if exp else RESULTS / "diffusion"
    return base / dataset / prior / f"seed{seed}"


def eval_dir(dataset, prior, seed, exp=None):
    base = RESULTS / "eval" / exp if exp else RESULTS / "eval"
    return base / dataset / prior / f"seed{seed}"


# --- run naming ---
# the wandb run name, the sbatch job name, and the slurm log file all share
# this string, so a run is easy to find across every system.

def run_name(stage, dataset, seed, prior=None):
    parts = [stage, dataset]
    if prior is not None:
        parts.append(prior)
    parts.append(f"seed{seed}")
    return "_".join(parts)


# --- preflight helpers ---
# call these at the very top of a job so it fails in the first seconds with a
# clear message, never deep into a long run or on a missing output directory.

def ensure_dirs(*dirs):
    """mkdir -p every directory passed in."""
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def require_inputs(items):
    """check required input files exist; exit cleanly if any are missing.

    items is a list of (path, hint) pairs; hint says how to produce the file.
    raises SystemExit (no traceback) listing everything that's missing.
    """
    missing = [(Path(p), hint) for p, hint in items if not Path(p).exists()]
    if missing:
        lines = ["[preflight] cannot start - missing required inputs:"]
        for p, hint in missing:
            lines.append(f"  - {p}")
            lines.append(f"      -> {hint}")
        raise SystemExit("\n".join(lines))
