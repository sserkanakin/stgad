# all file and folder locations for the project live here.
# import these helpers instead of hardcoding paths, so the layout is
# defined in one place and easy to change.
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RESULTS = ROOT / "results"

# six variants; raw is grouped by SOURCE (the F/S variants share one raw set).
_VARIANT_SOURCE = {
    "metrla": "metrla", "pemsbay": "pemsbay",
    "pems04f": "pems04", "pems04s": "pems04",
    "pems08f": "pems08", "pems08s": "pems08",
}
# raw source files keep their original names, grouped under data/raw/<source>/.
_RAW_PRIMARY = {
    "metrla": "metr-la.h5", "pemsbay": "pems-bay.h5",
    "pems04": "PEMS04.npz", "pems08": "PEMS08.npz",
}
_RAW_ADJ = {
    "metrla": "adj_mx.pkl", "pemsbay": "adj_mx_bay.pkl",
    "pems04": "PEMS04.csv", "pems08": "PEMS08.csv",
}


# --- raw (by source) + processed (by variant) data ---

def raw_dir(variant):
    return DATA / "raw" / _VARIANT_SOURCE[variant]


def raw_primary(variant):
    # speed/flow source file (.h5 for metr-la/pems-bay, .npz for pems04/08)
    return raw_dir(variant) / _RAW_PRIMARY[_VARIANT_SOURCE[variant]]


def raw_adj(variant):
    # adjacency source file (.pkl for metr-la/pems-bay, distance .csv for pems)
    return raw_dir(variant) / _RAW_ADJ[_VARIANT_SOURCE[variant]]


def processed_dir(dataset):
    return DATA / "processed" / dataset


def split_npz(dataset, split):
    # split is "train", "val" or "test"
    return processed_dir(dataset) / f"{split}.npz"


def adj_npz(dataset):
    # precomputed A / A_sym / L (built by scripts/build_datasets.py)
    return processed_dir(dataset) / "adj.npz"


def scaler_json(dataset):
    return processed_dir(dataset) / "scaler.json"


def gwn_native_dir(dataset):
    # GWN-repo-ready view: {train,val,test}.npz (B,12,N,2) + adj_mx.pkl
    return processed_dir(dataset) / "gwn_native"


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
