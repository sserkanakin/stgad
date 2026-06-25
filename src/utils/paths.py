# all file and folder locations for the project live here.
# import these helpers instead of hardcoding paths, so the layout is
# defined in one place and easy to change.
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# data + results roots are env-overridable so DAIC runs read/write from the large
# /tudelft.net data share (DAIC_DATA_DIR) instead of the tiny 8 GB home shares. set
# DATA_ROOT (raw/processed) and RESULTS_ROOT (residuals/backbone/diffusion/eval) to
# that share; both default to the repo's data/ and results/ for local dev.
DATA = Path(os.environ["DATA_ROOT"]) if os.environ.get("DATA_ROOT") else ROOT / "data"
RESULTS = Path(os.environ["RESULTS_ROOT"]) if os.environ.get("RESULTS_ROOT") else ROOT / "results"
OUTPUTS = RESULTS     # writes (diffusion/eval) go to the same root as the reads

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

# backbone namespaces the residual/diffusion/eval tree so the gwn/gcn/mlp grid
# never overwrites: results/<stage>/<dataset>/<backbone>/.../seed<seed>.

def backbone_dir(dataset, seed, backbone="gwn"):
    return RESULTS / "backbone" / dataset / backbone / f"seed{seed}"


def residual_dir(dataset, seed, backbone="gwn"):
    return RESULTS / "residuals" / dataset / backbone / f"seed{seed}"


def residual_npz(dataset, split, seed, backbone="gwn"):
    return residual_dir(dataset, seed, backbone) / f"{split}.npz"


def diffusion_dir(dataset, prior, seed, exp=None, backbone="gwn"):
    # exp namespaces a whole experiment run (<OUTPUTS>/diffusion/<exp>/...), so a
    # fresh launch is fully separate and a re-run never clobbers an old one.
    # OUTPUTS (writes) may differ from RESULTS (reads) — see RESULTS_ROOT above.
    base = OUTPUTS / "diffusion" / exp if exp else OUTPUTS / "diffusion"
    return base / dataset / backbone / prior / f"seed{seed}"


def eval_dir(dataset, prior, seed, exp=None, backbone="gwn"):
    base = OUTPUTS / "eval" / exp if exp else OUTPUTS / "eval"
    return base / dataset / backbone / prior / f"seed{seed}"


# --- run naming ---
# the wandb run name, the sbatch job name, and the slurm log file all share
# this string, so a run is easy to find across every system.

def run_name(stage, dataset, seed, prior=None, backbone=None):
    parts = [stage]
    if backbone is not None:
        parts.append(backbone)
    parts.append(dataset)
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
