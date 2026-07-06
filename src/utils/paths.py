# all file and folder locations for the project live here.
# import these helpers instead of hardcoding paths, so the layout is
# defined in one place and easy to change.
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# three env-overridable roots so DAIC runs are organized and quota-safe:
#   DATA_ROOT    inputs: data/raw, data/processed          (the data share)
#   RESULTS_ROOT inputs: residuals, backbone               (the data share, where sync put them)
#   OUTPUT_ROOT  WRITES: diffusion checkpoints, eval, logs (windowshome — one clean tree)
# each defaults to the repo (local dev). keeping reads on the data share and writes on
# windowshome means linuxhome only ever holds code.
DATA = Path(os.environ["DATA_ROOT"]) if os.environ.get("DATA_ROOT") else ROOT / "data"
RESULTS = Path(os.environ["RESULTS_ROOT"]) if os.environ.get("RESULTS_ROOT") else ROOT / "results"
OUTPUTS = Path(os.environ["OUTPUT_ROOT"]) if os.environ.get("OUTPUT_ROOT") else RESULTS

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


def laplacian_eigs(dataset, repaired=False):
    # separate cache for the repaired (single-component) graph so old analyses
    # built on the raw graph stay valid side by side.
    name = "laplacian_eigs_repaired.npz" if repaired else "laplacian_eigs.npz"
    return processed_dir(dataset) / name


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
    # with an exp tag the tree is EXPERIMENT-FIRST (one dated folder per launch,
    # per-dataset inside, train + eval side by side — easy to browse):
    #   <OUTPUTS>/<exp>/<dataset>/<backbone>/<prior>/seed<seed>/{train,eval}
    # without exp (local dev) the legacy flat layout is kept.
    # OUTPUTS (writes) may differ from RESULTS (reads) — see RESULTS_ROOT above.
    if exp:
        return OUTPUTS / exp / dataset / backbone / prior / f"seed{seed}" / "train"
    return OUTPUTS / "diffusion" / dataset / backbone / prior / f"seed{seed}"


def eval_dir(dataset, prior, seed, exp=None, backbone="gwn"):
    if exp:
        return OUTPUTS / exp / dataset / backbone / prior / f"seed{seed}" / "eval"
    return OUTPUTS / "eval" / dataset / backbone / prior / f"seed{seed}"


def exp_root(exp):
    """root of one experiment launch (connectivity.json, summary.md live here)."""
    return OUTPUTS / exp


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
