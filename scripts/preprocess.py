"""Preprocess raw traffic source files into sliding-window .npz files.

Three datasets, two source formats:
  - METR-LA / PEMS-BAY: DCRNN ``.h5`` with a DatetimeIndex; 70/10/20 split.
  - PEMS08F:            STSGCN ``.npz`` (T, N, 3) without timestamps;
                        60/20/20 split, channel 0 = flow; ToD synthesised
                        from sample index (288 slots/day).

Output (same for all):
  - x: (N_windows, 12, N, 2) — [speed_or_flow_raw, time_of_day_float01]
  - y: (N_windows, 12, N, 1) — primary channel raw, NEVER normalised
  - npz files store raw values; normalisation is applied at load time in TrafficDataset
  - scaler (mean, std) saved to {dataset}_scaler.yaml

Usage:
    python scripts/preprocess.py --dataset metrla
    python scripts/preprocess.py --dataset pemsbay
    python scripts/preprocess.py --dataset pems08f
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Make sure src is importable when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.preprocessing import (
    build_windows,
    compute_mask,
    compute_scaler,
    compute_time_of_day,
    split_timestamps,
)
from src.utils.config import load_config

X_OFFSETS = np.arange(-11, 1, 1)   # [-11, -10, ..., 0]  — 12 input steps
Y_OFFSETS = np.arange(1, 13, 1)    # [1, 2, ..., 12]     — 12 target steps

_DATASETS = ("metrla", "pemsbay", "pems08f")

# Source file paths — kept here so a missing file produces a clear error
# message before we try to read the data config.
_RAW_PATHS = {
    "metrla":  "data/raw/metr-la.h5",
    "pemsbay": "data/raw/pems-bay.h5",
    "pems08f": "data/raw/pems08/PEMS08.npz",
}

# Approximate expected window counts for sanity check.
# METR-LA / PEMS-BAY: DCRNN reference numbers (70/10/20).
# PEMS08F: T=17856, splits 60/20/20 → n_split - 23 windows each.
_EXPECTED = {
    "metrla":  {"train": 23974, "val": 3425, "test": 6850},
    "pemsbay": {"train": 36481, "val": 5211, "test": 10423},
    "pems08f": {"train": 10691, "val": 3548,  "test": 3548},
}


def _load_h5_source(raw_path: Path) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Load METR-LA / PEMS-BAY .h5 → (speed (T,N), tod (T,), T, N)."""
    df = pd.read_hdf(raw_path)         # index: DatetimeIndex, columns: sensor IDs
    T, N = df.shape
    speed = df.values.astype(np.float32)                  # (T, N)
    tod = compute_time_of_day(df.index)                   # (T,) ∈ [0, 1)
    return speed, tod, T, N


def _load_npz_source(
    raw_path: Path, channel: int, tod_size: int,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Load PEMS08F .npz → (primary (T,N), tod (T,), T, N).

    PEMS08.npz contains a single key ``data`` of shape (T, N, 3) with
    channels [flow, occupancy, speed]. The primary channel is selected by
    ``channel`` (config-driven; default 0 = flow). The file has no
    timestamps, so ToD is synthesised as ``(arange(T) mod tod_size) / tod_size``.
    """
    arr = np.load(raw_path)["data"]                       # (T, N, F) float64
    T, N, _ = arr.shape
    primary = arr[..., channel].astype(np.float32)        # (T, N)
    tod = ((np.arange(T) % tod_size) / float(tod_size)).astype(np.float32)
    return primary, tod, T, N


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess raw traffic data.")
    parser.add_argument(
        "--dataset", choices=list(_DATASETS), required=True,
        help="Dataset to preprocess."
    )
    args = parser.parse_args()
    dataset = args.dataset

    raw_path = Path(_RAW_PATHS[dataset])
    if not raw_path.exists():
        sys.exit(
            f"ERROR: {raw_path} not found.\n"
            "Run  bash scripts/download_data.sh  (METR-LA / PEMS-BAY) or "
            "download PEMS08 from STFGNN (PEMS08F)."
        )

    cfg = load_config(f"configs/data/{dataset}.yaml")
    train_frac = float(cfg.get("split_train", 0.7))
    test_frac = 1.0 - float(cfg.get("split_val", 0.8))
    tod_size = int(cfg.get("tod_size", 288))
    channel = int(cfg.get("channel", 0))   # only used for .npz source

    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load raw source — branch on extension
    # ------------------------------------------------------------------
    print(f"Loading {raw_path} ...")
    if raw_path.suffix == ".h5":
        primary, tod, T, N = _load_h5_source(raw_path)
        source_kind = "h5"
    elif raw_path.suffix == ".npz":
        primary, tod, T, N = _load_npz_source(raw_path, channel, tod_size)
        source_kind = f"npz ch={channel}"
    else:
        sys.exit(f"ERROR: unsupported source format {raw_path.suffix}")
    print(f"  Source: {source_kind}   T={T}, N={N}")

    # ------------------------------------------------------------------
    # 2. Build feature array (T, N, 2) = [primary_raw, time_of_day]
    # ------------------------------------------------------------------
    tod_broadcast = np.tile(tod[:, None], (1, N))          # (T, N)
    data = np.stack([primary, tod_broadcast], axis=-1)     # (T, N, 2)

    # ------------------------------------------------------------------
    # 3. Chronological split indices (ratios from config)
    # ------------------------------------------------------------------
    n_train, n_val_end = split_timestamps(T, train_frac=train_frac, test_frac=test_frac)
    n_val  = n_val_end - n_train
    n_test = T - n_val_end
    print(f"  Split ({train_frac:.0%}/{1-train_frac-test_frac:.0%}/{test_frac:.0%}): "
          f"n_train={n_train}, n_val={n_val}, n_test={n_test}")

    # ------------------------------------------------------------------
    # 4. Scaler from training-split primary channel only (before windowing)
    # ------------------------------------------------------------------
    scaler = compute_scaler(primary[:n_train])
    print(f"  Scaler: mean={scaler['mean']:.4f}, std={scaler['std']:.4f}")

    scaler_path = out_dir / f"{dataset}_scaler.yaml"
    with open(scaler_path, "w") as f:
        yaml.dump(scaler, f, default_flow_style=False)
    print(f"  Saved scaler → {scaler_path}")

    # ------------------------------------------------------------------
    # 5. Build windows per split and save raw .npz files
    # ------------------------------------------------------------------
    splits = {
        "train": data[:n_train],
        "val":   data[n_train:n_val_end],
        "test":  data[n_val_end:],
    }

    for split_name, data_split in splits.items():
        x, y = build_windows(data_split, X_OFFSETS, Y_OFFSETS)
        # Mask from raw primary channel (index 0) — before any normalisation
        x_mask = compute_mask(x[..., :1])   # (N_windows, 12, N, 1) bool
        y_mask = compute_mask(y)             # (N_windows, 12, N, 1) bool

        out_path = out_dir / f"{dataset}_{split_name}.npz"
        np.savez_compressed(out_path, x=x, y=y, x_mask=x_mask, y_mask=y_mask)

        n_windows = len(x)
        expected  = _EXPECTED[dataset][split_name]
        flag = "" if abs(n_windows - expected) / expected < 0.01 else "  ⚠ unexpected count"
        print(
            f"  {split_name:5s}: {n_windows:6d} windows  "
            f"x={x.shape}  y={y.shape}{flag}"
        )
        print(f"         saved → {out_path}")

    # ------------------------------------------------------------------
    # 6. Quick sanity checks
    # ------------------------------------------------------------------
    print("\nSanity checks:")

    # time_of_day range
    tod_sample = data[:5, 0, 1]
    print(f"  x[...,1] first 5 tod values: {tod_sample}  (expect floats in [0,1))")

    # y is raw primary channel (mph for METR-LA/PEMS-BAY, vehicles/5min for PEMS08F)
    train_y_mean = splits["train"][:, :, 0].mean()
    train_y_std  = splits["train"][:, :, 0].std()
    print(f"  raw primary (train): mean={train_y_mean:.2f}, std={train_y_std:.2f}  "
          f"(expect raw units, not ~0/1)")

    print("\nDone.")


if __name__ == "__main__":
    main()
