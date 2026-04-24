"""Preprocess raw .h5 files into sliding-window .npz files.

Implements the exact DCRNN/Graph-WaveNet protocol:
  - x: (N_windows, 12, N, 2) — [speed_raw, time_of_day_float01]
  - y: (N_windows, 12, N, 1) — speed_raw only, NEVER normalised
  - npz files store raw values; normalisation is applied at load time in TrafficDataset
  - scaler (mean, std) saved to {dataset}_scaler.yaml

Usage:
    python scripts/preprocess.py --dataset metrla
    python scripts/preprocess.py --dataset pemsbay
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

X_OFFSETS = np.arange(-11, 1, 1)   # [-11, -10, ..., 0]  — 12 input steps
Y_OFFSETS = np.arange(1, 13, 1)    # [1, 2, ..., 12]     — 12 target steps

_RAW_PATHS = {
    "metrla": "data/raw/metr-la.h5",
    "pemsbay": "data/raw/pems-bay.h5",
}

# Approximate expected window counts for sanity check (from DCRNN)
_EXPECTED = {
    "metrla":  {"train": 23974, "val": 3425, "test": 6850},
    "pemsbay": {"train": 36481, "val": 5211, "test": 10423},
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess raw traffic data.")
    parser.add_argument(
        "--dataset", choices=["metrla", "pemsbay"], required=True,
        help="Dataset to preprocess."
    )
    args = parser.parse_args()
    dataset = args.dataset

    raw_path = Path(_RAW_PATHS[dataset])
    if not raw_path.exists():
        sys.exit(
            f"ERROR: {raw_path} not found.\n"
            "Run  bash scripts/download_data.sh  first."
        )

    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load raw h5 → DataFrame (T × N)
    # ------------------------------------------------------------------
    print(f"Loading {raw_path} ...")
    df = pd.read_hdf(raw_path)          # index: DatetimeIndex, columns: sensor IDs
    T, N = df.shape
    print(f"  Shape: T={T}, N={N}")

    # ------------------------------------------------------------------
    # 2. Build feature array (T, N, 2) = [speed_raw, time_of_day]
    # ------------------------------------------------------------------
    speed = df.values.astype(np.float32)                   # (T, N)
    tod   = compute_time_of_day(df.index)                  # (T,) ∈ [0, 1)
    tod_broadcast = np.tile(tod[:, None], (1, N))          # (T, N)
    data = np.stack([speed, tod_broadcast], axis=-1)       # (T, N, 2)

    # ------------------------------------------------------------------
    # 3. Chronological split indices
    # ------------------------------------------------------------------
    n_train, n_val_end = split_timestamps(T)
    n_val  = n_val_end - n_train
    n_test = T - n_val_end
    print(f"  Split: n_train={n_train}, n_val={n_val}, n_test={n_test}")

    # ------------------------------------------------------------------
    # 4. Scaler from training speed only (before any windowing)
    # ------------------------------------------------------------------
    scaler = compute_scaler(speed[:n_train])
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
        # Mask from raw speed channel (channel 0) — before any normalisation
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

    # y is raw mph (not near 0)
    train_y_mean = splits["train"][:, :, 0].mean()
    train_y_std  = splits["train"][:, :, 0].std()
    print(f"  raw speed (train): mean={train_y_mean:.2f}, std={train_y_std:.2f}  "
          f"(expect raw mph, not ~0/1)")

    print("\nDone.")


if __name__ == "__main__":
    main()
