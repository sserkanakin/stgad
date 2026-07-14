#!/usr/bin/env python
"""build the 'raw signal' diffusion cache from the processed split.npz.

the pure-signal arm (backbone='raw') diffuses the FUTURE signal directly, conditioned
on the HISTORY window (DiffSTG-style), with NO trained backbone. to reuse the residual
pipeline unchanged we write the future signal under the "residuals" key (so the prior
fit + sigma_r + dataset all work as-is) and add a "history" field for the conditioning;
"y_pred" is zeros so y_true = y_pred + residual = the signal.

  python scripts/build_signal_cache.py --dataset pems08f
  python scripts/build_signal_cache.py --all

writes results/residuals/<ds>/raw/seed42/{train,val,test}.npz with:
  residuals (B,N,H)  the future signal, raw units (the diffusion target)
  history   (B,N,T_h) the history window, raw units (the conditioning)
  y_pred    (B,N,H)  zeros (so y_true = signal)
  mask      (B,N,H)  bool (future observation validity)
  mean,std           the signal scaler (train only), for z-scoring the conditioning
"""
from __future__ import annotations

import argparse

import numpy as np

from src.utils import paths

# the raw signal arm is backbone-free, so it is seed-independent; we still write under
# seed 42 to fit the residual path scheme (results/residuals/<ds>/raw/seed42/).
RAW_SEED = 42
ALL = ["metrla", "pemsbay", "pems04f", "pems04s", "pems08f", "pems08s"]


def _bnh(a: np.ndarray) -> np.ndarray:
    """(B, T, N, 1) -> (B, N, T)."""
    return np.ascontiguousarray(a[..., 0].transpose(0, 2, 1))


def build(dataset: str) -> None:
    # scaler from the TRAIN split's observed signal (z-score; matches the standard
    # DiffSTG/CSDI convention of train-only stats).
    tr = np.load(paths.split_npz(dataset, "train"))
    xtr = tr["x"][..., 0]
    mtr = tr["x_mask"][..., 0].astype(bool)
    mean = float(xtr[mtr].mean())
    std = float(xtr[mtr].std())
    print(f"[signal] {dataset}: scaler mean={mean:.3f} std={std:.3f}")

    out_dir = paths.residual_npz(dataset, "train", RAW_SEED, "raw").parent
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        d = np.load(paths.split_npz(dataset, split))
        signal = _bnh(d["y"]).astype(np.float32)            # future, raw (B,N,H)
        history = _bnh(d["x"]).astype(np.float32)           # history, raw (B,N,T_h)
        mask = _bnh(d["y_mask"]).astype(bool)               # future validity
        # CENTER the diffusion target: the zero-mean GAD prior only matches a zero-mean
        # target. we treat the train-signal MEAN as the trivial "backbone forecast" —
        # store residuals = signal - mean (zero-mean) and y_pred = mean, so the prior fit,
        # sigma_r, and the eval reconstruction (y_pred + σ·sample = mean + std·sample) all
        # stay correct with NO downstream code change. (uncentered, the prior's variance
        # was inflated up to ~20x by mean^2 and its mean was off by several sigma.)
        centered = (signal - np.float32(mean))              # (B,N,H) zero-mean target
        y_pred = np.full_like(signal, np.float32(mean))     # climatology mean = trivial backbone
        path = paths.residual_npz(dataset, split, RAW_SEED, "raw")
        np.savez_compressed(path, residuals=centered, history=history, y_pred=y_pred,
                            mask=mask, mean=np.float32(mean), std=np.float32(std))
        vr = centered[mask]
        print(f"  {split}: signal {signal.shape} history {history.shape}  "
              f"centered mean={vr.mean():+.3f} std={vr.std():.2f}  missing={1 - mask.mean():.4f}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset", choices=ALL)
    p.add_argument("--all", action="store_true")
    args = p.parse_args()
    if args.all:
        for ds in ALL:
            build(ds)
    elif args.dataset:
        build(args.dataset)
    else:
        p.error("give --dataset or --all")


if __name__ == "__main__":
    main()
