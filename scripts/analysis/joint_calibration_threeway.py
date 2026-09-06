r"""Joint (aggregate) calibration, three ways: DiffSTG (1M params, its own samples on
disk), our isotropic control and our Kronecker prior (both 17k, corrected sampler).

WHY. Per-point CRPS is won by the GWN backbone plus a standard deviation -- even a plain
Gaussian residual beats every published method (gaussian_no_diffusion_baseline.py). It
cannot separate priors. The JOINT distribution can: anything that sums errors (network
total, hour-ahead cumulative flow) exposes independence, and a covariance prior is the
only lever a small denoiser has there.

unit-free statistics (windows cannot be aligned across pipelines): range coverage of K
samples against its nominal (K-1)/(K+1), and predictive spread / actual error of a SUM
(1.0 = calibrated). see joint_stats.py.

result 2026-09-06 (network total, spread/err):   pems08f   pems08s
    DiffSTG 1M                                       1.12      0.30
    ours iso 17k                                     0.42      0.39
    ours kron 17k, gamma=0.8 (today)                 0.47      0.44
    ours kron 17k, matched (beta,gamma), prior-implied 0.98    1.01   (match_total_spread.py)

usage:
  python scripts/analysis/joint_calibration_threeway.py diffstg
  DS=pems08f RESULTS_ROOT=... OUTPUT_ROOT=... python scripts/analysis/joint_calibration_threeway.py ours
"""
import sys, os, glob, pickle, warnings
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2])); sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")
from joint_stats import stats, show

ARCH = "archive/20260902/results/model-ugnet-notcomplete/results"

def diffstg():
    for ds in ("pems08f", "pems08s", "pems04f", "pems04s", "metrla"):
        p = sorted(glob.glob(f"{ARCH}/*+{ds}+*43.pkl"))
        if not p: continue
        try: d = pickle.load(open(p[0], "rb"))
        except Exception as e: print(f"DiffSTG {ds}: unreadable ({e})"); continue
        S = d[0].numpy()[:, :, 12:, :, 0].transpose(1, 0, 3, 2)   # (K,B,N,H) future half
        Y = d[1].numpy()[:, 12:, :, 0].transpose(0, 2, 1)          # (B,N,H)
        show(f"DiffSTG  {ds}", stats(S, Y))

if __name__ == "__main__":
    if sys.argv[1:] == ["diffstg"]: diffstg()
    else:
        print("for 'ours' run the corrected-sampler variant in scripts/analysis/aggregate_metrics_check.py "
              "(same statistics via joint_stats.stats on its S / truth arrays)")
