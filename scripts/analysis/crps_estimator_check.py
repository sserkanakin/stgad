r"""Is `empirical_crps` the fair estimator its docstring claims? (It is not.)

fair (unbiased):  E|X-y| - (1/(2K(K-1))) * sum_ij |x_i - x_j|
naive  (biased):  E|X-y| - (1/(2K^2))    * sum_ij |x_i - x_j|

src/eval/crps.py uses w = (2i-K-1)/K^2, i.e. the 1/K^2 form. Its expectation is

    CRPS_true + E|X-X'| / (2K)

-- a POSITIVE bias proportional to ensemble SPREAD. At equal true skill the NARROWER
ensemble therefore scores better, which is a thumb on the scale for whichever arm the
sampler under-disperses. It matters most where K is smallest, and K is smallest exactly
where it does the most damage: `--val-k 4` drives best_crps.pt, early stopping and the
LR plateau.

usage: python scripts/analysis/crps_estimator_check.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.eval.crps import empirical_crps

TRUE = 1.0 / math.sqrt(math.pi)      # CRPS of N(0,1) predicting a N(0,1) draw


def fair_crps(samples, y):
    K = samples.shape[0]
    t1 = (samples - y.unsqueeze(0)).abs().mean(0)
    xs, _ = torch.sort(samples, dim=0)
    idx = torch.arange(1, K + 1, dtype=samples.dtype)
    w = (2 * idx - K - 1) / (K * (K - 1))
    return t1 - (w.view(K, 1, 1, 1) * xs).sum(0)


def main():
    torch.manual_seed(0)
    B = 20000
    y = torch.randn(B, 1, 1)
    print(f"analytic CRPS for a calibrated unit-normal forecast: {TRUE:.5f}\n")
    print(f"{'K':>5s}{'coded (1/K^2)':>16s}{'fair (1/K(K-1))':>18s}{'coded bias':>13s}")
    for K in (4, 8, 20, 50, 100):
        s = torch.randn(K, B, 1, 1)
        c = empirical_crps(s, y).mean().item()
        f = fair_crps(s, y).mean().item()
        print(f"{K:5d}{c:16.5f}{f:18.5f}{(c / TRUE - 1) * 100:12.2f}%")

    print("\n=== equal true skill, different spread ===")
    print("A is calibrated; B is under-dispersed x0.87 (what the oracle test measures")
    print("for the structured prior under the locked protocol).")
    for K in (4, 100):
        a = torch.randn(K, B, 1, 1)
        b = torch.randn(K, B, 1, 1) * 0.87
        ca, cb = empirical_crps(a, y).mean().item(), empirical_crps(b, y).mean().item()
        fa, fb = fair_crps(a, y).mean().item(), fair_crps(b, y).mean().item()
        print(f"  K={K:3d}  coded: B looks {(ca - cb) / ca * 100:+.2f}% better"
              f"   |  fair: B is {(fa - fb) / fa * 100:+.2f}%")


if __name__ == "__main__":
    main()
