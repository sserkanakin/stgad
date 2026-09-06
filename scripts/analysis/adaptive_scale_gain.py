r"""How much would a window-ADAPTIVE prior scale gain over the static prior?

WHY THIS EXISTS. Our prior is one Sigma for every window. PriorGrad's whole point
(Lee et al., ICLR 2022) is that the prior should depend on the conditioning. The
size of a window's forecast error varies a lot with the situation -- rush hour, an
incident, a quiet night -- and a strong backbone leaves little ELSE: its residual has
weak horizon correlation and a small graph component, so a static structured prior
barely beats white noise there (pems08f/gwn: +0.17 nats/point). The question is
whether the error SIZE is predictable from things known at forecast time.

Two predictors that are legitimately available when the forecast is issued:
  * the error of the last COMPLETED window (its truth is now observed), and
  * the hour of day.
A log-linear fit of the window's mean squared error on those two gives R^2, and the
held-out gain (nats per point) of a Gaussian whose scale follows that fit over a
static Gaussian of the same total variance. 'oracle' is the ceiling if each window's
own energy were known exactly.

This gain is a per-window SCALAR, so it multiplies every mode equally and adds on
top of whatever the covariance SHAPE (Sigma_T x Sigma_S) already buys.

usage: RESULTS_ROOT=$PWD/archive/20260902/results python scripts/analysis/adaptive_scale_gain.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
warnings.filterwarnings("ignore")

from src.data.residual_dataset import train_residual_sigma
from src.utils import paths

DS = ["pems08f", "pems08s", "pems04f", "pems04s", "metrla", "pemsbay"]
BB = ["mlp", "gcn", "gwn"]


def load(ds, bb, split, seed=42):
    d = np.load(paths.residual_npz(ds, split, seed, bb))
    R = d["residuals"].astype(np.float64)
    M = d["mask"].astype(bool)
    s = np.asarray(train_residual_sigma(ds, seed, bb, norm="pernode"))
    R = R / s.reshape(-1, 1)
    tod = np.load(paths.split_npz(ds, split))["y_tod"]
    while tod.ndim > 1:                       # one time-of-day per window
        tod = tod[:, 0]
    return R, M, tod


def energy(R, M):
    """per-window mean squared error over observed entries; windows under half
    observed are dropped (metrla has fully masked windows)."""
    cnt = M.sum(axis=(1, 2))
    e = (np.where(M, R, 0.0) ** 2).sum(axis=(1, 2)) / cnt.clip(min=1)
    return e, cnt >= 0.5 * M[0].size


def design(x_prev, hour):
    return np.column_stack([np.ones_like(x_prev), x_prev, np.eye(24)[hour]])


def cell(ds, bb):
    R, M, tod = load(ds, bb, "train")
    Rt, Mt, todt = load(ds, bb, "test")
    H = R.shape[2]
    e, ok = energy(R, M)
    et, okt = energy(Rt, Mt)
    lag = H                                   # the previous non-overlapping window
    k = ok[:-lag] & ok[lag:]
    kt = okt[:-lag] & okt[lag:]
    xp, y = np.log(e[:-lag][k]), np.log(e[lag:][k])
    hr = np.clip((tod[lag:][k] * 24).astype(int), 0, 23)
    xpt, yt = np.log(et[:-lag][kt]), np.log(et[lag:][kt])
    hrt = np.clip((todt[lag:][kt] * 24).astype(int), 0, 23)

    beta, *_ = np.linalg.lstsq(design(xp, hr), y, rcond=None)
    yhat = design(xpt, hrt) @ beta
    r2_train = 1 - np.var(y - design(xp, hr) @ beta) / np.var(y)
    r2_test = 1 - np.var(yt - yhat) / np.var(yt)

    # held-out nll per point: static gaussian (mle sigma^2 = mean e) vs the adaptive
    # one (scale follows the fit, one global factor so total variance matches)
    et_ = et[lag:][kt]
    static = 0.5 * np.log(et_.mean()) + 0.5
    s2 = np.exp(yhat)
    s2 *= (et_ / s2).mean()
    adaptive = 0.5 * np.mean(np.log(s2)) + 0.5 * np.mean(et_ / s2)
    oracle = 0.5 * np.mean(np.log(et_)) + 0.5
    return dict(ds=ds, bb=bb, r2_train=float(r2_train), r2_test=float(r2_test),
                gain=float(static - adaptive), gain_oracle=float(static - oracle),
                sd_log_energy=float(np.std(np.log(et_))))


def main():
    print(f"{'ds':9s}{'bb':5s}{'R2 train':>10s}{'R2 test':>9s}{'gain (nats/pt)':>16s}{'oracle':>9s}{'sd log e':>10s}")
    print("-" * 70)
    for ds in DS:
        for bb in BB:
            try:
                r = cell(ds, bb)
            except FileNotFoundError:
                continue
            print(f"{r['ds']:9s}{r['bb']:5s}{r['r2_train']:10.2f}{r['r2_test']:9.2f}"
                  f"{r['gain']:16.3f}{r['gain_oracle']:9.3f}{r['sd_log_energy']:10.2f}", flush=True)
    print("\nreference, the whole static kron SHAPE vs iso (prior_family_nll.py): "
          "pems08f mlp +1.07 / gcn +0.77 / gwn +0.17")


if __name__ == "__main__":
    main()
