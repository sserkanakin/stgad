"""how close is each fitted prior to the real (held-out) residuals?

three candidate closeness measures, structured kronecker prior vs the
scale-matched isotropic prior, on the VAL split (prior is fit on train):

  nll     mean negative log-likelihood per entry (nats) of held-out residuals
          under each gaussian prior. KL(data||prior) = NLL - H(data); since
          H(data) is the same for both priors, the NLL gap IS the KL gap.
  w2_mode 2-wasserstein between each prior and a gaussian matched to the
          measured residual energy per joint eigenmode (both diagonal in the
          eigenbasis -> W2 = sqrt(sum (sqrt v - sqrt s)^2)). measures how well
          the prior's per-mode variance tracks the data's per-mode energy.
  sw      sliced 2-wasserstein between real residual fields and prior draws
          (200 random directions), tail- and structure-aware, no covariance
          inversion.

usage: python scripts/prior_closeness.py
"""
from __future__ import annotations
import numpy as np
from src.data.residual_dataset import train_residual_sigma
from src.diffusion.prior import load_prior
from src.utils import paths

DATASETS = ["metrla", "pemsbay", "pems04f", "pems04s", "pems08f", "pems08s"]


def _load_val(ds):
    z = np.load(paths.residual_npz(ds, "val", 42, "gwn"))
    R = z["residuals"].astype(np.float64)
    m = z["mask"].astype(bool)
    keep = m.all(axis=(1, 2))
    X = R[keep] / float(train_residual_sigma(ds, 42, "gwn"))
    return X                                            # (W, N, H), standardized


def _rotate(X, pr):
    """R -> V_S^T R V_T for a stack (W,N,H); fast batched matmul."""
    return (pr.V_S.T @ X) @ pr.V_T                         # (W,N,H)


def _nll_per_entry(X, pr):
    """mean gaussian NLL per residual entry under prior pr (diagonal in eigbasis)."""
    C = _rotate(X, pr)
    s = pr.sigma2                                          # (N,H) per-mode variance
    nll = 0.5 * (C ** 2 / s + np.log(2 * np.pi * s))       # (W,N,H)
    return nll.mean()


def _w2_mode(X, pr):
    """W2 between prior and a gaussian matched to measured per-mode energy."""
    C = _rotate(X, pr)
    v = (C ** 2).mean(axis=0)                              # measured energy (N,H)
    s = pr.sigma2                                          # prior variance (N,H)
    return float(np.sqrt(((np.sqrt(v) - np.sqrt(s)) ** 2).sum()))


def _sliced_w2(X, pr, n_dir=200, rng=None):
    """sliced 2-wasserstein between real fields and prior draws, flattened."""
    rng = rng or np.random.default_rng(0)
    W = X.shape[0]
    draws = pr.V_S @ (np.sqrt(pr.sigma2) *
                      rng.standard_normal((W,) + pr.theta.shape)) @ pr.V_T.T
    A = X.reshape(W, -1)
    B = draws.reshape(W, -1)
    d = A.shape[1]
    dirs = rng.standard_normal((d, n_dir))
    dirs /= np.linalg.norm(dirs, axis=0, keepdims=True)
    pa = np.sort(A @ dirs, axis=0)                         # (W, n_dir)
    pb = np.sort(B @ dirs, axis=0)
    return float(np.sqrt(((pa - pb) ** 2).mean()))


if __name__ == "__main__":
    print(f"{'dataset':9s} | {'NLL/entry':>18s} | {'W2 per-mode':>16s} | {'sliced-W2':>16s}")
    print(f"{'':9s} | {'kron':>8s} {'iso':>8s} | {'kron':>7s} {'iso':>7s} | {'kron':>7s} {'iso':>7s}")
    print("-" * 74)
    for ds in DATASETS:
        X = _load_val(ds)
        prk = load_prior(ds, "kronecker", seed=42, backbone="gwn")
        pri = load_prior(ds, "isotropic", seed=42, backbone="gwn")
        nk, ni = _nll_per_entry(X, prk), _nll_per_entry(X, pri)
        wk, wi = _w2_mode(X, prk), _w2_mode(X, pri)
        sk, si = _sliced_w2(X, prk), _sliced_w2(X, pri)
        print(f"{ds:9s} | {nk:8.3f} {ni:8.3f} | {wk:7.2f} {wi:7.2f} | {sk:7.3f} {si:7.3f}"
              f"   (iso/kron sw={si/sk:.2f}x)")
