r"""Derive the prior's design choices from the data instead of inheriting them.

Every number the prior depends on is scored the same way: fit on TRAIN, evaluate the
Gaussian log-likelihood of the TEST residual field, per mode, in nats. Because every
prior in this family is diagonal in some (V_S, V_T) eigenbasis, that reduces to a
closed form with no sampling and no training:

    rotate      C = V_S^T R V_T                     (per window)
    energy      e[k,h] = mean_windows C[k,h]^2      (measured, on the split being scored)
    NLL/mode    (1/NH) sum_{k,h} [ 0.5 log(2 pi v[k,h]) + 0.5 e[k,h] / v[k,h] ]

with v[k,h] the prior's stationary variance for that mode. Reported as a DELTA against
the scale-matched isotropic prior, so positive = better than white noise. This is the
same estimator as the attribution table in the experiments chapter, applied to design
choices rather than to factors.

What it answers, in order:

  1. `family`  -- every member of the GTCNN product-graph family on one axis, including
                  the one the thesis builds. Theta[k,h] for each:
                     iso          const
                     spatial      lam_k + g
                     temporal     1 / mu_h
                     tensor       (lam_k + g) / mu_h        <- OURS (Kronecker product)
                     cartesian    (lam_k + g) + w_h         <- the heat equation
                     strong       (1+lam_k+g)(1+w_h) - 1
                  So the thesis's prior is not outside the product-graph family; it is
                  its TENSOR member, with a fitted temporal GSO in place of a Laplacian.
  2. `gamma`   -- the spatial shift, currently 0.8 inherited from GAD. Swept.
  3. `rho`     -- the temporal ridge, currently 0.1 chosen by conditioning number. Swept,
                  including rho = 0 (no ridge at all).
  4. `sigma_s` -- how Sigma_S is normalised. Unit MEAN EIGENVALUE (what the code does)
                  leaves the implied per-node variance varying by up to 1.31x across the
                  network, even though the residual was per-sensor standardised to unit
                  variance first. Unit DIAGONAL removes that. This is a specification
                  question, not a tuning one.

usage: python scripts/analysis/prior_design.py [family|gamma|rho|sigma_s|all]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.adjacency import compute_normalized_laplacian, load_adjacency  # noqa: E402
from src.data.residual_dataset import train_residual_sigma  # noqa: E402
from src.utils import paths  # noqa: E402

DATASETS = ["pems08f", "pems08s", "pems04f", "pems04s", "metrla", "pemsbay"]
BACKBONES = ["mlp", "gcn", "gwn"]
SEED = 42
LOG2PI = float(np.log(2.0 * np.pi))


# ─── data ────────────────────────────────────────────────────────────────────

def load_split(ds, bb, split):
    """(R, mask) in the per-sensor normalised space the diffusion runs in."""
    sig = np.asarray(train_residual_sigma(ds, SEED, bb, norm="pernode"))
    d = np.load(paths.residual_npz(ds, split, SEED, bb))
    R = d["residuals"].astype(np.float64)
    m = d["mask"].astype(bool)
    return R / sig.reshape(-1, 1), m


def laplacian_eig(ds):
    L = compute_normalized_laplacian(load_adjacency(ds, repair=True))
    lam, V = np.linalg.eigh(L)
    o = np.argsort(lam)
    return lam[o], V[:, o]


def path_laplacian_eig(H):
    A = np.zeros((H, H))
    i = np.arange(H - 1)
    A[i, i + 1] = A[i + 1, i] = 1.0
    w, V = np.linalg.eigh(compute_normalized_laplacian(A))
    return np.clip(w, 0.0, None), V


def fit_sigma_T(R, mask, rho):
    """Sigma_T = D^{1/2} R_H D^{1/2} with D the per-horizon second moment and R_H the
    Pearson horizon correlation, then eigen-ridged toward its own mean eigenvalue."""
    B, N, H = R.shape
    flat = R.reshape(B * N, H)
    keep = mask.reshape(B * N, H).all(axis=1)
    X = flat[keep]
    delta = (X ** 2).mean(axis=0)
    RH = np.corrcoef(X.T)
    S = (np.sqrt(delta)[:, None] * RH) * np.sqrt(delta)[None, :]
    mu, V = np.linalg.eigh(S)
    mu = np.clip(mu, 1e-12, None)
    if rho:
        mu = (1.0 - rho) * mu + rho * mu.mean()
    return mu, V


def mode_energy(R, mask, V_S, V_T, max_windows=4096):
    """e[k,h] = mean over windows of (V_S^T R V_T)[k,h]^2, masked entries zeroed.

    done as two BLAS matmuls rather than one einsum: the naive contraction is
    O(B N^2 H + B N H^2) with a bad access pattern and takes minutes on pemsbay.
    windows are subsampled because this is a mean of squares over 10^3-10^4 windows
    and converges long before the split is exhausted.
    """
    Rz = np.where(mask, R, 0.0)
    if Rz.shape[0] > max_windows:                    # deterministic stride, no RNG
        Rz = Rz[:: int(np.ceil(Rz.shape[0] / max_windows))]
    B, N, H = Rz.shape
    #  (N,N) @ (N, B*H) -> (N, B*H)
    tmp = V_S.T @ Rz.transpose(1, 0, 2).reshape(N, B * H)
    #  -> (B*N, H) @ (H,H) -> (B*N, H)
    C = tmp.reshape(N, B, H).transpose(1, 0, 2).reshape(B * N, H) @ V_T
    return (C ** 2).reshape(B, N, H).mean(axis=0)


def nll_per_mode(v, e):
    """Gaussian NLL per mode, nats. v = prior variance, e = measured energy."""
    v = np.clip(v, 1e-12, None)
    return float(np.mean(0.5 * (LOG2PI + np.log(v)) + 0.5 * e / v))


def scale_match(theta, target_var):
    """sigma^2 such that the prior's MEAN stationary variance equals target_var."""
    inv = 1.0 / theta
    return theta, float(target_var / inv.mean())


def score(theta, e_test, target_var):
    """delta-NLL per mode against the scale-matched isotropic prior; positive = better."""
    theta, s2 = scale_match(theta, target_var)
    ours = nll_per_mode(s2 / theta, e_test)
    iso = nll_per_mode(np.full_like(theta, target_var), e_test)
    return iso - ours


# ─── the product-graph family ────────────────────────────────────────────────

def family_thetas(lam, w, mu, gamma):
    """Theta[k,h] for every member. lam = graph Laplacian spectrum, w = horizon path
    Laplacian spectrum, mu = fitted temporal covariance eigenvalues."""
    a = (lam + gamma)[:, None]                 # spatial precision
    t = (1.0 / mu)[None, :]                    # fitted temporal precision
    p = (w + gamma)[None, :]                   # horizon-path precision (heat)
    return {
        "iso":              np.ones_like(a * t),
        "spatial only":     np.broadcast_to(a, (len(lam), len(mu))).copy(),
        "temporal only":    np.broadcast_to(t, (len(lam), len(mu))).copy(),
        "TENSOR (ours)":    a * t,
        "cartesian heat":   a + p,
        "strong heat":      (1.0 + a) * (1.0 + p) - 1.0,
        "tensor heat":      a * p,
        "cartesian fitted": a + t,
    }


# ─── reports ─────────────────────────────────────────────────────────────────

def _prep(ds, bb, gamma=0.8, rho=0.1):
    Rtr, mtr = load_split(ds, bb, "train")
    Rte, mte = load_split(ds, bb, "test")
    lam, V_S = laplacian_eig(ds)
    mu, V_T = fit_sigma_T(Rtr, mtr, rho)
    w, _ = path_laplacian_eig(Rtr.shape[-1])
    e_test = mode_energy(Rte, mte, V_S, V_T)
    target = float(Rtr[mtr].var())
    return lam, w, mu, e_test, target, V_S, V_T, Rtr, mtr, Rte, mte


def r_family():
    print("=== 1. the product-graph family, held-out delta-NLL per mode vs isotropic ===")
    print("    (fit on train, scored on test; positive = better than white noise)\n")
    names = None
    acc = {}
    for ds in DATASETS:
        for bb in BACKBONES:
            lam, w, mu, e, tgt, *_ = _prep(ds, bb)
            th = family_thetas(lam, w, mu, 0.8)
            if names is None:
                names = list(th)
                print(f"{'ds':9s}{'bb':5s}" + "".join(f"{n[:16]:>18s}" for n in names))
            row = {n: score(th[n], e, tgt) for n in names}
            for n in names:
                acc.setdefault(n, []).append(row[n])
            print(f"{ds:9s}{bb:5s}" + "".join(f"{row[n]:18.4f}" for n in names))
    print(f"\n{'MEAN':14s}" + "".join(f"{np.mean(acc[n]):18.4f}" for n in names))
    print(f"{'WINS vs iso':14s}" + "".join(f"{sum(x>0 for x in acc[n]):15d}/18" for n in names))


def r_gamma():
    print("=== 2. the spatial shift gamma, held-out delta-NLL per mode ===")
    print("    Theta = (lam_k + gamma) / mu_h, everything else fixed at rho=0.1\n")
    grid = [0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2]
    print(f"{'ds':9s}{'bb':5s}" + "".join(f"{g:>10}" for g in grid) + f"{'argmax':>10s}")
    best = []
    for ds in DATASETS:
        for bb in BACKBONES:
            lam, w, mu, e, tgt, *_ = _prep(ds, bb)
            vals = [score((lam + g)[:, None] * (1.0 / mu)[None, :], e, tgt) for g in grid]
            b = grid[int(np.argmax(vals))]
            best.append(b)
            print(f"{ds:9s}{bb:5s}" + "".join(f"{v:10.4f}" for v in vals) + f"{b:10.2f}")
    print(f"\nargmax gamma: median {np.median(best):.2f}, "
          f"range [{min(best):.2f}, {max(best):.2f}]   (code default 0.8, from GAD)")


def r_rho():
    print("=== 3. the temporal ridge rho, held-out delta-NLL per mode ===")
    print("    Theta = (lam_k + 0.8) / mu_h(rho); rho=0 is the unregularised fit\n")
    grid = [0.0, 0.01, 0.03, 0.1, 0.2, 0.4, 0.8]
    print(f"{'ds':9s}{'bb':5s}" + "".join(f"{g:>10}" for g in grid)
          + f"{'argmax':>8s}{'cond(S_T)':>11s}")
    best = []
    for ds in DATASETS:
        for bb in BACKBONES:
            Rtr, mtr = load_split(ds, bb, "train")
            Rte, mte = load_split(ds, bb, "test")
            lam, V_S = laplacian_eig(ds)
            tgt = float(Rtr[mtr].var())
            vals, cond0 = [], None
            for g in grid:
                mu, V_T = fit_sigma_T(Rtr, mtr, g)
                if cond0 is None:
                    cond0 = mu.max() / mu.min()
                e = mode_energy(Rte, mte, V_S, V_T)
                vals.append(score((lam + 0.8)[:, None] * (1.0 / mu)[None, :], e, tgt))
            b = grid[int(np.argmax(vals))]
            best.append(b)
            print(f"{ds:9s}{bb:5s}" + "".join(f"{v:10.4f}" for v in vals)
                  + f"{b:8.2f}{cond0:11.1f}")
    print(f"\nargmax rho: median {np.median(best):.2f}, "
          f"range [{min(best):.2f}, {max(best):.2f}]   (code default 0.1)")


def r_sigma_s():
    print("=== 4. how Sigma_S is normalised: unit mean eigenvalue vs unit diagonal ===")
    print("    the residual is per-sensor standardised to unit variance BEFORE the prior")
    print("    is built, so a correctly specified prior should give every node unit")
    print("    variance too. unit-mean-eigenvalue (the code) does not.\n")
    print(f"{'ds':9s}{'gamma':>7s}{'diag min':>10s}{'diag max':>10s}{'ratio':>8s}"
          f"{'meanEig dNLL':>14s}{'unitDiag dNLL':>15s}")
    for ds in DATASETS:
        lam, V_S = laplacian_eig(ds)
        for g in (0.8, 0.2):
            Sig = V_S @ np.diag(1.0 / (lam + g)) @ V_S.T
            d = np.diag(Sig)
            # arm A: unit mean eigenvalue (what load_prior does)
            tau2 = len(lam) / (1.0 / (lam + g)).sum()
            qA = (lam + g) / tau2
            # arm B: unit diagonal -- rescale the GMRF to a correlation matrix
            Dm = np.diag(1.0 / np.sqrt(d))
            SigB = Dm @ Sig @ Dm
            wB, VB = np.linalg.eigh(SigB)
            qB = 1.0 / np.clip(wB, 1e-12, None)
            outA, outB = [], []
            for bb in BACKBONES:
                Rtr, mtr = load_split(ds, bb, "train")
                Rte, mte = load_split(ds, bb, "test")
                mu, V_T = fit_sigma_T(Rtr, mtr, 0.1)
                tgt = float(Rtr[mtr].var())
                outA.append(score(qA[:, None] * (1.0 / mu)[None, :],
                                  mode_energy(Rte, mte, V_S, V_T), tgt))
                outB.append(score(qB[:, None] * (1.0 / mu)[None, :],
                                  mode_energy(Rte, mte, VB, V_T), tgt))
            print(f"{ds:9s}{g:7.2f}{d.min():10.4f}{d.max():10.4f}"
                  f"{d.max() / d.min():8.2f}{np.mean(outA):14.4f}{np.mean(outB):15.4f}")


REPORTS = {"family": r_family, "gamma": r_gamma, "rho": r_rho, "sigma_s": r_sigma_s}

if __name__ == "__main__":
    want = sys.argv[1:] or ["all"]
    if want == ["all"]:
        want = list(REPORTS)
    for k in want:
        REPORTS[k]()
        print()
