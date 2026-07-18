"""Legible spectra figures: WHERE does the signal's energy sit in frequency, and
does the structured prior put it in the same place?

Form = CUMULATIVE energy curves (read a value straight off: "X% of the energy sits
below frequency Y"), not raw spectra. Axes are consistent and labelled:
  - spatial  x = graph-Laplacian eigenvalue λ ∈ [0, 2]  (0 = smoothest, 2 = most
    oscillatory; the sym-normalized Laplacian is ALWAYS bounded by 2 -- that is why
    every spatial panel ends at ~2).
  - temporal x = temporal mode index 1..H (1 = the common/DC mode).

γ = 0.04 fixed = the Tikhonov shift that lifts the DC zero-mode so (L+γI) inverts;
NOT a smoothness knob.

Figures (results/figures/):
  spectrum_spatial.png   cumulative energy vs λ, all datasets: raw signal vs
                         (L+γI)^-1 prior vs isotropic. reads "N% below λ".
  spectrum_temporal.png  cumulative variance vs temporal mode: empirical Σ_T
                         (covariance) vs path-graph Σ_T.
  spectrum_joint_<ds>.png  marginal cumulative curves (spatial + temporal) of the
                         data vs the 4 joint constructions -- which one tracks the data.
Usage: .venv/bin/python scripts/plot_spectra.py
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.adjacency import load_adjacency, compute_normalized_laplacian   # noqa: E402
from src.utils import paths                                                    # noqa: E402

TRAFFIC = ["metrla", "pemsbay", "pems04f", "pems04s", "pems08f", "pems08s"]
GAMMA = 0.04
GAMMA_T = 0.04
# colorblind-safe (Wong), assigned by identity in fixed order
C = {"data": "#000000", "kron": "#0072B2", "iso": "#999999", "path": "#D55E00",
     "kroncov": "#0072B2", "krongraph": "#56B4E9", "cartesian": "#009E73", "strong": "#CC79A7"}
FIG = ROOT / "results" / "figures"
plt.rcParams.update({"font.size": 11, "axes.titlesize": 11, "figure.dpi": 140})


def spatial_eig(ds):
    if ds == "molene":
        L = np.load(paths.processed_dir("molene") / "adj.npz")["L"]
    else:
        L = compute_normalized_laplacian(load_adjacency(ds, repair=True))
    lam, V = np.linalg.eigh(L)
    o = np.argsort(lam)
    return lam[o], V[:, o]


def path_laplacian_eig(H):
    A = np.zeros((H, H))
    for i in range(H - 1):
        A[i, i + 1] = A[i + 1, i] = 1.0
    d = A.sum(1)
    dm = np.diag(1.0 / np.sqrt(np.clip(d, 1e-9, None)))
    w, Vt = np.linalg.eigh(np.eye(H) - dm @ A @ dm)
    o = np.argsort(w)
    return w[o], Vt[:, o]


def windows(ds, max_win=4000):
    if ds == "molene":
        d = np.load(paths.processed_dir("molene") / "signals.npz")
        return d["Xc"].astype(np.float64)[:, :, None], None
    tr = np.load(paths.split_npz(ds, "train"))
    Y = tr["y"][..., 0].astype(np.float64)            # (B, H, N)
    node_mean = Y.reshape(-1, Y.shape[2]).mean(0)
    W = (Y - node_mean[None, None, :]).transpose(0, 2, 1)   # (B, N, H) per-node centered
    if W.shape[0] > max_win:
        W = W[np.linspace(0, W.shape[0] - 1, max_win).astype(int)]
    return W, node_mean


def cum(energy):
    """cumulative fraction, ordered low->high frequency."""
    c = np.cumsum(energy) / energy.sum()
    return c


def _cross(x, c, frac):
    """x-value where cumulative curve first reaches frac."""
    i = int(np.searchsorted(c, frac))
    return x[min(i, len(x) - 1)]


# ---------------------------------------------------------------- spatial
def fig_spatial(datasets):
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharey=True)
    for ax, ds in zip(axes.ravel(), datasets):
        lam, V = spatial_eig(ds)
        W, _ = windows(ds)
        Cp = np.einsum("nk,wnh->wkh", V, W)
        e = (Cp ** 2).mean((0, 2))
        kron = 1.0 / (lam + GAMMA)
        iso = np.ones_like(lam)
        cd, ck, ci = cum(e), cum(kron), cum(iso)
        ax.plot(lam, cd * 100, color=C["data"], lw=2.4, label="raw signal")
        ax.plot(lam, ck * 100, color=C["kron"], lw=2.0, ls="--", label=r"$(L+\gamma I)^{-1}$ prior")
        ax.plot(lam, ci * 100, color=C["iso"], lw=1.6, ls=":", label="isotropic")
        # annotate where the raw signal reaches 50% and 90%
        for f in (0.5, 0.9):
            xf = _cross(lam, cd, f)
            ax.plot([xf, xf], [0, f * 100], color=C["data"], lw=0.8, alpha=0.4)
            ax.annotate(f"{int(f*100)}% by λ={xf:.2f}", (xf, f * 100),
                        textcoords="offset points", xytext=(4, -12), fontsize=8.5)
        ax.set_title(ds); ax.set_xlim(0, 2); ax.set_ylim(0, 100)
        ax.set_xlabel("graph frequency λ  (0=smooth, 2=oscillatory)")
        ax.grid(alpha=0.25)
    axes[0, 0].set_ylabel("cumulative energy (%)")
    axes[1, 0].set_ylabel("cumulative energy (%)")
    axes.ravel()[len(datasets) - 1].legend(loc="lower right", fontsize=9)
    for ax in axes.ravel()[len(datasets):]:
        ax.axis("off")
    fig.suptitle("Spatial: where the raw-signal energy sits across graph frequency "
                 "(steeper = more low-frequency / graph-smooth)", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG / "spectrum_spatial.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote spectrum_spatial.png")


# ---------------------------------------------------------------- temporal
def temporal_factors(ds):
    W, _ = windows(ds)
    H = W.shape[2]
    R = W.reshape(-1, H)
    delta = (R ** 2).mean(0)
    SigT = np.sqrt(delta)[:, None] * np.corrcoef((R - R.mean(0)).T) * np.sqrt(delta)[None, :]
    mu = np.sort(np.linalg.eigvalsh(SigT))[::-1].clip(1e-12)   # empirical variances, low->high freq
    w_path, _ = path_laplacian_eig(H)
    var_path = 1.0 / (w_path + GAMMA_T)
    return mu, var_path


def fig_temporal(datasets):
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.2 * len(datasets), 3.8),
                             sharey=True, squeeze=False)
    for ax, ds in zip(axes[0], datasets):
        emp, path = temporal_factors(ds)
        x = np.arange(1, len(emp) + 1)
        ce, cp = cum(emp), cum(path)
        ax.plot(x, ce * 100, "o-", color=C["data"], lw=2.2, ms=5, label="empirical Σ_T (data)")
        ax.plot(x, cp * 100, "s--", color=C["path"], lw=1.8, ms=5, label="path-graph Σ_T")
        ax.annotate(f"mode 1 = {ce[0]*100:.0f}%", (1, ce[0] * 100),
                    textcoords="offset points", xytext=(8, -4), fontsize=9, color=C["data"])
        ax.set_title(ds); ax.set_xlabel("temporal mode (1 = common/DC)")
        ax.set_ylim(0, 100); ax.grid(alpha=0.25)
    axes[0][0].set_ylabel("cumulative variance (%)")
    axes[0][-1].legend(loc="lower right", fontsize=9)
    fig.suptitle("Temporal: the data's variance is one common mode (empirical Σ_T jumps "
                 "to ~90% at mode 1); a path-graph spreads it out", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "spectrum_temporal.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote spectrum_temporal.png")


# ---------------------------------------------------------------- joint marginals
def joint_marginals(ds):
    lam_s, Vs = spatial_eig(ds)
    W, _ = windows(ds)
    H = W.shape[2]
    w_path, Vt = path_laplacian_eig(H)
    Cp = np.einsum("nk,wnh,hj->wkj", Vs, W, Vt)
    E = (Cp ** 2).mean(0)                              # (N, H) data joint energy

    R = W.reshape(-1, H)
    delta = (R ** 2).mean(0)
    SigT = np.sqrt(delta)[:, None] * np.corrcoef((R - R.mean(0)).T) * np.sqrt(delta)[None, :]
    mu_emp = np.sort(np.linalg.eigvalsh(SigT))[::-1].clip(1e-12)
    q_s = lam_s + GAMMA
    q_t_emp = 1.0 / mu_emp
    q_t_path = w_path + GAMMA_T

    def var(theta):
        return 1.0 / theta

    priors = {
        "kron-cov (Σ_T⊗Σ_S, thesis)": var(q_s[:, None] * q_t_emp[None, :]),
        "kron-graph (q_S·q_T)": var(q_s[:, None] * q_t_path[None, :]),
        "cartesian (λ_S+ω_T)": var(((lam_s[:, None] + w_path[None, :]) / 2) + GAMMA),
        "strong (q_S+q_T+q_S q_T)": var(q_s[:, None] + q_t_path[None, :]
                                        + q_s[:, None] * q_t_path[None, :]),
    }
    return lam_s, np.arange(1, H + 1), E, priors, ["kroncov", "krongraph", "cartesian", "strong"]


def fig_joint(ds):
    lam, tmode, E, priors, keys = joint_marginals(ds)
    e_spatial, e_temporal = E.sum(1), E.sum(0)         # marginals of the data
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

    # spatial marginal
    ax = axes[0]
    ax.plot(lam, cum(e_spatial) * 100, color=C["data"], lw=2.8, label="raw signal", zorder=5)
    for (name, S), k in zip(priors.items(), keys):
        ax.plot(lam, cum(S.sum(1)) * 100, color=C[k], lw=1.8, ls="--", label=name)
    ax.set_xlim(0, 2); ax.set_ylim(0, 100); ax.grid(alpha=0.25)
    ax.set_xlabel("spatial graph frequency λ"); ax.set_ylabel("cumulative energy (%)")
    ax.set_title("spatial marginal"); ax.legend(fontsize=8.5, loc="lower right")

    # temporal marginal
    ax = axes[1]
    ax.plot(tmode, cum(e_temporal) * 100, color=C["data"], lw=2.8, label="raw signal", zorder=5)
    for (name, S), k in zip(priors.items(), keys):
        ax.plot(tmode, cum(S.sum(0)) * 100, color=C[k], lw=1.8, ls="--", label=name)
    ax.set_ylim(0, 100); ax.grid(alpha=0.25)
    ax.set_xlabel("temporal mode (1 = common/DC)"); ax.set_title("temporal marginal")

    fig.suptitle(f"Joint fit — {ds}: which construction's cumulative energy tracks the "
                 f"raw signal (black)?", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / f"spectrum_joint_{ds}.png", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote spectrum_joint_{ds}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--joint-datasets", nargs="+", default=["pems08f", "pems08s", "metrla"])
    args = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    fig_spatial(TRAFFIC + ["molene"])
    fig_temporal(["metrla", "pems08f", "pems08s"])
    for ds in args.joint_datasets:
        fig_joint(ds)


if __name__ == "__main__":
    main()
