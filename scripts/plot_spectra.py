"""Figures: raw-signal graph-spectral energy vs the structured-prior constructions.

Answers, visually, per dataset:
  1. is the raw signal low graph-frequency (spatial) and low temporal-frequency?
  2. does the (L+γI)^-1 spatial prior concentrate energy the same way? (γ small,
     fixed at 0.04 -- just the Tikhonov shift that lifts the DC zero-mode, NOT a
     smoothness knob; GAD's 0.8 over-regularizes and whitens the prior.)
  3. which JOINT (space x time) construction matches the data's energy surface:
       - kron-cov : Σ = Σ_T(empirical) ⊗ Σ_S(graph)   -- the thesis prior (multiplicative)
       - kron-graph: both factors graph, θ = q_S · q_T  (tensor product graph)
       - cartesian: θ = (λ_S + ω_T)/2 + γ               (additive product graph, GAD-ST)
       - strong   : θ = q_S + q_T + q_S·q_T             (strong product graph)

Outputs PNGs to results/figures/.  No GPU.
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
# colorblind-safe (Wong)
C_DATA, C_KRON, C_ISO, C_PATH = "#000000", "#0072B2", "#999999", "#D55E00"
FIG = ROOT / "results" / "figures"


def spatial_eig(ds):
    if ds == "molene":
        L = np.load(paths.processed_dir("molene") / "adj.npz")["L"]
    else:
        L = compute_normalized_laplacian(load_adjacency(ds, repair=True))
    lam, V = np.linalg.eigh(L)
    o = np.argsort(lam)
    return lam[o], V[:, o]


def path_laplacian_eig(H):
    """path-graph Laplacian on H temporal nodes -> (ω sorted, V_T). eigs in [0,4]."""
    A = np.zeros((H, H))
    for i in range(H - 1):
        A[i, i + 1] = A[i + 1, i] = 1.0
    d = A.sum(1)
    L = np.diag(d) - A
    dm = np.diag(1.0 / np.sqrt(np.clip(d, 1e-9, None)))
    Ln = np.eye(H) - dm @ A @ dm                      # sym-norm, eigs in [0,2]
    w, Vt = np.linalg.eigh(Ln)
    o = np.argsort(w)
    return w[o], Vt[:, o]


def windows(ds, max_win=4000):
    """(W, N, H) per-node-centered signal windows + spatial per-node train mean."""
    if ds == "molene":
        d = np.load(paths.processed_dir("molene") / "signals.npz")
        X = d["Xc"].astype(np.float64)                # (T, N) per-station centered
        return X[:, :, None], None                    # (T, N, 1): each snapshot, H=1
    tr = np.load(paths.split_npz(ds, "train"))
    Y = tr["y"][..., 0].astype(np.float64)            # (B, H, N)
    node_mean = Y.reshape(-1, Y.shape[2]).mean(0)     # per-node train mean
    Y = Y - node_mean[None, None, :]                  # per-node center
    W = Y.transpose(0, 2, 1)                           # (B, N, H)
    if W.shape[0] > max_win:
        W = W[np.linspace(0, W.shape[0] - 1, max_win).astype(int)]
    return W, node_mean


def spatial_energy(ds, max_win=4000):
    """data spatial-mode energy (per-node centered), averaged over time+windows."""
    lam, V = spatial_eig(ds)
    W, _ = windows(ds, max_win)                       # (nW, N, H)
    C = np.einsum("nk,wnh->wkh", V, W)                # project onto spatial modes
    e = (C ** 2).mean((0, 2))                         # energy per spatial mode
    return lam, e / e.sum()


# ---------------------------------------------------------------- figure 1
def fig_spatial(datasets):
    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    for ax, ds in zip(axes.ravel(), datasets):
        lam, e = spatial_energy(ds)
        q = lam + GAMMA
        kron = 1.0 / q; kron /= kron.sum()            # (L+γI)^-1 spatial spectrum
        iso = np.ones_like(lam) / len(lam)
        ax.semilogy(lam, e, color=C_DATA, lw=1.8, label="raw signal")
        ax.semilogy(lam, kron, color=C_KRON, lw=1.5, ls="--", label=r"$(L+\gamma I)^{-1}$")
        ax.semilogy(lam, iso, color=C_ISO, lw=1.2, ls=":", label="isotropic")
        lowk = max(1, len(lam) // 10)
        ax.axvspan(lam.min(), lam[lowk], color=C_KRON, alpha=0.06)
        ax.set_title(f"{ds}  (low-λ: {e[:lowk].sum():.0%})", fontsize=10)
        ax.set_xlabel("graph frequency λ"); ax.grid(alpha=0.25)
    axes.ravel()[0].set_ylabel("normalized energy")
    axes.ravel()[len(datasets) - 1 if len(datasets) < 8 else 7].legend(fontsize=8)
    for ax in axes.ravel()[len(datasets):]:
        ax.axis("off")
    fig.suptitle(f"Spatial graph spectrum — raw signal concentrates at low λ  (γ={GAMMA})",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG / "spectrum_spatial.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("wrote spectrum_spatial.png")


# ---------------------------------------------------------------- figure 2
def temporal_factors(ds, max_win=4000):
    """empirical Σ_T eigenvalues (data-fit) vs path-graph Σ_T eigenvalues."""
    W, _ = windows(ds, max_win)                       # (nW, N, H)
    H = W.shape[2]
    R = W.transpose(0, 1, 2).reshape(-1, H)           # (nW*N, H) horizon vectors
    delta = (R ** 2).mean(0)                          # per-horizon variance
    Rc = R - R.mean(0)
    corr = np.corrcoef(Rc.T)
    SigT = np.sqrt(delta)[:, None] * corr * np.sqrt(delta)[None, :]
    mu_emp = np.sort(np.linalg.eigvalsh(SigT))[::-1]  # empirical temporal variances (desc)
    w_path, _ = path_laplacian_eig(H)
    var_path = 1.0 / (w_path + GAMMA_T)               # path-graph Σ_T variances
    return mu_emp / mu_emp.sum(), var_path / var_path.sum()


def fig_temporal(datasets):
    fig, axes = plt.subplots(1, len(datasets), figsize=(3.2 * len(datasets), 3.2), squeeze=False)
    for ax, ds in zip(axes[0], datasets):
        emp, path = temporal_factors(ds)
        x = np.arange(1, len(emp) + 1)
        ax.semilogy(x, emp, "o-", color=C_DATA, lw=1.6, ms=4, label="empirical Σ_T (data)")
        ax.semilogy(x, path, "s--", color=C_PATH, lw=1.4, ms=4, label="path-graph Σ_T")
        ax.set_title(ds, fontsize=10); ax.set_xlabel("temporal mode (low→high freq)")
        ax.grid(alpha=0.25)
    axes[0][0].set_ylabel("normalized variance")
    axes[0][-1].legend(fontsize=8)
    fig.suptitle("Temporal factor — empirical (covariance) vs path-graph", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "spectrum_temporal.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("wrote spectrum_temporal.png")


# ---------------------------------------------------------------- figure 3
def joint_surfaces(ds, max_win=4000):
    """data joint energy + 4 prior variance surfaces over (spatial λ, temporal freq)."""
    lam_s, Vs = spatial_eig(ds)
    W, _ = windows(ds, max_win)                       # (nW, N, H)
    H = W.shape[2]
    w_path, Vt = path_laplacian_eig(H)

    # data joint energy in (Vs, Vt_path)
    C = np.einsum("nk,wnh,hj->wkj", Vs, W, Vt)        # (nW, N, H)
    E = (C ** 2).mean(0)                              # (N, H) data energy
    E = E / E.sum()

    # temporal factors
    R = W.transpose(0, 1, 2).reshape(-1, H)
    delta = (R ** 2).mean(0); Rc = R - R.mean(0)
    SigT = np.sqrt(delta)[:, None] * np.corrcoef(Rc.T) * np.sqrt(delta)[None, :]
    mu_emp = np.sort(np.linalg.eigvalsh(SigT))[::-1].clip(1e-9)   # variances, desc = low->high freq
    q_s = lam_s + GAMMA                               # spatial precision
    q_t_emp = 1.0 / mu_emp                            # empirical temporal precision
    q_t_path = w_path + GAMMA_T                       # path-graph temporal precision

    def norm_var(theta):
        v = 1.0 / theta; return v / v.sum()

    surfaces = {
        "data (raw signal)": E,
        "kron-cov  Σ_T⊗Σ_S": norm_var(q_s[:, None] * q_t_emp[None, :]),
        "kron-graph  q_S·q_T": norm_var(q_s[:, None] * q_t_path[None, :]),
        "cartesian  λ_S+ω_T": norm_var(((lam_s[:, None] + w_path[None, :]) / 2) + GAMMA),
        "strong  q_S+q_T+q_Sq_T": norm_var(q_s[:, None] + q_t_path[None, :]
                                           + q_s[:, None] * q_t_path[None, :]),
    }
    return surfaces


def fig_joint(ds):
    surf = joint_surfaces(ds)
    n = len(surf)
    fig, axes = plt.subplots(1, n, figsize=(3.0 * n, 3.2))
    vmax = max(np.log10(s + 1e-12).max() for s in surf.values())
    vmin = vmax - 4
    for ax, (name, S) in zip(axes, surf.items()):
        im = ax.imshow(np.log10(S.T + 1e-12), origin="lower", aspect="auto",
                       cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("spatial mode (low→high λ)")
    axes[0].set_ylabel("temporal mode (low→high)")
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01, label="log₁₀ normalized energy")
    fig.suptitle(f"Joint space×time spectrum — {ds}: which product matches the data?",
                 fontsize=12)
    fig.savefig(FIG / f"spectrum_joint_{ds}.png", dpi=140, bbox_inches="tight")
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
