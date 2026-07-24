"""thesis figures for the heat-prior story (chapter 5).

regenerates, under the HEAT prior (cartesian product, learned + road graphs):
  figures/fig_heat_smoothness.pdf   A: low-third energy bars per dataset+graph
                                    B: corr-by-hop, real vs heat draw vs iso draw
  figures/fig_forward_convergence.pdf  variance error to stationary vs c-bar
  figures/fig_forward_decay.pdf        per-mode signal decay, measured vs closed form
  figures/fig_theta_spectrum.pdf       heat spectra (road, learned) vs iso flat

usage: .venv/bin/python scripts/analysis/thesis_heat_figures.py
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse.csgraph import shortest_path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.diffusion.prior import load_prior, _load_residuals_train, _knn_correlation_graph  # noqa: E402
from src.diffusion.forward_fcps import sample_prior                                         # noqa: E402
from src.data.adjacency import compute_normalized_laplacian, load_adjacency                 # noqa: E402

FIGS = ROOT / "tudelft_thesis_draft/figures"
DATASETS = ["pems08f", "pems08s", "pems04f", "pems04s"]
LABELS = {"pems08f": "PEMS08 flow", "pems08s": "PEMS08 speed",
          "pems04f": "PEMS04 flow", "pems04s": "PEMS04 speed"}
REF = "pems08s"
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "figure.dpi": 150})


def low_third(R, mask, lam_S, V_S):
    """joint product-spectrum statistic (same as residual_heat_protocol step 2):
    energy fraction in the lowest-frequency third of lam_ST = (lam_S + omega)/2."""
    H = R.shape[2]
    A_T = np.zeros((H, H)); i = np.arange(H - 1)
    A_T[i, i + 1] = 1.0; A_T[i + 1, i] = 1.0
    omega, V_T = np.linalg.eigh(compute_normalized_laplacian(A_T))
    lam_ST = 0.5 * (lam_S[:, None] + np.clip(omega, 0, None)[None, :])
    Rz = np.where(mask, R, 0.0)
    C = np.einsum("kn,bnh->bkh", V_S.T, Rz)
    C = np.einsum("bkh,hj->bkj", C, V_T)
    e = (C ** 2).mean(0).ravel()
    order = np.argsort(lam_ST.ravel())
    e = e[order]
    cut = len(e) // 3
    return float(e[:cut].sum() / e.sum())


def main():
    rng = np.random.default_rng(7)

    # ---------- panel data: smoothness fractions ----------
    frac_learn, frac_road = {}, {}
    for ds in DATASETS:
        R, M = _load_residuals_train(ds, 42, "gwn", norm="pernode")
        A_learn = _knn_correlation_graph(R, 5)
        lamL, VL = np.linalg.eigh(compute_normalized_laplacian(A_learn))
        A_road = np.asarray(load_adjacency(ds, repair=True)).astype(float)
        np.fill_diagonal(A_road, 0.0)
        lamR, VR = np.linalg.eigh(compute_normalized_laplacian(A_road))
        frac_learn[ds] = low_third(R, M, lamL, VL)
        frac_road[ds] = low_third(R, M, lamR, VR)
        print(f"{ds}: learned {frac_learn[ds]:.2f}  road {frac_road[ds]:.2f}")

    # ---------- reference dataset objects ----------
    R, M = _load_residuals_train(REF, 42, "gwn", norm="pernode")
    Rz = np.where(M, R, 0.0)
    A_learn = _knn_correlation_graph(R, 5)
    D_hop = shortest_path((A_learn > 0).astype(float), method="D", unweighted=True)
    load_prior.cache_clear()
    prL = load_prior(REF, "kronecker", seed=42, backbone="gwn", temporal="cartesian",
                     spatial_graph="correlation", knn_k=5, gamma=0.05, norm="pernode")
    load_prior.cache_clear()
    prR = load_prior(REF, "kronecker", seed=42, backbone="gwn", temporal="cartesian",
                     spatial_graph="road", gamma=0.8, norm="pernode")
    load_prior.cache_clear()
    prI = load_prior(REF, "isotropic", seed=42, backbone="gwn", norm="pernode")

    # ---------- fig A: smoothness + draw realism ----------
    fig, (a, b) = plt.subplots(1, 2, figsize=(8.6, 2.9))
    x = np.arange(len(DATASETS)); w = 0.27
    a.bar(x - w, [frac_learn[d] for d in DATASETS], w, label="learned graph", color="#0072B2")
    a.bar(x, [frac_road[d] for d in DATASETS], w, label="road graph", color="#009E73")
    a.bar(x + w, [1 / 3] * len(DATASETS), w, label="white field", color="#D55E00")
    a.set_xticks(x, [LABELS[d] for d in DATASETS], fontsize=8)
    a.set_ylabel("energy in lowest-frequency third")
    a.axhline(1 / 3, color="#999999", lw=0.8, ls=":")
    a.set_title("residual smoothness by graph")
    a.legend(fontsize=7.5, frameon=False)

    def corr_by_hop(X, maxhop=5):
        Xf = X.transpose(1, 0, 2).reshape(X.shape[1], -1)
        Xf = (Xf - Xf.mean(1, keepdims=True)) / (Xf.std(1, keepdims=True) + 1e-9)
        C = (Xf @ Xf.T) / Xf.shape[1]
        iu = np.triu_indices_from(C, k=1)
        d, c = D_hop[iu], C[iu]
        return [float(c[d == h].mean()) for h in range(1, maxhop + 1)]

    hops = range(1, 6)
    b.plot(hops, corr_by_hop(Rz[rng.integers(0, len(Rz), 500)]), "o-", color="#111111",
           lw=2.4, label="real residuals")
    b.plot(hops, corr_by_hop(sample_prior((500,), prL, rng)), "o-", color="#0072B2",
           lw=1.6, label="heat prior draw (learned)")
    b.plot(hops, corr_by_hop(sample_prior((500,), prI, rng)), "o-", color="#D55E00",
           lw=1.6, label="isotropic draw")
    b.axhline(0, color="#999999", lw=0.8)
    b.set_xlabel("hop distance on the learned graph"); b.set_ylabel("mean pair correlation")
    b.set_title(f"{LABELS[REF]}: does the noise decay like the data?")
    b.legend(fontsize=7.5, frameon=False)
    fig.tight_layout(); fig.savefig(FIGS / "fig_heat_smoothness.pdf"); plt.close(fig)

    # ---------- forward convergence (variance error to stationary vs c-bar) ----------
    theta = prL.theta * prL.sigma2_scale / prL.sigma2_scale  # (N,H) precisions
    th = prL.theta
    var_stat = prL.sigma2_scale / th
    C0 = np.einsum("kn,bnh->bkh", prL.V_S.T, Rz[rng.integers(0, len(Rz), 400)])
    C0 = np.einsum("bkh,hj->bkj", C0, prL.V_T)
    cbars = np.geomspace(0.02, 50.0, 40)
    err = []
    for cb in cbars:
        m = np.exp(-th * cb)[None]
        var_t = (m ** 2) * (C0 ** 2) + (1 - np.exp(-2 * th * cb))[None] * var_stat[None]
        err.append(float(np.abs(var_t.mean(0) - var_stat).sum() / var_stat.sum()))
    fig, ax = plt.subplots(figsize=(3.1, 2.6))
    ax.semilogx(cbars, np.array(err) * 100, color="#0072B2", lw=1.8)
    ax.axvline(50, color="#999999", lw=0.9, ls="--")
    ax.text(50, ax.get_ylim()[1] * 0.9, r" $c_0$", fontsize=8, color="#555555")
    ax.set_xlabel(r"accumulated heat $\bar c$")
    ax.set_ylabel("variance error to stationary (%)")
    fig.tight_layout(); fig.savefig(FIGS / "fig_forward_convergence.pdf"); plt.close(fig)

    # ---------- per-mode decay cascade ----------
    fig, ax = plt.subplots(figsize=(3.1, 2.6))
    qs = [0.0, 0.25, 0.5, 0.75, 1.0]
    th_sorted = np.sort(th.ravel())
    cols = plt.cm.viridis(np.linspace(0.15, 0.9, len(qs)))
    for q, col in zip(qs, cols):
        t = th_sorted[min(int(q * (len(th_sorted) - 1)), len(th_sorted) - 1)]
        ax.semilogx(cbars, np.exp(-t * cbars), color=col, lw=1.6,
                    label=fr"$\Theta={t:.2f}$")
    ax.set_xlabel(r"accumulated heat $\bar c$")
    ax.set_ylabel(r"signal retained $e^{-\Theta\bar c}$")
    ax.legend(fontsize=7, frameon=False, title="mode precision", title_fontsize=7)
    fig.tight_layout(); fig.savefig(FIGS / "fig_forward_decay.pdf"); plt.close(fig)

    # ---------- spectra ----------
    fig, ax = plt.subplots(figsize=(3.1, 2.6))
    for pr, lab, col in [(prL, r"heat, learned ($\gamma{=}0.05$)", "#0072B2"),
                         (prR, r"heat, road ($\gamma{=}0.8$)", "#009E73")]:
        v = np.sort((pr.sigma2_scale / pr.theta).ravel())[::-1]
        ax.semilogy(np.linspace(0, 1, len(v)), v, lw=1.7, color=col, label=lab)
    vi = float(np.mean(prI.sigma2_scale / prI.theta))
    ax.axhline(vi, color="#D55E00", lw=1.5, label="isotropic (flat)")
    ax.set_xlabel("modes, sorted by prior variance")
    ax.set_ylabel("stationary variance per mode")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(FIGS / "fig_theta_spectrum.pdf"); plt.close(fig)
    print("wrote 4 figures to", FIGS)


if __name__ == "__main__":
    main()
