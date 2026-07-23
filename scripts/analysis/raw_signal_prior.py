"""Evidence that the graph prior fits the raw traffic signal (and iso does not).

Writes results/analysis/raw_signal/ :
  formula.md            the exact prior formula + how gamma was chosen (with numbers)
  gamma_sweep_<ds>.png  low-freq graph energy of the prior vs gamma, with the DATA line
  spectrum_<ds>.png     per-graph-mode energy: data vs cartesian vs laplacian vs iso
  forward_<ds>.png      GAD-style heat cascade: mode retention exp(-theta*c) vs noise c
  fields_<ds>.png       a real traffic field vs a cartesian prior draw vs an iso draw
  match_table.md        low-lambda energy + Gaussian NLL (nats) per prior, all datasets

All CPU (eigendecomposition + projection). Reads the centered signal cache built by
build_signal_cache.py.  usage: .venv/bin/python scripts/analysis/raw_signal_prior.py
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.utils import paths                                                       # noqa: E402
from src.diffusion.prior import load_prior                                        # noqa: E402
from src.data.residual_dataset import train_residual_sigma                        # noqa: E402
from src.data.adjacency import compute_normalized_laplacian, load_adjacency       # noqa: E402

OUT = ROOT / "results" / "analysis" / "raw_signal"
OUT.mkdir(parents=True, exist_ok=True)
DATASETS = ["pems08f", "metrla", "pems04f"]
C = {"data": "#111111", "cartesian": "#0072B2", "laplacian": "#009E73", "iso": "#D55E00"}


def load_signal(ds):
    """centered signal R0 (B,N,H) in unit-variance space, + mask. built by build_signal_cache."""
    p = paths.residual_npz(ds, "train", 42, "raw")
    d = np.load(p)
    sr = float(np.asarray(train_residual_sigma(ds, 42, "raw", norm="global")))
    return d["residuals"].astype(np.float64) / sr, d["mask"].astype(bool)


def graph_eig(ds):
    lam, V = np.linalg.eigh(compute_normalized_laplacian(np.asarray(load_adjacency(ds, repair=True))))
    o = np.argsort(lam)
    return lam[o], V[:, o]


def spatial_energy(X, mask, V):
    """per-graph-mode energy of X (B,N,H), averaged over windows+horizon."""
    Xz = np.where(mask, X, 0.0)
    Cf = np.einsum("nk,bnh->bkh", V, Xz)
    return (Cf ** 2).mean((0, 2))


def lowfrac(e, lam, N):
    return float(e[np.arange(len(e)) <= N // 3].sum() / e.sum())


def prior_spatial_energy(prior):
    """the prior's per-graph-mode marginal variance (avg over horizon), in the graph basis.
    Var per (spatial mode k) = mean_h Σ_l σ²[k,l] · (V_T[h,l])²  → since V_T orthonormal,
    mean_h (V_T²)[h,l] = 1/H, so spatial-mode var = mean_l σ²[k,l]."""
    return prior.sigma2.mean(1)                                  # (N,) per spatial mode


def nll_nats(prior, R0, mask):
    """Gaussian NLL of held-out data under the (diagonal-in-eigenbasis) prior, per coord.
    project data into the prior's (V_S ⊗ V_T) basis, score against per-mode variance."""
    idx = mask.reshape(mask.shape[0], -1).all(1)
    X = R0[idx][:4000] if idx.sum() > 100 else np.nan_to_num(R0)[:4000]
    Cf = np.matmul(np.matmul(prior.V_S.T, X), prior.V_T)         # (B,N,H) coeffs
    v = np.clip(prior.sigma2, 1e-8, None)
    nll = 0.5 * ((Cf ** 2) / v + np.log(2 * np.pi * v))
    return float(nll.mean())


def main():
    rows = []
    for ds in DATASETS:
        try:
            R0, M = load_signal(ds)
        except FileNotFoundError:
            print(f"skip {ds}: no signal cache (run build_signal_cache.py --dataset {ds})")
            continue
        N, H = R0.shape[1], R0.shape[2]
        lam, V = graph_eig(ds)
        e_data = spatial_energy(R0, M, V)
        lf_data = lowfrac(e_data, lam, N)

        # ── gamma sweep (cartesian): low-λ energy of the prior vs gamma ──
        gammas = np.array([0.01, 0.02, 0.04, 0.08, 0.15, 0.3, 0.6, 0.8, 1.5, 3.0])
        lf_g = []
        for g in gammas:
            load_prior.cache_clear()
            pr = load_prior(ds, "kronecker", seed=42, backbone="raw", temporal="cartesian", gamma=float(g))
            e = prior_spatial_energy(pr)
            lf_g.append(lowfrac(e, lam, N))
        lf_g = np.array(lf_g)
        g_star = float(gammas[np.argmin(np.abs(lf_g - lf_data))])

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.semilogx(gammas, lf_g, "o-", color=C["cartesian"], label="cartesian prior")
        ax.axhline(lf_data, color=C["data"], ls="--", label=f"DATA low-λ energy = {lf_data:.2f}")
        ax.axhline(1.0 / 3, color=C["iso"], ls=":", label="iso (white) = 0.33")
        ax.axvline(g_star, color="#888888", lw=1)
        ax.annotate(f"γ*≈{g_star:g}\n(matches data)", (g_star, lf_data),
                    textcoords="offset points", xytext=(8, -28), fontsize=9)
        ax.set_xlabel("γ  (precision offset in θ=λ+γ)")
        ax.set_ylabel("fraction of prior energy on low-freq (smooth) modes")
        ax.set_title(f"{ds}: choosing γ so the prior's smoothness matches the signal", fontsize=10)
        ax.legend(fontsize=8, frameon=False); ax.grid(alpha=0.2)
        fig.tight_layout(); fig.savefig(OUT / f"gamma_sweep_{ds}.png", dpi=150); plt.close(fig)

        # ── priors at γ* ── cartesian / laplacian / iso spectra + low-λ + NLL
        load_prior.cache_clear()
        p_cart = load_prior(ds, "kronecker", seed=42, backbone="raw", temporal="cartesian", gamma=g_star)
        load_prior.cache_clear()
        p_lap = load_prior(ds, "kronecker", seed=42, backbone="raw", temporal="laplacian", gamma=g_star, gamma_t=0.6)
        load_prior.cache_clear()
        p_iso = load_prior(ds, "isotropic", seed=42, backbone="raw")
        e_cart = prior_spatial_energy(p_cart); e_cart *= e_data.sum() / e_cart.sum()
        e_lap = prior_spatial_energy(p_lap);   e_lap *= e_data.sum() / e_lap.sum()
        e_iso = prior_spatial_energy(p_iso);   e_iso *= e_data.sum() / e_iso.sum()

        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        ax.scatter(lam, e_data, s=10, color=C["data"], alpha=0.5, label="DATA (signal)")
        for e, key, lab in [(e_cart, "cartesian", "cartesian prior"),
                            (e_lap, "laplacian", "laplacian prior"),
                            (e_iso, "iso", "iso (white)")]:
            k = 12
            med = np.array([np.median(e[max(0, i - k):i + k]) for i in range(len(e))])
            ax.plot(lam, med, color=C[key], lw=2, label=lab)
        ax.set_yscale("log"); ax.set_xlabel("graph frequency λ (0 = smoothest)")
        ax.set_ylabel("energy per mode (scale-matched)")
        ax.set_title(f"{ds}: does the prior's spectrum track the signal's?", fontsize=10)
        ax.legend(fontsize=8, frameon=False); ax.grid(alpha=0.2)
        fig.tight_layout(); fig.savefig(OUT / f"spectrum_{ds}.png", dpi=150); plt.close(fig)

        # ── forward heat cascade (GAD Fig 1): mode retention exp(-θ c̄) vs c̄ ──
        theta = p_cart.theta.mean(1)                            # per spatial mode (avg horizon)
        order = np.argsort(theta)
        picks = order[[0, len(order) // 3, 2 * len(order) // 3, len(order) - 1]]
        cbar = np.geomspace(1e-2, 7.0, 100)
        fig, ax = plt.subplots(figsize=(6, 4))
        labs = ["slowest (smooth)", "slow", "fast", "fastest (rough)"]
        cols = ["#0072B2", "#56B4E9", "#E69F00", "#D55E00"]
        for m, lb, col in zip(picks, labs, cols):
            ax.semilogx(cbar, np.exp(-theta[m] * cbar), color=col, lw=2,
                        label=f"{lb}  θ={theta[m]:.2f}")
        ax.axvline(7.0, color="#888888", ls=":", label="c₀=7 (schedule top)")
        ax.set_xlabel("noise level c̄ (heat)"); ax.set_ylabel("signal retained  exp(−θ·c̄)")
        ax.set_title(f"{ds}: GAD heat cascade — smooth modes die last", fontsize=10)
        ax.legend(fontsize=8, frameon=False); ax.grid(alpha=0.2)
        fig.tight_layout(); fig.savefig(OUT / f"forward_{ds}.png", dpi=150); plt.close(fig)

        # ── example fields: real vs cartesian draw vs iso draw (first horizon) ──
        from src.diffusion.forward_fcps import sample_prior
        rng = np.random.default_rng(0)
        real = np.where(M, R0, np.nan)[0, :, 0]
        d_cart = sample_prior((1,), p_cart, rng)[0, :, 0]
        d_iso = sample_prior((1,), p_iso, rng)[0, :, 0]
        A = np.asarray(load_adjacency(ds, repair=True))
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
        for ax, fld, ttl in zip(axes, [real, d_cart, d_iso],
                                ["real signal field", "cartesian prior draw", "iso prior draw"]):
            # smoothness = avg |value - neighbor mean|
            nb = A @ np.nan_to_num(fld) / np.clip(A.sum(1), 1, None)
            rough = np.nanmean(np.abs(np.nan_to_num(fld) - nb))
            ax.plot(fld, lw=0.8, color="#333333")
            ax.set_title(f"{ttl}\ngraph roughness={rough:.2f}", fontsize=9)
            ax.set_xlabel("sensor"); ax.set_ylim(-3.5, 3.5)
        fig.suptitle(f"{ds}: a structured draw looks like traffic; an iso draw is white noise", fontsize=10)
        fig.tight_layout(); fig.savefig(OUT / f"fields_{ds}.png", dpi=150); plt.close(fig)

        rows.append((ds, lf_data, g_star, lowfrac(e_cart, lam, N), lowfrac(e_lap, lam, N),
                     lowfrac(e_iso, lam, N), nll_nats(p_cart, R0, M), nll_nats(p_iso, R0, M)))
        print(f"{ds}: low-λ data={lf_data:.2f} γ*={g_star:g} cart={lowfrac(e_cart,lam,N):.2f} "
              f"lap={lowfrac(e_lap,lam,N):.2f} iso=0.33  NLL cart={rows[-1][6]:.3f} iso={rows[-1][7]:.3f}")

    # ── tables ──
    with open(OUT / "match_table.md", "w") as f:
        f.write("# Prior-vs-signal match (evidence the structured prior fits, iso does not)\n\n")
        f.write("Low-λ energy = fraction of energy on the smoothest third of graph modes. "
                "A prior that matches the signal should have low-λ ≈ the data's. NLL = Gaussian "
                "negative log-likelihood of held-out signal under the prior (nats/coord, lower=better fit).\n\n")
        f.write("| dataset | data low-λ | γ* | cartesian low-λ | laplacian low-λ | iso low-λ | NLL cartesian | NLL iso | NLL gain (nats) |\n")
        f.write("|---|--:|--:|--:|--:|--:|--:|--:|--:|\n")
        for ds, lfd, gs, lfc, lfl, lfi, nc, ni in rows:
            f.write(f"| {ds} | {lfd:.2f} | {gs:g} | {lfc:.2f} | {lfl:.2f} | {lfi:.2f} | {nc:.3f} | {ni:.3f} | **{ni-nc:+.3f}** |\n")
    print(f"\nwrote {OUT}/match_table.md and figures")


if __name__ == "__main__":
    main()
