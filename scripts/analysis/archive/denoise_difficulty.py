"""WHY does iso beat kron even on smooth data? Test the denoising difficulty directly.

For a linear-Gaussian corruption, the BEST POSSIBLE denoiser is the Wiener filter, and its
residual error is computable per mode: MMSE[k] = s[k]·n[k]/(s[k]+n[k]) where s[k]=signal
variance, n[k]=noise variance on mode k. This is the floor NO denoiser (learned or not) can
beat. If kron's MMSE > iso's at matched TOTAL noise, then graph-correlated noise is
intrinsically harder to remove — the prior's geometry, not training, is why iso wins.

kron puts noise n[k]=σ²/θ[k] (HIGH on low-θ = smooth modes, where the signal lives).
iso puts noise flat. Both scale-matched to the same total. We sweep the forward noise level
c̄ (what the diffusion actually trains on) and report the Wiener MMSE for each.

usage: .venv/bin/python scripts/analysis/denoise_difficulty.py
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

OUT = ROOT / "results/analysis/raw_signal"; OUT.mkdir(parents=True, exist_ok=True)
DATASETS = ["pems08f", "metrla", "pems04f"]


def signal_mode_var(ds, prior):
    """empirical per-mode variance of the centered signal in the prior's (V_S⊗V_T) basis."""
    d = np.load(paths.residual_npz(ds, "train", 42, "raw"))
    sr = float(np.asarray(train_residual_sigma(ds, 42, "raw", norm="global")))
    R = d["residuals"].astype(np.float64) / sr
    M = d["mask"].astype(bool)
    Rz = np.where(M, R, 0.0)
    C = np.matmul(np.matmul(prior.V_S.T, Rz), prior.V_T)     # (B,N,H) coeffs
    return (C ** 2).mean(0)                                  # (N,H) per-mode signal variance


def main():
    print(f"{'dataset':9s} | {'total noise var (scale-match check)':36s} | Wiener MMSE (lower=easier to denoise)")
    print("-" * 100)
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(5 * len(DATASETS), 4.2))
    for ax, ds in zip(np.atleast_1d(axes), DATASETS):
        pk = load_prior(ds, "kronecker", seed=42, backbone="raw", temporal="cartesian", gamma=0.1)
        pi = load_prior(ds, "isotropic", seed=42, backbone="raw")
        s2 = signal_mode_var(ds, pk)                          # signal var in kron basis
        # stationary noise variance per mode: σ²/θ
        n2_k = pk.sigma2                                      # (N,H) kron: σ²/θ (structured)
        n2_i = np.full_like(n2_k, float(pi.sigma2.mean()))    # iso: flat, same mean
        # verify total noise matches (fair comparison)
        tot_k, tot_i = n2_k.mean(), n2_i.mean()
        # sweep forward noise level c̄ over the training range; noise var at c̄ = (σ²/θ)(1-e^{-2θc̄})
        cbars = np.geomspace(0.05, 25.0, 40)
        mmse_k, mmse_i = [], []
        for cb in cbars:
            nk = n2_k * (1 - np.exp(-2 * pk.theta * cb))      # kron noise var at c̄
            ni = n2_i * (1 - np.exp(-2 * pi.theta * cb))      # iso noise var at c̄
            # Wiener MMSE per mode, summed, normalized by signal energy
            mk = (s2 * nk / (s2 + nk)).sum() / s2.sum()
            mi = (s2 * ni / (s2 + ni)).sum() / s2.sum()
            mmse_k.append(mk); mmse_i.append(mi)
        mmse_k, mmse_i = np.array(mmse_k), np.array(mmse_i)
        worse = np.mean(mmse_k > mmse_i) * 100
        print(f"{ds:9s} | kron={tot_k:.4f}  iso={tot_i:.4f}  "
              f"(match={'YES' if abs(tot_k-tot_i)/tot_i<0.02 else 'NO'})          | "
              f"kron harder at {worse:.0f}% of noise levels  "
              f"(avg kron {mmse_k.mean():.3f} vs iso {mmse_i.mean():.3f})")
        ax.semilogx(cbars, mmse_k, "-", color="#0072B2", lw=2, label="kron (graph noise)")
        ax.semilogx(cbars, mmse_i, "-", color="#D55E00", lw=2, label="iso (white noise)")
        ax.set_xlabel("forward noise level c̄"); ax.set_ylabel("best-possible denoise error (Wiener MMSE)")
        ax.set_title(f"{ds}: which noise is harder to remove?", fontsize=10)
        ax.legend(fontsize=8, frameon=False); ax.grid(alpha=0.2)
    fig.suptitle("Optimal-denoiser error: graph-correlated (kron) vs white (iso) noise on the SAME signal",
                 fontsize=11)
    fig.tight_layout()
    f = OUT / "denoise_difficulty.png"; fig.savefig(f, dpi=150); plt.close(fig)
    print(f"\nwrote {f}")
    print("If kron's curve sits ABOVE iso's, graph noise is intrinsically harder to denoise → iso wins by geometry, not training.")


if __name__ == "__main__":
    main()
