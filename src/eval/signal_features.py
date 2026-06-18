"""gad-style scalar features for a residual field R ∈ ℝ^{N×H}.

each feature collapses a whole (N, H) field to a single number, so a *set* of
fields (prior samples, or val residuals) becomes a 1-d cloud per feature. comparing
those clouds with cheap 1-d distances (wasserstein / mmd / ks) is how we ask
"does the prior look like the data?" without an intractable NH-dim distance — this
is exactly gad's signal_metrics.py reduction (rozada et al. 2025).

all functions accept either a single field (N, H) or a batch (..., N, H) and
reduce over the last two axes, returning the leading (...) shape.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance

# machine names (stable, used as dict keys) and pretty names (for tables/plots).
FEATURE_NAMES = [
    "spatial_tv",
    "spectral_centroid",
    "degree_corr",
    "lag1_autocorr",
    "horizon_centroid",
]
NICE_NAMES = {
    "spatial_tv": "spatial TV (graph roughness)",
    "spectral_centroid": "spectral centroid (spatial)",
    "degree_corr": "degree correlation",
    "lag1_autocorr": "lag-1 autocorr (temporal)",
    "horizon_centroid": "horizon energy centroid",
}


# ─── individual features ─────────────────────────────────────────────────────


def total_variation(R: np.ndarray, L: np.ndarray) -> np.ndarray:
    """graph roughness: TV(R) = mean_h R[:,h]ᵀ L R[:,h]. higher = rougher on the graph."""
    LR = np.einsum("ij,...jh->...ih", L, R)
    per_h = np.einsum("...ih,...ih->...h", R, LR)   # quadratic form per horizon
    return per_h.mean(axis=-1)


def spectral_centroid(R: np.ndarray, lam: np.ndarray, V_S: np.ndarray) -> np.ndarray:
    """where spatial energy sits: SC(R) = Σ_k λ_k|û_k|² / Σ_k|û_k|², û = V_Sᵀ R, avg over h.
    low = energy in smooth (low-λ) modes; high = energy in rough modes."""
    U = np.einsum("ki,...ih->...kh", V_S.T, R)      # node space -> laplacian spectrum
    power = U ** 2
    num = (power * lam[:, None]).sum(axis=-2)        # weight modes by λ, sum -> (..., H)
    den = power.sum(axis=-2).clip(min=1e-12)
    return (num / den).mean(axis=-1)


def degree_correlation(R: np.ndarray, degree: np.ndarray) -> np.ndarray:
    """pearson corr between node degree and the residual, averaged over horizons.
    nonzero = residual energy is tied to how connected a sensor is."""
    d = degree - degree.mean()
    dn = np.sqrt((d ** 2).mean()).clip(min=1e-12)
    Rc = R - R.mean(axis=-2, keepdims=True)
    cov = (d[:, None] * Rc).mean(axis=-2)            # (..., H)
    sx = np.sqrt((Rc ** 2).mean(axis=-2)).clip(min=1e-12)
    return (cov / dn / sx).mean(axis=-1)


def lag1_autocorr(R: np.ndarray) -> np.ndarray:
    """temporal smoothness: lag-1 autocorr along the H horizons, averaged over nodes."""
    Rc = R - R.mean(axis=-1, keepdims=True)
    num = (Rc[..., :-1] * Rc[..., 1:]).sum(axis=-1)  # (..., N)
    den = (Rc ** 2).sum(axis=-1).clip(min=1e-12)
    return (num / den).mean(axis=-1)


def horizon_centroid(R: np.ndarray) -> np.ndarray:
    """temporal analogue of the spectral centroid: Σ_h h·E_h / Σ_h E_h, E_h = ‖R[:,h]‖².
    tells you how far out the error energy concentrates (1 = near horizon, H = far)."""
    H = R.shape[-1]
    E = (R ** 2).sum(axis=-2)                        # (..., H)
    h = np.arange(1, H + 1, dtype=float)
    return (E * h).sum(axis=-1) / E.sum(axis=-1).clip(min=1e-12)


# ─── reduction to a feature matrix ───────────────────────────────────────────


def compute_features(
    R: np.ndarray, L: np.ndarray, lam: np.ndarray, V_S: np.ndarray, degree: np.ndarray
) -> dict[str, np.ndarray]:
    """reduce fields R (..., N, H) to a dict {feature_name: (...,) values}."""
    return {
        "spatial_tv": total_variation(R, L),
        "spectral_centroid": spectral_centroid(R, lam, V_S),
        "degree_corr": degree_correlation(R, degree),
        "lag1_autocorr": lag1_autocorr(R),
        "horizon_centroid": horizon_centroid(R),
    }


def feature_matrix(
    R: np.ndarray, L: np.ndarray, lam: np.ndarray, V_S: np.ndarray, degree: np.ndarray
) -> np.ndarray:
    """stack the features of a batch R (M, N, H) into a (M, F) matrix in FEATURE_NAMES order."""
    f = compute_features(R, L, lam, V_S, degree)
    return np.stack([f[n] for n in FEATURE_NAMES], axis=-1)


# ─── two-sample distances between feature clouds ─────────────────────────────


def _mmd_rbf(x: np.ndarray, y: np.ndarray, max_n: int = 1500) -> float:
    """rbf-kernel MMD² between two 1-d samples, median-heuristic bandwidth.
    clouds are subsampled to max_n each to keep the kernel matrices small."""
    x = x[:: max(1, len(x) // max_n)]
    y = y[:: max(1, len(y) // max_n)]
    pool = np.concatenate([x, y])
    d = np.abs(pool[:, None] - pool[None, :])
    sigma = np.median(d[d > 0])
    sigma = float(sigma) if sigma > 0 else 1.0
    g = 1.0 / (2.0 * sigma ** 2)
    Kxx = np.exp(-g * (x[:, None] - x[None, :]) ** 2)
    Kyy = np.exp(-g * (y[:, None] - y[None, :]) ** 2)
    Kxy = np.exp(-g * (x[:, None] - y[None, :]) ** 2)
    m, n = len(x), len(y)
    mmd2 = (
        (Kxx.sum() - np.trace(Kxx)) / (m * (m - 1))
        + (Kyy.sum() - np.trace(Kyy)) / (n * (n - 1))
        - 2.0 * Kxy.mean()
    )
    return float(max(mmd2, 0.0))


def compare_feature_sets(
    A: np.ndarray, B: np.ndarray, names: list[str] = FEATURE_NAMES
) -> dict[str, dict[str, float]]:
    """per-feature distances between two feature clouds A (M1, F) and B (M2, F).

    for each feature returns:
      w1      raw 1-d wasserstein-1 (same units as the feature)
      w1_norm w1 divided by the std of B's feature column (scale-free, comparable
              across features; B is the reference, e.g. the val residuals)
      mmd     rbf-kernel MMD²
      ks      kolmogorov-smirnov statistic in [0, 1] (0 = identical, 1 = disjoint)
    """
    out: dict[str, dict[str, float]] = {}
    for j, name in enumerate(names):
        a, b = A[:, j], B[:, j]
        scale = float(b.std()) or 1.0
        w1 = float(wasserstein_distance(a, b))
        out[name] = {
            "w1": w1,
            "w1_norm": w1 / scale,
            "mmd": _mmd_rbf(a, b),
            "ks": float(ks_2samp(a, b).statistic),
        }
    return out
