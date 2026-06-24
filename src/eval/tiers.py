"""Tier 1-4 residual analysis: shared cell loader + the few metrics not already in
src/eval/signal_features.py or src/diffusion/.

The analysis runs identically on the ladder signal -> mlp -> gcn -> gwn so we can
watch the residual's structure change as the backbone removes more. A "cell" is
one (dataset, rung) pair, loaded as a raw-units field R (B, N, H) + bool mask.

  rung "signal" = climatology residual = y - per-(node, tod-bucket) train mean
                  (a zero-mean no-learning baseline, comparable to model residuals)
  rung mlp/gcn/gwn = the extracted backbone residuals (results/residuals/...)

Everything heavy (graph features, eigenbasis, prior fit) is imported from the
existing code; this module only adds: edge-ratio rho, GFT band profile,
nearest-Kronecker separability eta, masked excess kurtosis, plus the cell loader.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from src.data import adjacency as adj_mod
from src.utils import paths

RUNGS = ["signal", "mlp", "gcn", "gwn"]
DATASETS = ["metrla", "pemsbay", "pems04f", "pems04s", "pems08f", "pems08s"]
SLOTS_PER_DAY = 288


# ─── graph context (cached per dataset) ──────────────────────────────────────

@lru_cache(maxsize=8)
def graph_context(dataset: str) -> dict:
    """L, eigenpair (lam, V_S), node degree, and the edge / non-edge index masks."""
    A = adj_mod.load_adjacency(dataset)                      # (N,N) repaired
    L = adj_mod.compute_normalized_laplacian(A)              # (N,N) float64
    lam, V_S = adj_mod.laplacian_eigendecomposition(L, dataset, cache=True)
    Asym = (A + A.T) / 2.0
    np.fill_diagonal(Asym, 0.0)
    degree = (Asym > 0).sum(axis=1).astype(float)
    iu = np.triu_indices(A.shape[0], k=1)                    # upper-tri pairs
    is_edge = Asym[iu] > 0
    return {"L": L, "lam": lam, "V_S": V_S, "degree": degree,
            "iu": iu, "is_edge": is_edge, "N": A.shape[0]}


# ─── cell loading ────────────────────────────────────────────────────────────

def _signal_climatology(dataset: str, split: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """climatology residual for the 'signal' rung: y minus the per-(node, tod-slot)
    TRAIN mean, so it is zero-mean like the model residuals. Returns (R, mask) (B,N,H)."""
    def load(s):
        d = np.load(paths.split_npz(dataset, s))
        y = d["y"][..., 0].transpose(0, 2, 1).astype(np.float64)        # (B,N,H)
        tod = d["y_tod"][..., 0].transpose(0, 2, 1)                     # (B,N,H) in [0,1)
        m = d["y_mask"][..., 0].transpose(0, 2, 1).astype(bool)
        slot = np.rint(tod * SLOTS_PER_DAY).astype(int) % SLOTS_PER_DAY
        return y, slot, m

    ytr, slot_tr, mtr = load("train")
    N, H = ytr.shape[1], ytr.shape[2]
    node = np.broadcast_to(np.arange(N), ytr.shape[:2])    # (B,N)
    # per-(node, tod-slot) train mean over observed entries, one bincount per horizon.
    clim = np.full((N, SLOTS_PER_DAY, H), np.nan)
    for h in range(H):
        mm = mtr[:, :, h]
        lin = node[mm] * SLOTS_PER_DAY + slot_tr[:, :, h][mm]
        sums = np.bincount(lin, weights=ytr[:, :, h][mm], minlength=N * SLOTS_PER_DAY)
        cnts = np.bincount(lin, minlength=N * SLOTS_PER_DAY)
        mean = np.where(cnts > 0, sums / np.maximum(cnts, 1), np.nan).reshape(N, SLOTS_PER_DAY)
        node_mean = np.nanmean(mean, axis=1)               # fallback for unseen slots
        clim[:, :, h] = np.where(np.isnan(mean), node_mean[:, None], mean)

    y, slot, m = load(split)
    # gather prediction: pred[b,n,h] = clim[n, slot[b,n,h], h]
    pred = clim[np.arange(N)[None, :, None], slot, np.arange(H)[None, None, :]]
    return (y - pred).astype(np.float32), m


def load_cell(dataset: str, rung: str, split: str, seed: int = 42
              ) -> tuple[np.ndarray, np.ndarray]:
    """Return (R, mask) for one cell/split: R (B,N,H) float32 raw units, mask bool."""
    if rung == "signal":
        return _signal_climatology(dataset, split, seed)
    d = np.load(paths.residual_npz(dataset, split, seed, rung))
    R = d["residuals"].astype(np.float32)
    mask = d["mask"].astype(bool)
    return R, mask


# ─── helpers ──────────────────────────────────────────────────────────────────

def _node_corr(R: np.ndarray, mask: np.ndarray, max_samples: int = 8000) -> np.ndarray:
    """(N,N) node-node Pearson correlation over (subsampled) window x horizon samples.
    masked entries are dropped per node by mean-imputation before correlating."""
    B, N, H = R.shape
    X = R.transpose(1, 0, 2).reshape(N, B * H)             # (N, B*H)
    M = mask.transpose(1, 0, 2).reshape(N, B * H)
    if X.shape[1] > max_samples:                          # subsample columns
        idx = np.linspace(0, X.shape[1] - 1, max_samples).astype(int)
        X, M = X[:, idx], M[:, idx]
    X = np.where(M, X, np.nan)
    mu = np.nanmean(X, axis=1, keepdims=True)
    X = np.where(np.isnan(X), mu, X) - mu
    s = np.sqrt((X ** 2).mean(axis=1)); s[s == 0] = 1.0
    return (X @ X.T) / X.shape[1] / np.outer(s, s)


def edge_ratio_rho(R: np.ndarray, mask: np.ndarray, ctx: dict) -> dict:
    """spatial coupling: mean |corr| on edges vs non-edges and their ratio.
    ratio >> 1 means residual energy follows the graph; ~1 means white/iso."""
    C = _node_corr(R, mask)
    off = np.abs(C[ctx["iu"]])
    e, ne = ctx["is_edge"], ~ctx["is_edge"]
    rho_e = float(off[e].mean()) if e.any() else float("nan")
    rho_ne = float(off[ne].mean()) if ne.any() else float("nan")
    return {"rho_edge": rho_e, "rho_nonedge": rho_ne,
            "rho_ratio": rho_e / rho_ne if rho_ne > 1e-9 else float("nan"),
            "rho_global": float(off.mean())}


def gft_band_profile(R: np.ndarray, ctx: dict, n_bands: int = 3) -> dict:
    """fraction of spatial energy in low/mid/high Laplacian-eigenvalue bands (sum=1).
    low-heavy = smooth on the graph (L+γI prior fits); high-heavy = rough (wrong shape)."""
    V_S, lam = ctx["V_S"], ctx["lam"]
    U = np.einsum("ki,bih->bkh", V_S.T, R.astype(np.float64))   # (B,N,H) spectrum
    energy = (U ** 2).mean(axis=(0, 2))                         # (N,) per-mode energy
    edges = np.linspace(lam.min(), lam.max() + 1e-9, n_bands + 1)
    band = np.digitize(lam, edges[1:-1])
    tot = energy.sum() or 1.0
    frac = [float(energy[band == b].sum() / tot) for b in range(n_bands)]
    centroid = float((lam * energy).sum() / tot)
    return {"gft_low": frac[0], "gft_mid": frac[1], "gft_high": frac[2],
            "gft_lambda_centroid": centroid}


def kron_separability_eta(R: np.ndarray, mask: np.ndarray, max_windows: int = 4000) -> dict:
    """nearest-Kronecker separability: eta = ||Σ_emp - Σ_S⊗Σ_T||_F / ||Σ_emp||_F via the
    Van Loan-Pitsianis rearrangement (top singular value of R(Σ)). eta~0 = Kronecker-
    separable (factorization justified); eta~1 = not separable."""
    B, N, H = R.shape
    keep = mask.all(axis=2)                                   # fully-observed (window,node)? no
    # use windows where ALL (N,H) observed so vec covariance is clean; fall back to all
    full = mask.reshape(B, -1).all(axis=1)
    Rw = R[full] if full.sum() >= 50 else R
    if Rw.shape[0] > max_windows:
        Rw = Rw[np.linspace(0, Rw.shape[0] - 1, max_windows).astype(int)]
    X = Rw.reshape(Rw.shape[0], N * H).astype(np.float64)
    X -= X.mean(axis=0, keepdims=True)
    Sig = (X.T @ X) / X.shape[0]                              # (NH, NH) empirical cov
    fro2 = float((Sig ** 2).sum())
    # rearrange Σ[(i,h),(i',h')] -> Rmat[(i,i'),(h,h')], shape (N², H²)
    Rmat = Sig.reshape(N, H, N, H).transpose(0, 2, 1, 3).reshape(N * N, H * H)
    s1 = np.linalg.svd(Rmat, compute_uv=False)[0]            # top singular value
    eta = float(np.sqrt(max(1.0 - s1 ** 2 / fro2, 0.0)))
    return {"eta_separability": eta, "n_windows_eta": int(Rw.shape[0])}


def excess_kurtosis(R: np.ndarray, mask: np.ndarray) -> dict:
    """masked Fisher (excess) kurtosis overall and per-horizon. 0 = Gaussian; the
    Student-t branch's reason to exist scales with this."""
    from scipy.stats import kurtosis
    v = R[mask].astype(np.float64)
    H = R.shape[2]
    per_h = []
    for h in range(H):
        vh = R[:, :, h][mask[:, :, h]].astype(np.float64)
        per_h.append(float(kurtosis(vh, fisher=True)) if vh.size > 10 else float("nan"))
    return {"excess_kurtosis": float(kurtosis(v, fisher=True)) if v.size > 10 else float("nan"),
            "excess_kurtosis_h": per_h}


# ─── Tier 1: full per-cell descriptive bundle ─────────────────────────────────

def tier1_cell(dataset: str, rung: str, ctx: dict, split: str = "train",
               feat_windows: int = 3000) -> dict:
    """Every Tier-1 descriptive for one cell. Scale-invariant shape metrics make the
    ladder (signal->mlp->gcn->gwn) comparable; D_h is reported normalized too."""
    from src.diffusion.prior import _fit_sigma_T
    from src.eval import signal_features as sf

    R, mask = load_cell(dataset, rung, split)
    B, N, H = R.shape
    v = R[mask].astype(np.float64)

    out = {"dataset": dataset, "rung": rung, "N": N, "H": H, "n_windows": B,
           "missing_pct": round(100.0 * (1.0 - mask.mean()), 3),
           "value_std": float(v.std()), "value_bias": float(v.mean()),
           "value_absmax": float(np.abs(v).max())}
    out.update(edge_ratio_rho(R, mask, ctx))
    out.update(gft_band_profile(R, ctx))
    out.update(kron_separability_eta(R, mask))
    out.update(excess_kurtosis(R, mask))

    # per-window signal_features, averaged over a window subsample
    idx = np.linspace(0, B - 1, min(B, feat_windows)).astype(int)
    Rs = R[idx].astype(np.float64)
    out["spatial_tv"] = float(sf.total_variation(Rs, ctx["L"]).mean())
    out["spectral_centroid"] = float(sf.spectral_centroid(Rs, ctx["lam"], ctx["V_S"]).mean())
    out["degree_corr"] = float(sf.degree_correlation(Rs, ctx["degree"]).mean())
    out["lag1_autocorr"] = float(sf.lag1_autocorr(Rs).mean())
    out["horizon_centroid"] = float(sf.horizon_centroid(Rs).mean())

    # temporal structure: per-horizon variance D_h and adjacent-horizon correlation R_H
    delta_h, R_H, _ = _fit_sigma_T(R.astype(np.float64), mask)
    out["D_h"] = [float(x) for x in delta_h]
    out["D_h_growth"] = float(delta_h[-1] / delta_h[0]) if delta_h[0] > 0 else float("nan")
    out["R_H_adjacent"] = float(np.mean(np.diag(R_H, k=1)))   # mean lag-1 horizon corr
    out["R_H_far"] = float(R_H[0, -1])                        # h1 vs h12 corr
    return out


# ─── Tier 2: static spatial-covariance bake-off (Gaussian NLL on val) ─────────

def _spatial_cov(R: np.ndarray, mask: np.ndarray, max_cols: int = 20000) -> np.ndarray:
    """(N,N) marginal spatial covariance over (window x horizon) samples; missing
    entries mean-imputed per node (small fraction). The temporal factor is the same
    for every candidate so it cancels in the ranking — we compare spatial models."""
    B, N, H = R.shape
    X = R.transpose(1, 0, 2).reshape(N, B * H).astype(np.float64)
    M = mask.transpose(1, 0, 2).reshape(N, B * H)
    if X.shape[1] > max_cols:
        idx = np.linspace(0, X.shape[1] - 1, max_cols).astype(int)
        X, M = X[:, idx], M[:, idx]
    X = np.where(M, X, np.nan)
    mu = np.nanmean(X, axis=1, keepdims=True)
    X = np.where(np.isnan(X), mu, X) - mu
    return (X @ X.T) / X.shape[1]


def _candidates_S(S_train: np.ndarray, ctx: dict, shrink: float = 0.05) -> dict:
    """four spatial-covariance models built from the TRAIN spatial covariance.
    Each value is (Sigma_S, inv_Sigma_S, logdet) for the NLL."""
    N = S_train.shape[0]
    tr = np.trace(S_train)
    out = {}
    # isotropic sigma^2 I
    s2 = tr / N
    out["isotropic"] = (s2 * np.eye(N), np.eye(N) / s2, N * np.log(s2))
    # diagonal D_S
    d = np.clip(np.diag(S_train), 1e-9, None)
    out["diagonal"] = (np.diag(d), np.diag(1.0 / d), float(np.log(d).sum()))
    # graph (L + 0.8 I)^-1, scaled so trace matches S_train
    lam, V = ctx["lam"], ctx["V_S"]
    q = lam + 0.8                                            # precision eigenvalues
    raw_tr = float((1.0 / q).sum())
    c = tr / raw_tr                                          # variance match
    Sg = (V * (c / q)) @ V.T
    inv = (V * (q / c)) @ V.T
    logdet = float(np.log(c / q).sum())
    out["graph"] = (Sg, inv, logdet)
    # full empirical, ridge-shrunk to stay PD and generalize
    Se = (1 - shrink) * S_train + shrink * s2 * np.eye(N)
    ev, U = np.linalg.eigh(Se)
    ev = np.clip(ev, 1e-9, None)
    out["empirical"] = (Se, (U / ev) @ U.T, float(np.log(ev).sum()))
    return out


def _cov_nll(inv_S: np.ndarray, logdet: float, S_val: np.ndarray, N: int) -> float:
    """mean Gaussian NLL per N-dim spatial sample:
    0.5[ N log 2pi + log|Σ_S| + tr(Σ_S^-1 S_val) ]."""
    quad = float(np.einsum("ij,ji->", inv_S, S_val))        # tr(inv_S @ S_val)
    return 0.5 * (N * np.log(2 * np.pi) + logdet + quad)


def tier2_cell(dataset: str, rung: str, ctx: dict) -> dict:
    """rank the four spatial-covariance models by held-out (val) NLL + Mahalanobis."""
    Rtr, mtr = load_cell(dataset, rung, "train")
    Rva, mva = load_cell(dataset, rung, "val")
    S_train = _spatial_cov(Rtr, mtr)
    S_val = _spatial_cov(Rva, mva)
    N = S_train.shape[0]
    cands = _candidates_S(S_train, ctx)
    nll = {k: _cov_nll(inv, ld, S_val, N) for k, (S, inv, ld) in cands.items()}
    maha = {k: float(np.einsum("ij,ji->", inv, S_val)) / N   # tr(inv S_val)/N (=1 if perfect)
            for k, (S, inv, ld) in cands.items()}
    winner = min(nll, key=nll.get)
    return {"dataset": dataset, "rung": rung, "N": N,
            "nll": {k: round(v, 4) for k, v in nll.items()},
            "mahalanobis": {k: round(v, 4) for k, v in maha.items()},
            "winner": winner,
            "graph_vs_iso_nll": round(nll["isotropic"] - nll["graph"], 4),
            "ranking": sorted(nll, key=nll.get)}


# ─── Tier 3: forward SDE simulation — math validation + mixing ─────────────────

def tier3_cell(dataset: str, rung: str, ctx: dict, n_base: int = 6, K: int = 200,
               n_c: int = 8) -> dict:
    """GAD forward on real residuals: (a) the mixing trajectory is analytic — fraction
    of modes mixed (exp(-2θc̄)<0.05) along the c̄ grid, and whether GAD's c₀ mixes the
    slowest mode; (b) a single-c̄ MC sanity that the REAL forward_fcps sampler's per-mode
    mean/var match closed-form (median over modes, so a V_S⊗V_T bug stands out)."""
    from src.diffusion import forward_fcps as F
    from src.diffusion.prior import build_prior_from_R
    from src.eval.prior_diagnostics import conditioning_row

    Rtr, mtr = load_cell(dataset, rung, "train")
    prior, sigma_r = build_prior_from_R(dataset, Rtr, mtr, spatial="structured", gamma=0.8)
    cond = conditioning_row(prior)
    c_lo, c_hi = F.noise_level_bounds(prior)
    c_grid = np.geomspace(c_lo, c_hi, n_c)
    theta = prior.theta
    # analytic mixing trajectory: fraction of modes with exp(-2θc̄) < 0.05
    mix_traj = [float((np.exp(-2.0 * theta * cb) < 0.05).mean()) for cb in c_grid]

    # one-c̄ MC sanity at the level where the median mode is half-decayed (both mean
    # and var meaningful): cb* = ln2 / median(theta).
    cb = float(np.log(2.0) / np.median(theta))
    base = (Rtr[:n_base].astype(np.float64) / sigma_r)
    C0 = F._rotate_to_eigenbasis(base, prior)
    rng = np.random.default_rng(0)
    Rt = F.sample_R_t_from_c_bar(np.repeat(base, K, axis=0), cb, prior, rng)
    Ct = F._rotate_to_eigenbasis(Rt, prior).reshape(n_base, K, prior.N, prior.H)
    mean_cf = np.exp(-theta * cb)[None] * C0
    var_cf = (1.0 - np.exp(-2.0 * theta * cb)) / theta
    # variance is the robust math check (it pins θ AND the V_S⊗V_T rotation: a wrong
    # rotation mixes modes and breaks the per-mode variances). The per-mode MEAN is
    # MC-noisy on the half-decayed modes at finite K, so it is reported, not gated.
    mean_relerr = float(np.median(np.abs(Ct.mean(1) - mean_cf)
                                  / (np.abs(mean_cf) + 1e-6)))
    var_relerr = float(np.median(np.abs(Ct.var(1) - var_cf[None]) / (var_cf[None] + 1e-9)))
    return {"dataset": dataset, "rung": rung,
            "theta_min": cond["θ_min"], "theta_max": cond["θ_max"],
            "kappa_Q": cond["κ(Q)"], "c0_mix_budget": cond["c₀ (mix budget)"],
            "c0_theta_min": cond["c₀ (mix budget)"] * cond["θ_min"],
            "c_hi": float(c_hi), "frac_modes_mixed_at_chi": round(mix_traj[-1], 4),
            "mix_trajectory": [round(x, 3) for x in mix_traj],
            "c_grid": [float(x) for x in c_grid],
            "marginal_mean_relerr": round(mean_relerr, 4),
            "marginal_var_relerr": round(var_relerr, 4),
            "math_validated": bool(var_relerr < 0.15)}


# ─── Tier 4: analytic-score generation (CRPS / PIT) ───────────────────────────

def analytic_gaussian_denoiser(prior, c_lo: float, c_hi: float, device):
    """Closed-form optimal R0-denoiser for N(0,Σ): in the eigenbasis the posterior
    mean of the stationary OU is Ĉ0 = exp(-θ·c̄)·C_t. Maps the reverse sampler's
    normalised tau back to c̄ = c_lo·(c_hi/c_lo)^tau. Returns a denoiser(R_t,tau,Y_hat)."""
    import math
    import torch
    Vs = torch.as_tensor(prior.V_S, device=device, dtype=torch.float32)
    Vt = torch.as_tensor(prior.V_T, device=device, dtype=torch.float32)
    th = torch.as_tensor(prior.theta, device=device, dtype=torch.float32)
    log_lo, log_span = math.log(c_lo), math.log(c_hi / c_lo)

    def denoiser(R_t, tau, Y_hat):           # R_t,(B,N,H); tau,(B,)
        cb = torch.exp(tau * log_span + log_lo).view(-1, 1, 1)
        C = torch.einsum("ki,bih,hj->bkj", Vs.T, R_t, Vt)
        C = torch.exp(-th.unsqueeze(0) * cb) * C
        return torch.einsum("ik,bkh,hj->bij", Vs, C, Vt.T)
    return denoiser


def _crps_pit(ens: np.ndarray, obs: np.ndarray) -> tuple[float, float]:
    """ensemble CRPS (lower=better) and PIT-calibration error (0=perfectly uniform).
    ens (M,N,H) generated samples; obs (W,N,H) held-out residuals. Both per (node,h),
    averaged. CRPS = E_m|x-y| - 0.5 E_mm'|x-x'|; PIT_cal = mean|hist - uniform|."""
    M, N, H = ens.shape
    W = obs.shape[0]
    # subsample to keep the (M x W) per-cell outer products small
    if W > 400:
        obs = obs[np.linspace(0, W - 1, 400).astype(int)]; W = 400
    crps_nh, pit_ranks = [], []
    spread = np.abs(ens[:, None] - ens[None, :]).mean(axis=(0, 1))      # (N,H)
    for w in range(W):
        e1 = np.abs(ens - obs[w][None]).mean(axis=0)                     # (N,H)
        crps_nh.append(e1 - 0.5 * spread)
        pit_ranks.append((ens < obs[w][None]).mean(axis=0))             # (N,H) PIT
    crps = float(np.mean(crps_nh))
    pit = np.asarray(pit_ranks).ravel()
    hist, _ = np.histogram(pit, bins=10, range=(0, 1), density=False)
    pit_cal = float(np.abs(hist / pit.size - 0.1).sum())                # L1 vs uniform
    return crps, pit_cal


def tier4_cell(dataset: str, rung: str, ctx: dict, steps=(10, 25, 50, 100),
               M: int = 128, n_windows: int = 400, device: str = "cpu") -> dict:
    """Generate samples with the ANALYTIC Gaussian score for the structured (γ=0.8)
    and isotropic priors, across step counts, and score CRPS+PIT vs held-out residuals.
    Isolates the prior's contribution (no net). GAD claim: structured gap widens as
    steps shrink. The structured ceiling-to-reality gap is what the net is for."""
    import torch
    from src.diffusion import forward_fcps as F
    from src.diffusion.prior import build_prior_from_R
    from src.diffusion.reverse import reverse_sample

    Rtr, mtr = load_cell(dataset, rung, "train")
    Rte, mte = load_cell(dataset, rung, "test")
    dev = torch.device(device)
    res = {"dataset": dataset, "rung": rung, "steps": list(steps),
           "crps": {"structured": {}, "isotropic": {}}, "pit_cal": {"structured": {}, "isotropic": {}}}
    sigma_r = float(Rtr[mtr].std())
    obs = (Rte[:max(n_windows * 3, 1200)].astype(np.float32) / sigma_r)   # unit-space held-out

    for name, spatial in [("structured", "structured"), ("isotropic", "iso")]:
        prior, _ = build_prior_from_R(dataset, Rtr, mtr, spatial=spatial, gamma=0.8)
        c_lo, c_hi = F.noise_level_bounds(prior)
        sched = F.FCPSSchedule.for_prior(prior)
        den = analytic_gaussian_denoiser(prior, c_lo, c_hi, dev)
        Yh = torch.zeros((M, prior.N, prior.H), device=dev)               # unconditional
        for ns in steps:
            g = torch.Generator(device=device).manual_seed(0)
            R0, _ = reverse_sample(prior, sched, den, Yh, n_steps=ns, eta=1.0, generator=g)
            ens = R0.cpu().numpy()
            crps, pit = _crps_pit(ens, obs)
            res["crps"][name][str(ns)] = round(crps, 4)
            res["pit_cal"][name][str(ns)] = round(pit, 4)
    res["crps_gap_pct"] = {s: round(100 * (res["crps"]["isotropic"][s] - res["crps"]["structured"][s])
                                    / res["crps"]["isotropic"][s], 2) for s in map(str, steps)}
    return res
