"""Raw signal vs backbone residual: which one does the Kronecker prior Σ_T⊗Σ_S actually
fit, decomposed into a SEPARATE spatial analysis, a SEPARATE temporal analysis, and their
combination — plus a from-scratch verification that doing the forward SDE in the joint
eigenspace is exact (not an approximation).

Fields (same npz, y_true = y_pred + residual):
  raw    = (y_true − mean)/std          the centered raw future window (GAD's setting)
  resid  = residual / std_resid         the frozen-backbone forecast residual (our setting)

Sections (all offline, train fit / test held out):
  1. SEPARABILITY     Van Loan–Pitsianis index — is the field's covariance Kronecker at all?
  2. SPATIAL          graph-spectral energy: Dirichlet (white=1), DC-mode share, smooth-10%,
                      alignment with the heat prior 1/(λ+γ). low = graph-smooth (prior fits).
  3. TEMPORAL         Σ_T structure: per-horizon variance growth, lag-1 autocorrelation,
                      common-mode share, and off-band precision mass (is Q_T a path graph?).
  4. ATTRIBUTION      held-out per-mode NLL of iso / T-only / S-only / full, so ΔT (time buys),
                      ΔS (graph buys), graph|T (graph on top of time) — raw vs resid side by side.
  5. EIGENSPACE CHECK the forward is run per-mode in the (V_S⊗V_T) basis; verify that basis
                      really diagonalises the full precision Q=Σ_T⁻¹⊗Σ_S⁻¹ and that the per-mode
                      marginal equals the full-covariance matrix-function marginal (rel err ~1e-12).

usage: .venv/bin/python scripts/analysis/raw_vs_residual_prior.py [--datasets metrla pems08f pems04f]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.utils import paths                                                        # noqa: E402
from src.diffusion.prior import GAMMA_DEFAULT, SHRINK_T                            # noqa: E402
from scripts.analysis.kronecker_audit import separability, _nll                   # noqa: E402

DATASETS = ["metrla", "pems08f", "pems04f"]
BB = "gwn"
CAP = 6000
GAMMA = GAMMA_DEFAULT
OUT = ROOT / "results/analysis/paper_results"
FIG = OUT / "figs"


# ── loading: raw vs residual, both (B,N,H), train scaler ────────────────────────

def _scaler(ds, bb, field):
    d = np.load(paths.residual_npz(ds, "train", 42, bb))
    if field == "resid":
        R = d["residuals"].astype(np.float64); M = d["mask"].astype(bool)
        return float(R[M].std()), 0.0
    return float(d["std"]), float(d["mean"])                       # raw: stored global mean/std


def load_field(ds, split, bb, field, scale, shift):
    d = np.load(paths.residual_npz(ds, split, 42, bb))
    R = d["residuals"].astype(np.float64); M = d["mask"].astype(bool)
    if field == "resid":
        X = R / scale
    else:
        X = (d["y_pred"].astype(np.float64) + R - shift) / scale   # (y_pred+resid)=y_true, centered
    X = np.where(M, X, 0.0)
    if len(X) > CAP:
        X = X[np.linspace(0, len(X) - 1, CAP).astype(int)]
    return X


# ── graph + temporal fits (symmetric for raw and resid) ─────────────────────────

def graph_eig(ds, repair=True):
    from src.data.adjacency import load_adjacency, compute_normalized_laplacian
    L = compute_normalized_laplacian(load_adjacency(ds, repair=repair))
    lam, V = np.linalg.eigh(L)
    o = np.argsort(lam)
    return lam[o], V[:, o]


def fit_temporal(X, rho=SHRINK_T):
    """pool every (window,node) H-vector → empirical temporal covariance Σ_T = D^½ R_H D^½,
    ridge-shrunk (μ←(1-ρ)μ+ρμ̄), then eigendecompose. mirrors prior._fit_sigma_T."""
    B, N, H = X.shape
    F = X.reshape(B * N, H)
    F = F[np.abs(F).sum(1) > 0]                                    # drop all-zero (masked) rows
    d_h = (F ** 2).mean(0)
    R_H = np.corrcoef(F.T)
    Ds = np.sqrt(d_h)
    Sig = (Ds[:, None] * R_H) * Ds[None, :]
    mu, V = np.linalg.eigh(Sig)
    mu = np.clip(mu, 1e-12, None)
    if rho:
        mu = (1.0 - rho) * mu + rho * mu.mean()
    return d_h, R_H, mu, V, Sig


def _modevar(X, V_S, V_T):
    C = X
    if V_S is not None:
        C = np.einsum("ki,bih->bkh", V_S.T, C, optimize=True)
    if V_T is not None:
        C = np.einsum("bkh,hj->bkj", C, V_T, optimize=True)
    return (C ** 2).mean(0)


# ── 2. spatial structure ────────────────────────────────────────────────────────

def spatial_metrics(X, lam_S, V_S):
    C = np.einsum("ki,bih->bkh", V_S.T, X, optimize=True)
    E = (C ** 2).mean(axis=(0, 2))                                 # per graph-mode energy
    meanlam = lam_S.mean()
    dirichlet = float((lam_S * E).sum() / E.sum() / meanlam)       # <1 smoother than white
    p = E / E.sum()
    o = np.argsort(lam_S)                                          # ascending graph frequency
    dc = float(p[o[0]])                                            # smoothest (near-DC) mode share
    k10 = max(1, int(0.10 * len(lam_S)))
    smooth10 = float(p[o][:k10].sum())                             # share in smoothest 10% (white=.10)
    align = float(np.corrcoef(E, 1.0 / (lam_S + GAMMA))[0, 1])
    return dict(dirichlet=dirichlet, dc=dc, smooth10=smooth10, align=align)


# ── 3. temporal structure ───────────────────────────────────────────────────────

def temporal_metrics(d_h, R_H, Sig):
    H = len(d_h)
    D = np.sqrt(d_h)
    growth = float(D[-1] / D[0])                                   # std(step H) / std(step 1)
    lag1 = float(np.mean([R_H[h, h + 1] for h in range(H - 1)]))
    ev = np.sort(np.linalg.eigvalsh(R_H))[::-1]
    common = float(ev[0] / ev.sum())
    Q = np.linalg.inv(Sig)
    off = np.abs(Q).copy(); np.fill_diagonal(off, 0.0)
    band = np.abs(np.subtract.outer(np.arange(H), np.arange(H))) == 1
    offband = float(off[~band & ~np.eye(H, dtype=bool)].sum() / off.sum())
    return dict(growth=growth, lag1=lag1, common=common, offband=offband)


# ── 4. attribution (held-out NLL of the nested family) ──────────────────────────

def attribution(Xtr, Xte, lam_S, V_S, mu_T, V_T):
    N, H = len(lam_S), len(mu_T)
    s_S = 1.0 / (lam_S + GAMMA); s_S /= s_S.mean()
    t_T = mu_T / mu_T.mean()
    one_N, one_H = np.ones(N), np.ones(H)
    cands = [("iso", None, None, np.outer(one_N, one_H)),
             ("T-only", None, V_T, np.outer(one_N, t_T)),
             ("S-only", V_S, None, np.outer(s_S, one_H)),
             ("full", V_S, V_T, np.outer(s_S, t_T))]
    out = {}
    for name, vs, vt, shape in cands:
        nll, oct = _nll(_modevar(Xtr, vs, vt), _modevar(Xte, vs, vt), shape)
        out[f"nll_{name}"] = nll; out[f"oct_{name}"] = oct
    out["dT"] = out["nll_iso"] - out["nll_T-only"]
    out["dS"] = out["nll_iso"] - out["nll_S-only"]
    out["graph_on_time"] = out["nll_T-only"] - out["nll_full"]
    return out


# ── 5. eigenspace-forward correctness (from scratch, full covariance) ───────────

def eigenspace_check(ds):
    """verify the per-mode eigenspace forward equals the true full-covariance forward.
    build the FULL precision Q=Σ_T⁻¹⊗Σ_S⁻¹ (NH×NH) and check (a) V_T⊗V_S diagonalises it,
    (b) the marginal covariance at c̄ from per-mode σ² equals the full matrix-function marginal."""
    lam_S, V_S = graph_eig(ds)
    N = len(lam_S)
    Xtr = load_field(ds, "train", BB, "resid", *_scaler(ds, BB, "resid"))
    _, _, mu_T, V_T, _ = fit_temporal(Xtr)
    H = len(mu_T)
    q_S = lam_S + GAMMA
    tau2 = N / (1.0 / q_S).sum()
    theta = q_S[:, None] / (tau2 * mu_T[None, :])                  # (N,H) per-mode precision
    sig2 = 1.0

    # full precision Q = kron(Σ_T⁻¹, Σ_S⁻¹) with vec convention h·N+i (temporal outer)
    QS = V_S @ np.diag(q_S) @ V_S.T                                # Σ_S⁻¹
    QT = V_T @ np.diag(1.0 / (tau2 * mu_T)) @ V_T.T                # (τ²Σ_T)⁻¹
    Qfull = np.kron(QT, QS)                                        # (NH, NH)
    W = np.kron(V_T, V_S)                                          # tensor eigvecs

    # (a) does W diagonalise Qfull with eigenvalues vec(theta) (order h·N+i)?
    theta_vec = theta.T.reshape(-1)                                # [h*N+i] = theta[i,h]
    diag_err = np.linalg.norm(Qfull @ W - W * theta_vec[None, :]) / np.linalg.norm(Qfull @ W)
    orth = max(np.linalg.norm(V_S.T @ V_S - np.eye(N)), np.linalg.norm(V_T.T @ V_T - np.eye(H)))

    # (b) marginal at a mid c̄: per-mode σ²_t vs full matrix-function (I−e^{−2c̄Q})Q⁻¹
    cbar = 0.5
    lamQ, U = np.linalg.eigh(Qfull)                                # full eig (independent path)
    Sig_t_full = (U * ((1.0 - np.exp(-2 * cbar * lamQ)) / lamQ)[None, :]) @ U.T
    s2_mode = sig2 * (1.0 - np.exp(-2 * cbar * theta)) / theta     # (N,H) per-mode
    Sig_t_eig = (W * s2_mode.T.reshape(-1)[None, :]) @ W.T         # reconstruct full from per-mode
    marg_err = np.linalg.norm(Sig_t_eig - Sig_t_full) / np.linalg.norm(Sig_t_full)

    # (c) empirical: draw via per-mode sampler, compare cov to analytic
    rng = np.random.default_rng(0)
    Bn = 20000
    Z = rng.standard_normal((Bn, N, H))
    C = np.sqrt(s2_mode)[None] * Z
    Rt = np.einsum("ik,bkh,hj->bij", V_S, C, V_T.T, optimize=True)  # forward-from-zero at c̄
    vec = Rt.transpose(0, 2, 1).reshape(Bn, H * N)
    emp_err = np.linalg.norm(np.cov(vec, rowvar=False) - Sig_t_full) / np.linalg.norm(Sig_t_full)
    return dict(N=N, H=H, orth=float(orth), diag_err=float(diag_err),
                marg_err=float(marg_err), emp_err=float(emp_err))


# ── driver ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=DATASETS)
    args = ap.parse_args()

    rows = {}
    for ds in args.datasets:
        lam_S, V_S = graph_eig(ds)
        rows[ds] = {}
        for field in ("raw", "resid"):
            sc = _scaler(ds, BB, field)
            Xtr = load_field(ds, "train", BB, field, *sc)
            Xte = load_field(ds, "test", BB, field, *sc)
            d_h, R_H, mu_T, V_T, Sig = fit_temporal(Xtr)
            rows[ds][field] = dict(
                sep=separability(Xtr),
                spat=spatial_metrics(Xtr, lam_S, V_S),
                temp=temporal_metrics(d_h, R_H, Sig),
                attr=attribution(Xtr, Xte, lam_S, V_S, mu_T, V_T),
            )

    # ---- print + markdown ----
    md = ["# Raw signal vs residual — Kronecker prior structure analysis\n",
          f"backbone **{BB}**, γ={GAMMA}, held-out **test**. field: raw=(y_true−mean)/std, resid=residual/σ.\n"]

    print("\n=== 1. SEPARABILITY (Van Loan–Pitsianis top-1; 1.0 = perfectly Kronecker) ===")
    md += ["## 1. Separability (VLP top-1 / top-3)\n",
           "| dataset | raw | resid |", "|---|--:|--:|"]
    print(f"{'dataset':9s} | {'raw sep1/sep3':>16s} | {'resid sep1/sep3':>16s}")
    for ds in args.datasets:
        r, s = rows[ds]["raw"]["sep"], rows[ds]["resid"]["sep"]
        print(f"{ds:9s} | {r['sep1']:.3f} / {r['sep3']:.3f}    | {s['sep1']:.3f} / {s['sep3']:.3f}")
        md.append(f"| {ds} | {r['sep1']:.3f} / {r['sep3']:.3f} | {s['sep1']:.3f} / {s['sep3']:.3f} |")

    print("\n=== 2. SPATIAL (Dirichlet white=1; DC share; smooth-10% white=.10; align w/ 1/(λ+γ)) ===")
    md += ["\n## 2. Spatial structure (graph spectrum)\n",
           "Dirichlet <1 = graph-smooth (low-frequency); align = corr of residual energy with the heat prior 1/(λ+γ).\n",
           "| dataset | field | Dirichlet | DC-mode | smooth-10% | align 1/(λ+γ) |", "|---|---|--:|--:|--:|--:|"]
    print(f"{'dataset':9s} {'field':5s} | {'Dirichlet':>9s} {'DC':>6s} {'smth10':>6s} {'align':>6s}")
    for ds in args.datasets:
        for f in ("raw", "resid"):
            m = rows[ds][f]["spat"]
            print(f"{ds:9s} {f:5s} | {m['dirichlet']:9.3f} {m['dc']:6.3f} {m['smooth10']:6.3f} {m['align']:6.3f}")
            md.append(f"| {ds} | {f} | {m['dirichlet']:.3f} | {m['dc']:.3f} | {m['smooth10']:.3f} | {m['align']:.3f} |")

    print("\n=== 3. TEMPORAL (var growth H/1; lag-1 autocorr; common-mode share; off-band precision) ===")
    md += ["\n## 3. Temporal structure (Σ_T)\n",
           "growth = std(step12)/std(step1); common = top eig share of R_H; offband = precision mass beyond nearest-neighbour (path graph = 0).\n",
           "| dataset | field | var growth | lag-1 ρ | common-mode | off-band precision |", "|---|---|--:|--:|--:|--:|"]
    print(f"{'dataset':9s} {'field':5s} | {'growth':>6s} {'lag1':>6s} {'common':>6s} {'offband':>7s}")
    for ds in args.datasets:
        for f in ("raw", "resid"):
            m = rows[ds][f]["temp"]
            print(f"{ds:9s} {f:5s} | {m['growth']:6.2f} {m['lag1']:6.2f} {m['common']:6.2f} {m['offband']:7.2f}")
            md.append(f"| {ds} | {f} | {m['growth']:.2f} | {m['lag1']:.2f} | {m['common']:.2f} | {m['offband']:.2f} |")

    print("\n=== 4. ATTRIBUTION (held-out per-mode NLL; ΔT time buys, ΔS graph buys, graph|T on top of time) ===")
    md += ["\n## 4. Attribution (held-out per-mode NLL, lower=better)\n",
           "ΔT = iso−T-only (time buys) · ΔS = iso−S-only (graph buys) · graph|T = T-only−full (graph on top of time). nats/mode.\n",
           "| dataset | field | iso | T-only | S-only | full | ΔT | ΔS | graph\\|T |", "|---|---|--:|--:|--:|--:|--:|--:|--:|"]
    print(f"{'dataset':9s} {'field':5s} | {'iso':>6s} {'Tonly':>6s} {'Sonly':>6s} {'full':>6s} | {'ΔT':>6s} {'ΔS':>6s} {'grph|T':>6s}")
    for ds in args.datasets:
        for f in ("raw", "resid"):
            a = rows[ds][f]["attr"]
            print(f"{ds:9s} {f:5s} | {a['nll_iso']:6.3f} {a['nll_T-only']:6.3f} {a['nll_S-only']:6.3f} "
                  f"{a['nll_full']:6.3f} | {a['dT']:6.3f} {a['dS']:6.3f} {a['graph_on_time']:6.3f}")
            md.append(f"| {ds} | {f} | {a['nll_iso']:.3f} | {a['nll_T-only']:.3f} | {a['nll_S-only']:.3f} | "
                      f"{a['nll_full']:.3f} | {a['dT']:.3f} | {a['dS']:.3f} | {a['graph_on_time']:.3f} |")

    print("\n=== 5. EIGENSPACE-FORWARD CORRECTNESS (rel errors → ~0 mean the per-mode forward is EXACT) ===")
    md += ["\n## 5. Eigenspace-forward correctness\n",
           "The forward runs per-mode in the (V_S⊗V_T) basis. These rel-errors verify that basis IS the exact "
           "eigenbasis of the full precision Q=Σ_T⁻¹⊗Σ_S⁻¹ and reproduces the full-covariance marginal.\n",
           "| dataset | N·H | orthonormality | Q diagonalised | marginal vs full | empirical cov |", "|---|--:|--:|--:|--:|--:|"]
    print(f"{'dataset':9s} | {'N·H':>5s} | {'orth':>9s} {'diag_err':>9s} {'marg_err':>9s} {'emp_err':>9s}")
    for ds in args.datasets:
        c = eigenspace_check(ds)
        print(f"{ds:9s} | {c['N']*c['H']:5d} | {c['orth']:9.2e} {c['diag_err']:9.2e} {c['marg_err']:9.2e} {c['emp_err']:9.3f}")
        md.append(f"| {ds} | {c['N']*c['H']} | {c['orth']:.1e} | {c['diag_err']:.1e} | {c['marg_err']:.1e} | {c['emp_err']:.3f} |")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "raw_vs_residual_prior.md").write_text("\n".join(md) + "\n")
    print(f"\nwrote {OUT / 'raw_vs_residual_prior.md'}")

    # ---- figure: spatial spectrum raw vs resid ----
    fig, axes = plt.subplots(1, len(args.datasets), figsize=(5.2 * len(args.datasets), 4.2))
    if len(args.datasets) == 1:
        axes = [axes]
    for ax, ds in zip(axes, args.datasets):
        lam_S, V_S = graph_eig(ds)
        N = len(lam_S); o = np.argsort(lam_S); rank = np.arange(1, N + 1) / N
        for f, col in (("raw", "#0072B2"), ("resid", "#D55E00")):
            X = load_field(ds, "train", BB, f, *_scaler(ds, BB, f))
            C = np.einsum("ki,bih->bkh", V_S.T, X, optimize=True)
            E = (C ** 2).mean(axis=(0, 2))
            ax.plot(rank, np.cumsum(E[o]) / E.sum(), lw=2, color=col, label=f)
        # x-axis is fraction of MODES (rank), so white noise (uniform energy per mode) IS the
        # diagonal. plotting vs the λ VALUE instead would distort this: normalized-Laplacian
        # eigenvalues bunch near λ≈1, so 90% of the modes sit in the right half of the λ axis.
        ax.plot([0, 1], [0, 1], "k:", lw=1, alpha=0.6, label="white (uniform per mode)")
        ax.set_title(f"{ds}: cumulative graph-spectral energy")
        ax.set_xlabel("fraction of graph modes (smooth → rough)")
        ax.set_ylabel("cumulative share"); ax.legend(fontsize=8, frameon=False); ax.grid(alpha=0.25)
    fig.suptitle("Raw signal piles its energy in the few smoothest modes (curve hugs top-left); "
                 "residual sits near the white diagonal", fontsize=11)
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIG / "raw_vs_residual_spectrum.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {FIG / 'raw_vs_residual_spectrum.png'}")


if __name__ == "__main__":
    main()
