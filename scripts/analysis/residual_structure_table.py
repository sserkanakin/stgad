"""one row per (dataset, backbone): everything the prior claims about the residual.

this is the RQ1 table. it measures, on the same residuals the prior is fitted to,
the two temporal properties (scale growth, non-stationary horizon correlation),
the spatial property (graph-smoothness), the separability the Kronecker form
assumes, the tail the Gaussian cannot carry, the stiffness of the resulting
spectrum, and a held-out attribution of how much each factor is worth.

train split fits the prior; test split scores it.

usage: .venv/bin/python scripts/analysis/residual_structure_table.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.utils import paths                                                     # noqa: E402
from src.diffusion.prior import GAMMA_DEFAULT                                    # noqa: E402
from scripts.analysis.kronecker_audit import separability                        # noqa: E402
from scripts.analysis.raw_vs_residual_prior import (                             # noqa: E402
    _scaler, load_field, graph_eig, fit_temporal, spatial_metrics, attribution,
)

DATASETS = ["pems08f", "pems08s", "pems04f", "pems04s", "metrla", "pemsbay"]
BB = ["mlp", "gcn", "gwn"]
OUT = ROOT / "results/analysis/paper_results/residual_structure.md"


def tail_and_scale(X):
    """excess kurtosis and per-horizon scale, on observed entries only."""
    F = X.reshape(-1, X.shape[-1])
    F = F[np.abs(F).sum(1) > 0]
    v = F.ravel()
    v = v[v != 0.0]
    m, s = v.mean(), v.std()
    kurt = float(((v - m) ** 4).mean() / s ** 4 - 3.0)
    sd = F.std(0)
    return kurt, float(sd[-1] / sd[0])


def horizon_corr(R_H):
    """lag-one correlation at the start and at the end of the window: a stationary
    kernel is forced to make these equal."""
    return float(R_H[0, 1]), float(R_H[-2, -1])


def row(ds, bb):
    try:
        sc = _scaler(ds, bb, "resid")
        Xtr = load_field(ds, "train", bb, "resid", *sc)
        Xte = load_field(ds, "test", bb, "resid", *sc)
    except (FileNotFoundError, KeyError):
        return None
    lam_S, V_S = graph_eig(ds)
    d_h, R_H, mu_T, V_T, _ = fit_temporal(Xtr)
    sp = spatial_metrics(Xtr, lam_S, V_S)
    at = attribution(Xtr, Xte, lam_S, V_S, mu_T, V_T)
    kurt, growth = tail_and_scale(Xtr)
    r_early, r_late = horizon_corr(R_H)

    # per-mode precision the forward process actually uses, and its dynamic range
    q_S = lam_S + GAMMA_DEFAULT
    tau2 = len(lam_S) / (1.0 / q_S).sum()
    theta = q_S[:, None] / (tau2 * mu_T[None, :])
    stiff = float(theta.max() / theta.min())

    sep1 = separability(Xtr)["sep1"]
    return dict(ds=ds, bb=bb, growth=growth, r_early=r_early, r_late=r_late,
                kurt=kurt, dirichlet=sp["dirichlet"], dc=sp["dc"],
                smooth10=sp["smooth10"], align=sp["align"], sep=sep1,
                stiff=stiff, dT=at["dT"], dS=at["dS"], gT=at["graph_on_time"])


rows = [r for ds in DATASETS for bb in BB if (r := row(ds, bb))]

L = ["# Residual structure, one row per (dataset, backbone)", "",
     "Fitted on train, attribution scored on held-out test. `growth` = std at horizon 12 /",
     "std at horizon 1. `rho early/late` = lag-one horizon correlation at the two ends of the",
     "window (a stationary kernel must make them equal). `kurt` = excess kurtosis.",
     "`Dirichlet` = graph-spectral roughness, white = 1, lower = smoother. `DC` = share of",
     "spatial energy in the smoothest graph mode. `smooth10` = share in the smoothest tenth",
     "of the spectrum (white = 0.10). `align` = correlation of the spatial energy profile",
     "with the prior's 1/(lambda+gamma). `sep` = Van Loan-Pitsianis separability (rank-1).",
     "`stiff` = theta_max/theta_min of the joint spectrum. `dT`/`dS` = held-out per-mode NLL",
     "the temporal / spatial factor buys over isotropic; `g|T` = the spatial factor on top of",
     "the temporal one (nats per mode).", "",
     "| ds | bb | growth | rho early | rho late | kurt | Dirichlet | DC | smooth10 | align | sep | stiff | dT | dS | g\\|T |",
     "|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
for r in rows:
    L.append(
        f"| {r['ds']} | {r['bb']} | {r['growth']:.2f} | {r['r_early']:.2f} | {r['r_late']:.2f} | "
        f"{r['kurt']:.1f} | {r['dirichlet']:.3f} | {r['dc']:.3f} | {r['smooth10']:.3f} | "
        f"{r['align']:.2f} | {r['sep']:.3f} | {r['stiff']:.0f} | {r['dT']:+.3f} | "
        f"{r['dS']:+.3f} | {r['gT']:+.3f} |")

OUT.write_text("\n".join(L) + "\n")

cols = ["ds", "bb", "growth", "r_early", "r_late", "kurt", "dirichlet", "dc",
        "smooth10", "align", "sep", "stiff", "dT", "dS", "gT"]
csv = [",".join(cols)] + [",".join(str(r[c]) for c in cols) for r in rows]
OUT.with_suffix(".csv").write_text("\n".join(csv) + "\n")
print("\n".join(L))
