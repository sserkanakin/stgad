"""Prior-vs-residual SHAPE MATCH, per cell — the cheap test of whether a structured
prior can help, and the setup for the reshaped-prior contribution.

For each (dataset, backbone) and each prior (iso / kron-empirical / kron-cartesian):
  * project the REAL training residual into that prior's joint eigenbasis (V_S ⊗ V_T)
  * v_emp[k,h] = the residual's actual per-mode variance
  * v_model[k,h] = the prior's per-mode variance σ²/θ
  * both are variance distributions over the N·H joint modes.

Mismatch numbers (lower = the prior's SHAPE already matches the residual):
  TV       = ½Σ|p_model − p_emp|    fraction of variance the prior mis-allocates (0..1)
  oct      = std of log2(v_emp / v_model)   how many OCTAVES off, per mode (whitening error)

Decision: a structured prior can only help where it matches the residual BETTER than
white noise, i.e. TV(kron) < TV(iso). The reshaped prior sets v = v_emp exactly (TV→0,
GAD's "per-eigenvalue c̄" future-work, Remark 1) — its headroom over iso is TV(iso).

Figure (per dataset): spatial marginal (energy vs graph frequency λ_S) and temporal
marginal, PRIOR vs RESIDUAL, for both kron variants — SEE whether the prior's variance
sits at low frequency and whether the residual actually does too.

usage: .venv/bin/python scripts/analysis/prior_residual_match.py [--datasets metrla ...]
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.diffusion.prior import load_prior, _load_residuals_train                 # noqa: E402
from src.utils import paths                                                       # noqa: E402

VARIANTS = ["metrla", "pemsbay", "pems04f", "pems04s", "pems08f", "pems08s"]
BACKBONES = ["mlp", "gcn", "gwn"]
FIG = paths.analysis_dir("structure")


def per_mode_var(R, prior):
    """empirical per-mode variance of residual R (B,N,H) in the prior eigenbasis.
    projection V_S^T R V_T done as two BLAS matmuls (broadcast over batch)."""
    C = np.matmul(prior.V_S.T, R)                                    # (B,N,H)
    C = np.matmul(C, prior.V_T)                                      # (B,N,H)
    return (C ** 2).mean(0)                                          # (N,H)


def mismatch(R, prior):
    v_emp = per_mode_var(R, prior)
    v_model = prior.sigma2                                           # σ²/θ  (N,H)
    p_e = v_emp / v_emp.sum()
    p_m = v_model / v_model.sum()
    tv = 0.5 * float(np.abs(p_e - p_m).sum())
    oct = float(np.std(np.log2(np.clip(v_emp, 1e-30, None) / np.clip(v_model, 1e-30, None))))
    return tv, oct, v_emp, v_model


def table():
    rows = []
    for ds in VARIANTS:
        for bb in BACKBONES:
            if not paths.residual_npz(ds, "train", 42, bb).exists():
                continue
            R, M = _load_residuals_train(ds, 42, bb)
            R = np.where(M, R, 0.0)                              # mask: missing entries are -y_pred garbage
            R = R[np.linspace(0, len(R) - 1, min(1000, len(R))).astype(int)]
            out = {"ds": ds, "bb": bb}
            for name, kw in (("iso", dict(prior="isotropic")),
                             ("kronE", dict(prior="kronecker", temporal="empirical")),
                             ("kronC", dict(prior="kronecker", temporal="cartesian"))):
                pr = load_prior(ds, seed=42, backbone=bb, **kw)
                tv, oct, _, _ = mismatch(R, pr)
                out[f"tv_{name}"] = tv
                out[f"oct_{name}"] = oct
            rows.append(out)
    return rows


def print_table(rows):
    hdr = (f"{'dataset':9s} {'bb':4s} | {'TV_iso':>7s} {'TV_kronE':>8s} {'TV_kronC':>8s} | "
           f"{'oct_kE':>6s} {'oct_kC':>6s} | {'best prior':>11s}  {'headroom':>8s}")
    print(hdr); print("-" * len(hdr))
    last = None
    md = ["| dataset | backbone | TV iso | TV kronE | TV kronC | best structured | reshape headroom (=TV iso) |",
          "|---|---|--:|--:|--:|:--|--:|"]
    for r in rows:
        if last and r["ds"] != last:
            print()
        last = r["ds"]
        tvs = {"iso": r["tv_iso"], "kronE": r["tv_kronE"], "kronC": r["tv_kronC"]}
        best = min(tvs, key=tvs.get)
        struct_best = "kronE" if r["tv_kronE"] <= r["tv_kronC"] else "kronC"
        beats = "yes" if tvs[struct_best] < r["tv_iso"] else "NO (iso ties/wins)"
        print(f"{r['ds']:9s} {r['bb']:4s} | {r['tv_iso']:7.3f} {r['tv_kronE']:8.3f} {r['tv_kronC']:8.3f} | "
              f"{r['oct_kronE']:6.2f} {r['oct_kronC']:6.2f} | {best:>11s}  {r['tv_iso']:8.3f}")
        md.append(f"| {r['ds']} | {r['bb']} | {r['tv_iso']:.3f} | {r['tv_kronE']:.3f} | "
                  f"{r['tv_kronC']:.3f} | {struct_best} ({'beats iso' if beats=='yes' else 'iso wins'}) | {r['tv_iso']:.3f} |")
    print("\nTV = fraction of prior variance placed on the WRONG modes (0 = perfect shape match).")
    print("A structured prior can help only where TV(kron) < TV(iso). reshape headroom = TV(iso)"
          " (reshaped prior sets v=v_emp, TV→0).")
    out = FIG / "prior_residual_match.md"
    out.write_text("\n".join(md) + "\n")
    print(f"wrote {out}")


def figure(ds, bb="gwn"):
    R, M = _load_residuals_train(ds, 42, bb)
    R = np.where(M, R, 0.0)
    R = R[np.linspace(0, len(R) - 1, min(1500, len(R))).astype(int)]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for col, (name, kw) in enumerate((("kron-empirical", dict(temporal="empirical")),
                                      ("kron-cartesian", dict(temporal="cartesian")))):
        pr = load_prior(ds, prior="kronecker", seed=42, backbone=bb, **kw)
        v_emp = per_mode_var(R, pr)
        v_model = pr.sigma2
        # spatial marginal vs graph frequency λ_S
        o = np.argsort(pr.lam_S)
        ax = axes[0, col]
        em = v_emp.sum(1)[o]; mo = v_model.sum(1)[o]
        ax.plot(pr.lam_S[o], mo / mo.sum(), "-", lw=2, label="prior variance")
        ax.plot(pr.lam_S[o], em / em.sum(), "-", lw=2, label="residual energy")
        ax.set_title(f"{name}: spatial (graph) spectrum"); ax.set_xlabel("graph frequency λ_S")
        ax.set_ylabel("share of energy"); ax.legend(fontsize=8); ax.grid(alpha=0.25)
        # temporal marginal (modes ordered by prior variance = smooth→rough)
        ax = axes[1, col]
        ot = np.argsort(-v_model.sum(0))
        et = v_emp.sum(0)[ot]; mt = v_model.sum(0)[ot]
        x = np.arange(len(ot))
        ax.bar(x - 0.2, mt / mt.sum(), 0.4, label="prior variance")
        ax.bar(x + 0.2, et / et.sum(), 0.4, label="residual energy")
        ax.set_title(f"{name}: temporal spectrum"); ax.set_xlabel("temporal mode (smooth → rough)")
        ax.set_ylabel("share of energy"); ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.25)
    fig.suptitle(f"{ds}/{bb}: does the Kronecker prior's variance sit where the residual energy is?",
                 fontsize=12)
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(FIG / f"prior_match_{ds}_{bb}.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote results/figures/prior_match_{ds}_{bb}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--figure-datasets", nargs="*", default=["metrla", "pems08f"])
    args = ap.parse_args()
    global VARIANTS
    if args.datasets:
        VARIANTS = args.datasets
    rows = table()
    print_table(rows)
    for ds in args.figure_datasets:
        figure(ds)


if __name__ == "__main__":
    main()
