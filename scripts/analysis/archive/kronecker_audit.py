"""Kronecker audit: is Σ = Σ_T ⊗ Σ_S the right TARGET for the residual's second moment?

verify_process.py already checks the MECHANICS (the code applies Σ_T⊗Σ_S faithfully and
the forward reaches N(0,Σ)). this script checks the MODELLING question the mechanics
can't touch: given the forward drift pushes every graph×time eigenmode toward zero at
rate θ[k,h] = q_S[k]/(τ²μ_T[h]), is that per-mode rate actually right for the residual?
three offline diagnostics (saved residuals + the production prior fit):

  1. SEPARABILITY  — is Cov(vec(r)) even a Kronecker product at all? Van Loan–Pitsianis:
     rearrange the empirical covariance, take its top singular value; sep = σ1²/‖Σ‖_F² is
     the fraction of covariance energy ANY Σ_T⊗Σ_S can represent. an upper bound on the
     whole approach, independent of how we fit the two factors.

  2. ATTRIBUTION  — held-out gaussian fit of a nested family that shares ONE total
     variance and differs only in SHAPE:
         iso     Σ ∝ I ⊗ I           (no structure)
         T-only  Σ ∝ Σ_T ⊗ I         (temporal structure only)
         S-only  Σ ∝ I ⊗ Σ_S         (graph structure only)
         full    Σ ∝ Σ_T ⊗ Σ_S
     each candidate's overall scale α is the train MLE; we report per-mode held-out NLL
     (a proper score → lower = the drift pushes the right modes). iso→T-only is what TIME
     buys, iso→S-only what the GRAPH buys, T-only→full what the graph adds ON TOP of time.

  3. WHITENING MAP — for the full prior, the held-out per-mode variance ratio w/v over the
     (N,H) modes. =1 everywhere ⇒ θ is exactly right; ≠1 marks a mode the drift over- or
     under-crushes. shown as the spatial marginal (vs graph frequency λ_S) + the temporal
     marginal + the full (N,H) ratio.

key asymmetry this exposes: Σ_T is FIT to the residual (D_H,R_H) so T-only ~matches by
construction; Σ_S is IMPOSED from the graph (L+γI)^{-1}, never fit — so S-only is the
honest test of whether the graph structure is meaningful for THIS residual.

usage: .venv/bin/python scripts/analysis/kronecker_audit.py \
         [--datasets metrla ...] [--backbones gwn ...] [--split test] [--figure-datasets metrla pems08f]
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
from src.diffusion.prior import load_prior, GAMMA_DEFAULT                          # noqa: E402
from src.utils import paths                                                        # noqa: E402

VARIANTS = ["metrla", "pemsbay", "pems04f", "pems04s", "pems08f", "pems08s"]
BACKBONES = ["mlp", "gcn", "gwn"]
OUT = ROOT / "results/analysis/paper_results"
FIG = OUT / "figs"
COV_CAP = 8000          # samples used for the covariance / per-mode variance estimates
COLS = ["#0072B2", "#E69F00", "#009E73", "#D55E00"]


def _load(ds: str, split: str, bb: str, seed: int = 42):
    """residual field (B,N,H) in the prior's unit-variance space, mask-zeroed, subsampled.
    matches prior.py's fit space (divide by the same train scaler)."""
    from src.diffusion.prior import _load_residuals_train
    from src.data.residual_dataset import train_residual_sigma
    if split == "train":
        R, M = _load_residuals_train(ds, seed, bb)
    else:
        d = np.load(paths.residual_npz(ds, split, seed, bb))
        R = d["residuals"].astype(np.float64)
        M = d["mask"].astype(bool)
        sig = np.asarray(train_residual_sigma(ds, seed, bb, norm="global"))
        R = R / (sig if sig.ndim == 2 else sig.reshape(-1, 1))
    R = np.where(M, R, 0.0)                                  # mask: missing -> 0 (small fraction)
    if len(R) > COV_CAP:
        R = R[np.linspace(0, len(R) - 1, COV_CAP).astype(int)]
    return R


def _rotate(R, V_S, V_T):
    """(B,N,H) -> per-mode coeffs in the (V_S ⊗ V_T) basis. V_S/V_T = None means that
    axis is the identity (iso / flat factor) and its rotation is skipped — this makes
    the big-N identity axes free (pemsbay's 325×325 no-op was the whole cost)."""
    C = R
    if V_S is not None:
        C = np.einsum("ki,bih->bkh", V_S.T, C, optimize=True)
    if V_T is not None:
        C = np.einsum("bkh,hj->bkj", C, V_T, optimize=True)
    return C


def _modevar(R, V_S, V_T):
    return (_rotate(R, V_S, V_T) ** 2).mean(0)              # (N,H)


# ── 1. separability (Van Loan–Pitsianis) ───────────────────────────────────────

def separability(R):
    """fraction of Cov(vec r) energy representable as a single Σ_T⊗Σ_S (top-1) and by
    the best 3 kronecker terms (top-3). vec convention h·N+i, so the temporal factor is
    the OUTER kron block index — matches prior.py's np.kron(Σ_T,Σ_S)."""
    B, N, H = R.shape
    vecR = np.ascontiguousarray(R.transpose(0, 2, 1)).reshape(B, H * N)   # index h*N+i
    Sig = np.cov(vecR, rowvar=False)                                      # (HN,HN)
    fro2 = float((Sig ** 2).sum())
    rearr = Sig.reshape(H, N, H, N).transpose(0, 2, 1, 3).reshape(H * H, N * N).astype(np.float32)
    s = np.linalg.svd(rearr, compute_uv=False)
    s2 = (s.astype(np.float64) ** 2)
    return dict(sep1=float(s2[0] / fro2), sep3=float(s2[:3].sum() / fro2),
                N=N, H=H, fro=float(np.sqrt(fro2)))


# ── 2. attribution (held-out per-mode NLL of the nested family) ─────────────────

def _candidates(prior_full):
    """(name, V_S, V_T, shape[N,H]) for iso / T-only / S-only / full, from the production
    fit. shape is the per-mode variance PATTERN (mean 1); the scale α is fit separately so
    the four differ only in how they distribute one total variance."""
    N, H = prior_full.N, prior_full.H
    V_S, V_T = prior_full.V_S, prior_full.V_T                # None below = identity axis (skipped)
    s_S = 1.0 / (prior_full.lam_S + prior_full.gamma)        # (L+γI)^{-1} eigenvalues
    s_S = s_S / s_S.mean()
    # empirical temporal variance pattern = the full prior's sigma2 marginalised over space,
    # taken in V_T (spatial-averaged), normalised to mean 1.
    t_T = prior_full.sigma2.mean(0)                           # (H,) avg over spatial modes
    t_T = t_T / t_T.mean()
    one_N, one_H = np.ones(N), np.ones(H)
    return [
        ("iso",    None, None, np.outer(one_N, one_H)),
        ("T-only", None, V_T,  np.outer(one_N, t_T)),
        ("S-only", V_S,  None, np.outer(s_S, one_H)),
        ("full",   V_S,  V_T,  np.outer(s_S, t_T)),
    ]


def _nll(w_tr, w_te, shape):
    """train-MLE scale α = mean(w_tr/shape), then per-mode held-out NLL 0.5·mean(w_te/v+ln v)
    (dropping the common 0.5 ln2π). also the whitening octaves std(log2 w_te/v)."""
    shape = np.clip(shape, 1e-12, None)
    alpha = float((w_tr / shape).mean())
    v = np.clip(alpha * shape, 1e-12, None)
    nll = 0.5 * float((w_te / v + np.log(v)).mean())
    oct = float(np.std(np.log2(np.clip(w_te, 1e-30, None) / v)))
    return nll, oct


def attribution(ds, bb, split):
    prior = load_prior(ds, "kronecker", seed=42, backbone=bb,
                       spatial="structured", temporal="empirical")
    R_tr = _load(ds, "train", bb)
    R_te = _load(ds, split, bb)
    out = {"ds": ds, "bb": bb}
    for name, V_S, V_T, shape in _candidates(prior):
        w_tr = _modevar(R_tr, V_S, V_T)
        w_te = _modevar(R_te, V_S, V_T)
        nll, oct = _nll(w_tr, w_te, shape)
        out[f"nll_{name}"] = nll
        out[f"oct_{name}"] = oct
    # decompositions (positive = improvement over iso, in nats/mode)
    out["dT"] = out["nll_iso"] - out["nll_T-only"]           # time buys
    out["dS"] = out["nll_iso"] - out["nll_S-only"]           # graph buys
    out["dfull"] = out["nll_iso"] - out["nll_full"]          # full buys
    out["graph_on_time"] = out["nll_T-only"] - out["nll_full"]   # graph ON TOP of time
    return out


# ── 3. whitening map ────────────────────────────────────────────────────────────

def whitening_figure(ds, bb, split):
    prior = load_prior(ds, "kronecker", seed=42, backbone=bb,
                       spatial="structured", temporal="empirical")
    V_S, V_T = prior.V_S, prior.V_T
    R_tr, R_te = _load(ds, "train", bb), _load(ds, split, bb)
    w_tr, w_te = _modevar(R_tr, V_S, V_T), _modevar(R_te, V_S, V_T)
    shape = prior.sigma2 / prior.sigma2.mean()               # full prior pattern, mean 1
    alpha = float((w_tr / shape).mean())
    v = np.clip(alpha * shape, 1e-12, None)                  # best-scaled full prior variance
    ratio = np.log2(np.clip(w_te, 1e-30, None) / v)          # octaves off, held-out (N,H)

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6))
    aS, aT, aM = axes

    # spatial marginal: prior variance vs held-out energy vs graph frequency λ_S
    o = np.argsort(prior.lam_S)
    pm = v.sum(1)[o]; em = w_te.sum(1)[o]
    aS.plot(prior.lam_S[o], pm / pm.sum(), "-", color=COLS[0], lw=2, label="prior variance σ²/θ")
    aS.plot(prior.lam_S[o], em / em.sum(), "-", color=COLS[3], lw=2, label="held-out residual energy")
    aS.set_title(f"spatial: does the graph prior sit where the residual is?\n(structured Σ_S, γ={GAMMA_DEFAULT})",
                 fontsize=9)
    aS.set_xlabel("graph frequency λ_S  (smooth → rough)"); aS.set_ylabel("share of energy")
    aS.legend(fontsize=8, frameon=False); aS.grid(alpha=0.25)

    # temporal marginal (modes ordered by prior variance)
    ot = np.argsort(-v.sum(0))
    x = np.arange(len(ot))
    aT.bar(x - 0.2, (v.sum(0)[ot]) / v.sum(), 0.4, color=COLS[0], label="prior variance")
    aT.bar(x + 0.2, (w_te.sum(0)[ot]) / w_te.sum(), 0.4, color=COLS[3], label="held-out energy")
    aT.set_title("temporal: prior vs held-out residual (empirical Σ_T)", fontsize=9)
    aT.set_xlabel("temporal mode (smooth → rough)"); aT.set_ylabel("share of energy")
    aT.legend(fontsize=8, frameon=False); aT.grid(axis="y", alpha=0.25)

    # full (N,H) whitening ratio: log2(held-out var / prior var); 0 = drift rate exactly right
    vmax = float(np.nanpercentile(np.abs(ratio), 98)) or 1.0
    im = aM.imshow(ratio[o], aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   origin="lower", extent=(0, prior.H, 0, prior.N))
    aM.set_title("whitening residual  log₂(held-out var / prior var)\nblue = drift over-crushes, red = under-crushes",
                 fontsize=9)
    aM.set_xlabel("temporal mode"); aM.set_ylabel("spatial mode (smooth → rough)")
    fig.colorbar(im, ax=aM, fraction=0.046, pad=0.04)

    med = float(np.median(np.abs(ratio)))
    fig.suptitle(f"{ds}/{bb} [{split}]: whitening audit of the full Σ_T⊗Σ_S prior "
                 f"(median |octaves off| = {med:.2f})", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    FIG.mkdir(parents=True, exist_ok=True)
    png = FIG / f"kron_audit_whitening_{ds}_{bb}.png"
    fig.savefig(png, dpi=160, bbox_inches="tight")
    fig.savefig(png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return png, med


# ── driver ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--backbones", nargs="*", default=None)
    ap.add_argument("--split", default="test", choices=("val", "test"))
    ap.add_argument("--figure-datasets", nargs="*", default=["metrla", "pems08f"])
    args = ap.parse_args()
    datasets = args.datasets or VARIANTS
    backbones = args.backbones or BACKBONES

    sep_rows, att_rows = [], []
    for ds in datasets:
        for bb in backbones:
            if not paths.residual_npz(ds, "train", 42, bb).exists():
                continue
            R_tr = _load(ds, "train", bb)
            sep = separability(R_tr); sep.update(ds=ds, bb=bb); sep_rows.append(sep)
            att_rows.append(attribution(ds, bb, args.split))

    # separability table
    print("\n=== 1. SEPARABILITY: is Cov(vec r) even a Kronecker product? ===")
    print(f"{'dataset':9s} {'bb':4s} | {'N':>4s} {'H':>3s} | {'sep(top-1)':>10s} {'sep(top-3)':>10s}   verdict")
    print("-" * 66)
    for r in sep_rows:
        v = ("separable" if r["sep1"] > 0.9 else "mostly-sep" if r["sep1"] > 0.75 else "NON-separable")
        print(f"{r['ds']:9s} {r['bb']:4s} | {r['N']:>4d} {r['H']:>3d} | {r['sep1']:>10.3f} {r['sep3']:>10.3f}   {v}")

    # attribution table
    print(f"\n=== 2. ATTRIBUTION: held-out [{args.split}] per-mode NLL (lower=better) + what each factor buys ===")
    hdr = (f"{'dataset':9s} {'bb':4s} | {'iso':>7s} {'T-only':>7s} {'S-only':>7s} {'full':>7s} | "
           f"{'ΔT':>6s} {'ΔS':>6s} {'grph|T':>7s} | {'oct_full':>8s}")
    print(hdr); print("-" * len(hdr))
    for r in att_rows:
        print(f"{r['ds']:9s} {r['bb']:4s} | {r['nll_iso']:7.3f} {r['nll_T-only']:7.3f} "
              f"{r['nll_S-only']:7.3f} {r['nll_full']:7.3f} | {r['dT']:6.3f} {r['dS']:6.3f} "
              f"{r['graph_on_time']:7.3f} | {r['oct_full']:8.2f}")
    print("\nΔT = iso−(T-only) time buys · ΔS = iso−(S-only) graph buys · grph|T = (T-only)−full graph ON TOP of time")
    print("positive = improvement over iso (nats/mode). oct_full = std of log2(held-out var / prior var), 0 = perfect.")

    # markdown dump
    OUT.mkdir(parents=True, exist_ok=True)
    md = ["# Kronecker audit\n", f"held-out split: **{args.split}**\n",
          "## 1. Separability (Van Loan–Pitsianis)\n",
          "sep = fraction of Cov(vec r) energy representable as Σ_T⊗Σ_S.\n",
          "| dataset | bb | N | H | sep top-1 | sep top-3 |", "|---|---|--:|--:|--:|--:|"]
    for r in sep_rows:
        md.append(f"| {r['ds']} | {r['bb']} | {r['N']} | {r['H']} | {r['sep1']:.3f} | {r['sep3']:.3f} |")
    md += ["\n## 2. Attribution (held-out per-mode NLL, lower = better)\n",
           "| dataset | bb | iso | T-only | S-only | full | ΔT (time) | ΔS (graph) | graph\\|time | oct full |",
           "|---|---|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for r in att_rows:
        md.append(f"| {r['ds']} | {r['bb']} | {r['nll_iso']:.3f} | {r['nll_T-only']:.3f} | "
                  f"{r['nll_S-only']:.3f} | {r['nll_full']:.3f} | {r['dT']:.3f} | {r['dS']:.3f} | "
                  f"{r['graph_on_time']:.3f} | {r['oct_full']:.2f} |")
    (OUT / "kronecker_audit.md").write_text("\n".join(md) + "\n")
    print(f"\nwrote {OUT / 'kronecker_audit.md'}")

    # figures
    print("\n=== 3. WHITENING MAP ===")
    for ds in args.figure_datasets:
        if not paths.residual_npz(ds, "train", 42, "gwn").exists():
            continue
        png, med = whitening_figure(ds, "gwn", args.split)
        print(f"  {ds}/gwn: median |octaves off| = {med:.2f}   wrote {png}")


if __name__ == "__main__":
    main()
