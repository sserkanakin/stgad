"""Is the residual graph-smooth? Show it, per backbone, in the prior's own basis.

Three answers per dataset, all computed in the exact space the diffusion runs in
(per-node normalized residuals, repaired road-graph Laplacian, zero-filled mask —
identical to the prior fit):

  A. spatial spectrum: per-graph-mode energy vs Laplacian eigenvalue λ. graph-SMOOTH
     data slopes DOWN with λ (energy on low-frequency modes, what the GAD 1/(λ+γ)
     prior assumes); WHITE data is flat (what iso assumes). where the curve actually
     sits answers the question.
  B. correlation vs hop distance: neighbors more correlated than strangers = the
     graph-coupling that exists REGARDLESS of where the energy sits in the spectrum.
  C. smoothness index s = (Σ e_k λ_k / Σ e_k) / mean(λ): 1 = white, <1 = smoother
     than white, >1 = rougher. one number per backbone.

Plus headroom_vs_gain.png: joint-spectrum spread (AM/GM of v_emp, dB) vs the REALIZED
test CRPS gain of the best structured arm over iso (5-seed means from the DAIC evals).

usage: .venv/bin/python scripts/analysis/graph_smoothness.py
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse.csgraph import shortest_path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.utils import paths                                                       # noqa: E402
from src.diffusion.prior import _load_residuals_train, _spatial_graph_eig, _fit_sigma_T  # noqa: E402
from src.data.adjacency import load_adjacency                                     # noqa: E402

FIG = paths.analysis_dir("structure")
GAMMA_GAD = 0.8
# fixed colors (Okabe-Ito, colorblind-safe): backbone identity everywhere
C_BB = {"mlp": "#E69F00", "gcn": "#009E73", "gwn": "#0072B2"}
CELLS = {"pems08f": ["mlp", "gcn", "gwn"], "pems08s": ["mlp", "gcn", "gwn"],
         "pems04f": ["mlp", "gwn"], "pems04s": ["mlp", "gwn"]}


def spectrum(ds, bb):
    """(lam_S, e_k normalized to mean 1, v_emp joint spectrum) in prior conventions."""
    R, mask = _load_residuals_train(ds, 42, bb, norm="pernode")
    lam_S, V_S = _spatial_graph_eig(ds, R, "road", 15, repair=True)
    Rz = np.where(mask, R, 0.0)
    C = np.einsum("kn,bnh->bkh", V_S.T, Rz)
    e = (C ** 2).mean(axis=(0, 2))
    # joint (graph x temporal) spectrum, the reshaped prior's v_emp
    _, _, Sigma_T = _fit_sigma_T(R, mask)
    mu_T, V_T = np.linalg.eigh(Sigma_T)
    Cj = np.matmul(np.matmul(V_S.T, Rz), V_T)
    v_emp = np.clip((Cj ** 2).mean(0), 1e-12, None)
    return lam_S, e / e.mean(), v_emp


def hop_corr(ds, bb, max_hop=5):
    """mean |correlation| between node pairs bucketed by graph hop distance."""
    R, mask = _load_residuals_train(ds, 42, bb, norm="pernode")
    A = np.asarray(load_adjacency(ds)).astype(float)
    np.fill_diagonal(A, 0.0)
    D = shortest_path((A > 0).astype(float), method="D", unweighted=True)
    Rz = np.where(mask, R, 0.0)
    X = Rz.transpose(1, 0, 2).reshape(R.shape[1], -1)         # (N, B*H)
    Cc = np.corrcoef(X)
    iu = np.triu_indices_from(Cc, k=1)
    d, c = D[iu], np.abs(Cc[iu])
    buckets, labels = [], []
    for h in range(1, max_hop + 1):
        buckets.append(c[d == h].mean()); labels.append(str(h))
    buckets.append(c[(d > max_hop) & np.isfinite(d)].mean()); labels.append(f">{max_hop}")
    return labels, buckets


def main():
    idx_rows = []
    for ds, bbs in CELLS.items():
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
        fig.suptitle(f"{ds} — residual graph structure in the prior's basis "
                     f"(per-node norm, repaired road Laplacian, train seed42)", fontsize=11)
        ax = axes[0]
        for bb in bbs:
            lam, e, v = spectrum(ds, bb)
            # scatter + 15-mode rolling median so the trend is readable
            ax.scatter(lam, e, s=6, alpha=0.25, color=C_BB[bb])
            k = 15
            med = [np.median(e[max(0, i - k):i + k]) for i in range(len(e))]
            ax.plot(lam, med, color=C_BB[bb], lw=2, label=bb)
            s_idx = float((e * lam).sum() / e.sum() / lam.mean())
            am_gm = 10.0 * np.log10(v.mean() / np.exp(np.log(v).mean()))
            idx_rows.append((ds, bb, s_idx, am_gm))
        lam_ref = np.linspace(max(lam.min(), 1e-3), lam.max(), 200)
        gad = 1.0 / (lam_ref + GAMMA_GAD)
        ax.plot(lam_ref, gad / gad.mean(), "k--", lw=1.5, label=f"GAD prior 1/(λ+{GAMMA_GAD})")
        ax.axhline(1.0, color="#888888", lw=1.5, ls=":", label="iso (white)")
        ax.set_yscale("log"); ax.set_xlabel("Laplacian eigenvalue λ (0=smoothest)")
        ax.set_ylabel("mode energy / mean"); ax.legend(fontsize=8, frameon=False)
        ax.set_title("A. spatial spectrum: smooth slopes down, white is flat", fontsize=9)
        ax.grid(alpha=0.2)

        ax = axes[1]
        for bb in bbs:
            labels, hc = hop_corr(ds, bb)
            ax.plot(range(len(hc)), hc, "o-", color=C_BB[bb], lw=2, ms=5, label=bb)
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
        ax.set_xlabel("graph hop distance between sensor pair")
        ax.set_ylabel("mean |correlation|"); ax.legend(fontsize=8, frameon=False)
        ax.set_title("B. graph coupling: neighbors vs strangers", fontsize=9)
        ax.grid(alpha=0.2)

        ax = axes[2]
        rows = [r for r in idx_rows if r[0] == ds]
        ax.bar([r[1] for r in rows], [r[2] for r in rows],
               color=[C_BB[r[1]] for r in rows], width=0.55)
        ax.axhline(1.0, color="#888888", ls=":", lw=1.5)
        ax.text(len(rows) - 0.45, 1.01, "white noise", fontsize=8, color="#555555")
        ax.set_ylabel("smoothness index  s = ⟨λ⟩_energy / ⟨λ⟩")
        ax.set_title("C. <1 = graph-smoother than white", fontsize=9)
        for i, r in enumerate(rows):
            ax.text(i, r[2] + 0.01, f"{r[2]:.2f}", ha="center", fontsize=9)
        ax.grid(alpha=0.2, axis="y")
        fig.tight_layout()
        f = FIG / f"graph_smoothness_{ds}.png"
        fig.savefig(f, dpi=150); plt.close(fig)
        print(f"wrote {f}")

    # headroom vs realized gain: spread of the matched spectrum vs what tests actually paid
    gains = {}
    for f in glob.glob(str(ROOT / "results/20260721_ladder/*/*/*/*/seed*/eval/test_metrics_bestcrps_ancestral50.json")):
        p = f.split("/")
        arm, ds, bb = p[-7], p[-6], p[-5]
        if arm not in ("iso", "emp", "rsh"):
            continue
        gains.setdefault((ds, bb, arm), []).append(json.load(open(f))["crps"])
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for ds, bb, s_idx, am_gm in idx_rows:
        try:
            iso = np.mean(gains[(ds, bb, "iso")])
            best = min(np.mean(gains[(ds, bb, "emp")]), np.mean(gains[(ds, bb, "rsh")]))
        except KeyError:
            continue
        g = (iso - best) / iso * 100.0
        ax.scatter(am_gm, g, s=60, color=C_BB[bb], zorder=3)
        ax.annotate(f"{ds[4:]}·{bb}", (am_gm, g), textcoords="offset points",
                    xytext=(6, 4), fontsize=8)
    ax.axhline(0.0, color="#888888", ls=":", lw=1.5)
    ax.set_xlabel("prior-exploitable spread: AM/GM of v_emp (dB)   [0 = white, nothing to match]")
    ax.set_ylabel("realized TEST gain of best structured arm vs iso (%)")
    ax.set_title("what the spectrum offers vs what the test set paid", fontsize=10)
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=b) for b, c in C_BB.items()]
    ax.legend(handles=handles, fontsize=8, frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    f = FIG / "headroom_vs_gain.png"
    fig.savefig(f, dpi=150); plt.close(fig)
    print(f"wrote {f}")

    print("\n| dataset | bb | smoothness s (<1=smooth) | spread AM/GM dB |")
    print("|---|---|--:|--:|")
    for ds, bb, s, a in idx_rows:
        print(f"| {ds} | {bb} | {s:.3f} | {a:.1f} |")


if __name__ == "__main__":
    main()
