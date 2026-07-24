"""denoiser-free premise check: does the prior alone describe the residuals
better than matched isotropic noise?

For each (dataset, backbone): draw k unconditional samples from each prior's
stationary law (no training, no denoiser), treat them as a residual ensemble,
and score against the TEST residuals with CRPS, the energy score, and a
variogram score (order 0.5). All arms share the per-node scales, so scoring in
normalized residual space preserves the between-prior comparison.

usage: .venv/bin/python scripts/analysis/prior_only_ensemble.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.diffusion.prior import load_prior  # noqa: E402
from src.diffusion.forward_fcps import sample_prior  # noqa: E402

K = 100
NWIN = 400          # scored test windows (subsampled, fixed seed)
NPAIR = 4000        # coordinate pairs for the variogram score

CELLS = [("pems08s", "gwn"), ("pems08f", "gwn"), ("pems08s", "mlp"), ("pems08f", "mlp")]
ARMS = {
    "heat-learned": dict(kind="kronecker", temporal="cartesian",
                         spatial_graph="correlation", knn_k=5, gamma=0.05),
    "heat-road":    dict(kind="kronecker", temporal="cartesian",
                         spatial_graph="road", gamma=0.8),
    "iso":          dict(kind="isotropic"),
}


def crps_ens(E, y, m):
    """ensemble CRPS per masked entry. E (K,N,H), y (N,H), m bool (N,H)."""
    a = np.abs(E - y[None]).mean(0)
    Es = np.sort(E, axis=0)
    w = 2 * np.arange(1, K + 1) - K - 1          # pairwise |Xi-Xj| via sorted form
    b = (w[:, None, None] * Es).sum(0) / (K * K)
    return float(np.mean((a - 0.5 * b)[m])) if m.any() else np.nan


def energy(E, y, m):
    """energy score on the masked flattened window."""
    v = E[:, m]; t = y[m]
    if t.size == 0:
        return np.nan
    a = np.linalg.norm(v - t[None], axis=1).mean()
    idx = np.random.default_rng(0).permutation(K)
    b = np.linalg.norm(v - v[idx], axis=1).mean()
    return float(a - 0.5 * b)


def variogram(E, y, m, pairs, p=0.5):
    """variogram score of order p over fixed coordinate pairs (masked windows)."""
    i, j = pairs
    ok = m.ravel()[i] & m.ravel()[j]
    i, j = i[ok], j[ok]
    if i.size == 0:
        return np.nan
    yv = y.ravel()
    d_obs = np.abs(yv[i] - yv[j]) ** p
    Ev = E.reshape(K, -1)
    d_ens = (np.abs(Ev[:, i] - Ev[:, j]) ** p).mean(0)
    return float(((d_obs - d_ens) ** 2).mean())


def main():
    rng = np.random.default_rng(7)
    print(f"prior-only ensembles: k={K}, {NWIN} test windows, VS order 0.5, {NPAIR} pairs")
    for ds, bb in CELLS:
        f = ROOT / f"results/residuals/{ds}/{bb}/seed42/test.npz"
        if not f.exists():
            print(f"{ds}/{bb}: no test residuals locally, skipped"); continue
        d = np.load(f)
        R, M = d["residuals"].astype(np.float64), d["mask"].astype(bool)
        # normalize per node with the TRAIN scales the priors use
        tr = np.load(ROOT / f"results/residuals/{ds}/{bb}/seed42/train.npz")
        Rt, Mt = tr["residuals"].astype(np.float64), tr["mask"].astype(bool)
        scale = np.sqrt(np.array([(Rt[:, i][Mt[:, i]] ** 2).mean() if Mt[:, i].any() else 1.0
                                  for i in range(R.shape[1])]))
        Rn = R / np.clip(scale, 1e-9, None)[None, :, None]
        B, N, H = Rn.shape
        sel = rng.choice(B, size=min(NWIN, B), replace=False)
        pairs = (rng.integers(0, N * H, NPAIR), rng.integers(0, N * H, NPAIR))

        print(f"\n{ds}/{bb} ({len(sel)} windows)")
        for name, cfg in ARMS.items():
            load_prior.cache_clear()
            kind = cfg.pop("kind") if "kind" in cfg else "kronecker"
            pr = load_prior(ds, kind, seed=42, backbone=bb, norm="pernode", **cfg)
            cfg["kind"] = kind
            E = sample_prior((K,), pr, np.random.default_rng(11))     # shared draw seed
            cr, es, vs = [], [], []
            for b in sel:
                y, m = Rn[b], M[b]
                cr.append(crps_ens(E, y, m))
                es.append(energy(E, y, m))
                vs.append(variogram(E, y, m, pairs))
            print(f"  {name:13s} CRPS {np.nanmean(cr):.4f}  ES {np.nanmean(es):.3f}  "
                  f"VS0.5 {np.nanmean(vs):.4f}")

        # dependence-targeted variogram: learned-graph edges (all horizons) + lag-1 pairs.
        # this is where the priors differ; random pairs above are mostly long-range.
        from src.diffusion.prior import _knn_correlation_graph
        A = _knn_correlation_graph(Rt, 5)
        iu, ju = np.nonzero(np.triu(A > 0, 1))
        hcol = np.repeat(np.arange(H), len(iu))
        ei = np.tile(iu, H) * H + hcol; ej = np.tile(ju, H) * H + hcol
        ti = np.arange(N * H).reshape(N, H)[:, :-1].ravel()
        tj = np.arange(N * H).reshape(N, H)[:, 1:].ravel()
        pi, pj = np.concatenate([ei, ti]), np.concatenate([ej, tj])
        for name, cfg in ARMS.items():
            load_prior.cache_clear()
            kind = cfg.pop("kind") if "kind" in cfg else "kronecker"
            pr = load_prior(ds, kind, seed=42, backbone=bb, norm="pernode", **cfg)
            cfg["kind"] = kind
            E = sample_prior((K,), pr, np.random.default_rng(11)).reshape(K, -1)
            d_ens = (np.abs(E[:, pi] - E[:, pj]) ** 0.5).mean(0)
            vse = []
            for b in sel:
                y, m = Rn[b].ravel(), M[b].ravel()
                ok = m[pi] & m[pj]
                vse.append(float((((np.abs(y[pi[ok]] - y[pj[ok]]) ** 0.5) - d_ens[ok]) ** 2).mean()))
            print(f"  {name:13s} VS0.5(edges+lag1) {np.mean(vse):.4f}")


if __name__ == "__main__":
    main()
