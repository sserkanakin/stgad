"""the prior used on its own: draw K residual fields from N(0, Σ) (kronecker, matched) or from
white noise, add them to the frozen backbone's point forecast, and score exactly like a trained
model (per-node normalised quantile CRPS, MAE of the ensemble mean, 90 % coverage; network-total
fair CRPS, coverage, spread/error). no denoiser, no conditioning. cpu, mac.
usage: python scripts/analysis/prior_alone_baseline.py [pems08f ...]
"""
import sys, json, numpy as np
sys.path.insert(0, ".")
from src.diffusion.prior import load_prior
from src.utils import paths
DS = sys.argv[1:] or ["pems08f", "pems08s", "pems04f", "pems04s", "metrla", "pemsbay"]
K, BATCH, Q = 100, 64, np.linspace(0.05, 0.95, 19)
rng = np.random.default_rng(0)
def fair_crps(S, y):                      # S (K, ...) samples, y (...) truth; fair estimator
    K_ = S.shape[0]; t1 = np.abs(S - y[None]).mean(0); xs = np.sort(S, axis=0)
    w = (2 * np.arange(1, K_ + 1) - K_ - 1) / (K_ * (K_ - 1)); return t1 - np.tensordot(w, xs, axes=(0, 0))
import os
out = json.load(open("results/analysis/prior_alone.json")) if os.path.exists("results/analysis/prior_alone.json") else {}
for ds in DS:
    if ds in out: print(ds, "already done"); continue
    cfg = json.load(open(f"results/20260907_matched/kron-hist/{ds}/gwn/kronecker/seed42/train/config.json"))
    pk = load_prior(ds, "kronecker", seed=42, backbone="gwn", norm=cfg["norm"], gamma=cfg["gamma_resolved"], global_edge=cfg["global_edge_beta"] or None)
    pi = load_prior(ds, "isotropic", seed=42, backbone="gwn", norm=cfg["norm"])
    tr = np.load(paths.residual_npz(ds, "train", 42, "gwn")); Rtr = tr["residuals"].astype(np.float32); Mtr = tr["mask"].astype(bool)
    sig = np.array([Rtr[:, i, :][Mtr[:, i, :]].std() for i in range(Rtr.shape[1])], dtype=np.float32)
    te = np.load(paths.residual_npz(ds, "test", 42, "gwn")); R = te["residuals"].astype(np.float32); Yh = te["y_pred"].astype(np.float32); M = te["mask"].astype(bool)
    Y = Yh + R; M = M & (np.abs(Y) > 1e-6)                                   # y == 0 masked, as in the eval
    B, N, H = R.shape; out[ds] = {}
    for arm, pr in (("kron", pk), ("iso", pi)):
        sd = np.sqrt(pr.sigma2).astype(np.float32)                               # (N,H) per-mode std
        qnum = 0.0; qden = 0.0; mae = 0.0; cov = 0.0; cnt = 0; tot_crps = 0.0; tot_cov = 0.0; sp2 = 0.0; er2 = 0.0; ntot = 0
        for b0 in range(0, B, BATCH):
            b1 = min(B, b0 + BATCH); nb = b1 - b0
            C = sd[None, None] * rng.standard_normal((K, nb, N, H)).astype(np.float32)
            r = (pr.V_S.astype(np.float32) @ C @ pr.V_T.T.astype(np.float32)) if arm == "kron" else C   # (K,nb,N,H), matmul broadcasts over K,nb
            F = Yh[None, b0:b1] + sig[None, None, :, None] * r                   # (K, nb, N, H) forecast samples
            y = Y[b0:b1]; m = M[b0:b1]
            q = np.quantile(F, Q, axis=0)                                        # (19, nb, N, H)
            ind = (y[None] <= q).astype(np.float32)
            qnum += float((2 * np.abs((q - y[None]) * (ind - Q[:, None, None, None])) * m[None]).sum() / len(Q)); qden += float((np.abs(y) * m).sum())
            mean = F.mean(0); mae += float((np.abs(mean - y) * m).sum()); cnt += int(m.sum())
            lo, hi = np.quantile(F, 0.05, axis=0), np.quantile(F, 0.95, axis=0); cov += float((((y >= lo) & (y <= hi)) * m).sum())
            T = (F * m[None]).sum(2); Tt = (y * m).sum(1)                          # totals (K, nb, H), (nb, H)
            tot_crps += float(fair_crps(T, Tt).sum()); ntot += Tt.size
            tlo, thi = np.quantile(T, 0.05, axis=0), np.quantile(T, 0.95, axis=0); tot_cov += float(((Tt >= tlo) & (Tt <= thi)).sum())
            sp2 += float(T.var(0, ddof=1).sum()); er2 += float(((T.mean(0) - Tt) ** 2).sum())
        out[ds][arm] = {"crps": qnum / qden, "mae": mae / cnt, "coverage": cov / cnt, "joint_tot_crps": tot_crps / ntot, "joint_tot_coverage": tot_cov / ntot, "joint_tot_spread_ratio": float(np.sqrt(sp2 / er2)), "windows": B, "K": K}
        print(ds, arm, {k: round(v, 4) for k, v in out[ds][arm].items()}, flush=True)
    json.dump(out, open("results/analysis/prior_alone.json", "w"), indent=1)
print("saved results/analysis/prior_alone.json")
