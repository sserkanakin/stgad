"""Two-moment match of the spatial prior: beta for the NETWORK total, gamma for the road-NEIGHBOUR
correlation -- against the one-moment (total only) match the campaign uses. prior level only.
prints, per dataset, the (beta, gamma) chosen by each rule with the total ratio, the neighbour
correlation, the held-out NLL and the multi-scale ratio at the cluster scale, plus the frontier
'gamma that matches the total, for each beta'. light on memory: test NLL on a 2000-window subsample.

usage: RESULTS_ROOT=results python scripts/analysis/two_moment_match.py pems08f pems08s
"""
import sys, os, math, json
sys.path.insert(0, "."); os.environ.setdefault("RESULTS_ROOT", "results")
import numpy as np
from src.diffusion import prior as P
from src.data.adjacency import load_adjacency
from src.data.residual_dataset import train_residual_sigma
from src.utils import paths

BETAS = (0.0, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0); GAMMAS = np.geomspace(1e-3, 3.0, 24)
out = {}
for ds in (sys.argv[1:] or ["pems08f", "pems08s"]):
    R, M = P._load_residuals_train(ds, 42, "gwn", "pernode"); R = R.astype(np.float32)
    var_data = float((R[M] ** 2).mean()); N, H = R.shape[1], R.shape[2]
    delta, RH, ST = P._fit_sigma_T(R, M); muT, VT = np.linalg.eigh(ST); muT = np.clip(muT, 1e-12, None); muT = 0.9 * muT + 0.1 * muT.mean()
    sig = np.asarray(train_residual_sigma(ds, 42, "gwn", norm="pernode")).reshape(-1)
    ok = M.reshape(len(M), -1).mean(1) > 0.5
    truth = (np.where(M, R, 0) * sig[None, :, None]).sum(1)[ok].std(0)
    A = load_adjacency(ds, repair=True).astype(float); np.fill_diagonal(A, 0); adj = A > 0; off = ~np.eye(N, dtype=bool)
    X = np.where(M, R, 0)[ok].transpose(0, 2, 1).reshape(-1, N); X = X - X.mean(0); s = np.sqrt((X ** 2).mean(0)); C = (X.T @ X) / len(X) / np.outer(s, s)
    c_nb = float(C[adj & off].mean()); del X, C
    dte = np.load(paths.residual_npz(ds, "test", 42, "gwn")); Rte = (dte["residuals"].astype(np.float32) / sig.reshape(1, -1, 1)); Mte = dte["mask"].astype(bool)
    sub = np.linspace(0, len(Rte) - 1, min(2000, len(Rte))).astype(int); Rte, Mte = np.where(Mte[sub], Rte[sub], 0), Mte[sub]
    del R, M
    rows = []
    for beta in BETAS:
        for g in GAMMAS:
            muS, VS, _ = P._spatial_factor(ds, True, beta, float(g)); th = (1 / muS)[:, None] / muT[None, :]; th, s2 = P._normalize_scale(th); s2 *= var_data / float((s2 / th).mean()); var = s2 / th
            r = float((P._implied_total_sd(VS, var, sig) / truth).mean())
            nb = float(((VS * muS) @ VS.T)[adj & off].mean())
            Cm = np.einsum("ki,bih,hj->bkj", VS.T, Rte, VT); d2 = (Cm ** 2).mean(0); nll = float(0.5 * np.mean(np.log(var) + d2 / var))
            rows.append((beta, float(g), r, nb, nll))
    rows = np.array(rows)
    one = rows[np.argmin(abs(rows[:, 2] - 1))]
    two = rows[np.argmin(np.log(rows[:, 2]) ** 2 + np.log(rows[:, 3] / c_nb) ** 2)]
    bn = rows[np.argmin(rows[:, 4])]
    print(f"\n{ds}: data neighbour corr {c_nb:.3f}")
    print(f"  {'rule':24s} {'beta':>6s} {'gamma':>7s} {'total':>6s} {'nb corr':>8s} {'NLL test':>9s}")
    for lab, b in (("one-moment (campaign)", one), ("two-moment", two), ("best NLL", bn)):
        print(f"  {lab:24s} {b[0]:6.3f} {b[1]:7.3f} {b[2]:6.2f} {b[3]:8.3f} {b[4]:9.4f}")
    print("  frontier — for each beta, the gamma that matches the total:")
    for beta in BETAS:
        sb = rows[rows[:, 0] == beta]; b = sb[np.argmin(abs(sb[:, 2] - 1))]
        print(f"    beta={beta:<5} gamma={b[1]:.3f} total={b[2]:.2f} nb={b[3]:.3f} nll={b[4]:.4f}")
    out[ds] = {"neighbour_corr_data": c_nb, "one_moment": one.tolist(), "two_moment": two.tolist(), "best_nll": bn.tolist(), "grid": rows.tolist()}
p = paths.ROOT / "results/analysis/prior/two_moment_match.json"; p.write_text(json.dumps(out)); print("->", p)
