"""fit of four noise priors to the gwn residual at every aggregation scale, with the EXACT
per-lead-time covariance (the temporal factor mixed into lead times through V_T), plus the
product-graph (cartesian, kronecker-SUM precision) prior matched by the same rule.
priors: iso (white), road08 (road graph, gamma 0.8, no regularising link), matched (the prior
used in the thesis, beta/gamma from the run configs), cartesian (L_S(+beta) ⊕ L_T(path) + gamma I,
beta/gamma matched on the training total with the exact formula).
writes results/analysis/prior_structures.json. cpu, mac.
"""
import sys, json, math, numpy as np
sys.path.insert(0, ".")
import scipy.sparse.csgraph as cg
from src.diffusion import prior as P
from src.data.adjacency import load_adjacency, compute_normalized_laplacian, spectral_clusters
from src.data.residual_dataset import train_residual_sigma
from src.utils import paths
DS = [a for a in sys.argv[1:] if a != "--strong"] or ["pems08f", "pems08s", "pems04f", "pems04s", "metrla", "pemsbay"]
def unit(ds, split, sig):
    d = np.load(paths.residual_npz(ds, split, 42, "gwn")); R = d["residuals"].astype(np.float64) / sig[None, :, None]; M = d["mask"].astype(bool)
    ok = M.reshape(len(M), -1).mean(1) > 0.5; return np.where(M, R, 0.0)[ok], M[ok]
def cov_h_exact(VS, VT, var, h):                      # Cov(r[:,h]) in sensor coordinates for a prior diagonal in VS⊗VT
    v = (VT[h] ** 2) @ var.T                            # (N,): sum_{h'} VT[h,h']^2 var[k,h']
    return (VS * v) @ VS.T
def nll_test(R, M, VS, VT, var):
    C = VS.T @ R @ VT; return float(0.5 * np.mean(C ** 2 / var + np.log(2 * np.pi * var)))
import os
out = json.load(open("results/analysis/prior_structures.json")) if os.path.exists("results/analysis/prior_structures.json") else {}
ONLY_STRONG = "--strong" in sys.argv          # add the strong product prior to a finished file
for ds in DS:
    if ds in out and not ONLY_STRONG: print(ds, "already done"); continue
    if ds in out and ONLY_STRONG and "strong" in out[ds]["priors"]: print(ds, "strong already done"); continue
    cfg = json.load(open(f"results/20260907_matched/kron-hist/{ds}/gwn/kronecker/seed42/train/config.json"))
    sig = np.asarray(train_residual_sigma(ds, 42, "gwn", norm="pernode")).reshape(-1)
    Rtr, Mtr = unit(ds, "train", sig); Rte, Mte = unit(ds, "test", sig); N, H = Rtr.shape[1], Rtr.shape[2]; var_data = float((Rtr[Mtr] ** 2).mean())
    tot_tr = (Rtr * sig[None, :, None]).sum(1).std(0); tot_te = (Rte * sig[None, :, None]).sum(1).std(0)
    A = load_adjacency(ds, global_edge=0.0); hops = cg.shortest_path((A > 0).astype(float), unweighted=True)
    sets = {"sensor": [np.array([i]) for i in range(N)], "1-hop": [np.where(hops[i] <= 1)[0] for i in range(N)]}
    for k in (64, 16, 4):
        lab = spectral_clusters(ds, k); sets[f"cluster{k}"] = [np.where(lab == c)[0] for c in range(k) if (lab == c).sum() > 0]
    sets["network"] = [np.arange(N)]
    def ratio(R, VS, VT, var, S):
        num, den = [], []
        for S_ in S:
            w = np.zeros(N); w[S_] = sig[S_]; data_sd = (R[:, S_, :] * sig[None, S_, None]).sum(1).std(0)
            prior_sd = np.array([math.sqrt(max(w @ cov_h_exact(VS, VT, var, h) @ w, 1e-12)) for h in range(H)])
            num.append(prior_sd); den.append(data_sd)
        return float(np.mean(np.array(num) / np.maximum(np.array(den), 1e-9)))
    def total_profile(VS, VT, var, tot):
        return [math.sqrt(sig @ cov_h_exact(VS, VT, var, h) @ sig) / tot[h] for h in range(H)]
    # temporal factor (as load_prior)
    delta, R_H, S_T = P._fit_sigma_T(Rtr, Mtr); mu_T, V_T = np.linalg.eigh(S_T); mu_T = np.clip(mu_T, 1e-12, None); mu_T = 0.9 * mu_T + 0.1 * mu_T.mean()
    def kron_var(mu_S, mu_T_):
        th = (1.0 / mu_S)[:, None] / mu_T_[None, :]; th, s2 = P._normalize_scale(th); s2 *= var_data / float((s2 / th).mean()); return s2 / th
    priors = {}
    priors["iso"] = (np.eye(N), np.eye(H), np.full((N, H), var_data))
    mu8, V8, _ = P._spatial_factor(ds, True, 0.0, 0.8); priors["road08"] = (V8, V_T, kron_var(mu8, mu_T))
    beta, gamma = float(cfg["global_edge_beta"]), float(cfg["gamma_resolved"]); muM, VM, _ = P._spatial_factor(ds, True, beta, gamma); priors["matched"] = (VM, V_T, kron_var(muM, mu_T))
    # kronecker prior with (beta, gamma) re-matched with the EXACT total (for reference: what the rule would have picked)
    best = None
    for b in (0.003, 0.01, 0.03, 0.1, 0.3):
        for g in np.geomspace(1e-3, 3.0, 40):
            mS, VS_, _ = P._spatial_factor(ds, True, b, float(g)); var = kron_var(mS, mu_T); r = float(np.mean(total_profile(VS_, V_T, var, tot_tr)))
            if best is None or abs(r - 1) < abs(best[0] - 1): best = (r, b, float(g), VS_, var)
    priors["kron_exact"] = (best[3], V_T, best[4]); kron_exact_pair = (best[1], best[2], best[0])
    # cartesian product-graph GMRF: precision eigenvalues lam_S(k) + lam_T(h) + gamma, eigenvectors V_S ⊗ V_T(path)
    LT = compute_normalized_laplacian(np.eye(H, k=1) + np.eye(H, k=-1)); lamT, VTp = np.linalg.eigh(LT)
    best = None
    for b in (0.0, 0.003, 0.01, 0.03, 0.1, 0.3):
        lamS, VSb = np.linalg.eigh(compute_normalized_laplacian(load_adjacency(ds, global_edge=b)))
        for g in np.geomspace(1e-3, 3.0, 40):
            var = 1.0 / (lamS[:, None] + lamT[None, :] + g); var *= var_data / var.mean()
            r = float(np.mean(total_profile(VSb, VTp, var, tot_tr)))
            if best is None or abs(r - 1) < abs(best[0] - 1): best = (r, b, float(g), VSb, var)
    priors["cartesian"] = (best[3], VTp, best[4]); cart_pair = (best[1], best[2], best[0])
    # strong product (adds the λ_S·λ_T term), built by the live loader so it is exactly the trained arm's prior
    ps = P.load_prior(ds, "strong", 42, backbone="gwn", norm="pernode", global_edge="match"); priors["strong"] = (ps.V_S, ps.V_T, ps.sigma2); strong_pair = (ps.global_edge, ps.gamma, ps.match_ratio)
    if ONLY_STRONG and ds in out:
        res = out[ds]; res["strong_pair"] = list(strong_pair); priors = {"strong": priors["strong"]}
    else:
        res = {"N": N, "matched_pair": [beta, gamma], "kron_exact_pair": list(kron_exact_pair), "cartesian_pair": list(cart_pair), "strong_pair": list(strong_pair), "scales": list(sets), "priors": {}}
    for name, (VS, VT, var) in priors.items():
        res["priors"][name] = {"ratio_train": [ratio(Rtr, VS, VT, var, S) for S in sets.values()], "ratio_test": [ratio(Rte, VS, VT, var, S) for S in sets.values()],
                               "total_profile_train": total_profile(VS, VT, var, tot_tr), "total_profile_test": total_profile(VS, VT, var, tot_te), "nll_test": nll_test(Rte, Mte, VS, VT, var)}
        print(ds, f"{name:10s} nll {res['priors'][name]['nll_test']:.3f} | ratio test " + " ".join(f"{x:.2f}" for x in res["priors"][name]["ratio_test"]) + f" | total profile test h1/h6/h12 {res['priors'][name]['total_profile_test'][0]:.2f}/{res['priors'][name]['total_profile_test'][5]:.2f}/{res['priors'][name]['total_profile_test'][11]:.2f}", flush=True)
    print(ds, "pairs: matched", res["matched_pair"], "kron re-matched exact", res["kron_exact_pair"], "cartesian", res["cartesian_pair"], flush=True)
    out[ds] = res; json.dump(out, open("results/analysis/prior_structures.json", "w"), indent=1)
print("saved")
