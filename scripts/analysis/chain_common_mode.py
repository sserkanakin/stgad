"""Where does the network-total spread go along the reverse chain? (mac, cpu)

the prior gives the common mode ("all sensors err together") the right variance, yet the
trained kron cell's samples spread only ~0.6x the truth on the network total. this walks
the corrected ancestral chain on a saved checkpoint and records, at every noise level,
the spread across samples of the network total for (a) the chain state and (b) the
denoiser's x0_hat, for three denoisers:
    learned  the trained GCNN
    gauss    the Bayes denoiser of the Gaussian prior itself (oracle for the sampler)
    hybrid   learned, but the common-mode coefficient replaced by the Gaussian one
if 'gauss' ends at ~1.0 the sampler is fine and the loss is the denoiser's; if 'hybrid'
recovers it, the loss is specifically in how the net treats the common mode.

usage: RESULTS_ROOT=results OUTPUT_ROOT=results python scripts/analysis/chain_common_mode.py pems08f 42
"""
import sys, os, json, math, importlib.util
sys.path.insert(0, ".")
os.environ.setdefault("RESULTS_ROOT", "results"); os.environ.setdefault("OUTPUT_ROOT", "results")
import numpy as np, torch
from src.data.residual_dataset import train_residual_sigma, sigma_to_torch, make_dataset
from src.diffusion.reverse import reverse_sample_ancestral, _prior_sample_torch, _rotate_to_eig_torch, _rotate_from_eig_torch

spec = importlib.util.spec_from_file_location("evald", "scripts/eval_diffusion.py")
evald = importlib.util.module_from_spec(spec); spec.loader.exec_module(evald)

ds = sys.argv[1] if len(sys.argv) > 1 else "pems08f"
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
B, K = int(os.environ.get("B", 48)), int(os.environ.get("K", 24))
ARMS = os.environ.get("ARMS", "iso kron").split(); MODES = os.environ.get("MODES", "learned gauss hybrid").split()
device = torch.device("cpu")
torch.manual_seed(0)


def run_chain(prior, schedule, denoiser, Y, n_steps, mode, k_common, gen, probe):
    """the corrected ancestral sampler, step for step (reverse.py), with hooks."""
    N, H = prior.N, prior.H
    V_S = torch.as_tensor(prior.V_S, dtype=torch.float32); V_T = torch.as_tensor(prior.V_T, dtype=torch.float32)
    theta = torch.as_tensor(prior.theta, dtype=torch.float32); s2 = prior.sigma2_scale
    sigma2_stat = s2 / theta
    R_t = _prior_sample_torch((Y.shape[0],), prior, V_S, V_T, sigma2_stat, generator=gen, device=device, dtype=torch.float32)
    c_grid = np.geomspace(schedule.c_0, schedule.c_lo, n_steps + 1); t_grid = schedule.t_of_c_bar(c_grid)
    C = _rotate_to_eig_torch(R_t, V_S, V_T)
    trace = []
    for i in range(n_steps + 1):
        c_cur = float(c_grid[i])
        a_c = torch.exp(-theta * c_cur); s_c = torch.clamp(s2 * (1 - torch.exp(-2 * theta * c_cur)) / theta, min=1e-20)
        sig_p2 = 1.0 / (a_c * a_c / s_c + 1.0 / sigma2_stat)
        R_t = _rotate_from_eig_torch(C, V_S, V_T)
        C0_gauss = sig_p2 * (a_c / s_c) * C                               # E[x0 | x_c] under the Gaussian prior
        if mode == "gauss":
            C0_hat = C0_gauss
        else:
            t_b = torch.full((Y.shape[0],), float(t_grid[i]))
            C0_hat = _rotate_to_eig_torch(denoiser(R_t, t_b, Y), V_S, V_T)
            if mode == "hybrid":
                C0_hat = C0_hat.clone(); C0_hat[:, k_common, :] = C0_gauss[:, k_common, :]
        trace.append((c_cur,) + probe(C, C0_hat))                   # numbers only: the full state at 88 nodes is ~4 GB
        if i == n_steps:
            break
        c_nxt = float(c_grid[i + 1])
        a_d = torch.exp(-theta * (c_cur - c_nxt)); s_d = torch.clamp(s2 * (1 - torch.exp(-2 * theta * (c_cur - c_nxt))) / theta, min=1e-20)
        a_p = torch.exp(-theta * c_nxt); s_p = torch.clamp(s2 * (1 - torch.exp(-2 * theta * c_nxt)) / theta, min=1e-20)
        v = s_d * s_p / (a_d * a_d * s_p + s_d)
        mean = v * (a_d * C / s_d + a_p * C0_hat / s_p)
        var = v + (v * a_p / s_p) ** 2 * sig_p2
        z = torch.randn(C.shape, generator=gen, dtype=torch.float32)
        C = mean + var.sqrt() * z
    return _rotate_from_eig_torch(C0_hat, V_S, V_T), trace


out = {}
for arm, prior_name in (("iso", "isotropic"), ("kron", "kronecker")):
    if arm not in ARMS: continue
    exp = f"20260907_matched/{arm}-hist"
    den, prior, schedule, cfg, ep = evald.load_run(ds, prior_name, seed, exp, device, backbone="gwn", ckpt_name="best_crps.pt")
    den.eval()
    res_seed = cfg["residual_seed"]
    sigma_r = train_residual_sigma(ds, res_seed, "gwn", norm=cfg["norm"]); sig = sigma_to_torch(sigma_r, device)  # (N,1)
    dset = make_dataset(ds, "test", res_seed, "gwn", sigma_r, cond_past=bool(cfg["cond_past"]), zero_masked=bool(cfg["zero_masked_input"]))
    idx = torch.linspace(0, len(dset) - 1, B).long()
    R0, Y, M = dset.R0[idx], dset.Y_hat[idx], dset.mask[idx] & (dset.y_true_raw[idx].abs() > 1e-6)
    n_steps = int(cfg["anc_steps_derived"])
    V_S = torch.as_tensor(prior.V_S, dtype=torch.float32)
    u = torch.ones(prior.N) / math.sqrt(prior.N); k_common = int(torch.argmax((V_S.T @ u).abs()))
    w = (sig.view(1, -1, 1) * M.float()).unsqueeze(0)                  # (1,B,N,H) raw-unit weights, masked
    truth_tot = (R0 * sig.view(1, -1, 1) * M).sum(1)                     # (B,H) raw total residual
    res = {"ckpt_epoch": ep, "n_steps": n_steps, "c_lo": schedule.c_lo, "k_common": k_common,
           "common_overlap": float((V_S.T @ u)[k_common].abs())}
    with torch.no_grad():
        Yk = Y.repeat(K, 1, 1)
        for mode in MODES:
            gen = torch.Generator().manual_seed(123)
            V_T = torch.as_tensor(prior.V_T, dtype=torch.float32)
            def probe(Cs, C0):                                             # spread (std over K, mean over B,H) of the raw network total
                st = (_rotate_from_eig_torch(Cs, V_S, V_T).view(K, B, prior.N, prior.H) * w).sum(2).std(0).mean()
                x0 = (_rotate_from_eig_torch(C0, V_S, V_T).view(K, B, prior.N, prior.H) * w).sum(2).std(0).mean()
                return (float(st), float(x0))
            xhat, trace = run_chain(prior, schedule, den, Yk, n_steps, mode, k_common, gen, probe)
            if mode == "learned":                                          # guard: identical to the production sampler
                gen2 = torch.Generator().manual_seed(123)
                ref, _ = reverse_sample_ancestral(prior, schedule, den, Yk, n_steps=n_steps, generator=gen2)
                assert torch.allclose(xhat, ref, atol=1e-4), "diagnostic chain diverged from reverse_sample_ancestral"
            S = xhat.view(K, B, prior.N, prior.H)
            tot = (S * w).sum(2)                                           # (K,B,H)
            sp2 = tot.var(0, unbiased=True).mean(); er2 = (truth_tot - tot.mean(0)).pow(2).mean()
            cov = ((truth_tot >= tot.quantile(0.05, 0)) & (truth_tot <= tot.quantile(0.95, 0))).float().mean()
            rows = [list(r) for r in trace]
            res[mode] = {"spread_ratio_total": float((sp2 / er2).sqrt()), "coverage90_total": float(cov),
                         "rmse_total_of_mean": float(er2.sqrt()), "trace": rows}
            print(f"{ds} {arm:4s} {mode:7s}: total spread/err = {res[mode]['spread_ratio_total']:.3f}   "
                  f"cov90 = {cov:.3f}   rmse(mean) = {er2.sqrt():.1f}", flush=True)
    res["truth_total_sd"] = float(truth_tot.std(0).mean())
    out[arm] = res
p = f"results/analysis/chain_common_mode_{ds}_s{seed}_{'-'.join(ARMS)}.json"
json.dump(out, open(p, "w"), indent=1); print("->", p)
# compact view of the chain: where the spread is, per arm/mode, at a few noise levels
levels = [7, 1, 0.1, 0.01, 1e-3]
print(f"\nspread of the network total (std over samples, raw units; truth sd = {out['kron']['truth_total_sd']:.0f})")
print("arm  mode     " + "".join(f"{'c='+str(c):>16s}" for c in levels) + f"{'end':>16s}")
for arm in out:
    for mode in MODES:
        tr = out[arm][mode]["trace"]
        cells = []
        for c in levels:
            j = int(np.argmin([abs(math.log(r[0]) - math.log(c)) for r in tr])); cells.append(f"{tr[j][1]:7.0f}/{tr[j][2]:<7.0f}")
        cells.append(f"{'':7s} {tr[-1][2]:<7.0f}")
        print(f"{arm:4s} {mode:8s} " + "".join(f"{x:>16s}" for x in cells))
print("   (each cell: state-total spread / x0_hat-total spread)")
