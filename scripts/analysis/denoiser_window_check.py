"""what the denoiser adds on one test window of pems08f: one sample and the ensemble mean against the truth. cpu, mac."""
import sys, json, numpy as np, torch
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
from eval_diffusion import load_run
from src.data.residual_dataset import make_dataset, train_residual_sigma
from src.diffusion.reverse import reverse_sample_ancestral
ds, exp, seed, w, K = "pems08f", "20260907_matched/kron-hist", 42, 3189, 100
device = torch.device("cpu")
denoiser, prior, schedule, cfg, _ = load_run(ds, "kronecker", seed, exp, device, ckpt_name="best_crps.pt")
res_seed = cfg.get("residual_seed", seed); sigma_r = train_residual_sigma(ds, res_seed, "gwn", norm=cfg.get("norm", "global"))
data = make_dataset(ds, "test", res_seed, "gwn", sigma_r, cond_past=bool(cfg.get("cond_past")), zero_masked=bool(cfg.get("zero_masked_input")))
r0 = data.R0[w].numpy(); m = data.mask[w].numpy().astype(bool); Y = data.Y_hat[w:w + 1].to(device).repeat(K, 1, 1)
torch.manual_seed(7)
with torch.no_grad():
    S, _ = reverse_sample_ancestral(prior, schedule, denoiser, Y, n_steps=int(cfg["anc_steps_derived"]), c_lo=schedule.c_lo, corrected=True)
S = S.numpy()  # (K, N, H) samples in scaled units
one = S[0]; mean = S.mean(0); lo, hi = np.quantile(S, 0.05, 0), np.quantile(S, 0.95, 0)
c = lambda a, b: float(np.corrcoef(a[m], b[m])[0, 1])
print(f"window {w}: truth std {r0[m].std():.2f} | one sample std {one[m].std():.2f} | ensemble mean std {mean[m].std():.2f}")
print(f"correlation with the truth: one sample {c(one, r0):.2f} | ensemble mean {c(mean, r0):.2f} | another sample vs the first {c(S[1], one):.2f}")
print(f"MAE vs truth (scaled units): one sample {np.abs(one - r0)[m].mean():.2f} | ensemble mean {np.abs(mean - r0)[m].mean():.2f} | zero forecast {np.abs(r0)[m].mean():.2f}")
print(f"90 % interval of the {K} samples covers {((r0 >= lo) & (r0 <= hi))[m].mean():.2f} of the true entries; interval half-width mean {((hi - lo) / 2)[m].mean():.2f}")
# the network total for this window at each lead time: truth vs ensemble
sig = np.asarray(sigma_r).reshape(-1)
T = (S * sig[None, :, None] * m[None]).sum(1); Tt = (r0 * sig[:, None] * m).sum(0)
print("network total by lead time: truth", np.round(Tt[[0, 5, 11]], 0), " ensemble mean", np.round(T.mean(0)[[0, 5, 11]], 0), " ensemble sd", np.round(T.std(0)[[0, 5, 11]], 0))
np.savez("results/analysis/denoiser_window3189.npz", S=S, r0=r0, m=m)
