"""does the ensemble width follow the window's true error size, and does the ensemble mean carry any sign? 240 pems08f test windows. cpu, mac."""
import sys, numpy as np, torch
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
from eval_diffusion import load_run
from src.data.residual_dataset import make_dataset, train_residual_sigma
from src.diffusion.reverse import reverse_sample_ancestral
ds, exp, seed, K, W = "pems08f", "20260907_matched/kron-hist", 42, 20, 240
device = torch.device("cpu")
denoiser, prior, schedule, cfg, _ = load_run(ds, "kronecker", seed, exp, device, ckpt_name="best_crps.pt")
res_seed = cfg.get("residual_seed", seed); sigma_r = train_residual_sigma(ds, res_seed, "gwn", norm=cfg.get("norm", "global"))
data = make_dataset(ds, "test", res_seed, "gwn", sigma_r, cond_past=bool(cfg.get("cond_past")), zero_masked=bool(cfg.get("zero_masked_input")))
idx = np.linspace(0, data.R0.shape[0] - 1, W).astype(int)
R0 = data.R0[idx].numpy(); M = data.mask[idx].numpy().astype(bool); Y = data.Y_hat[idx].to(device)
torch.manual_seed(1); S = []
with torch.no_grad():
    for b0 in range(0, W, 40):
        Yb = Y[b0:b0 + 40].repeat(K, 1, 1)
        s, _ = reverse_sample_ancestral(prior, schedule, denoiser, Yb, n_steps=int(cfg["anc_steps_derived"]), c_lo=schedule.c_lo, corrected=True)
        S.append(s.numpy().reshape(K, -1, *R0.shape[1:]))
S = np.concatenate(S, axis=1)                     # (K, W, N, H)
mean = S.mean(0); sd = S.std(0)
tw = np.array([np.sqrt((R0[i][M[i]] ** 2).mean()) for i in range(W)])          # truth rms per window
sw = np.array([sd[i][M[i]].mean() for i in range(W)])                           # ensemble sd per window
mw = np.array([np.sqrt((mean[i][M[i]] ** 2).mean()) for i in range(W)])          # ensemble-mean rms per window
pooled_corr = np.corrcoef(mean[M], R0[M])[0, 1]
print(f"{W} test windows, K={K}: truth rms per window ranges {tw.min():.2f}-{tw.max():.2f} (scaled units)")
print(f"ensemble sd per window ranges {sw.min():.2f}-{sw.max():.2f}; correlation(ensemble sd, truth rms) across windows = {np.corrcoef(sw, tw)[0,1]:.2f}")
print(f"ensemble mean: rms per window {mw.mean():.2f} (truth {tw.mean():.2f}); pooled correlation with the truth over all entries = {pooled_corr:.3f}")
z = R0 / np.maximum(sd, 1e-6); print(f"standardised residual (truth / ensemble sd): std over all entries {z[M].std():.2f} (1 = calibrated scale); share of entries inside the 90 % interval {((R0 >= np.quantile(S, 0.05, 0)) & (R0 <= np.quantile(S, 0.95, 0)))[M].mean():.2f}")
