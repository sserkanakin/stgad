"""one reverse chain of the trained pems08f kronecker model on a real test window, drawn as a
single row: the denoiser's estimate of the clean field at five heat levels, then the true
residual. sensors are ordered along the road graph (fiedler vector) so neighbours are adjacent
rows. cpu/mps, mac. writes tudelft_thesis_paper/figures/results/exp_reverse.pdf"""
import sys, json, numpy as np, torch
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from eval_diffusion import load_run
from src.data.residual_dataset import make_dataset, train_residual_sigma
from src.data.adjacency import load_adjacency, compute_normalized_laplacian
from src.diffusion.reverse import reverse_sample_ancestral
INK, MUTED, ACC, WARM = "#1B2430", "#8A94A0", "#00A6D6", "#C8321E"
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.edgecolor": INK, "pdf.fonttype": 42})
ds, exp, seed, w = "pems08f", "20260907_matched/kron-hist", 42, 3189
device = torch.device("cpu")
denoiser, prior, schedule, cfg, _ = load_run(ds, "kronecker", seed, exp, device, ckpt_name="best_crps.pt")
res_seed = cfg.get("residual_seed", seed)
sigma_r = train_residual_sigma(ds, res_seed, "gwn", norm=cfg.get("norm", "global"))
data = make_dataset(ds, "test", res_seed, "gwn", sigma_r, cond_past=bool(cfg.get("cond_past")), zero_masked=bool(cfg.get("zero_masked_input")))
r0 = data.R0[w].numpy() * data.mask[w].numpy(); Y = data.Y_hat[w:w + 1].to(device)
n_steps = int(cfg["anc_steps_derived"]); est = {}
def cb(i, c, R0_hat): est[i] = (float(c), R0_hat[0].detach().cpu().numpy())
torch.manual_seed(7)
with torch.no_grad():
    reverse_sample_ancestral(prior, schedule, denoiser, Y, n_steps=n_steps, c_lo=schedule.c_lo, corrected=True, step_callback=cb)
steps = sorted(est); cs = np.array([est[i][0] for i in steps])
want = [steps[0], steps[int(np.argmin(np.abs(np.log(cs) - np.log(0.05))))], steps[-1]]
lam, V = np.linalg.eigh(compute_normalized_laplacian(load_adjacency(ds, global_edge=0.0))); order = np.argsort(V[:, 1])
div = LinearSegmentedColormap.from_list("div", [ACC, "#FFFFFF", WARM]); vmax = 3.0
fig, axs = plt.subplots(1, 4, figsize=(6.4, 2.6), constrained_layout=True, gridspec_kw={"width_ratios": [1, 1, 1, 1]})
titles = ["start of the chain\n(heat 7, the prior's draw)", "middle of the chain\n(heat 0.05)", "end of the chain\n(the sample)", "the true residual"]
fields = [est[want[0]][1], est[want[1]][1], est[want[2]][1], r0]
for ax, f, t in zip(axs, fields, titles):
    im = ax.imshow(f[order], aspect="auto", cmap=div, vmin=-vmax, vmax=vmax, interpolation="nearest", extent=[0.5, 12.5, len(order) + 0.5, 0.5])
    ax.set_title(t, fontsize=7); ax.set_xticks([1, 6, 12]); ax.set_xlabel("lead time (5-min steps)", fontsize=6.5); ax.tick_params(labelsize=6)
axs[0].set_ylabel("sensor (ordered along the road)", fontsize=6.5)
for ax in axs[1:]: ax.set_yticklabels([])
cb = fig.colorbar(im, ax=axs, shrink=0.8, pad=0.02); cb.set_label("estimated error (per-sensor units)", fontsize=6.5); cb.ax.tick_params(labelsize=6)
fig.savefig("tudelft_thesis_paper/figures/results/exp_reverse.pdf", bbox_inches="tight", pad_inches=0.03); print("written; levels", [round(est[i][0], 4) for i in want], "of", n_steps, "steps")
