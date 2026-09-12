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
want = [steps[0]] + [steps[int(np.argmin(np.abs(np.log(cs) - np.log(t))))] for t in (0.5, 0.05, 0.005)] + [steps[-1]]
lam, V = np.linalg.eigh(compute_normalized_laplacian(load_adjacency(ds, global_edge=0.0))); order = np.argsort(V[:, 1])
div = LinearSegmentedColormap.from_list("div", [ACC, "#FFFFFF", WARM]); vmax = 3.0
fig, axs = plt.subplots(1, 6, figsize=(6.4, 2.35), constrained_layout=True)
for ax, i in zip(axs[:5], want):
    c, f = est[i]; ax.imshow(f[order], aspect="auto", cmap=div, vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax.set_title(("start, $\\bar c=%g$" % c) if i == steps[0] else ("end, at the floor" if i == steps[-1] else "$\\bar c=%.3g$, step %d of %d" % (c, i, n_steps)), fontsize=7)
    ax.set_xticks([]); ax.set_yticks([])
axs[5].imshow(r0[order], aspect="auto", cmap=div, vmin=-vmax, vmax=vmax, interpolation="nearest"); axs[5].set_title("true residual", fontsize=7); axs[5].set_xticks([]); axs[5].set_yticks([])
for ax in axs: ax.set_xlabel("lead time $\\to$", fontsize=6.5)
axs[0].set_ylabel("sensors, ordered along the road graph $\\downarrow$", fontsize=6.5)
fig.suptitle("the denoiser's estimate of the clean field along the reverse chain (PEMS08 flow, one test window)", fontsize=8)
fig.savefig("tudelft_thesis_paper/figures/results/exp_reverse.pdf"); print("written; levels", [round(est[i][0], 4) for i in want], "of", n_steps, "steps")
