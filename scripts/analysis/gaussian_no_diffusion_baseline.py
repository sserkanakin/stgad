"""GWN + a plain Gaussian residual, NO diffusion: the per-point floor every diffusion
arm has to beat.

per sensor the residual is modelled as N(0, sigma_i^2) with sigma_i the per-sensor std of
the TRAIN residuals (the same scaler the diffusion runs under), so the forecast quantiles
are y_pred + sigma_i * z_q. scored with the papers' normalised quantile-CRPS (19 quantiles,
divided by sum|y|, y==0 masked), directly comparable to tab:exp_sota and to the campaign's
`crps` key. the iso / kron diffusion columns are NOT here on purpose: they come from the
fresh campaign metrics (scripts/analysis/thesis_tables.py), never from an older run.

usage: python scripts/analysis/gaussian_no_diffusion_baseline.py   (reads results/residuals/<ds>/gwn/seed42/test.npz)
writes results/analysis/gaussian_baseline.json
"""
import sys, json, warnings; sys.path.insert(0, '.'); warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
from scipy.stats import norm
from src.utils import paths
from src.data.residual_dataset import train_residual_sigma

QS = np.linspace(0.05, 0.95, 19)
# best published normalised CRPS on the same split convention (SpecSTG / DiffSTG tables)
PUB = {"pems04f": 0.0723, "pems04s": 0.0241}

print(f"{'dataset':9s}{'GWN+Gaussian (no diffusion)':>30s}{'best published':>16s}")
out = {}
for ds in ("pems08f", "pems08s", "pems04f", "pems04s", "metrla", "pemsbay"):
    d = np.load(paths.residual_npz(ds, "test", 42, "gwn"))
    r = d["residuals"].astype(np.float64); yp = d["y_pred"].astype(np.float64)
    m = d["mask"].astype(bool); yt = yp + r
    m &= (yt != 0)                                           # PEMS convention: drop y==0
    sig = np.asarray(train_residual_sigma(ds, 42, "gwn", norm="pernode")).reshape(1, -1, 1)
    num = 0.0
    for q in QS:
        qhat = yp + sig * norm.ppf(q)
        pin = np.abs((qhat - yt) * ((yt <= qhat).astype(float) - q))
        num += 2 * (pin * m).sum()
    crps = float(num / (19 * (np.abs(yt) * m).sum()))
    mae = float((np.abs(r) * m).sum() / m.sum())
    out[ds] = {"crps": crps, "mae": mae, "published_best_crps": PUB.get(ds)}
    b = PUB.get(ds)
    print(f"{ds:9s}{crps:30.4f}{(f'{b:.4f}' if b else '   (n/a)'):>16s}")
p = Path("results/analysis/gaussian_baseline.json"); p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps(out, indent=1)); print("->", p)
