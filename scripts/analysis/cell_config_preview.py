"""Preview, on the Mac (cpu), exactly what each diffusion cell will derive from the
fresh residual caches before the gpu campaign runs: the iso scale, the kron (beta,
gamma) match, the derived noise floor and ancestral step count for both arms.

usage: python scripts/analysis/cell_config_preview.py [datasets...]   (reads results/residuals/<ds>/gwn/seed42)
"""
import math, sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from src.diffusion.prior import load_prior
from src.diffusion.forward_fcps import FCPSSchedule, derived_floor, derived_steps, undenoised_fraction

datasets = sys.argv[1:] or ["pems08f", "pems08s", "pems04f", "pems04s", "metrla", "pemsbay"]
rows = []
for ds in datasets:
    for arm, kw in (("iso", dict(prior="isotropic")),
                    ("kron", dict(prior="kronecker", global_edge="match"))):
        p = load_prior(ds, seed=42, backbone="gwn", norm="pernode", **kw)
        c_lo = derived_floor(p)
        sch = FCPSSchedule(c_min=0.1, c_0=7.0, alpha=4.0, c_lo_override=c_lo)
        n = derived_steps(p, sch)
        th = np.asarray(p.theta, dtype=float)
        rows.append(dict(dataset=ds, arm=arm,
                         beta=float(getattr(p, "global_edge", 0.0) or 0.0), gamma=float(p.gamma),
                         theta_min=float(th.min()), theta_max=float(th.max()),
                         c_lo=float(c_lo), decades=math.log10(7.0 / c_lo), anc_steps=int(n),
                         f_at_0_02=float(undenoised_fraction(p, 0.02))))
        r = rows[-1]
        print(f"{ds:8s} {arm:4s} beta={r['beta']:<6.3g} gamma={r['gamma']:<6.3g} "
              f"theta [{r['theta_min']:.2f}, {r['theta_max']:.3g}]  c_lo={r['c_lo']:.3g} "
              f"({r['decades']:.2f} dec) -> {r['anc_steps']} steps | old c_lo=0.02 left {100*r['f_at_0_02']:.1f}% un-denoised")
out = Path("results/analysis/cell_config_preview.json"); out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(rows, indent=1)); print("->", out)
