"""aggregate per-cell test metrics of 20260907_matched into mean/sd over seeds, for any arm and
optional eval tag. used for the product-graph arms (cart, strong) and the horizon pass (--tag horizon,
which adds joint_tot_*_h). the metrics files are synced from the cluster with sync_from_daic.sh.

    .venv/bin/python scripts/analysis/aggregate_arms.py cart strong            -> results/analysis/product_summary.json
    .venv/bin/python scripts/analysis/aggregate_arms.py iso kron --tag horizon -> results/analysis/horizon_totals.json
"""
import sys, json, glob, numpy as np
args = [a for a in sys.argv[1:] if not a.startswith("--")]; tag = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else ""
prior_of = {"iso": "isotropic", "kron": "kronecker", "cart": "cartesian", "strong": "strong"}
DS = ["pems08f", "pems08s", "pems04f", "pems04s", "metrla", "pemsbay"]
out = {}
for arm in args:
    for ds in DS:
        pat = f"results/20260907_matched/{arm}-hist/{ds}/gwn/{prior_of[arm]}/seed*/eval/test_metrics_bestcrps_ancestral*{('_' + tag) if tag else ''}.json"
        fs = [f for f in glob.glob(pat) if tag or ("_" not in f.split("ancestral")[-1].replace(".json", ""))]
        ms = [json.load(open(f)) for f in fs]
        if not ms: print("missing", arm, ds); continue
        cell = {"n": len(ms)}
        for k, v in ms[0].items():
            if any(k not in m for m in ms): continue                     # keys added to the eval mid-campaign
            if isinstance(v, (int, float)) and not isinstance(v, bool): cell[k] = {"mean": float(np.mean([m[k] for m in ms])), "sd": float(np.std([m[k] for m in ms], ddof=1)) if len(ms) > 1 else 0.0}
            elif isinstance(v, list) and v and isinstance(v[0], (int, float)): cell[k] = np.mean([m[k] for m in ms], axis=0).tolist()
        out[f"{ds}/{arm}"] = cell; print(arm, ds, "seeds", len(ms), "crps %.4f" % cell["crps"]["mean"], "tot crps %.1f" % cell["joint_tot_crps"]["mean"], "tot cov %.3f" % cell["joint_tot_coverage"]["mean"])
name = "horizon_totals.json" if tag == "horizon" else "product_summary.json"
json.dump(out, open(f"results/analysis/{name}", "w"), indent=1); print("saved", name)
