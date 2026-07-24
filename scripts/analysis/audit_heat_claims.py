"""audit every headline heat-campaign claim from the primary eval jsons.
prints per-seed values, protocol fields, and config provenance; flags mismatches."""
import json, glob, os
import numpy as np

ROOT = "/Users/serkan/Projects/research-tud"

def load(pat):
    rows = []
    for p in sorted(glob.glob(os.path.join(ROOT, pat))):
        d = json.load(open(p))
        seed = p.split("/seed")[1].split("/")[0]
        rows.append((seed, d, p))
    return rows

def proto_ok(d):
    issues = []
    if d.get("n_steps") != 50: issues.append(f"n_steps={d.get('n_steps')}")
    if d.get("k_samples") != 100: issues.append(f"k={d.get('k_samples')}")
    if d.get("mask_zero") not in (1, True): issues.append(f"mask_zero={d.get('mask_zero')}")
    return issues

def config_of(evalpath, keys=("gamma","spatial_graph","cond_past","norm","residual_seed","temporal","knn_k","c_0")):
    cfg = os.path.join(os.path.dirname(os.path.dirname(evalpath)), "train", "config.json")
    if not os.path.exists(cfg): return "NO-CONFIG"
    c = json.load(open(cfg))
    return {k: c.get(k) for k in keys if k in c}

def cell(label, heat_pat, base_pat, metric="crps", show_cfg=False, protocol=True):
    H, B = load(heat_pat), load(base_pat)
    hv = [(s, d[metric]) for s, d, _ in H if metric in d]
    bv = [(s, d[metric]) for s, d, _ in B if metric in d]
    if not hv or not bv:
        print(f"  {label}: MISSING (heat n={len(hv)}, base n={len(bv)})"); return
    hm, bm = np.mean([v for _, v in hv]), np.mean([v for _, v in bv])
    flags = []
    for _, d, _ in H + B:
        if protocol: flags += proto_ok(d)
    nw = {d.get("n_test_windows") for _, d, _ in H + B if "n_test_windows" in d}
    if len(nw) > 1: flags.append(f"n_windows differ {nw}")
    fl = ("  !! " + ",".join(sorted(set(flags)))) if flags else ""
    print(f"  {label}: heat {hm:.4f} {['%s:%.4f'%x for x in hv]}  vs base {bm:.4f} (n={len(bv)})"
          f"  margin {100*(bm-hm)/bm:+.2f}%{fl}")
    if show_cfg:
        print(f"      cfg: {config_of(H[0][2])}")

T = "eval/test_metrics_bestcrps_ancestral50.json"
print("== 1. THE SIX WINS (test@50 k=100) ==")
cell("pems08s/gwn cart-past  ", f"results/20260724_heat/cart-past/pems08s/gwn/*/seed*/{T}",
     f"results/20260721_ladder/iso-past/pems08s/gwn/*/seed*/{T}", show_cfg=True)
cell("pems08s/gwn cartL3-past", f"results/20260724_heat/cartL3-past/pems08s/gwn/*/seed*/{T}",
     f"results/20260721_ladder/iso-past/pems08s/gwn/*/seed*/{T}", show_cfg=True)
cell("pems08s/gwn cartLg2-past", f"results/20260724_heat/cartLg2-past/pems08s/gwn/*/seed*/{T}",
     f"results/20260721_ladder/iso-past/pems08s/gwn/*/seed*/{T}", show_cfg=True)
cell("pems04f/gcn cart-past  ", f"results/20260724_heat/cart-past/pems04f/gcn/*/seed*/{T}",
     f"results/20260721_ladder/iso-past/pems04f/gcn/*/seed*/{T}", show_cfg=True)
cell("pems04f/gcn cart-fo    ", f"results/20260724_heat/cart-fo/pems04f/gcn/*/seed*/{T}",
     f"results/20260721_ladder/iso/pems04f/gcn/*/seed*/{T}")
cell("pems04s/gcn cart-past  ", f"results/20260724_heat/cart-past/pems04s/gcn/*/seed*/{T}",
     f"results/20260721_ladder/iso-past/pems04s/gcn/*/seed*/{T}")

print("== 2. pems08s/gwn cart-past secondary metrics ==")
for m in ["winkler","w1","energy","crps_energy","mae","rmse","coverage"]:
    cell(f"  {m:12s}", f"results/20260724_heat/cart-past/pems08s/gwn/*/seed*/{T}",
         f"results/20260721_ladder/iso-past/pems08s/gwn/*/seed*/{T}", metric=m, protocol=False)

print("== 3. HEAT vs EMP (strong backbone, past) ==")
for ds, bb in [("pems08f","gwn"),("pems08s","gwn"),("pems04f","gwn"),("pems04s","gwn")]:
    cell(f"{ds}/{bb} cart vs emp", f"results/20260724_heat/cart-past/{ds}/{bb}/*/seed*/{T}",
         f"results/20260721_ladder/emp-past/{ds}/{bb}/*/seed*/{T}")

print("== 4. LOSSES / boundary ==")
cell("metrla/gwn cart-past   ", f"results/20260724_heat/cart-past/metrla/gwn/*/seed*/{T}",
     f"results/20260721_ladder/iso-past/metrla/gwn/*/seed*/{T}")
cell("pemsbay/gwn cart-past  ", f"results/20260724_heat/cart-past/pemsbay/gwn/*/seed*/{T}",
     f"results/20260721_ladder/iso-past/pemsbay/gwn/*/seed*/{T}")
cell("metrla cartLw-past     ", f"results/20260724_heat/cartLw-past/metrla/gwn/*/seed*/{T}",
     f"results/20260721_ladder/iso-past/metrla/gwn/*/seed*/{T}")
cell("pems04s/gcn cartLp-past", f"results/20260724_heat/cartLp-past/pems04s/gcn/*/seed*/{T}",
     f"results/20260721_ladder/iso-past/pems04s/gcn/*/seed*/{T}")

print("== 5. TIES (strong backbone) ==")
for ds in ["pems08f","pems04f","pems04s"]:
    cell(f"{ds}/gwn cart-past", f"results/20260724_heat/cart-past/{ds}/gwn/*/seed*/{T}",
         f"results/20260721_ladder/iso-past/{ds}/gwn/*/seed*/{T}")

print("== 6. MLP (weak backbone) ==")
for arm, base in [("cart-fo","iso"),("cartL-fo","iso"),("cart-past","iso-past")]:
    for ds in ["pems08f","pems08s","pems04f"]:
        cell(f"{ds}/mlp {arm:9s}", f"results/20260724_heat/{arm}/{ds}/mlp/*/seed*/{T}",
             f"results/20260721_ladder/{base}/{ds}/mlp/*/seed*/{T}")
cell("pems08f/mlp EMP-fo     ", f"results/20260721_ladder/emp/pems08f/mlp/*/seed*/{T}",
     f"results/20260721_ladder/iso/pems08f/mlp/*/seed*/{T}")

print("== 7. EM SWEEP (val, k=32 — paired) ==")
for arm, pat in [("iso  ","results/20260721_ladder/iso-past/pems08s/gwn/*/seed*/eval/val_metrics_bestcrps_em250.json"),
                 ("cart ","results/20260724_heat/cart-past/pems08s/gwn/*/seed*/eval/val_metrics_bestcrps_em250.json"),
                 ("cartL","results/20260724_heat/cartL-past/pems08s/gwn/*/seed*/eval/val_metrics_bestcrps_em250.json")]:
    rows = load(pat)
    if rows:
        ks = {d.get("k_samples") for _, d, _ in rows}
        print(f"  em250 {arm}: {np.mean([d['crps'] for _,d,_ in rows]):.4f} "
              f"{['%s:%.4f'%(s,d['crps']) for s,d,_ in rows]} (k={ks})")
    else:
        print(f"  em250 {arm}: MISSING")

print("== 8. FEW-STEP (val, ancestral 50 vs 5, paired) ==")
for arm, base in [("cart-past","results/20260724_heat/cart-past"),
                  ("cartL-past","results/20260724_heat/cartL-past"),
                  ("iso-past","results/20260721_ladder/iso-past")]:
    r50 = load(f"{base}/pems08s/gwn/*/seed*/eval/val_metrics_bestcrps_ancestral50.json")
    r5  = load(f"{base}/pems08s/gwn/*/seed*/eval/val_metrics_bestcrps_ancestral5.json")
    if r50 and r5:
        m50, m5 = np.mean([d['crps'] for _,d,_ in r50]), np.mean([d['crps'] for _,d,_ in r5])
        print(f"  {arm:10s}: @50 {m50:.4f}  @5 {m5:.4f}  degradation {100*(m5-m50)/m50:+.1f}%")
    else:
        print(f"  {arm:10s}: MISSING (50:{len(r50)} 5:{len(r5)})")
