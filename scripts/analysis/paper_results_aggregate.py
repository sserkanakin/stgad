r"""Aggregate every test-grade diffusion eval on disk into one place for the paper.

Walks results/ for the locked-protocol test metrics (ancestral@50, k=100, y==0 masked,
best_crps.pt) and emits, under results/analysis/paper_results/:

  master_cells.csv   one row per (arm, cond, dataset, backbone, prior, seed) cell
  agg.csv            seed-aggregated mean/std per (arm, cond, dataset, backbone)
  tables.md          human-readable markdown tables (T1..T-sampling, EM, DiffSTG)
  *.tex              latex twins of the headline tables (\input-ready)

arm taxonomy (parsed from the eval json 'exp' = "<campaign>/<armdir>"):
  base   iso | emp | rsh | cart | cartL | cartL3 | cartLg2 | cartLp | cartLw
  cond   fo (forecast-only, C=Yhat) | past (C=[R_past|Yhat])
  whit   whitened-loss variant (_w)
  ugnet  ~996k UGnet denoiser instead of the 17k GCNN (no test evals on disk -> skipped)

  emp  = Sigma_S structured graph-Kronecker  ⊗  Sigma_T empirical per-horizon variance
         (this is the "spatial kronecker + variance temporal, kronecker product" arm)
  cart = Sigma = sigma^2 (L_ST + gamma I)^-1 on the space⊕time CARTESIAN product graph
         (the spatio-temporal heat prior; the only construction that stays a heat eqn)

usage: .venv/bin/python scripts/analysis/paper_results_aggregate.py
"""
from __future__ import annotations

import csv
import glob
import json
import math
import os
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "results", "analysis", "paper_results")
os.makedirs(OUT, exist_ok=True)

# metrics carried through (headline first). crps = normalized 19-quantile CRPS (the
# number baselines report); crps_energy = raw-unit energy CRPS; coverage nominal 0.90.
METRICS = ["crps", "mae", "rmse", "coverage", "winkler", "crps_energy", "w1", "energy",
           "s_per_window", "gen_seconds"]
DATASETS = ["pems08f", "pems08s", "pems04f", "pems04s", "metrla", "pemsbay"]
BACKBONES = ["mlp", "gcn", "gwn"]


def parse_arm(exp: str):
    """(campaign, base, cond, whit, ugnet) from exp='20260721_ladder/emp-past'.

    some campaigns (the 2026-07-28 product-graph runs) write a flat exp with no
    "<campaign>/<arm>" split; treat the whole string as the campaign then.
    """
    campaign, _, armdir = exp.partition("/")
    d = armdir or campaign
    ugnet = d.endswith("-ugnet")
    if ugnet:
        d = d[: -len("-ugnet")]
    if d.endswith("-past"):
        cond, d = "past", d[: -len("-past")]
    elif d.endswith("-fo"):
        cond, d = "fo", d[: -len("-fo")]
    else:
        cond = "fo"  # bare ladder arms (emp/iso/rsh/emp_w/rsh_w) are forecast-only
    whit = d.endswith("_w")
    if whit:
        d = d[:-2]
    return campaign, d, cond, whit, ugnet


def load_cells():
    rows = []
    for f in glob.glob(os.path.join(ROOT, "results", "**",
                                    "test_metrics_bestcrps_ancestral50.json"), recursive=True):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        campaign, base, cond, whit, ugnet = parse_arm(d["exp"])
        rec = dict(campaign=campaign, base=base, cond=cond, whit=int(whit), ugnet=int(ugnet),
                   dataset=d["dataset"], backbone=d["backbone"], prior=d["prior"],
                   seed=d["seed"], ckpt_epoch=d.get("ckpt_epoch"),
                   n_test_windows=d.get("n_test_windows"), k=d.get("k_samples"),
                   n_steps=d.get("n_steps"), path=os.path.relpath(f, ROOT))
        for m in METRICS:
            rec[m] = d.get(m)
        rows.append(rec)
    return rows


def arm_label(base, whit, ugnet):
    lbl = base + ("_w" if whit else "") + ("+ugnet" if ugnet else "")
    return lbl


def aggregate(rows):
    """mean/std over seeds per (campaign, base, cond, whit, ugnet, dataset, backbone)."""
    groups = defaultdict(list)
    for r in rows:
        key = (r["campaign"], r["base"], r["cond"], r["whit"], r["ugnet"],
               r["dataset"], r["backbone"])
        groups[key].append(r)
    agg = {}
    for key, cells in groups.items():
        a = dict(campaign=key[0], base=key[1], cond=key[2], whit=key[3], ugnet=key[4],
                 dataset=key[5], backbone=key[6], nseeds=len(cells),
                 seeds=",".join(str(c["seed"]) for c in sorted(cells, key=lambda c: c["seed"])))
        for m in METRICS:
            vals = [c[m] for c in cells if c[m] is not None]
            if vals:
                mean = sum(vals) / len(vals)
                std = math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals)) if len(vals) > 1 else 0.0
                a[m + "_mean"], a[m + "_std"] = mean, std
            else:
                a[m + "_mean"], a[m + "_std"] = None, None
        agg[key] = a
    return agg


def fmt(v, nd=4):
    return "—" if v is None else f"{v:.{nd}f}"


def cell_lookup(agg, campaign, base, cond, whit, ugnet, ds, bb, metric="crps"):
    key = (campaign, base, cond, whit, ugnet, ds, bb)
    a = agg.get(key)
    if not a or a.get(metric + "_mean") is None:
        return None
    return a


# --------- table builders -----------------------------------------------------

LADDER = "20260721_ladder"
HEAT = "20260724_heat"


def ladder_table(agg, cond, metric="crps", nd=4):
    """iso / emp / rsh / emp_w / rsh_w for a given conditioning, per (ds, bb)."""
    arms = [("iso", 0, "iso"), ("emp", 0, "emp"), ("rsh", 0, "rsh"),
            ("emp", 1, "emp_w"), ("rsh", 1, "rsh_w")]
    lines = [f"| dataset | bb | " + " | ".join(a[2] for a in arms) + " |",
             "|---|---|" + "|".join(["--"] * len(arms)) + "|"]
    for ds in DATASETS:
        for bb in BACKBONES:
            vals = []
            for base, whit, _ in arms:
                a = cell_lookup(agg, LADDER, base, cond, whit, 0, ds, bb, metric)
                vals.append((a[metric + "_mean"], a[metric + "_std"], a["nseeds"]) if a else None)
            if all(v is None for v in vals):
                continue
            present = [v[0] for v in vals if v]
            best = min(present) if present and metric != "coverage" else (
                max(present, key=lambda x: -abs(x - 0.90)) if present else None)
            cells = []
            for v in vals:
                if v is None:
                    cells.append("—")
                else:
                    s = f"{v[0]:.{nd}f}±{v[1]:.{nd}f}" if v[2] > 1 else f"{v[0]:.{nd}f}"
                    if v[0] == best and metric != "coverage":
                        s = f"**{s}**"
                    cells.append(s + f"<sub>{v[2]}</sub>")
            lines.append(f"| {ds} | {bb} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def heat_table(agg, cond, metric="crps", nd=4):
    """cart (heat) vs emp vs iso for a given conditioning."""
    lines = [f"| dataset | bb | iso | emp | rsh | cart (heat) |",
             "|---|---|--|--|--|--|"]
    for ds in DATASETS:
        for bb in BACKBONES:
            row = {}
            row["iso"] = cell_lookup(agg, LADDER, "iso", cond, 0, 0, ds, bb, metric)
            row["emp"] = cell_lookup(agg, LADDER, "emp", cond, 0, 0, ds, bb, metric)
            row["rsh"] = cell_lookup(agg, LADDER, "rsh", cond, 0, 0, ds, bb, metric)
            row["cart"] = cell_lookup(agg, HEAT, "cart", cond, 0, 0, ds, bb, metric)
            if all(v is None for v in row.values()):
                continue
            present = [v[metric + "_mean"] for v in row.values() if v]
            best = min(present) if present and metric != "coverage" else None
            cells = []
            for k in ["iso", "emp", "rsh", "cart"]:
                v = row[k]
                if v is None:
                    cells.append("—")
                else:
                    m, s, n = v[metric + "_mean"], v[metric + "_std"], v["nseeds"]
                    txt = f"{m:.{nd}f}±{s:.{nd}f}" if n > 1 else f"{m:.{nd}f}"
                    if best is not None and m == best and metric != "coverage":
                        txt = f"**{txt}**"
                    cells.append(txt + f"<sub>{n}</sub>")
            lines.append(f"| {ds} | {bb} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def sampling_table(agg):
    """per-dataset ours ancestral@50 wall-time (s/window & total), for the gwn arm as
    the representative (time is prior-independent, ~identical across arms)."""
    lines = ["| dataset | N | test windows | ours s/window (k=100) | ours total (s) | notes |",
             "|---|--:|--:|--:|--:|--|"]
    for ds in DATASETS:
        # pick any present arm (iso preferred) on gwn for the representative timing
        a = None
        for base in ["iso", "emp", "rsh"]:
            a = cell_lookup(agg, LADDER, base, "fo", 0, 0, ds, "gwn", "s_per_window")
            if a:
                break
        if not a:
            continue
        spw = a["s_per_window_mean"]
        tot = a["gen_seconds_mean"]
        lines.append(f"| {ds} | — | — | {spw:.4f} | {tot:.0f} | ancestral@50, k=100 |")
    return "\n".join(lines)


def em_table():
    """the 3 cells that have EM-1500 alongside ancestral-50 (pems08f/gwn)."""
    lines = ["| arm | sampler | steps | CRPS | MAE | coverage | s/window | total (s) |",
             "|---|---|--:|--:|--:|--:|--:|--:|"]
    for arm, pdir in [("iso", "isotropic"), ("emp", "kronecker"), ("rsh", "kronecker")]:
        base = os.path.join(ROOT, "results/20260721_ladder", arm,
                            "pems08f/gwn", pdir, "seed42/eval")
        for tag, steps in [("ancestral50", 50), ("em1500", 1500)]:
            f = os.path.join(base, f"test_metrics_bestcrps_{tag}.json")
            if not os.path.exists(f):
                continue
            d = json.load(open(f))
            lines.append(f"| {arm} | {d['sampler']} | {d['n_steps']} | {d['crps']:.5f} | "
                         f"{d['mae']:.4f} | {d['coverage']:.3f} | {d['s_per_window']:.4f} | "
                         f"{d['gen_seconds']:.0f} |")
    return "\n".join(lines)


def diffstg_table():
    """DiffSTG native re-run (their code, seed43, TEST split). NB the csv 'is_test' column is
    DiffSTG's smoke-mode flag, NOT the split selector — train.py runs the final eval on the
    TEST split (n_samples=8, ddim_multi@40). window counts match ours within 0.3%."""
    f = os.path.join(ROOT, "results/model-ugnet-notcomplete/research-tud-diffstg/results/diffstg_ref/DiffSTG.csv")
    if not os.path.exists(f):
        return "_(DiffSTG.csv not found)_"
    lines = ["| dataset | MAE | RMSE | CRPS | MIS | sampling time (s) | steps | k | split |",
             "|---|--:|--:|--:|--:|--:|--:|--:|--|"]
    for r in csv.DictReader(open(f)):
        lines.append(f"| {r['data.name']} | {float(r['mae']):.2f} | {float(r['rmse']):.2f} | "
                     f"{float(r['crps']):.4f} | {float(r['mis']):.1f} | {float(r['time']):.0f} | "
                     f"{r['model.sample_steps']}({r['model.sample_strategy']}) | {r.get('n_samples','8')} | test |")
    return "\n".join(lines)


def _val_best_crps(armdir):
    """mean val best_crps per (base, ds, bb) over seeds, from validation_history.json."""
    rows = defaultdict(list)
    for f in glob.glob(os.path.join(ROOT, "results/20260721_ladder", armdir,
                                    "*/*/*/*/train/validation_history.json")):
        parts = f.split(os.sep)
        i = parts.index("20260721_ladder")
        ds, bb = parts[i + 2], parts[i + 3]
        try:
            bc = json.load(open(f)).get("best_crps")
        except Exception:
            bc = None
        if bc is not None:
            rows[(ds, bb)].append(bc)
    return {k: sum(v) / len(v) for k, v in rows.items()}


def ugnet_val_table():
    """capacity axis at MATCHED conditioning (past) on VAL: 17k GCNN denoiser vs 996k UGnet,
    per arm. val best_crps mean over seeds. (User: UGnet val numbers are enough — no test evals.)"""
    lines = ["| arm | dataset | bb | GCNN 17k (val) | UGnet 996k (val) | Δ (17k→996k) |",
             "|---|---|---|--:|--:|--:|"]
    for base in ["iso", "emp", "rsh"]:
        g17 = _val_best_crps(f"{base}-past")          # 17k GCNN, past conditioning
        g996 = _val_best_crps(f"{base}-past-ugnet")   # 996k UGnet, past conditioning
        for key in sorted(g996):
            ds, bb = key
            a, b = g17.get(key), g996.get(key)
            if b is None:
                continue
            if a is None:
                lines.append(f"| {base} | {ds} | {bb} | — | {b:.4f} | — |")
            else:
                lines.append(f"| {base} | {ds} | {bb} | {a:.4f} | {b:.4f} | {100*(a-b)/a:+.1f}% |")
    return "\n".join(lines)


def _tex_num(a, metric="crps", nd=4):
    if a is None or a.get(metric + "_mean") is None:
        return "--"
    return f"{a[metric + '_mean']:.{nd}f}"


def write_latex_heat(agg):
    r"""heat (Cartesian product-graph) prior vs iso/emp, booktabs, matching the draft's
    tab:exp_main style. UNCALIBRATED test CRPS, seed-mean. \method = emp (structured Kron),
    \heat = cart. Only cells where cart exists; both conditionings."""
    ds_names = {"pems08f": "PEMS08 flow", "pems08s": "PEMS08 speed",
                "pems04f": "PEMS04 flow", "pems04s": "PEMS04 speed",
                "metrla": "METR-LA", "pemsbay": "PEMS-BAY"}
    body = []
    for cond, condlab in [("fo", "forecast-only"), ("past", "past")]:
        first = True
        for ds in DATASETS:
            for bb in ["gcn", "gwn", "mlp"]:
                cart = cell_lookup(agg, HEAT, "cart", cond, 0, 0, ds, bb, "crps")
                if cart is None:
                    continue
                iso = cell_lookup(agg, LADDER, "iso", cond, 0, 0, ds, bb, "crps")
                emp = cell_lookup(agg, LADDER, "emp", cond, 0, 0, ds, bb, "crps")
                vals = {"iso": iso, "emp": emp, "cart": cart}
                present = [v["crps_mean"] for v in vals.values() if v]
                best = min(present) if present else None
                def cell(v):
                    if v is None:
                        return "--"
                    s = f"{v['crps_mean']:.4f}"
                    return f"\\textbf{{{s}}}" if v["crps_mean"] == best else s
                cline = "\\addlinespace\n" if (first and cond == "past") else ""
                body.append(f"{cline}{ds_names[ds]} & {bb.upper()} & {condlab} & "
                            f"{cell(iso)} & {cell(emp)} & {cell(cart)} \\\\")
                first = False
    tex = r"""% auto-generated by scripts/analysis/paper_results_aggregate.py
% Cartesian product-graph HEAT prior vs isotropic and the empirical Kronecker prior.
% UNCALIBRATED test CRPS (ancestral@50, k=100, y==0 masked, best_crps.pt), seed-mean.
% \method{} = empirical Kronecker (structured spatial x empirical temporal); \heat{} = Cartesian heat prior.
\begin{table}[H]
\centering
\caption{The spatio-temporal heat prior on the Cartesian product graph
($\Sigma=\sigma^2(L_{ST}+\gamma I)^{-1}$, the only construction that stays a
space--time heat equation) against the isotropic control and the empirical
Kronecker prior. Uncalibrated test CRPS, seed-mean; bold marks the best prior
in each row. The heat prior ties or edges the isotropic control on the strong
backbone and takes the best-ever PEMS08-speed/GWN cell under past conditioning.}
\label{tab:exp_heat}
\small
\begin{tabular}{lll rrr}
\toprule
Dataset & Backbone & Cond. & isotropic & \method{} & \heat{} \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    open(os.path.join(OUT, "heat_prior.tex"), "w").write(tex)
    print(f"wrote {OUT}/heat_prior.tex ({len(body)} rows)")


def main():
    rows = load_cells()
    agg = aggregate(rows)
    write_latex_heat(agg)
    print(f"loaded {len(rows)} test cells, {len(agg)} (arm,cond,ds,bb) groups")

    # master_cells.csv
    cols = ["campaign", "base", "cond", "whit", "ugnet", "dataset", "backbone", "prior",
            "seed", "ckpt_epoch", "n_test_windows", "k", "n_steps"] + METRICS + ["path"]
    with open(os.path.join(OUT, "master_cells.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["campaign"], r["base"], r["cond"], r["dataset"], r["backbone"], r["seed"])):
            w.writerow({c: r.get(c) for c in cols})

    # agg.csv
    aggcols = ["campaign", "base", "cond", "whit", "ugnet", "dataset", "backbone", "nseeds", "seeds"] + \
              [m + s for m in METRICS for s in ("_mean", "_std")]
    with open(os.path.join(OUT, "agg.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=aggcols)
        w.writeheader()
        for a in sorted(agg.values(), key=lambda a: (a["campaign"], a["base"], a["cond"], a["dataset"], a["backbone"])):
            w.writerow({c: a.get(c) for c in aggcols})

    # tables.md
    md = []
    md.append("# Paper results — test-grade, locked protocol\n")
    md.append("_ancestral@50 · k=100 · y==0 masked · best_crps.pt · seed-mean±std, "
              "<sub>n</sub>=#seeds · CRPS = normalized 19-quantile (baseline-comparable). "
              "Bold = best arm in the row._\n")
    md.append("**arm key:** `emp` = spatial graph-Kronecker Σ_S ⊗ empirical-variance Σ_T "
              "(the 'spatial kronecker + variance temporal' prior) · `iso` = scale-matched "
              "white noise · `rsh` = reshaped-to-residual-spectrum · `_w` = whitened loss · "
              "`cart` = spatio-temporal heat prior on the Cartesian product graph.\n")

    for metric, title in [("crps", "CRPS (19-quantile, ↓)"), ("mae", "MAE (↓)"),
                          ("coverage", "coverage (nominal 0.90)"), ("energy", "energy distance (↓)")]:
        md.append(f"\n## T1 — Forecast-only conditioning · {title}\n")
        md.append(ladder_table(agg, "fo", metric))
    for metric, title in [("crps", "CRPS (19-quantile, ↓)"), ("mae", "MAE (↓)"),
                          ("coverage", "coverage (nominal 0.90)"), ("energy", "energy distance (↓)")]:
        md.append(f"\n## T2 — Past (history) conditioning · {title}\n")
        md.append(ladder_table(agg, "past", metric))

    md.append("\n## T3 — Heat (Cartesian product-graph) prior vs iso/emp/rsh · CRPS\n")
    md.append("### forecast-only\n")
    md.append(heat_table(agg, "fo", "crps"))
    md.append("\n### past conditioning\n")
    md.append(heat_table(agg, "past", "crps"))

    md.append("\n## T-time — Sampling wall-time (ours, ancestral@50, k=100)\n")
    md.append(sampling_table(agg))
    md.append("\n## T-EM — EM vs ancestral sampler (pems08f/gwn, the 3 cells with EM data)\n")
    md.append("_EM barely moves CRPS but fixes coverage — at ~30× wall-time._\n")
    md.append(em_table())
    md.append("\n## T-DiffSTG — DiffSTG native re-run (their code, seed43)\n")
    md.append(diffstg_table())
    md.append("\n## T-UGnet — capacity axis (VAL): 17k GCNN vs 996k UGnet denoiser, past cond\n")
    md.append("_val best_crps (no test evals — val is sufficient here per decision). Δ>0 = UGnet "
              "better. Capacity helps on the weak MLP (+26–36%) but not GWN (+0.2–1.5%) — the same "
              "backbone-dependence as the prior: capacity and structure are substitutable channels._\n")
    md.append(ugnet_val_table())

    open(os.path.join(OUT, "tables.md"), "w").write("\n".join(md) + "\n")
    print(f"wrote {OUT}/master_cells.csv, agg.csv, tables.md")

    # quick console sanity: headline gwn CRPS fo
    print("\n=== sanity: fo gwn CRPS (iso / emp / rsh) ===")
    for ds in DATASETS:
        cells = []
        for base, whit in [("iso", 0), ("emp", 0), ("rsh", 0)]:
            a = cell_lookup(agg, LADDER, base, "fo", whit, 0, ds, "gwn", "crps")
            cells.append(f"{base}={a['crps_mean']:.4f}(n{a['nseeds']})" if a else f"{base}=—")
        print(f"  {ds:8s} " + "  ".join(cells))


if __name__ == "__main__":
    main()
