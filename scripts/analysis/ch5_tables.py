"""tables of the experiments chapter and of the appendices, generated from the aggregated
result files so that every number in the text has a file behind it. cpu, mac.

    .venv/bin/python scripts/analysis/ch5_tables.py

inputs:  results/analysis/{ladder_summary,pooled_summary,results_dossier,prior_structures,prior_alone}.json,
         results/analysis/prior/<ds>.json, the run configs of 20260907_matched
outputs: tudelft_thesis_paper/tables/*.tex, tudelft_thesis_paper/appendix/derived_table.tex
tables whose inputs are missing are skipped with a note.
"""
import json, os, pathlib, numpy as np

ROOT = pathlib.Path("."); T = ROOT / "tudelft_thesis_paper/tables"; T.mkdir(exist_ok=True)
names = {"pems08f": "PEMS08 flow", "pems08s": "PEMS08 speed", "pems04f": "PEMS04 flow", "pems04s": "PEMS04 speed", "metrla": "METR-LA", "pemsbay": "PEMS-BAY"}
DS = list(names)
def load(name):
    p = ROOT / "results/analysis" / name
    return json.load(open(p)) if p.exists() else None
lad = load("ladder_summary.json"); pooled = load("pooled_summary.json"); dos = load("results_dossier.json")
struct = load("prior_structures.json"); alone = load("prior_alone.json"); prior = {ds: load(f"prior/{ds}.json") for ds in DS}
g = lad["gwn"]
def m(c, k): return c[k]["mean"]
def sd(c, k): return c[k]["sd"]
def pct(a, b): return 100 * (b / a - 1)
def tab(cols, head, rows, name):
    (T / f"{name}.tex").write_text("\\begin{tabular}{" + cols + "}\n\\toprule\n" + head + "\n\\midrule\n" + "\n".join(r + " \\\\" for r in rows) + "\n\\bottomrule\n\\end{tabular}\n"); print("wrote", name)

# ---- 5.1 datasets ----
rows = [("PEMS08 flow", "170", "flow", "10{,}691 / 3{,}548 / 3{,}548", "60/20/20"), ("PEMS08 speed", "170", "speed", "10{,}691 / 3{,}548 / 3{,}548", "60/20/20"),
        ("PEMS04 flow", "307", "flow", "10{,}173 / 3{,}375 / 3{,}375", "60/20/20"), ("PEMS04 speed", "307", "speed", "10{,}173 / 3{,}375 / 3{,}375", "60/20/20"),
        ("METR-LA", "207", "speed", "23{,}974 / 3{,}425 / 6{,}850", "70/10/20"), ("PEMS-BAY", "325", "speed", "36{,}465 / 5{,}209 / 10{,}419", "70/10/20")]
tab("lrlll", "dataset & $N$ & quantity & windows train / val / test & split \\\\", [" & ".join(r) for r in rows], "exp_datasets")

# ---- 5.3 the prior's fit (exact per-lead-time covariance) ----
if struct and len(struct) == 6:
    rows = []
    for ds in DS:
        s = struct[ds]["priors"]; p = prior[ds]
        rows.append(f"{names[ds]} & {p['A2_corr_by_hops']['test'][0]:.2f} & {p['B7_match']['neighbour_corr_prior']:.2f} & "
                    f"{s['matched']['ratio_train'][-1]:.2f} & {s['matched']['ratio_test'][-1]:.2f} & {s['cartesian']['ratio_train'][-1]:.2f} & {s['cartesian']['ratio_test'][-1]:.2f} & {s['strong']['ratio_train'][-1]:.2f} & {s['strong']['ratio_test'][-1]:.2f} & "
                    f"{100 * p['B3_separability']['kron_rank1_fraction']:.0f} & {s['iso']['nll_test'] - s['matched']['nll_test']:.2f} & {s['iso']['nll_test'] - s['cartesian']['nll_test']:.2f} & {s['iso']['nll_test'] - s['strong']['nll_test']:.2f}")
    tab("lrrrrrrrrrrrr", " & \\multicolumn{2}{c}{neighbour corr.} & \\multicolumn{6}{c}{spread of the network total, prior / measured} & Kron. & \\multicolumn{3}{c}{likelihood gain over white (nats)} \\\\\n"
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-9}\\cmidrule(lr){11-13}\n & & & \\multicolumn{2}{c}{Kronecker} & \\multicolumn{2}{c}{Cartesian} & \\multicolumn{2}{c}{strong} & share & & & \\\\\n"
        "dataset & data & prior & train & test & train & test & train & test & (\\%) & Kronecker & Cartesian & strong \\\\", rows, "exp_priorfit")
else: print("skip exp_priorfit (prior_structures incomplete)")

# ---- 5.4 per node ----
rows = []
for ds in DS:
    pub = dos["published"].get(ds); best = min(pub.items(), key=lambda kv: kv[1]["crps"]) if pub else None
    i, k = g[f"{ds}/iso"], g[f"{ds}/kron"]; a = alone.get(ds) if alone else None
    al = f"{a['iso']['crps']:.4f} & {a['kron']['crps']:.4f}" if a else "--- & ---"
    rows.append(f"{names[ds]} & " + (f"{best[1]['crps']:.4f} ({best[0]})" if best else "---") + f" & {al} & {m(i, 'crps'):.4f} & {m(k, 'crps'):.4f} & {pct(m(i, 'crps'), m(k, 'crps')):+.1f}")
tab("lrrrrrr", " & best published & \\multicolumn{2}{c}{prior alone, no denoiser} & \\multicolumn{3}{c}{diffusion} \\\\\n\\cmidrule(lr){3-4}\\cmidrule(lr){5-7}\n"
    "dataset & CRPS (method) & white & Kronecker & white & Kronecker & $\\Delta$ (\\%) \\\\", rows, "exp_pernode")

# ---- 5.5 sums ----
rows = []
for ds in DS:
    i, k = g[f"{ds}/iso"], g[f"{ds}/kron"]
    rows.append(f"{names[ds]} & {m(i, 'joint_tot_crps'):.0f} & {pct(m(i, 'joint_tot_crps'), m(k, 'joint_tot_crps')):+.1f} & {m(i, 'joint_tot_coverage'):.2f} & {m(k, 'joint_tot_coverage'):.2f} & "
                f"{m(i, 'joint_tot_spread_ratio'):.2f} & {m(k, 'joint_tot_spread_ratio'):.2f} & {pct(m(i, 'joint_cum_crps'), m(k, 'joint_cum_crps')):+.1f} & {m(i, 'joint_cum_coverage'):.2f} & {m(k, 'joint_cum_coverage'):.2f}")
tab("lrrrrrrrrr", " & \\multicolumn{6}{c}{network total at each lead time} & \\multicolumn{3}{c}{cumulative sum, per sensor} \\\\\n\\cmidrule(lr){2-7}\\cmidrule(lr){8-10}\n"
    " & CRPS$_{\\mathrm{sum}}$ & & \\multicolumn{2}{c}{coverage} & \\multicolumn{2}{c}{spread / error} & & \\multicolumn{2}{c}{coverage} \\\\\n"
    "dataset & white & $\\Delta$ (\\%) & white & Kron. & white & Kron. & $\\Delta$ (\\%) & white & Kron. \\\\", rows, "exp_sums")

# ---- 5.5 the prior alone against the trained model, network total ----
if alone and len(alone) == 6:
    rows = []
    for ds in DS:
        a = alone[ds]["kron"]; k = g[f"{ds}/kron"]; i = g[f"{ds}/iso"]
        rows.append(f"{names[ds]} & {pct(m(i, 'joint_tot_crps'), a['joint_tot_crps']):+.1f} & {pct(m(i, 'joint_tot_crps'), m(k, 'joint_tot_crps')):+.1f} & {a['joint_tot_coverage']:.2f} & {m(k, 'joint_tot_coverage'):.2f} & "
                    f"{a['joint_tot_spread_ratio']:.2f} & {m(k, 'joint_tot_spread_ratio'):.2f} & {pct(m(i, 'crps'), a['crps']):+.1f} & {pct(m(i, 'crps'), m(k, 'crps')):+.1f}")
    tab("lrrrrrrrr", " & \\multicolumn{2}{c}{CRPS$_{\\mathrm{sum}}$ $\\Delta$ (\\%)} & \\multicolumn{2}{c}{coverage} & \\multicolumn{2}{c}{spread / error} & \\multicolumn{2}{c}{per-node CRPS $\\Delta$ (\\%)} \\\\\n"
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}\ndataset & prior alone & diffusion & prior alone & diffusion & prior alone & diffusion & prior alone & diffusion \\\\", rows, "exp_alone")
else: print("skip exp_alone (prior_alone incomplete)")

# ---- 5.7 chain diagnostic ----
c = dos["chain"]["pems08f"]
tab("lrrrr", " & \\multicolumn{2}{c}{white noise} & \\multicolumn{2}{c}{Kronecker prior} \\\\\n\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\ndenoiser used in the chain & spread / error & coverage & spread / error & coverage \\\\",
    [f"{n} & {c['iso'][k][0]:.2f} & {c['iso'][k][1]:.2f} & {c['kron'][k][0]:.2f} & {c['kron'][k][1]:.2f}" for n, k in
     (("the learned network", "learned"), ("the prior's filter for every pattern (oracle)", "gauss"), ("the prior's filter for the network-wide pattern only", "hybrid"))], "exp_chain")

# ---- appendix B: derived quantities, with the exact spread ratio ----
rows = []
for ds in DS:
    ck = json.load(open(ROOT / f"results/20260907_matched/kron-hist/{ds}/gwn/kronecker/seed42/train/config.json")); ci = json.load(open(ROOT / f"results/20260907_matched/iso-hist/{ds}/gwn/isotropic/seed42/train/config.json"))
    p = prior[ds]; th = p["C3_theta"]["matched_theta_quantiles"][-1]
    ex = f"{struct[ds]['priors']['matched']['ratio_train'][-1]:.2f} & {struct[ds]['priors']['matched']['ratio_test'][-1]:.2f}" if struct and ds in struct else "--- & ---"
    def sci(v):
        e = int(np.floor(np.log10(v))); return f"${v / 10 ** e:.1f}\\times10^{{{e}}}$"
    rows.append(f"{names[ds]} & {p['N']} & {ck['global_edge_beta']:.3f} & {ck['gamma_resolved']:.3f} & {ck['global_edge_match_ratio']:.3f} & {ex} & {th:.0f} & {sci(ck['c_lo'])} & {ck['anc_steps_derived']} & {sci(ci['c_lo'])} & {ci['anc_steps_derived']}")
(ROOT / "tudelft_thesis_paper/appendix/derived_table.tex").write_text(
    "\\begin{table}[H]\n\\centering\n\\caption{Derived quantities of the prior and the sampler, Graph WaveNet residuals, one row per dataset. $N$ is the number of sensors; $\\beta$ the weight of the all-to-all regularising link relative to the mean road-edge weight; $\\gamma$ the Laplacian shift; "
    "rule: the spread ratio of the network total as the matching rule computes it on the training residuals; exact: the same ratio with the temporal mixing taken into account, on the training and on the test residuals; $\\theta_{\\max}$ the precision of the fastest joint mode, with the slowest fixed at $0.8$; "
    "$c_{\\mathrm{lo}}$ the noise floor and $n$ the number of sampling steps, for the Kronecker prior and for white noise.}\n\\label{tab:derived}\n\\footnotesize\n\\begin{tabular}{lrrrrrrrrrrr}\n\\toprule\n"
    " & & \\multicolumn{8}{c}{Kronecker prior} & \\multicolumn{2}{c}{white noise} \\\\\n\\cmidrule(lr){3-10}\\cmidrule(lr){11-12}\n & & & & \\multicolumn{3}{c}{spread ratio of the total} & & & & & \\\\\n\\cmidrule(lr){5-7}\n"
    "dataset & $N$ & $\\beta$ & $\\gamma$ & rule & exact, train & exact, test & $\\theta_{\\max}$ & $c_{\\mathrm{lo}}$ & $n$ & $c_{\\mathrm{lo}}$ & $n$ \\\\\n\\midrule\n" + "\n".join(r + " \\\\" for r in rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"); print("wrote derived_table")

# ---- appendix C: full grids ----
rows = [f"{names[ds]} & " + " & ".join(f"{dos['backbone_mae'][f'{ds}/{b}']:.2f}" for b in ("mlp", "gcn", "gwn")) for ds in DS]
tab("lrrr", "dataset & linear & GCN & Graph WaveNet \\\\", rows, "app_backbone")
rows = []
for ds in DS:
    i, k = g[f"{ds}/iso"], g[f"{ds}/kron"]
    rows.append(f"{names[ds]} & " + " & ".join(f"{m(i, key):{f}} & {m(k, key):{f}}" for key, f in (("mae", ".2f"), ("rmse", ".2f"), ("coverage", ".3f"), ("winkler", ".1f"), ("energy", ".1f"))))
tab("lrrrrrrrrrr", " & \\multicolumn{2}{c}{MAE} & \\multicolumn{2}{c}{RMSE} & \\multicolumn{2}{c}{coverage} & \\multicolumn{2}{c}{interval score} & \\multicolumn{2}{c}{energy distance} \\\\\n"
    "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}\\cmidrule(lr){10-11}\ndataset & white & Kron. & white & Kron. & white & Kron. & white & Kron. & white & Kron. \\\\", rows, "app_pernode_full")
rows = []
for ds in DS:
    i, k = g[f"{ds}/iso"], g[f"{ds}/kron"]
    rows.append(f"{names[ds]} & {m(i, 'crps'):.4f} $\\pm$ {sd(i, 'crps'):.4f} & {m(k, 'crps'):.4f} $\\pm$ {sd(k, 'crps'):.4f} & {m(i, 'joint_tot_crps'):.1f} $\\pm$ {sd(i, 'joint_tot_crps'):.1f} & {m(k, 'joint_tot_crps'):.1f} $\\pm$ {sd(k, 'joint_tot_crps'):.1f} & "
                f"{m(i, 'joint_tot_coverage'):.3f} $\\pm$ {sd(i, 'joint_tot_coverage'):.3f} & {m(k, 'joint_tot_coverage'):.3f} $\\pm$ {sd(k, 'joint_tot_coverage'):.3f}")
tab("lrrrrrr", " & \\multicolumn{2}{c}{per-node CRPS} & \\multicolumn{2}{c}{CRPS$_{\\mathrm{sum}}$, network total} & \\multicolumn{2}{c}{coverage of the total} \\\\\n\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\n"
    "dataset & white & Kronecker & white & Kronecker & white & Kronecker \\\\", rows, "app_seeds")
rows = []
for ds in DS:
    p = pooled[ds]; i, k = g[f"{ds}/iso"], g[f"{ds}/kron"]
    rows.append(f"{names[ds]} & {pct(m(i, 'crps'), m(k, 'crps')):+.1f} & " + " & ".join(f"{pct(p['iso'][f'joint_{s}_crps'], p['kron'][f'joint_{s}_crps']):+.1f}" for s in ("c64", "c16", "c4", "tot")) + " & " +
                " & ".join(f"{p['iso'][f'joint_{s}_coverage']:.2f} / {p['kron'][f'joint_{s}_coverage']:.2f}" for s in ("c64", "c16", "c4", "tot")))
tab("lrrrrrrrrr", " & \\multicolumn{5}{c}{CRPS change, Kronecker vs white (\\%)} & \\multicolumn{4}{c}{coverage of the 90\\,\\% interval, white / Kronecker} \\\\\n\\cmidrule(lr){2-6}\\cmidrule(lr){7-10}\n"
    "dataset & sensor & 64 regions & 16 regions & 4 regions & network & 64 regions & 16 regions & 4 regions & network \\\\", rows, "app_pooled")
rows = []
for ds in DS:
    cells = []
    for bb in ("mlp", "gcn", "gwn"):
        i, k = lad[bb][f"{ds}/iso"], lad[bb][f"{ds}/kron"]
        cells += [f"{m(i, 'crps'):.4f}", f"{m(k, 'crps'):.4f}", f"{pct(m(i, 'joint_tot_crps'), m(k, 'joint_tot_crps')):+.1f}", f"{m(i, 'joint_tot_coverage'):.2f} / {m(k, 'joint_tot_coverage'):.2f}"]
    rows.append(names[ds] + " & " + " & ".join(cells))
tab("lrrrrrrrrrrrr", " & \\multicolumn{4}{c}{linear (2 seeds)} & \\multicolumn{4}{c}{GCN (2 seeds)} & \\multicolumn{4}{c}{Graph WaveNet (4 seeds)} \\\\\n\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\\cmidrule(lr){10-13}\n"
    " & \\multicolumn{2}{c}{node CRPS} & total & cov. & \\multicolumn{2}{c}{node CRPS} & total & cov. & \\multicolumn{2}{c}{node CRPS} & total & cov. \\\\\n"
    "dataset & white & Kron. & $\\Delta$ (\\%) & w. / K. & white & Kron. & $\\Delta$ (\\%) & w. / K. & white & Kron. & $\\Delta$ (\\%) & w. / K. \\\\", rows, "app_ladder")
rows = []
for ds in DS:
    d = dos["dc_rule"][ds]; a, b = d["learned"], d["dc"]
    rows.append(f"{names[ds]} & {pct(a['joint_tot_crps'], b['joint_tot_crps']):+.1f} & {a['joint_tot_coverage']:.2f} & {b['joint_tot_coverage']:.2f} & {a['joint_tot_spread_ratio']:.2f} & {b['joint_tot_spread_ratio']:.2f} & {pct(a['crps'], b['crps']):+.1f}")
tab("lrrrrrr", " & CRPS$_{\\mathrm{sum}}$ & \\multicolumn{2}{c}{coverage of the total} & \\multicolumn{2}{c}{spread / error of the total} & per-node CRPS \\\\\n\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\n"
    "dataset & $\\Delta$ (\\%) & learned & with the rule & learned & with the rule & $\\Delta$ (\\%) \\\\", rows, "app_dcrule")
rows = []
for ds in DS:
    i, k = dos["cells"][f"{ds}/gwn/iso"], dos["cells"][f"{ds}/gwn/kron"]
    rows.append(f"{names[ds]} & {m(i, 'n_steps'):.0f} & {m(k, 'n_steps'):.0f} & {m(i, 'n_test_windows'):.0f} & {m(i, 'gen_seconds') / 60:.0f} & {m(k, 'gen_seconds') / 60:.0f}")
tab("lrrrrr", " & \\multicolumn{2}{c}{sampling steps} & test & \\multicolumn{2}{c}{sampling time (min)} \\\\\n\\cmidrule(lr){2-3}\\cmidrule(lr){5-6}\ndataset & white & Kron. & windows & white & Kron. \\\\", rows, "app_sampling")
