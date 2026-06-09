# interactive explorer for the backbone residuals, the graph, and the
# diagnostics that say what is going wrong.
#
# run it with:
#   .venv/bin/streamlit run scripts/analysis/dashboard.py
#
# one dataset at a time: the network in space, the error growing over the
# forecast horizon, the residual distributions, and the train→val gate.
import os
import sys

# make the sibling core module importable and run from the repo root.
_here = os.path.abspath(__file__)
sys.path.insert(0, os.path.dirname(_here))

import numpy as np
import plotly.graph_objects as go
import streamlit as st

import dashboard_core as core
from dashboard_core import SIGNAL, DATASETS, SEED, H_MIN, ROOT

os.chdir(ROOT)
W = "stretch"  # st width keyword: fill the container

# ----------------------------------------------------------------------------
# sidebar — the shared controls
# ----------------------------------------------------------------------------
st.set_page_config(page_title="residual & graph explorer", layout="wide")
st.sidebar.title("controls")
dataset = st.sidebar.selectbox("dataset", DATASETS,
                               format_func=lambda d: SIGNAL[d]["name"])
split = st.sidebar.radio("split", ["train", "val", "test"], horizontal=True)
info = SIGNAL[dataset]
unit = info["unit"]

stats = core.node_horizon_stats(dataset, split, SEED)
N, H, B = stats["N"], stats["H"], stats["B"]

if "node" not in st.session_state:
    st.session_state.node = 0
# keep the stored node in range when the dataset (and so N) changes
st.session_state.node = int(np.clip(st.session_state.node, 0, N - 1))
node = st.sidebar.number_input("focus node", 0, N - 1, key="node")

horizon = st.sidebar.slider("horizon (steps ahead)", 1, H, 1)
st.sidebar.caption(f"horizon {horizon} = +{horizon * H_MIN} min")

st.sidebar.divider()
st.sidebar.caption(
    f"**{info['name']}** · {info['what']}\n\n"
    f"{B:,} windows · {N} sensors · {H} horizons\n\n"
    f"signal unit: **{unit}**")

# headline numbers
node_rmse_h = stats["rmse"][:, horizon - 1]
node_bias_h = stats["bias"][:, horizon - 1]
overall_rmse = float(np.sqrt(np.nansum(stats["rmse"] ** 2 * stats["count"]) /
                             np.nansum(stats["count"])))
overall_bias = float(np.nansum(stats["bias"] * stats["count"]) /
                     np.nansum(stats["count"]))
cov = float(np.nansum(stats["count"]) / (B * N * H))


# ----------------------------------------------------------------------------
# header
# ----------------------------------------------------------------------------
st.title(f"{info['name']} — residuals, graph, diagnostics")
c1, c2, c3, c4 = st.columns(4)
c1.metric("overall RMSE", f"{overall_rmse:.2f} {unit}")
c2.metric("overall bias", f"{overall_bias:+.3f} {unit}",
          help="mean residual; should sit near 0 if the backbone is unbiased")
c3.metric("valid coverage", f"{cov * 100:.1f}%",
          help="share of (window,node,horizon) cells that are not masked-out missing readings")
c4.metric(f"RMSE @ +{horizon * H_MIN}min", f"{float(np.nanmean(node_rmse_h)):.2f} {unit}")

tab_graph, tab_st, tab_resid, tab_wrong = st.tabs(
    ["🗺️ graph & space", "🎞️ spatio-temporal", "📈 residuals", "🚨 what's wrong"])


# ----------------------------------------------------------------------------
# tab 1 — graph in space
# ----------------------------------------------------------------------------
with tab_graph:
    st.markdown(
        "the network laid out in 2-d. no real sensor coordinates exist for these "
        "datasets, so the layout comes from the graph itself — connected sensors "
        "are drawn close together.")
    lc, rc = st.columns([3, 2])
    with rc:
        layout_kind = st.radio("layout", ["spectral eigenmap", "spring"],
                               help="spectral = laplacian eigenmap (fast, stable). "
                                    "spring = force-directed on the thresholded edges.")
        default_thr = 0.1 if dataset != "pems08f" else 0.5
        thr_max = 0.95 if dataset != "pems08f" else 1.0
        thr = st.slider("edge threshold", 0.0, thr_max, default_thr,
                        help="only edges with weight above this are drawn / used for spring")
        color_kind = st.selectbox(
            "colour nodes by",
            [f"RMSE @ +{horizon * H_MIN}min", f"bias @ +{horizon * H_MIN}min",
             "mean RMSE (all horizons)", "node degree"])

    ei, ej, _ = core.edge_segments(dataset, thr)
    xs, ys = (core.layout_spring(dataset, thr) if layout_kind == "spring"
              else core.layout_spectral(dataset))

    is_bias = color_kind.startswith("bias @")
    if color_kind.startswith("RMSE @"):
        color, clabel = node_rmse_h, f"rmse ({unit})"
    elif is_bias:
        color, clabel = node_bias_h, f"bias ({unit})"
    elif color_kind.startswith("mean RMSE"):
        color, clabel = np.nanmean(stats["rmse"], axis=1), f"rmse ({unit})"
    else:
        color, clabel = core.node_degrees(N, ei, ej), "degree"

    fig = core.graph_figure(xs, ys, ei, ej, color, clabel,
                            f"{color_kind} · {len(ei)} edges shown",
                            picked=node, bias=is_bias)
    with lc:
        ev = st.plotly_chart(fig, width=W, key="graph", on_select="rerun")
        # clicking a node updates the focus node
        try:
            pts = ev["selection"]["points"] if ev else None
            if pts:
                cd = pts[0].get("customdata")
                idx = int(cd[0]) if isinstance(cd, (list, tuple)) else int(pts[0]["point_index"])
                if idx != node:
                    st.session_state.node = idx
                    st.rerun()
        except Exception:
            pass

    with rc:
        order = np.argsort(-np.nan_to_num(node_rmse_h))
        st.markdown(f"**worst sensors @ +{horizon * H_MIN}min**")
        st.dataframe(
            {"node": order[:10].tolist(),
             f"rmse ({unit})": [round(float(node_rmse_h[i]), 2) for i in order[:10]],
             f"bias ({unit})": [round(float(node_bias_h[i]), 2) for i in order[:10]]},
            hide_index=True, width=W)


# ----------------------------------------------------------------------------
# tab 2 — spatio-temporal
# ----------------------------------------------------------------------------
with tab_st:
    st.markdown(
        "**how the forecast error spreads over the network as the horizon grows.** "
        "press play to sweep +5 → +60 min; node colour is RMSE.")
    if st.checkbox("render animation (heavier)", value=True):
        d_thr = 0.1 if dataset != "pems08f" else 0.5
        xs2, ys2 = core.layout_spectral(dataset)
        ei2, ej2, _ = core.edge_segments(dataset, d_thr)
        st.plotly_chart(core.animated_graph(xs2, ys2, ei2, ej2, stats["rmse"], unit),
                        width=W, key="anim")

    st.divider()
    st.markdown("**the full picture: RMSE for every sensor at every horizon.** "
                "rows are sensors (sorted worst-first), columns are +5…+60 min.")
    order = np.argsort(-np.nanmean(stats["rmse"], axis=1))
    xh = [f"+{(h + 1) * H_MIN}" for h in range(H)]
    st.plotly_chart(
        core.heatmap(stats["rmse"][order], xh, [str(int(i)) for i in order],
                     "per-sensor RMSE by horizon", f"rmse ({unit})"),
        width=W, key="nh_heat")

    st.divider()
    st.markdown("**one window, up close.** pick a window and see the residual for "
                "every sensor across the 12 horizons.")
    w = st.slider("window index", 0, B - 1, 0, key="win")
    R, yp, yt, M, _, _ = core.load_residuals(dataset, split, SEED)
    Rw = np.where(M[w], R[w], np.nan)  # (N, H)
    order_w = np.argsort(-np.nan_to_num(np.nanmean(np.abs(Rw), axis=1)))
    st.plotly_chart(
        core.heatmap(Rw[order_w], xh, [str(int(i)) for i in order_w],
                     f"window {w}: residual (true − pred)", f"residual ({unit})",
                     colorscale="RdBu", zmid=0),
        width=W, key="win_heat")


# ----------------------------------------------------------------------------
# tab 3 — residual analysis
# ----------------------------------------------------------------------------
with tab_resid:
    R, yp, yt, M, _, _ = core.load_residuals(dataset, split, SEED)

    st.markdown("### per-horizon residual scale and bias")
    lc, rc = st.columns(2)
    xh = [(h + 1) * H_MIN for h in range(H)]
    bias_h = np.array([np.nansum(stats["bias"][:, h] * stats["count"][:, h]) /
                       np.nansum(stats["count"][:, h]) for h in range(H)])
    f1 = go.Figure(go.Scatter(x=xh, y=np.sqrt(stats["delta"]),
                              mode="lines+markers", name="rmse"))
    f1.update_layout(title="residual RMSE vs horizon", height=360,
                     xaxis_title="minutes ahead", yaxis_title=unit,
                     margin=dict(l=10, r=10, t=40, b=10))
    lc.plotly_chart(f1, width=W, key="rmse_h")
    f2 = go.Figure(go.Scatter(x=xh, y=bias_h, mode="lines+markers", name="bias"))
    f2.add_hline(y=0, line_dash="dot", line_color="grey")
    f2.update_layout(title="residual bias vs horizon (want ≈ 0)", height=360,
                     xaxis_title="minutes ahead", yaxis_title=unit,
                     margin=dict(l=10, r=10, t=40, b=10))
    rc.plotly_chart(f2, width=W, key="bias_h")

    st.divider()
    st.markdown(f"### residual distribution @ +{horizon * H_MIN}min")
    Rh = R[:, :, horizon - 1][M[:, :, horizon - 1]].astype(np.float64)
    if Rh.size == 0:
        st.warning("no valid residuals at this horizon for this split.")
    else:
        Rh_plot = (Rh[np.linspace(0, Rh.size - 1, 200_000).astype(int)]
                   if Rh.size > 200_000 else Rh)
        lc, rc = st.columns([3, 1])
        fh = go.Figure(go.Histogram(x=Rh_plot, nbinsx=120, marker_color="steelblue"))
        fh.add_vline(x=0, line_dash="dot", line_color="black")
        fh.update_layout(title="residuals (true − pred)", height=380,
                         xaxis_title=unit, margin=dict(l=10, r=10, t=40, b=10))
        lc.plotly_chart(fh, width=W, key="hist")
        mu, sd = Rh.mean(), Rh.std()
        rc.metric("mean", f"{mu:+.3f}")
        rc.metric("std", f"{sd:.3f}")
        if sd > 0:
            rc.metric("skew", f"{float(((Rh - mu) ** 3).mean() / sd ** 3):+.2f}",
                      help="0 for a symmetric residual")
            rc.metric("kurtosis", f"{float(((Rh - mu) ** 4).mean() / sd ** 4):.2f}",
                      help="3 for a gaussian; >3 = heavy tails")

    st.divider()
    st.markdown(f"### sensor {node} over time — truth vs forecast")
    st.caption("each window forecasts the next hour. fixing the horizon and "
               "sweeping the window index gives a continuous time series of the "
               f"{horizon * H_MIN}-min-ahead forecast against what happened.")
    rng = st.slider("window range", 0, B - 1, (0, min(2000, B - 1)), key="trng")
    sl = slice(rng[0], rng[1] + 1)
    h = horizon - 1
    valid = M[sl, node, h]
    t = np.arange(rng[0], rng[1] + 1)
    ft = go.Figure()
    ft.add_trace(go.Scatter(x=t, y=np.where(valid, yt[sl, node, h], np.nan),
                            mode="lines", name="truth",
                            line=dict(color="black", width=1.4)))
    ft.add_trace(go.Scatter(x=t, y=np.where(valid, yp[sl, node, h], np.nan),
                            mode="lines", name="forecast",
                            line=dict(color="crimson", width=1.2)))
    ft.update_layout(title=f"sensor {node}, +{horizon * H_MIN}min ahead",
                     height=380, xaxis_title="window index", yaxis_title=unit,
                     margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(ft, width=W, key="ts")


# ----------------------------------------------------------------------------
# tab 4 — what's going wrong
# ----------------------------------------------------------------------------
with tab_wrong:
    diag = core.load_diagnostic(dataset, SEED)
    st.markdown("### gate 1 — does the residual scale match between train and val?")
    st.caption("the kronecker prior is fit on the train residuals and reused on "
               "val/test. if the per-horizon scale δₕ drifts more than 10% from "
               "train to val, that assumption is broken.")
    if diag is None:
        st.warning("no diagnostic.json for this dataset.")
    else:
        dtr = np.array(diag["delta_train"])
        dvl = np.array(diag["delta_val"])
        dte = np.array(diag.get("delta_test", [np.nan] * len(dtr)))
        ratio = np.array(diag["ratio_vl_over_tr_minus_one"])
        if diag["pass_overall"]:
            st.success("PASS — train and val residual scales agree within 10% at every horizon.")
        else:
            worst = float(np.max(np.abs(ratio))) * 100
            st.error(f"FAIL — val residual scale departs from train by up to "
                     f"{worst:.0f}% (gate is 10%). the prior fit on train will be "
                     f"mis-scaled on val/test. this is the known train→val step jump.")
        xh = [(h + 1) * H_MIN for h in range(len(dtr))]
        fg = go.Figure()
        fg.add_trace(go.Scatter(x=xh, y=np.sqrt(dtr), name="train", mode="lines+markers"))
        fg.add_trace(go.Scatter(x=xh, y=np.sqrt(dvl), name="val", mode="lines+markers"))
        fg.add_trace(go.Scatter(x=xh, y=np.sqrt(dte), name="test", mode="lines+markers",
                                line=dict(dash="dot")))
        fg.update_layout(title="√δₕ — residual scale by split", height=360,
                         xaxis_title="minutes ahead", yaxis_title=unit,
                         margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fg, width=W, key="gate")
        fr = go.Figure(go.Bar(x=xh, y=ratio * 100,
                              marker_color=["crimson" if abs(r) >= 0.10 else "seagreen"
                                            for r in ratio]))
        fr.add_hline(y=10, line_dash="dot", line_color="grey")
        fr.add_hline(y=-10, line_dash="dot", line_color="grey")
        fr.update_layout(title="val/train scale gap per horizon (%) · grey lines = ±10% gate",
                         height=320, xaxis_title="minutes ahead", yaxis_title="% gap",
                         margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fr, width=W, key="gateratio")

    st.divider()
    st.markdown("### is the backbone biased?")
    st.caption("residuals should be roughly zero-mean. a persistent sign means the "
               "graph-wavenet forecast leans high or low, and the diffusion model "
               "would have to learn that offset instead of just the spread.")
    bias_h = np.array([np.nansum(stats["bias"][:, h] * stats["count"][:, h]) /
                       np.nansum(stats["count"][:, h]) for h in range(H)])
    rel = np.abs(bias_h) / np.sqrt(stats["delta"])
    if np.nanmax(rel) < 0.05:
        st.success(f"OK — |bias| stays under 5% of RMSE (max {np.nanmax(rel) * 100:.1f}%).")
    else:
        st.warning(f"bias reaches {np.nanmax(rel) * 100:.0f}% of RMSE at some horizon — "
                   "the backbone has a systematic offset.")

    st.divider()
    st.markdown("### missing-data coverage")
    st.caption("masked-out cells are missing readings (speed==0). low coverage at a "
               "sensor means its residual statistics rest on little data.")
    node_cov = stats["count"].sum(1) / (B * H)
    fc = go.Figure(go.Histogram(x=node_cov * 100, nbinsx=40, marker_color="teal"))
    fc.update_layout(title="per-sensor valid coverage (%)", height=320,
                     xaxis_title="% of cells valid", margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fc, width=W, key="cov")
    bad = int((node_cov < 0.5).sum())
    if bad:
        st.warning(f"{bad} sensor(s) have under 50% valid data on the {split} split.")
    else:
        st.success("every sensor has at least 50% valid data on this split.")

    st.divider()
    st.markdown("### prior sign-off summary (`STATE.md`)")
    state_p = os.path.join(ROOT, "STATE.md")
    if os.path.exists(state_p):
        with open(state_p) as f:
            st.markdown(f.read())
    else:
        st.info("no STATE.md found.")
