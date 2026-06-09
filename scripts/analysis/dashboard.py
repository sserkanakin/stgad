# interactive dashboard for the residuals, the graph, and the diagnostics.
#
# run it with:
#   .venv/bin/streamlit run scripts/analysis/dashboard.py
#
# it loads the cached backbone residuals (results/residuals/<ds>/seed0) and the
# adjacency, and lets you poke at one dataset at a time: the graph in space, the
# error growing over the forecast horizon, the residual distributions, and the
# diagnostics that say what is going wrong.
import os
import sys

# run from the repo root so the `src` imports work no matter where streamlit
# is launched from.
_here = os.path.abspath(__file__)
_root = _here
while not os.path.exists(os.path.join(_root, "pyproject.toml")):
    _root = os.path.dirname(_root)
if _root not in sys.path:
    sys.path.insert(0, _root)
os.chdir(_root)

import json

import numpy as np
import networkx as nx
import plotly.graph_objects as go
import streamlit as st

from src.data import adjacency
from src.utils import paths

# --- dataset facts (units, what the signal is) ---
DATASETS = ["metrla", "pems08f", "pemsbay"]
SIGNAL = {
    "metrla":  {"name": "METR-LA",  "unit": "mph",            "what": "loop-detector speed"},
    "pems08f": {"name": "PEMS08F",  "unit": "veh / 5 min",    "what": "traffic flow (channel 0)"},
    "pemsbay": {"name": "PEMS-BAY", "unit": "mph",            "what": "loop-detector speed"},
}
SEED = 0
H_MIN = 5  # one horizon step = 5 minutes


# ----------------------------------------------------------------------------
# loaders (cached so flipping tabs does not re-read the big npz files)
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="loading residuals…")
def load_residuals(dataset, split, seed):
    """returns residuals, y_pred, y_true, mask — all (B, N, H) — plus mean/std.

    older files (pems-bay) do not store y_true; reconstruct it as y_pred + R.
    """
    d = np.load(paths.residual_npz(dataset, split, seed))
    R = d["residuals"].astype(np.float32)
    yp = d["y_pred"].astype(np.float32)
    yt = d["y_true"].astype(np.float32) if "y_true" in d.files else (yp + R)
    M = d["mask"].astype(bool)
    mean = float(d["mean"]) if "mean" in d.files else float("nan")
    std = float(d["std"]) if "std" in d.files else float("nan")
    return R, yp, yt, M, mean, std


@st.cache_data(show_spinner=False)
def load_graph(dataset):
    """raw adjacency (N, N) plus the cached laplacian eigenpairs."""
    A = adjacency.load_adjacency(dataset).astype(np.float64)
    A = 0.5 * (A + A.T)              # symmetrise the directed dcrnn matrices
    np.fill_diagonal(A, 0.0)        # drop self-loops for display
    eig = np.load(paths.laplacian_eigs(dataset))
    return A, eig["eigenvalues"], eig["eigenvectors"]


@st.cache_data(show_spinner=False)
def load_diagnostic(dataset, seed):
    p = paths.residual_dir(dataset, seed) / "diagnostic.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


# ----------------------------------------------------------------------------
# summary statistics (the heavy reduction over B windows, done once and cached)
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="crunching residual stats…")
def node_horizon_stats(dataset, split, seed):
    """per (node, horizon) rmse / bias / count, plus the per-horizon scale delta_h.

    everything is masked: positions where mask is false are missing readings and
    never enter a mean. the loop is over the 12 horizons so we never materialise
    a second copy of the (B, N, H) array.
    """
    R, _, _, M, _, _ = load_residuals(dataset, split, seed)
    B, N, H = R.shape
    cnt = np.zeros((N, H))
    s1 = np.zeros((N, H))
    s2 = np.zeros((N, H))
    for h in range(H):
        Rh = R[:, :, h]
        Mh = M[:, :, h]
        w = Mh.astype(np.float64)
        cnt[:, h] = w.sum(0)
        s1[:, h] = (Rh * w).sum(0, dtype=np.float64)
        s2[:, h] = ((Rh.astype(np.float64) ** 2) * w).sum(0)
    safe = cnt > 0
    bias = np.where(safe, s1 / np.where(safe, cnt, 1), np.nan)
    msq = np.where(safe, s2 / np.where(safe, cnt, 1), np.nan)
    rmse = np.sqrt(msq)
    # per-horizon scale: masked mean of R^2 over all nodes and windows
    col = cnt.sum(0)
    delta = np.where(col > 0, s2.sum(0) / np.where(col > 0, col, 1), np.nan)
    # fraction of (window, node) positions that are valid, per horizon
    coverage = cnt.sum(0) / (B * N)
    return {
        "rmse": rmse, "bias": bias, "count": cnt,
        "delta": delta, "coverage": coverage,
        "B": B, "N": N, "H": H,
    }


@st.cache_data(show_spinner=False)
def layout_spectral(dataset):
    """2-d laplacian eigenmap: columns 1 and 2 of the eigenvectors.

    connected nodes sit near each other, so it reads as a rough spatial map even
    though we have no real sensor lat/lon.
    """
    _, evals, evecs = load_graph(dataset)
    return evecs[:, 1], evecs[:, 2]


@st.cache_data(show_spinner=False)
def layout_spring(dataset, thr):
    """force-directed layout of the thresholded graph (deterministic seed)."""
    A, _, _ = load_graph(dataset)
    G = nx.Graph()
    G.add_nodes_from(range(A.shape[0]))
    iu = np.triu_indices_from(A, k=1)
    for i, j in zip(*iu):
        if A[i, j] > thr:
            G.add_edge(int(i), int(j), weight=float(A[i, j]))
    pos = nx.spring_layout(G, seed=0, k=None, iterations=60)
    xs = np.array([pos[i][0] for i in range(A.shape[0])])
    ys = np.array([pos[i][1] for i in range(A.shape[0])])
    return xs, ys


@st.cache_data(show_spinner=False)
def edge_segments(dataset, thr):
    """edge endpoints above the weight threshold, as flat upper-triangle pairs."""
    A, _, _ = load_graph(dataset)
    iu = np.triu_indices_from(A, k=1)
    keep = A[iu] > thr
    return iu[0][keep], iu[1][keep], A[iu][keep]


# ----------------------------------------------------------------------------
# figure builders
# ----------------------------------------------------------------------------
def graph_figure(xs, ys, ei, ej, color, clabel, title, picked=None):
    """network scatter: grey edges underneath, nodes coloured by `color`."""
    fig = go.Figure()
    # edges as one trace with None separators
    ex, ey = [], []
    for a, b in zip(ei, ej):
        ex += [xs[a], xs[b], None]
        ey += [ys[a], ys[b], None]
    fig.add_trace(go.Scatter(
        x=ex, y=ey, mode="lines",
        line=dict(width=0.4, color="rgba(150,150,150,0.4)"),
        hoverinfo="skip", showlegend=False,
    ))
    deg = np.zeros(len(xs))
    for a, b in zip(ei, ej):
        deg[a] += 1
        deg[b] += 1
    text = [f"node {i}<br>{clabel}: {color[i]:.2f}<br>degree: {int(deg[i])}"
            for i in range(len(xs))]
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers",
        marker=dict(size=8, color=color, colorscale="Inferno",
                    showscale=True, colorbar=dict(title=clabel),
                    line=dict(width=0.5, color="rgba(0,0,0,0.3)")),
        text=text, hoverinfo="text",
        customdata=np.arange(len(xs)), showlegend=False,
    ))
    if picked is not None and 0 <= picked < len(xs):
        fig.add_trace(go.Scatter(
            x=[xs[picked]], y=[ys[picked]], mode="markers",
            marker=dict(size=16, color="rgba(0,0,0,0)",
                        line=dict(width=2.5, color="cyan")),
            hoverinfo="skip", showlegend=False,
        ))
    fig.update_layout(
        title=title, height=560, margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x"),
        plot_bgcolor="white",
    )
    return fig


def animated_graph(xs, ys, ei, ej, rmse, unit):
    """one frame per horizon: nodes recoloured by rmse[:, h], fixed colour scale."""
    H = rmse.shape[1]
    vmax = float(np.nanmax(rmse))
    ex, ey = [], []
    for a, b in zip(ei, ej):
        ex += [xs[a], xs[b], None]
        ey += [ys[a], ys[b], None]
    edges = go.Scatter(x=ex, y=ey, mode="lines",
                       line=dict(width=0.4, color="rgba(150,150,150,0.4)"),
                       hoverinfo="skip", showlegend=False)

    def nodes(h):
        return go.Scatter(
            x=xs, y=ys, mode="markers",
            marker=dict(size=8, color=rmse[:, h], colorscale="Inferno",
                        cmin=0, cmax=vmax, showscale=True,
                        colorbar=dict(title=f"rmse<br>({unit})")),
            text=[f"node {i}: {rmse[i, h]:.2f}" for i in range(len(xs))],
            hoverinfo="text", showlegend=False)

    frames = [go.Frame(data=[edges, nodes(h)],
                       name=f"+{(h + 1) * H_MIN} min") for h in range(H)]
    fig = go.Figure(data=[edges, nodes(0)], frames=frames)
    steps = [dict(method="animate", label=f"+{(h + 1) * H_MIN}",
                  args=[[f"+{(h + 1) * H_MIN} min"],
                        dict(mode="immediate",
                             frame=dict(duration=0, redraw=True),
                             transition=dict(duration=0))]) for h in range(H)]
    fig.update_layout(
        title="forecast error across the network as the horizon grows",
        height=600, margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x"),
        plot_bgcolor="white",
        updatemenus=[dict(type="buttons", showactive=False, x=0.0, y=1.12,
                          buttons=[dict(label="▶ play", method="animate",
                                        args=[None, dict(frame=dict(duration=700, redraw=True),
                                                         fromcurrent=True)]),
                                   dict(label="⏸ pause", method="animate",
                                        args=[[None], dict(mode="immediate",
                                                           frame=dict(duration=0, redraw=False))])])],
        sliders=[dict(active=0, currentvalue=dict(prefix="horizon "),
                      pad=dict(t=40), steps=steps)],
    )
    return fig


def heatmap(z, x, y, title, cbar, colorscale="Inferno", zmid=None):
    fig = go.Figure(go.Heatmap(z=z, x=x, y=y, colorscale=colorscale,
                               zmid=zmid, colorbar=dict(title=cbar)))
    fig.update_layout(title=title, height=440,
                      margin=dict(l=10, r=10, t=40, b=10))
    return fig


# ----------------------------------------------------------------------------
# sidebar — the shared controls
# ----------------------------------------------------------------------------
st.set_page_config(page_title="residual & graph explorer", layout="wide")
st.sidebar.title("controls")
dataset = st.sidebar.selectbox(
    "dataset", DATASETS,
    format_func=lambda d: SIGNAL[d]["name"])
split = st.sidebar.radio("split", ["train", "val", "test"], horizontal=True)
info = SIGNAL[dataset]
unit = info["unit"]

stats = node_horizon_stats(dataset, split, SEED)
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
        thr_max = 0.95 if dataset != "pems08f" else 0.5
        thr = st.slider("edge threshold", 0.0, thr_max, 0.1 if dataset != "pems08f" else 0.5,
                        help="only edges with weight above this are drawn / used for spring")
        color_kind = st.selectbox(
            "colour nodes by",
            [f"RMSE @ +{horizon * H_MIN}min", f"bias @ +{horizon * H_MIN}min",
             "mean RMSE (all horizons)", "node degree"])

    ei, ej, ew = edge_segments(dataset, thr)
    if layout_kind == "spring":
        xs, ys = layout_spring(dataset, thr)
    else:
        xs, ys = layout_spectral(dataset)

    if color_kind.startswith("RMSE @"):
        color, clabel = node_rmse_h, f"rmse ({unit})"
    elif color_kind.startswith("bias @"):
        color, clabel = stats["bias"][:, horizon - 1], f"bias ({unit})"
    elif color_kind.startswith("mean RMSE"):
        color, clabel = np.nanmean(stats["rmse"], axis=1), f"rmse ({unit})"
    else:
        deg = np.zeros(N)
        for a, b in zip(ei, ej):
            deg[a] += 1
            deg[b] += 1
        color, clabel = deg, "degree"

    cs = "RdBu" if color_kind.startswith("bias") else "Inferno"
    fig = graph_figure(xs, ys, ei, ej, color, clabel,
                       f"{color_kind} · {len(ei)} edges shown", picked=node)
    if color_kind.startswith("bias"):
        fig.data[1].marker.colorscale = "RdBu"
        fig.data[1].marker.cmid = 0
    with lc:
        ev = st.plotly_chart(fig, use_container_width=True, key="graph",
                             on_select="rerun")
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
             f"bias ({unit})": [round(float(stats['bias'][i, horizon - 1]), 2) for i in order[:10]]},
            hide_index=True, use_container_width=True)


# ----------------------------------------------------------------------------
# tab 2 — spatio-temporal
# ----------------------------------------------------------------------------
with tab_st:
    st.markdown(
        "**how the forecast error spreads over the network as the horizon grows.** "
        "press play to sweep +5 → +60 min; node colour is RMSE.")
    if st.checkbox("render animation (heavier)", value=True):
        if "spring" in st.session_state.get("layout_kind", "") :
            xs2, ys2 = layout_spring(dataset, 0.1 if dataset != "pems08f" else 0.5)
        else:
            xs2, ys2 = layout_spectral(dataset)
        ei2, ej2, _ = edge_segments(dataset, 0.1 if dataset != "pems08f" else 0.5)
        st.plotly_chart(animated_graph(xs2, ys2, ei2, ej2, stats["rmse"], unit),
                        use_container_width=True, key="anim")

    st.divider()
    st.markdown("**the full picture: RMSE for every sensor at every horizon.** "
                "rows are sensors (sorted worst-first), columns are +5…+60 min.")
    order = np.argsort(-np.nanmean(stats["rmse"], axis=1))
    xh = [f"+{(h + 1) * H_MIN}" for h in range(H)]
    st.plotly_chart(
        heatmap(stats["rmse"][order], xh, [str(int(i)) for i in order],
                "per-sensor RMSE by horizon", f"rmse ({unit})"),
        use_container_width=True, key="nh_heat")

    st.divider()
    st.markdown("**one window, up close.** pick a window and see the residual for "
                "every sensor across the 12 horizons.")
    w = st.slider("window index", 0, B - 1, 0, key="win")
    R, yp, yt, M, _, _ = load_residuals(dataset, split, SEED)
    Rw = np.where(M[w], R[w], np.nan)  # (N, H)
    order_w = np.argsort(-np.nanmean(np.abs(Rw), axis=1))
    st.plotly_chart(
        heatmap(Rw[order_w], xh, [str(int(i)) for i in order_w],
                f"window {w}: residual (true − pred)", f"residual ({unit})",
                colorscale="RdBu", zmid=0),
        use_container_width=True, key="win_heat")


# ----------------------------------------------------------------------------
# tab 3 — residual analysis
# ----------------------------------------------------------------------------
with tab_resid:
    R, yp, yt, M, _, _ = load_residuals(dataset, split, SEED)

    st.markdown("### per-horizon residual scale and bias")
    lc, rc = st.columns(2)
    delta = stats["delta"]
    bias_h = np.array([np.nansum(stats["bias"][:, h] * stats["count"][:, h]) /
                       np.nansum(stats["count"][:, h]) for h in range(H)])
    xh = [(h + 1) * H_MIN for h in range(H)]
    f1 = go.Figure()
    f1.add_trace(go.Scatter(x=xh, y=np.sqrt(delta), mode="lines+markers",
                            name="rmse"))
    f1.update_layout(title="residual RMSE vs horizon", height=360,
                     xaxis_title="minutes ahead", yaxis_title=unit,
                     margin=dict(l=10, r=10, t=40, b=10))
    lc.plotly_chart(f1, use_container_width=True, key="rmse_h")
    f2 = go.Figure()
    f2.add_trace(go.Scatter(x=xh, y=bias_h, mode="lines+markers", name="bias"))
    f2.add_hline(y=0, line_dash="dot", line_color="grey")
    f2.update_layout(title="residual bias vs horizon (want ≈ 0)", height=360,
                     xaxis_title="minutes ahead", yaxis_title=unit,
                     margin=dict(l=10, r=10, t=40, b=10))
    rc.plotly_chart(f2, use_container_width=True, key="bias_h")

    st.divider()
    st.markdown(f"### residual distribution @ +{horizon * H_MIN}min")
    Rh = R[:, :, horizon - 1][M[:, :, horizon - 1]].astype(np.float64)
    # subsample for the histogram so the browser stays responsive
    if Rh.size > 200_000:
        idx = np.linspace(0, Rh.size - 1, 200_000).astype(int)
        Rh_plot = Rh[idx]
    else:
        Rh_plot = Rh
    lc, rc = st.columns([3, 1])
    fh = go.Figure(go.Histogram(x=Rh_plot, nbinsx=120,
                                marker_color="steelblue"))
    fh.add_vline(x=0, line_dash="dot", line_color="black")
    fh.update_layout(title="residuals (true − pred)", height=380,
                     xaxis_title=unit, margin=dict(l=10, r=10, t=40, b=10))
    lc.plotly_chart(fh, use_container_width=True, key="hist")
    mu, sd = Rh.mean(), Rh.std()
    sk = float(((Rh - mu) ** 3).mean() / sd ** 3)
    ku = float(((Rh - mu) ** 4).mean() / sd ** 4)
    rc.metric("mean", f"{mu:+.3f}")
    rc.metric("std", f"{sd:.3f}")
    rc.metric("skew", f"{sk:+.2f}", help="0 for a symmetric residual")
    rc.metric("kurtosis", f"{ku:.2f}", help="3 for a gaussian; >3 = heavy tails")

    st.divider()
    st.markdown(f"### sensor {node} over time — truth vs forecast")
    st.caption("each window forecasts the next hour. fixing the horizon and "
               "sweeping the window index gives a continuous time series of the "
               f"{horizon * H_MIN}-min-ahead forecast against what happened.")
    npts = min(B, 2000)
    rng = st.slider("window range", 0, B - 1, (0, min(npts, B - 1)), key="trng")
    sl = slice(rng[0], rng[1] + 1)
    h = horizon - 1
    valid = M[sl, node, h]
    t = np.arange(rng[0], rng[1] + 1)
    yt_n = np.where(valid, yt[sl, node, h], np.nan)
    yp_n = np.where(valid, yp[sl, node, h], np.nan)
    ft = go.Figure()
    ft.add_trace(go.Scatter(x=t, y=yt_n, mode="lines", name="truth",
                            line=dict(color="black", width=1.4)))
    ft.add_trace(go.Scatter(x=t, y=yp_n, mode="lines", name="forecast",
                            line=dict(color="crimson", width=1.2)))
    ft.update_layout(title=f"sensor {node}, +{horizon * H_MIN}min ahead",
                     height=380, xaxis_title="window index",
                     yaxis_title=unit, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(ft, use_container_width=True, key="ts")


# ----------------------------------------------------------------------------
# tab 4 — what's going wrong
# ----------------------------------------------------------------------------
with tab_wrong:
    diag = load_diagnostic(dataset, SEED)
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
        passed = diag["pass_overall"]
        if passed:
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
        st.plotly_chart(fg, use_container_width=True, key="gate")
        fr = go.Figure(go.Bar(x=xh, y=ratio * 100,
                              marker_color=["crimson" if abs(r) >= 0.10 else "seagreen"
                                            for r in ratio]))
        fr.add_hline(y=10, line_dash="dot", line_color="grey")
        fr.add_hline(y=-10, line_dash="dot", line_color="grey")
        fr.update_layout(title="val/train scale gap per horizon (%) · grey lines = ±10% gate",
                         height=320, xaxis_title="minutes ahead", yaxis_title="% gap",
                         margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fr, use_container_width=True, key="gateratio")

    st.divider()
    st.markdown("### is the backbone biased?")
    st.caption("residuals should be roughly zero-mean. a persistent sign means the "
               "graph-wavenet forecast leans high or low, and the diffusion model "
               "would have to learn that offset instead of just the spread.")
    bias_h = np.array([np.nansum(stats["bias"][:, h] * stats["count"][:, h]) /
                       np.nansum(stats["count"][:, h]) for h in range(H)])
    rmse_h = np.sqrt(stats["delta"])
    rel = np.abs(bias_h) / rmse_h
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
    st.plotly_chart(fc, use_container_width=True, key="cov")
    bad = int((node_cov < 0.5).sum())
    if bad:
        st.warning(f"{bad} sensor(s) have under 50% valid data on the {split} split.")
    else:
        st.success("every sensor has at least 50% valid data on this split.")

    st.divider()
    st.markdown("### prior sign-off summary (`STATE.md`)")
    state_p = os.path.join(_root, "STATE.md")
    if os.path.exists(state_p):
        with open(state_p) as f:
            st.markdown(f.read())
    else:
        st.info("no STATE.md found.")
