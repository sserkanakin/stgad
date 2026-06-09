# pure data + figure logic behind the dashboard.
#
# kept separate from the streamlit ui so it can be imported and tested directly
# (the ui file scripts/analysis/dashboard.py is just widgets on top of this).
import os
import sys
import json

import numpy as np
import networkx as nx
import plotly.graph_objects as go
import streamlit as st

# run from the repo root so the `src` imports resolve wherever we are launched.
_root = os.path.abspath(__file__)
while not os.path.exists(os.path.join(_root, "pyproject.toml")):
    _root = os.path.dirname(_root)
if _root not in sys.path:
    sys.path.insert(0, _root)
ROOT = _root

from src.data import adjacency
from src.utils import paths

# --- dataset facts (units, what the signal is) ---
DATASETS = ["metrla", "pems08f", "pemsbay"]
SIGNAL = {
    "metrla":  {"name": "METR-LA",  "unit": "mph",         "what": "loop-detector speed"},
    "pems08f": {"name": "PEMS08F",  "unit": "veh / 5 min", "what": "traffic flow (channel 0)"},
    "pemsbay": {"name": "PEMS-BAY", "unit": "mph",         "what": "loop-detector speed"},
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
        w = M[:, :, h].astype(np.float64)
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
    coverage = col / (B * N)
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
    _, _evals, evecs = load_graph(dataset)
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
    pos = nx.spring_layout(G, seed=0, iterations=60)
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


def node_degrees(n, ei, ej):
    """display degree = number of drawn edges touching each node."""
    deg = np.zeros(n)
    np.add.at(deg, ei, 1)
    np.add.at(deg, ej, 1)
    return deg


# ----------------------------------------------------------------------------
# figure builders
# ----------------------------------------------------------------------------
def graph_figure(xs, ys, ei, ej, color, clabel, title, picked=None, bias=False):
    """network scatter: grey edges underneath, nodes coloured by `color`."""
    fig = go.Figure()
    ex, ey = [], []
    for a, b in zip(ei, ej):
        ex += [xs[a], xs[b], None]
        ey += [ys[a], ys[b], None]
    fig.add_trace(go.Scatter(
        x=ex, y=ey, mode="lines",
        line=dict(width=0.4, color="rgba(150,150,150,0.4)"),
        hoverinfo="skip", showlegend=False))
    deg = node_degrees(len(xs), ei, ej)
    text = [f"node {i}<br>{clabel}: {color[i]:.2f}<br>degree: {int(deg[i])}"
            for i in range(len(xs))]
    marker = dict(size=8, color=color,
                  colorscale="RdBu" if bias else "Inferno",
                  showscale=True, colorbar=dict(title=clabel),
                  line=dict(width=0.5, color="rgba(0,0,0,0.3)"))
    if bias:
        marker["cmid"] = 0
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers", marker=marker,
        text=text, hoverinfo="text",
        customdata=np.arange(len(xs)), showlegend=False))
    if picked is not None and 0 <= picked < len(xs):
        fig.add_trace(go.Scatter(
            x=[xs[picked]], y=[ys[picked]], mode="markers",
            marker=dict(size=16, color="rgba(0,0,0,0)",
                        line=dict(width=2.5, color="cyan")),
            hoverinfo="skip", showlegend=False))
    fig.update_layout(
        title=title, height=560, margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x"),
        plot_bgcolor="white")
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

    labels = [f"+{(h + 1) * H_MIN} min" for h in range(H)]
    frames = [go.Frame(data=[edges, nodes(h)], name=labels[h]) for h in range(H)]
    fig = go.Figure(data=[edges, nodes(0)], frames=frames)
    steps = [dict(method="animate", label=f"+{(h + 1) * H_MIN}",
                  args=[[labels[h]], dict(mode="immediate",
                                          frame=dict(duration=0, redraw=True),
                                          transition=dict(duration=0))])
             for h in range(H)]
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
                      pad=dict(t=40), steps=steps)])
    return fig


def heatmap(z, x, y, title, cbar, colorscale="Inferno", zmid=None):
    fig = go.Figure(go.Heatmap(z=z, x=x, y=y, colorscale=colorscale,
                               zmid=zmid, colorbar=dict(title=cbar)))
    fig.update_layout(title=title, height=440,
                      margin=dict(l=10, r=10, t=40, b=10))
    return fig
