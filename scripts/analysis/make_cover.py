# cover art: sample from the thesis's spatial prior (L+gI)^-1 on the real
# pems-bay sensor graph (largest component), dark full-bleed a4 render
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import LineCollection

rng = np.random.default_rng(7)

d = np.load("/Users/serkan/Projects/research-tud/data/processed/pemsbay/adj.npz")
G0 = nx.from_numpy_array(d["A_sym"])
lcc = max(nx.connected_components(G0), key=len)
G = nx.convert_node_labels_to_integers(G0.subgraph(lcc).copy())
n = G.number_of_nodes()

pos = nx.spring_layout(G, k=1.6 / np.sqrt(n), iterations=600, seed=3)
xy = np.array([pos[i] for i in range(n)])
xy -= xy.mean(0)
# pca-rotate: long axis vertical for the portrait page
_, _, Vt = np.linalg.svd(xy, full_matrices=False)
xy = xy @ Vt.T[:, ::-1]
xy /= np.abs(xy).max(0)  # fill both axes

# gmrf draw on the component's own sym normalized laplacian, gamma=0.8
A = nx.to_numpy_array(G)
deg = A.sum(1)
Dm = 1.0 / np.sqrt(np.maximum(deg, 1e-12))
L = np.eye(n) - (Dm[:, None] * A * Dm[None, :])
lam, V = np.linalg.eigh(L)
z = V @ (rng.standard_normal(n) / np.sqrt(np.clip(lam, 0, None) + 0.8))
z = (z - z.mean()) / z.std()

# a4 canvas in axes coords (x 0..1, y 0..297/210)
H = 297 / 210
X = 0.5 + 0.44 * xy[:, 0]
Y = 0.50 * H + 0.42 * H * xy[:, 1]

fig = plt.figure(figsize=(210 / 25.4, 297 / 25.4), dpi=300)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 1); ax.set_ylim(0, H); ax.axis("off")

# background: near-black navy with a faint glow under the graph
bg = LinearSegmentedColormap.from_list("bg", ["#03060e", "#081530", "#03060e"])
gy, gx = np.mgrid[0:400, 0:283]
r = np.hypot((gx / 283 - 0.5) * 0.9, (gy / 400 * H - 0.48 * H) / 1.1) / 0.75
ax.imshow(np.clip(1 - r, 0, 1) * 0.6, extent=(0, 1, 0, H), origin="lower",
          cmap=bg, vmin=0, vmax=1, aspect="auto", zorder=0)

cmap = LinearSegmentedColormap.from_list(
    "prior", ["#0b2a5e", "#2563c4", "#4884d6", "#37c2e8", "#d8f7ff"])

edges = np.array(G.edges())
segs = np.stack([np.column_stack([X[edges[:, 0]], Y[edges[:, 0]]]),
                 np.column_stack([X[edges[:, 1]], Y[edges[:, 1]]])], axis=1)
for lw, al in [(2.2, 0.04), (1.0, 0.09), (0.4, 0.22)]:  # glow passes
    ax.add_collection(LineCollection(segs, colors="#59b7e8", linewidths=lw,
                                     alpha=al, zorder=1))

c = cmap((z - z.min()) / (z.max() - z.min()))
sz = 8 + 46 * (np.abs(z) / np.abs(z).max()) ** 1.5
ax.scatter(X, Y, s=sz * 3.4, c=c, alpha=0.10, lw=0, zorder=2)  # halo
ax.scatter(X, Y, s=sz, c=c, alpha=0.95, lw=0, zorder=3)

fig.savefig(
    "/Users/serkan/Projects/research-tud/tudelft_thesis_draft/figures/cover_graph.jpg",
    dpi=300, pil_kwargs={"quality": 95})
print("done", n, "nodes,", len(edges), "edges")
