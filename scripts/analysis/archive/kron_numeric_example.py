"""A readable numeric example of what the spatial, temporal, and Kronecker prior matrices
are RIGHT NOW — and a real-data check of why the spatial/temporal split is incoherent.

Part 1: a tiny (N=3 nodes, H=3 horizons) fully-printable example — see what each matrix
        looks like and how the Kronecker glues them.
Part 2: the real pems08f prior — show that the TEMPORAL factor equals the residual's
        measured temporal covariance (fit), while the SPATIAL factor (graph Laplacian)
        does NOT match the residual's measured spatial covariance (assumed, and wrong).

usage: .venv/bin/python scripts/analysis/kron_numeric_example.py
"""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.diffusion.prior import load_prior, _load_residuals_train, GAMMA_DEFAULT
from src.data.adjacency import load_adjacency, compute_normalized_laplacian

np.set_printoptions(precision=2, suppress=True, linewidth=120)
G = GAMMA_DEFAULT
DS, BB = "pems08f", "gwn"


def corr(M):
    d = np.sqrt(np.clip(np.diag(M), 1e-12, None))
    return M / np.outer(d, d)


def line(s=""):
    print(s)


# ============================ PART 1 — tiny printable example ============================
line("=" * 78)
line("PART 1  —  tiny example (N=3 nodes on a path graph, H=3 horizons)")
line("=" * 78)

# --- spatial factor: built from the GRAPH ONLY (never sees the residual) ---
A = np.array([[0, 1, 0],          # 3-node path graph 1—2—3
              [1, 0, 1],
              [0, 1, 0]], float)
L = compute_normalized_laplacian(A)
Sig_S = np.linalg.inv(L + G * np.eye(3))
line("\nSPATIAL  Σ_S = (L + γI)^-1   (γ=0.8) — this is the graph Laplacian, DATA NEVER ENTERS")
line("normalized L (graph topology only):")
line(str(L))
line("\nΣ_S as a correlation matrix (what it claims about node-node correlation):")
line(str(corr(Sig_S)))
line("→ neighbors 1-2 and 2-3 correlated ~equally, 1-3 less. A pure smoothness pattern,")
line("  decided by the graph. Change the residual all you want — this matrix does not move.")

# --- temporal factor: the MEASURED residual covariance (real, top-left 3x3) ---
pr = load_prior(DS, "kronecker", seed=42, backbone=BB, spatial="structured", temporal="empirical")
Sig_T_full = pr.V_T @ np.diag(pr.mu_T) @ pr.V_T.T
Sig_T = Sig_T_full[:3, :3]
line("\nTEMPORAL  Σ_T  (real, first 3 horizons of pems08f) — this IS the measured residual cov")
line("Σ_T as a correlation matrix:")
line(str(corr(Sig_T)))
line("→ adjacent horizons correlated, decaying with lag. This is READ OFF the residual.")

# --- Kronecker: glue them ---
Sig = np.kron(corr(Sig_T), corr(Sig_S))   # 9x9, using correlations for readability
line("\nKRONECKER  Σ = Σ_T ⊗ Σ_S   (9×9, correlation form; index = horizon-block × node)")
line(str(Sig))
line("→ block (a,c) = Σ_T[a,c] · Σ_S. The SAME spatial pattern Σ_S, repeated in every")
line("  temporal block, scaled by the temporal correlation. That is what 'separable' means:")
line("  the space pattern is identical at every horizon; the time pattern identical at every node.")

# ============================ PART 2 — why it doesn't make sense ============================
line("\n" + "=" * 78)
line(f"PART 2  —  real {DS}/{BB}: is each factor consistent with the residual it diffuses?")
line("=" * 78)

R, mask = _load_residuals_train(DS, 42, BB)
R = np.where(mask, R, 0.0)
R = R[np.linspace(0, len(R) - 1, 4000).astype(int)]
N, H = pr.N, pr.H

# --- TEMPORAL: factor-we-use vs residual's measured temporal covariance ---
F_t = R.reshape(-1, H)
F_t = F_t[np.abs(F_t).sum(1) > 0]
emp_T = corr(np.cov(F_t.T))                         # measured temporal correlation
used_T = corr(Sig_T_full)                           # the temporal factor we actually use
iu = np.triu_indices(H, 1)
c_T = np.corrcoef(used_T[iu], emp_T[iu])[0, 1]
line(f"\nTEMPORAL factor vs measured temporal covariance:")
line(f"  correlation between their off-diagonal entries = {c_T:.3f}   (1.0 = identical)")
line(f"  e.g. horizon(1,2): factor says ρ={used_T[0,1]:.2f}, residual actually has ρ={emp_T[0,1]:.2f}")
line(f"       horizon(1,6): factor says ρ={used_T[0,5]:.2f}, residual actually has ρ={emp_T[0,5]:.2f}")
line("  → the temporal factor IS the residual's temporal covariance. Consistent by construction.")

# --- SPATIAL: factor-we-use vs residual's measured spatial covariance ---
lam, V = np.linalg.eigh(compute_normalized_laplacian(load_adjacency(DS, repair=True)))
Sig_S_full = V @ np.diag(1.0 / (lam + G)) @ V.T
used_S = corr(Sig_S_full)                           # the spatial factor we actually use
F_s = R.transpose(1, 0, 2).reshape(N, -1)
emp_S = corr(np.cov(F_s))                            # measured spatial correlation
iuN = np.triu_indices(N, 1)
c_S = np.corrcoef(used_S[iuN], emp_S[iuN])[0, 1]
line(f"\nSPATIAL factor vs measured spatial covariance:")
line(f"  correlation between their off-diagonal entries = {c_S:.3f}   (1.0 = identical)")

# edge vs non-edge: does the residual honor graph adjacency?
Aadj = np.asarray(load_adjacency(DS, repair=True)); np.fill_diagonal(Aadj, 0)
edge = Aadj > 0
line(f"  residual |corr| on graph EDGES     = {np.abs(emp_S[edge]).mean():.3f}")
line(f"  residual |corr| on NON-edges       = {np.abs(emp_S[~edge & ~np.eye(N,dtype=bool)]).mean():.3f}")
line(f"  factor   |corr| on graph EDGES     = {np.abs(used_S[edge]).mean():.3f}")
line(f"  factor   |corr| on NON-edges       = {np.abs(used_S[~edge & ~np.eye(N,dtype=bool)]).mean():.3f}")
# a couple of concrete disagreement examples
off = used_S.copy(); np.fill_diagonal(off, 0)
i, j = np.unravel_index(np.argmax(off), off.shape)          # a pair the FACTOR calls most correlated
line(f"  e.g. nodes the FACTOR calls most correlated ({i},{j}): factor ρ={used_S[i,j]:.2f}, "
     f"residual actually ρ={emp_S[i,j]:.2f}")
offe = np.abs(emp_S).copy(); np.fill_diagonal(offe, 0)
p, q = np.unravel_index(np.argmax(offe), offe.shape)        # a pair the DATA says most correlated
line(f"       nodes the RESIDUAL calls most correlated ({p},{q}): residual ρ={emp_S[p,q]:.2f}, "
     f"factor says ρ={used_S[p,q]:.2f}")
line("  → the spatial factor is a graph GUESS. It largely disagrees with the residual's real")
line("    spatial correlation — it never looked at the residual.")

line("\n" + "=" * 78)
line("WHY IT DOESN'T MAKE SENSE")
line("=" * 78)
line(f"  TEMPORAL factor  ≈ residual temporal covariance   (entry-corr {c_T:.2f})  → FIT to the data")
line(f"  SPATIAL  factor  ≠ residual spatial covariance    (entry-corr {c_S:.2f})  → ASSUMED, and wrong")
line("  One axis is measured from the residual; the other is a graph assumption the residual")
line("  does not honor. Same prior, two contradictory philosophies — that is the incoherence.")
