"""Graph operators for the backbones, matching the canonical conventions.

Kept separate so every graph-aware backbone uses the SAME normalized adjacency
and the SAME convention as the diffusion prior downstream.
"""
import numpy as np
import torch


def kipf_norm_adj(A: np.ndarray) -> np.ndarray:
    """Kipf & Welling renormalization: Â = D̃^-1/2 (A+I) D̃^-1/2, D̃ = deg(A+I).

    This is the 1-hop propagation used by the plain GCN and by STGCN's
    first-order graph conv. Self-loops ADDED (that is the renormalization trick),
    unlike the prior's Laplacian which strips them.
    """
    A = np.asarray(A, dtype=np.float64)
    np.fill_diagonal(A, 0.0)
    A = np.maximum(A, A.T)                 # symmetric
    A_tilde = A + np.eye(A.shape[0])       # add self-loops
    d = A_tilde.sum(1)
    d_inv_sqrt = np.power(d, -0.5)
    d_inv_sqrt[~np.isfinite(d_inv_sqrt)] = 0.0
    return (d_inv_sqrt[:, None] * A_tilde) * d_inv_sqrt[None, :]


def to_tensor(M: np.ndarray, device=None) -> torch.Tensor:
    return torch.as_tensor(np.asarray(M), dtype=torch.float32, device=device)
