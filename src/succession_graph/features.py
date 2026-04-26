"""Feature builders for the succession graph: skills + adjacency."""
from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd


def skills_matrix(df: pd.DataFrame) -> np.ndarray:
    """Stack the per-employee skill vectors into an (n, 40) matrix."""
    arr = np.array([np.asarray(v, dtype=float) for v in df["skills"].values])
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def adjacency_matrix(G: nx.DiGraph, emp_index: list[str]) -> np.ndarray:
    """Symmetric, row-normalised adjacency in the order of `emp_index`."""
    idx_map = {e: i for i, e in enumerate(emp_index)}
    n = len(emp_index)
    A = np.zeros((n, n), dtype=float)
    for u, v in G.edges():
        if u in idx_map and v in idx_map:
            A[idx_map[u], idx_map[v]] = 1.0
            A[idx_map[v], idx_map[u]] = 1.0
    rs = A.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    return A / rs
