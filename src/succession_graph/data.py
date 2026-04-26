"""Synthetic 2,000-employee org tree + skills profile.

Writes:
    data/processed/employee_attrs.parquet
    data/processed/org_graph.gpickle    (a networkx.DiGraph)

Manager hierarchy is a tree. Skills are a 40-dim vector drawn from a per-role
centroid plus Gaussian noise, then row-normalised.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RAW = DATA_DIR / "raw"
PROCESSED = DATA_DIR / "processed"

ROLES = [
    "CEO", "VP Engineering", "VP Sales", "VP HR",
    "Director Eng", "Director Sales", "Director Ops", "Director Marketing",
    "Manager Eng", "Manager Sales", "Senior IC", "IC",
]
SKILL_DIM = 40
N_TOTAL = 2_000


def _role_centroids(seed: int = 42) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    centroids = {}
    for r in ROLES:
        v = rng.normal(0.0, 1.0, size=SKILL_DIM)
        # Bias certain dimensions per role for separability
        h = abs(hash(r)) % SKILL_DIM
        v[h] += 3.0
        v[(h + 7) % SKILL_DIM] += 2.0
        centroids[r] = v
    return centroids


def make_org(n: int = N_TOTAL, seed: int = 42) -> tuple[pd.DataFrame, nx.DiGraph]:
    """Generate the employee table + the networkx DiGraph (manager → report)."""
    rng = np.random.default_rng(seed)
    centroids = _role_centroids(seed)

    # Build the tree top-down
    rows: list[dict] = []
    G = nx.DiGraph()

    def emp_id(i: int) -> str:
        return f"E-{i:04d}"

    # Root: CEO
    rows.append(dict(emp_id=emp_id(1), manager_id=None, role="CEO", level=7))
    G.add_node(emp_id(1))
    next_id = 2

    # Layer plan (from L7 root downward)
    layer_plan = [
        ("VP Engineering", "VP Sales", "VP HR"),       # L6 — 3 VPs
        ("Director Eng", "Director Sales", "Director Ops", "Director Marketing"),  # L5 — Directors
        ("Manager Eng", "Manager Sales"),              # L4 — Managers
        ("Senior IC",),                                # L3
        ("IC",),                                       # L2
        ("IC",),                                       # L1
    ]
    branching = [3, 4, 8, 6, 8, 10]
    levels_per_layer = [6, 5, 4, 3, 2, 1]

    parents_at_layer: list[list[str]] = [[emp_id(1)]]
    for layer_i, roles_pool in enumerate(layer_plan):
        new_parents = []
        for parent in parents_at_layer[-1]:
            n_children = max(1, int(rng.poisson(branching[layer_i])))
            for _ in range(n_children):
                if next_id > n:
                    break
                role = rng.choice(roles_pool)
                lvl = levels_per_layer[layer_i]
                e = emp_id(next_id)
                rows.append(dict(emp_id=e, manager_id=parent, role=str(role), level=int(lvl)))
                G.add_node(e)
                G.add_edge(parent, e)
                new_parents.append(e)
                next_id += 1
            if next_id > n:
                break
        parents_at_layer.append(new_parents)
        if next_id > n:
            break

    # Pad up to n with extra ICs under random managers in the deepest non-empty layer
    deepest_managers = next(
        (lst for lst in reversed(parents_at_layer[:-1]) if lst), [emp_id(1)]
    )
    while next_id <= n:
        parent = str(rng.choice(deepest_managers))
        e = emp_id(next_id)
        rows.append(dict(emp_id=e, manager_id=parent, role="IC", level=1))
        G.add_node(e)
        G.add_edge(parent, e)
        next_id += 1

    df = pd.DataFrame(rows)
    n_actual = len(df)

    # Skills
    skills = np.zeros((n_actual, SKILL_DIM), dtype=float)
    for i, role in enumerate(df["role"].values):
        centroid = centroids[role]
        v = centroid + rng.normal(0.0, 0.7, size=SKILL_DIM)
        # Random sparsity: ~30% of dimensions zeroed
        mask = rng.binomial(1, 0.7, size=SKILL_DIM)
        v = v * mask
        norm = np.linalg.norm(v)
        if norm > 0:
            v = v / norm
        skills[i] = v
    df["skills"] = list(skills)

    # Tenure + performance, mildly correlated with level
    df["tenure_yrs"] = np.clip(
        rng.gamma(2.0, 1.5, size=n_actual) + 0.4 * df["level"], 0.1, 25.0
    ).round(2)
    df["performance_rating"] = np.clip(
        rng.normal(loc=3.0 + 0.1 * (df["level"] - 1), scale=0.7), 1, 5
    ).round().astype(int)

    return df, G


def write_processed() -> tuple[Path, Path]:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    df, G = make_org()
    attrs_out = PROCESSED / "employee_attrs.parquet"
    graph_out = PROCESSED / "org_graph.gpickle"
    # Parquet does not love object-arrays of np.ndarray — store as list of lists.
    df_to_write = df.copy()
    df_to_write["skills"] = df_to_write["skills"].apply(lambda v: list(map(float, v)))
    df_to_write.to_parquet(attrs_out, index=False)
    with open(graph_out, "wb") as fh:
        pickle.dump(G, fh)
    return attrs_out, graph_out


if __name__ == "__main__":
    a, g = write_processed()
    print(f"wrote -> {a}")
    print(f"wrote -> {g}")
