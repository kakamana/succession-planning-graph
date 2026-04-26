"""Spectral embeddings + 3-component readiness scoring for succession planning.

Uses Truncated SVD on `[skills | adjacency]` as a stand-in for a full GNN.
Scoring is the convex combination defined in `docs/03_methodology.md`:

    readiness = 0.6 * skill_match + 0.3 * structural_proximity + 0.1 * performance.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD

from .data import make_org
from .features import adjacency_matrix, skills_matrix

MODEL_DIR = Path(__file__).resolve().parents[2] / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

SCORING_WEIGHTS = dict(skill=0.6, structural=0.3, performance=0.1)


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
def fit_embeddings(
    df: pd.DataFrame, G: nx.DiGraph, k: int = 16, seed: int = 42,
) -> tuple[np.ndarray, list[str]]:
    """Truncated SVD on `[S | A]` → (n, k) embedding."""
    emp_index = list(df["emp_id"].values)
    S = skills_matrix(df)
    A = adjacency_matrix(G, emp_index)
    M = np.hstack([S, A])
    k_eff = min(k, max(M.shape[1] - 1, 1))
    svd = TruncatedSVD(n_components=k_eff, random_state=seed)
    Z = svd.fit_transform(M)
    return Z, emp_index


# ---------------------------------------------------------------------------
# Sub-scores
# ---------------------------------------------------------------------------
def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _shortest_path_distances(
    G: nx.DiGraph, source: str, max_d: int = 6,
) -> dict[str, int]:
    """Undirected shortest-path distance from source, capped at max_d."""
    UG = G.to_undirected(as_view=True)
    out: dict[str, int] = {}
    for node, d in nx.single_source_shortest_path_length(UG, source, cutoff=max_d).items():
        out[node] = int(d)
    return out


# ---------------------------------------------------------------------------
# Successor ranking
# ---------------------------------------------------------------------------
def succession_for(
    df: pd.DataFrame,
    G: nx.DiGraph,
    Z: np.ndarray,
    emp_index: list[str],
    manager_id: str,
    k: int = 5,
    weights: dict | None = None,
) -> list[dict]:
    """Return the top-k succession candidates for a given manager."""
    if manager_id not in set(emp_index):
        raise ValueError(f"unknown manager_id: {manager_id}")
    weights = weights or SCORING_WEIGHTS
    idx = {e: i for i, e in enumerate(emp_index)}

    incumbent = df.loc[df["emp_id"] == manager_id].iloc[0]
    inc_idx = idx[manager_id]
    inc_skills = np.asarray(incumbent["skills"], dtype=float)
    inc_emb = Z[inc_idx]

    spd = _shortest_path_distances(G, manager_id, max_d=6)

    rows = []
    for i, row in df.iterrows():
        emp_id = row["emp_id"]
        if emp_id == manager_id:
            continue
        cand_skills = np.asarray(row["skills"], dtype=float)
        skill_match = _cosine(cand_skills, inc_skills)
        struct_d = spd.get(emp_id, 6)
        structural_proximity = 1.0 / (1.0 + struct_d)
        emb_sim = _cosine(Z[idx[emp_id]], inc_emb)
        # Blend skills and embedding similarity (0.7/0.3 inside the "skill" leg)
        skill_blend = 0.7 * skill_match + 0.3 * emb_sim
        perf = (float(row["performance_rating"]) - 1.0) / 4.0
        readiness = (
            weights["skill"] * skill_blend
            + weights["structural"] * structural_proximity
            + weights["performance"] * perf
        )
        rows.append(dict(
            emp_id=emp_id,
            role=str(row["role"]),
            level=int(row["level"]),
            tenure_yrs=float(row["tenure_yrs"]),
            skill_match=round(float(skill_blend), 4),
            structural_proximity=round(float(structural_proximity), 4),
            performance=round(float(perf), 4),
            readiness_score=round(float(readiness), 4),
            shortest_path_distance=int(struct_d),
        ))
    rows.sort(key=lambda r: r["readiness_score"], reverse=True)
    return rows[:k]


def enforce_slate_diversity(
    candidates: list[dict],
    attr_lookup: dict[str, str],
    max_share: float = 0.8,
    pool: list[dict] | None = None,
) -> list[dict]:
    """If any subgroup's share exceeds `max_share`, swap the lowest-ranked
    over-represented candidate for the next-best under-represented candidate."""
    if not candidates:
        return candidates
    K = len(candidates)
    counts: dict[str, int] = {}
    for c in candidates:
        a = attr_lookup.get(c["emp_id"], "Other")
        counts[a] = counts.get(a, 0) + 1
    over = [a for a, n in counts.items() if n / K > max_share]
    if not over:
        return candidates
    if pool is None:
        return candidates
    target_attr = over[0]
    # Drop lowest-ranked candidate whose attr matches target_attr
    drop_idx = None
    for i in range(len(candidates) - 1, -1, -1):
        if attr_lookup.get(candidates[i]["emp_id"], "Other") == target_attr:
            drop_idx = i
            break
    if drop_idx is None:
        return candidates
    out = candidates.copy()
    out.pop(drop_idx)
    chosen_ids = {c["emp_id"] for c in out}
    for cand in pool:
        if cand["emp_id"] in chosen_ids:
            continue
        if attr_lookup.get(cand["emp_id"], "Other") != target_attr:
            out.append(cand)
            break
    return out


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def save(obj, name: str) -> Path:
    path = MODEL_DIR / name
    joblib.dump(obj, path)
    return path


def load(name: str):
    return joblib.load(MODEL_DIR / name)


def fit_and_save(df: pd.DataFrame | None = None, G: nx.DiGraph | None = None) -> dict:
    if df is None or G is None:
        df, G = make_org()
    Z, emp_index = fit_embeddings(df, G)
    save(dict(Z=Z, emp_index=emp_index), "embeddings.pkl")
    import json
    with open(MODEL_DIR / "scoring_config.json", "w") as fh:
        json.dump(SCORING_WEIGHTS, fh, indent=2)
    return dict(
        n_employees=int(len(df)),
        n_edges=int(G.number_of_edges()),
        embedding_dim=int(Z.shape[1]),
    )


if __name__ == "__main__":
    import json
    out = fit_and_save()
    print(json.dumps(out, indent=2))
