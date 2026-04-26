"""Inference wrapper for the succession recommender."""
from __future__ import annotations

import pickle
from functools import lru_cache
from pathlib import Path

import networkx as nx
import pandas as pd

from . import models
from .data import PROCESSED, make_org

ATTRS = PROCESSED / "employee_attrs.parquet"
GRAPH = PROCESSED / "org_graph.gpickle"


def _load() -> tuple[pd.DataFrame, nx.DiGraph]:
    if ATTRS.exists() and GRAPH.exists():
        df = pd.read_parquet(ATTRS)
        with open(GRAPH, "rb") as fh:
            G = pickle.load(fh)
        return df, G
    return make_org()


@lru_cache(maxsize=1)
def _artifacts():
    df, G = _load()
    Z, emp_index = models.fit_embeddings(df, G)
    return df, G, Z, emp_index


def succession(manager_id: str, k: int = 5) -> list[dict]:
    df, G, Z, emp_index = _artifacts()
    return models.succession_for(df, G, Z, emp_index, manager_id=manager_id, k=k)


def known_managers(limit: int = 30) -> list[str]:
    df, G, _, _ = _artifacts()
    # Surface the deeper managers (level >= 4), not just the CEO.
    pool = df.loc[df["level"] >= 4, "emp_id"].tolist()
    return pool[:limit] if pool else df["emp_id"].head(limit).tolist()
