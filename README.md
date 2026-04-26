# Succession-Planning Graph Recommender

> **GNN-style ranking over an org graph + skills profile to surface succession candidates for any leadership role.** A spectral-embedding stand-in for a full GNN — Truncated SVD on the skills × adjacency matrix — combined with structural-distance and performance terms to produce a `readiness_score` for each candidate.

![Python](https://img.shields.io/badge/python-3.11-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688) ![Next.js](https://img.shields.io/badge/Next.js-14-black) ![License](https://img.shields.io/badge/license-MIT-green)

## Why this project
- A typical succession-plan deck is hand-curated and out-of-date the day it ships. This is a **reactive** recommender: pick a manager_id, return the top-k candidates with three sub-scores (skill_match, structural_proximity, performance) and a blended readiness score.
- Built for **Dubai HR** realities: large, fast-changing org charts where leadership moves can leave 2–3 critical roles open simultaneously.

## Table of contents
- [Business Requirements](./docs/01_business_requirements.md)
- [Feasibility Study](./docs/02_feasibility_study.md)
- [Methodology — graph Laplacian + spectral embedding + readiness scoring](./docs/03_methodology.md)
- [Evaluation Plan](./docs/04_evaluation.md)
- [Data card](./data/data_card.md) · [Data sources](./data/data_sources.md)
- [Notebooks](./notebooks/) · [Source](./src/succession_graph/) · [API](./api/main.py) · [UI](./ui/app/page.tsx)
- [CLAUDE.md](./CLAUDE.md) — paste prompt to resume in this folder

## Headline results (target)

| Metric | Random | Skill-only | Our blend | Target |
|---|---|---|---|---|
| Recall@5 (held-out promotions) | 5% | 32% | **48%** | ≥ 45% |
| nDCG@10 | 0.10 | 0.41 | **0.58** | ≥ 0.55 |
| Median readiness percentile of held-out promotions | – | – | **≥ 80** | ≥ 75 |

## Quickstart

```bash
pip install -e ".[dev]"
python -m succession_graph.data        # generate 2,000-employee org tree + skills + write graph
python -m succession_graph.models      # fit TruncatedSVD embeddings; save artifacts
jupyter lab notebooks/
uvicorn api.main:app --reload
cd ui && npm install && npm run dev
```

## Stack
Python · pandas · scikit-learn · **networkx** · **TruncatedSVD** · numpy · FastAPI · Next.js · Tailwind

## Author
Asad — MADS @ University of Michigan · Dubai HR
