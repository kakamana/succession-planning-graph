# Feasibility Study — Succession-Planning Graph Recommender

## 1. Data feasibility

### Synthetic dataset
- **Generator:** `src/succession_graph/data.py::make_org()` — 2,000 employees with `emp_id, manager_id, role, level, skills:[40-vec], tenure_yrs, performance_rating`. Manager hierarchy is a tree.
- **Why synthetic:** real org charts and skill profiles are deeply sensitive. The synthetic frame mirrors the schema and the dynamics needed for an open, reproducible portfolio.

### Real-world equivalent
- **HRIS** (Workday / SuccessFactors): `emp_id`, `manager_id`, `role`, `level`.
- **Skills inventory:** O*NET-style 40-dim vector or self-rated skill profile.
- **Performance system:** rating + recent-promotion flag (used for held-out evaluation).

## 2. Technical feasibility
- **Algorithmic shortlist**
  - Skills cosine similarity — baseline
  - **TruncatedSVD on (skills × adjacency-aggregate)** — main (this stub)
  - Graph Laplacian eigenmaps — equivalent spectral alternative
  - GraphSAGE / GCN — stretch goal
- **Compute:** 1 CPU; SVD on a 2,000 × 80 matrix is sub-second.
- **Serving:** FastAPI + numpy embedding matrix (~tens of KB) + a single nearest-neighbour search per request.

## 3. Economic feasibility
| Line item | Monthly cost |
|-----------|--------------|
| 1× small container | ~$8 |
| Storage | ~$1 |
| MLflow (self-hosted) | $0 |
| **Total** | **~$9 / mo** |

**Value:** even a 25% reduction in time-to-shortlist for unexpected leadership exits has a measurable cost-of-vacancy benefit. For roles ≥ Director, days-of-vacancy savings dwarf the run-cost.

## 4. Operational feasibility
- **Refresh cadence:** nightly batch on the org snapshot.
- **Monitoring:** Recall@5 on a rolling 90-day held-out promotion set.
- **Human-in-the-loop:** every shortlist requires HRBP + senior-leader signoff; D&I review for Director+ slates.

## 5. Ethical / legal feasibility
- **Decision-aid framing on every API response.**
- **Slate-diversity gate** (max subgroup share ≤ 80%) prevents lookalike entrenchment.
- **PII:** anonymized `emp_id` in the model layer; the HRIS never exposes names downstream.
- **Explainability:** three sub-scores + a one-line "why" — never a black-box single number.

## 6. Recommendation
**Go.** The TruncatedSVD stub is enough to demonstrate the pattern, the GNN replacement is a clean follow-up, and the fairness + signoff workflow is built in from day one.
