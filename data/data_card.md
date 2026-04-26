# Data Card — H20 Succession-Planning Graph

## Dataset composition

| Layer | Source | Rows × cols | Purpose |
|-------|--------|-------------|---------|
| Synthetic org | `src/succession_graph/data.py::make_org()` | 2,000 × 7 | Reproducible employee table |
| Org graph | derived from `manager_id` | 2,000 nodes / 1,999 edges | Tree structure for proximity |

## Fields

| Field | Type | Description |
|---|---|---|
| `emp_id` | str | E-0001 … E-2000 |
| `manager_id` | str / null | Pointer to direct manager; null for the CEO |
| `role` | str | one of 12 roles |
| `level` | int (1–7) | 1 = Junior … 7 = VP |
| `skills` | float[40] | normalised skill vector |
| `tenure_yrs` | float | 0.1 – 25.0 |
| `performance_rating` | int (1–5) | discrete |

## Generative process
- Recursive tree construction with branching factor sampled per level (3 at top, 6–10 at line-management, 12 at IC layer).
- 40-dim skills vector drawn from a per-role centroid + Gaussian noise; row-normalised.
- Performance and tenure mildly correlated with level.

## Files written
- `data/processed/employee_attrs.parquet` — the employee table.
- `data/processed/org_graph.gpickle` — `networkx.DiGraph`.

## PII
None.

## Reproducing
```bash
python -m succession_graph.data
```
Deterministic seed = 42.

## Licensing
- MIT (this repo).
