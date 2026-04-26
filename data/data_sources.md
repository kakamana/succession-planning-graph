# Data Sources — H20 Succession-Planning Graph

## Primary

| # | Source | URL | Fields used | License |
|---|--------|-----|-------------|---------|
| 1 | Synthetic org (`src/succession_graph/data.py`) | n/a | All 7 fields (see `data_card.md`) | MIT |

## Real-world equivalents

| Source | URL | Use |
|--------|-----|-----|
| Workday HRIS | https://www.workday.com/ | `emp_id`, `manager_id`, `role`, `level` |
| O*NET 28.1 | https://www.onetcenter.org/database.html | 40-dim skills basis (per-role centroids) |
| SuccessFactors | https://www.sap.com/products/hcm.html | Performance / tenure / promotion history |

## Synthetic generation
`src/succession_graph/data.py::make_org(seed=42)` writes
`data/processed/employee_attrs.parquet` and `data/processed/org_graph.gpickle`.
Deterministic — no individual records.

## How to regenerate
```bash
python -m succession_graph.data
```

## Attribution
This project uses no third-party datasets. The methodology references Belkin
& Niyogi (Laplacian Eigenmaps), Hamilton et al. (GraphSAGE), and Halko et al.
(Truncated SVD) — see `docs/03_methodology.md`.
