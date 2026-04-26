# Notebook 02 — Feature Engineering

>>> `from succession_graph.features import skills_matrix, adjacency_matrix`

## 1. Skills matrix
>>> `S = skills_matrix(df)` — shape (n, 40), rows L2-normalised.

## 2. Adjacency
>>> `A = adjacency_matrix(G)` — symmetric, row-normalised.

## 3. Stacked feature matrix
>>> `M = np.hstack([S, A])` — shape (n, 40 + n).

## 4. SVD components
>>> Inspect explained-variance curve; pick `k = 16` as the elbow.
