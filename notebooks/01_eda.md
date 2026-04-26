# Notebook 01 — EDA: synthetic org + skills

>>> `from succession_graph.data import make_org; df, G = make_org()`

## 1. Headcount + tree shape
- Total employees, max depth, branching-factor distribution.
- Verify the directed graph is a tree (one root, no cycles).

## 2. Role / level / dept
- Counts by `role`, `level`. Confirm pyramid shape.

## 3. Skills
- Distribution of skill-vector entries; top 5 most-prevalent skills per role.
- t-SNE of the 40-dim skill vectors coloured by role (visual sanity check).

## 4. Performance × tenure
- Joint distribution; per-level breakdown.

## 5. Hypotheses for modeling
1. Role centroids in skill space are well-separated → cosine similarity is informative.
2. Direct reports + same-manager peers should dominate the structural-proximity term.
3. The blend `(0.6, 0.3, 0.1)` will beat any single-component recommender.
