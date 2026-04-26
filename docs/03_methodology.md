# Methodology — Spectral Embeddings + Skills + Structural Proximity

A full GNN is overkill for the scope of this project. We use a **spectral-embedding stand-in** — Truncated SVD on the concatenation of node-feature (skills) and graph-structure (aggregated adjacency) — that delivers most of the practical signal while remaining a one-CPU, sub-second fit.

---

## 1. Graph

Build the org as `G = (V, E)` where each $v \in V$ is an employee, and an edge $(u, v) \in E$ exists iff $u$ is the manager of $v$. Verify that the resulting graph is a tree at load time.

For each node, attach a **skills profile** $s_v \in \mathbb{R}^{40}$, normalised to unit L2.

## 2. Graph Laplacian — intuition

For an undirected graph with adjacency $A \in \mathbb{R}^{n \times n}$ and degree $D = \text{diag}(\sum_j A_{ij})$, the (unnormalised) Laplacian is

$$
L = D - A.
$$

The eigenvectors $\phi_1, \ldots, \phi_k$ corresponding to the smallest non-zero eigenvalues of $L$ are the **Laplacian eigenmaps** — a $k$-dimensional embedding that places strongly-connected nodes near each other. These have been used as a stand-in for shallow GNN message-passing since long before the deep-graph era.

## 3. Truncated SVD on (adjacency × skills)

We avoid a separate eigendecomposition by stacking node-features and structure into a single matrix and SVD-ing once:

$$
M \;=\; \big[\, S \;\;\, \tilde A \,\big] \in \mathbb{R}^{n \times (40 + n)},
$$

where $S \in \mathbb{R}^{n \times 40}$ are the row-normalised skill vectors and $\tilde A \in \mathbb{R}^{n \times n}$ is the symmetrised adjacency normalised by row-sum. Truncated SVD with $k = 16$ components yields embeddings $Z \in \mathbb{R}^{n \times 16}$ that combine **what each employee can do** with **where they sit in the hierarchy**.

$$
M \approx U_k \Sigma_k V_k^\top, \qquad Z = U_k \Sigma_k.
$$

## 4. Three sub-scores

For an incumbent $v^*$ (the manager whose role is to be back-filled) and a candidate $c$:

- **Skill match** — cosine similarity of skills:
  $$\text{skill\_match}(c, v^*) = \frac{s_c \cdot s_{v^*}}{\|s_c\| \, \|s_{v^*}\|}.$$
- **Structural proximity** — inverse shortest-path distance on the undirected hierarchy (no self-edges, capped at 6):
  $$\text{struct\_prox}(c, v^*) = \frac{1}{1 + \text{spd}(c, v^*)}.$$
- **Performance** — `(performance_rating - 1) / 4 ∈ [0, 1]`.

We additionally use the spectral embedding cosine $\cos(z_c, z_{v^*})$ as a tiebreaker / regulariser; it correlates with `skill_match` but adds the structural signal.

## 5. Readiness score

$$
\text{readiness}(c, v^*) \;=\; 0.6 \cdot \text{skill\_match} \;+\; 0.3 \cdot \text{struct\_prox} \;+\; 0.1 \cdot \text{performance}.
$$

The weights are configurable in `models.SCORING_WEIGHTS` and were chosen to match the per-stakeholder priorities in `docs/01_business_requirements.md`. They are tunable on backtest.

## 6. Slate-diversity gate

After ranking, walk the top-K candidates and ensure no single subgroup (`nationality_group` or `gender`, when available) exceeds 80% of the slate. If the gate trips, swap the lowest-ranked over-represented candidate for the next-best under-represented candidate.

## 7. References
- Belkin & Niyogi, *Laplacian Eigenmaps for Dimensionality Reduction and Data Representation*, NC 2003.
- Hamilton, Ying, Leskovec, *Inductive Representation Learning on Large Graphs (GraphSAGE)*, NeurIPS 2017.
- Kipf & Welling, *Semi-Supervised Classification with Graph Convolutional Networks*, ICLR 2017.
- Halko, Martinsson, Tropp, *Finding Structure with Randomness: Probabilistic Algorithms for Constructing Approximate Matrix Decompositions* (Truncated SVD), SIAM 2011.
