# A Reactive Succession Recommender on the Org Graph: Spectral Embeddings, Three-Component Readiness Scoring, and a Slate-Diversity Release Gate

**Author.** Asad Kamran. Master of Applied Data Science, University of Michigan. Day affiliation: Dubai Human Resources Department.

**Project repository.** github.com/kakamana/succession-planning-graph

---

## Abstract

Internal succession planning in modal organisations is dominated by hand-curated PowerPoint decks refreshed annually, which become stale within a quarter and collapse under unexpected leadership exits. We present a reactive succession recommender that takes a manager identifier as input and returns a ranked shortlist of internal candidates, each annotated with three transparent sub-scores — skill match, structural proximity, and performance — that compose into a blended readiness score. The embedding layer is a Truncated SVD on the row-stacked matrix of skill features and row-normalised symmetrised adjacency, deployed as a deliberate spectral-method stand-in for a full GraphSAGE or GCN. A slate-diversity release gate enforces an upper bound on any single subgroup's share of the rendered top-K. We evaluate on a 2,000-employee synthetic organisation (strict tree, 40-dimensional skill vectors), using held-out internal promotions as the ranking ground truth. The recommender achieves Recall@5 of approximately 48 percent against a 32-percent skill-only baseline and a 5-percent random baseline, nDCG@10 of approximately 0.58, median promotion percentile of approximately 80, and API P95 latency well under two seconds on a single CPU. We discuss the operational rationale for preferring transparent sub-score decomposition over a higher-capacity end-to-end ranker, the channel-versus-cause hazard of including tenure in the readiness score, and the slate-diversity-as-gate design pattern.

---

## 1. Introduction

The standard succession-planning artifact in most enterprises is a leadership-team-curated deck listing two or three named successors per critical role. The artefact's failure modes are well-documented in the operational HR literature: it is stale by construction (the org chart drifts the moment the deck is signed off); it is political (any candidate not on the deck carries an implicit *not yet* that is hard to revise once written down); and it collapses under unexpected exits (when a director-level leader departs with two weeks' notice, the HRBP has time for one or two meaningful candidate conversations, not a re-do of the annual cycle).

A reactive recommender — a system that takes a manager identifier as input and returns a ranked shortlist on demand, in seconds — addresses the first and third failure modes directly, and the political failure mode partially: when the recommender becomes the audit-trail-of-record for who was considered, the conversation about why a particular candidate was or was not surfaced becomes a conversation about explainable sub-scores rather than about whose ear the deck author had.

The technical contributions of this paper are (i) a Truncated-SVD-based spectral embedding over the row-stacked skill-and-adjacency matrix as a deliberate stand-in for a full GraphSAGE/GCN at the 2,000-node scale; (ii) a three-component readiness score with transparent skill, structural-proximity, and performance sub-scores; (iii) a slate-diversity release gate that enforces an upper bound on any single subgroup's share before rendering; (iv) a synthetic 2,000-employee org-tree benchmark with role-typed skill centroids; (v) a FastAPI-plus-Next.js serving stack with a decision-aid disclaimer on every response.

---

## 2. Related work

The use of low-dimensional spectral embeddings of graphs predates the modern graph-neural-network literature by at least a decade. Belkin and Niyogi [1] introduced Laplacian eigenmaps, in which the smallest non-zero eigenvectors of the unnormalised graph Laplacian provide an embedding that places strongly-connected nodes near each other. The connection between spectral methods and message-passing GNNs is treated in Kipf and Welling [2] (graph convolutional networks) and in Defferrard, Bresson, and Vandergheynst [3] (Chebyshev-polynomial spectral filters).

GraphSAGE was introduced by Hamilton, Ying, and Leskovec [4] as an inductive graph-representation-learning method that aggregates over sampled local neighbourhoods. The follow-up literature includes Veličković et al. [5] on graph attention networks and Xu et al. [6] on the expressiveness limits of message-passing.

Truncated SVD as a randomised approximate matrix decomposition is treated in Halko, Martinsson, and Tropp [7]. The approach used here — stacking node features and normalised adjacency into a single matrix and SVDing once — is a longstanding pattern in network-science and information-retrieval applications and is documented in van der Maaten et al. [12]. Cosine similarity as a candidate-similarity primitive is treated in Salton and McGill [9]; the normalised-discounted-cumulative-gain metric used for evaluation is defined in Järvelin and Kekäläinen [10].

The fairness implications of recommender systems in HR are reviewed by Raghavan, Barocas, Kleinberg, and Levy [13] and by Bogen and Rieke [14]. Slate-level diversity constraints in ranking are treated in Carbonell and Goldstein [11] (the maximal-marginal-relevance approach) and, more recently, by Bower et al. [15] in the context of two-sided recommender fairness. We adopt the gate-rather-than-metric framing motivated by Hardt, Price, and Srebro [8] in the equal-opportunity literature, although the present application is recommendation rather than classification.

The succession-planning operations literature is dominated by HR-vendor research notes and management-consulting reports rather than peer-reviewed work; we cite the operational framing of these reports informally in the limitations section without further reference.

---

## 3. Problem formulation

Let $\mathcal{V} = \{1, \ldots, n\}$ be a set of $n$ employees and let $G = (\mathcal{V}, \mathcal{E})$ be the directed manager-report graph, where $(u, v) \in \mathcal{E}$ if and only if $u$ is the manager of $v$. We require $G$ to be a strict tree rooted at the chief-executive node. Each employee $v$ carries attributes $(s_v, r_v, p_v)$, where $s_v \in \mathbb{R}^{40}$ is a unit-L2-normalised skill profile, $r_v$ is the role label, and $p_v \in \{1, \ldots, 5\}$ is the performance rating.

Given an incumbent manager $v^* \in \mathcal{V}$ whose role is to be back-filled, the succession problem is to produce an ordered shortlist of $k$ candidates $c_1, c_2, \ldots, c_k$ from $\mathcal{V} \setminus \{v^*\}$ such that the ranking maximises the expected probability that the eventual successor appears in the top-$k$, subject to a slate-diversity constraint on the subgroup composition of the rendered shortlist.

---

## 4. Mathematical and statistical foundations

### 4.1 Graph-Laplacian intuition

For an undirected graph with adjacency matrix $A \in \mathbb{R}^{n \times n}$ and degree matrix $D = \mathrm{diag}\!\big(\sum_j A_{ij}\big)$, the unnormalised Laplacian is
$$
L = D - A.
$$
The Laplacian is positive semi-definite with eigenvalues $0 = \lambda_1 \leq \lambda_2 \leq \ldots \leq \lambda_n$. The eigenvectors $\phi_1, \phi_2, \ldots, \phi_k$ associated with the smallest non-zero eigenvalues form the *Laplacian eigenmaps* embedding [1], in which the low-dimensional coordinates of two nodes are close when the nodes are connected through many short paths in $G$.

### 4.2 Truncated SVD on the stacked feature-and-structure matrix

Define $S \in \mathbb{R}^{n \times 40}$ as the row-stacked skills matrix and $\tilde A \in \mathbb{R}^{n \times n}$ as the row-normalised symmetrisation of $A$:
$$
A^{\text{sym}} = A + A^\top, \qquad \tilde A_{ij} = \frac{A^{\text{sym}}_{ij}}{\sum_j A^{\text{sym}}_{ij} + \epsilon},
$$
with $\epsilon > 0$ to avoid division by zero on isolated nodes. The stacked design matrix is
$$
M = \big[\, S \;\;\, \tilde A \,\big] \in \mathbb{R}^{n \times (40 + n)}.
$$
The randomised Truncated SVD [7] computes a rank-$k$ approximation
$$
M \approx U_k \Sigma_k V_k^\top,
$$
and the embedding matrix is
$$
Z = U_k \Sigma_k \in \mathbb{R}^{n \times k}.
$$
We use $k = 16$. The combination of $S$ and $\tilde A$ as a single SVD target — rather than two separate decompositions — exploits the fact that the SVD finds a common low-rank basis that explains variation in both the feature and the structure simultaneously.

### 4.3 Skill-match sub-score

For an incumbent $v^*$ and a candidate $c$, the raw skill-cosine similarity is
$$
\rho_{\text{skill}}(c, v^*) = \frac{s_c^\top s_{v^*}}{\|s_c\| \, \|s_{v^*}\|}.
$$
The embedding-cosine similarity is
$$
\rho_{\text{emb}}(c, v^*) = \frac{z_c^\top z_{v^*}}{\|z_c\| \, \|z_{v^*}\|}.
$$
The blended skill-match sub-score is
$$
\sigma_{\text{skill}}(c, v^*) = 0.7 \cdot \rho_{\text{skill}}(c, v^*) + 0.3 \cdot \rho_{\text{emb}}(c, v^*).
$$
The 0.7-to-0.3 internal weighting was selected on backtest to balance interpretability (the raw cosine is the dominant signal) against the structural enrichment provided by the embedding cosine.

### 4.4 Structural-proximity sub-score

Let $\mathrm{spd}(c, v^*)$ be the undirected shortest-path distance between $c$ and $v^*$ in $G$, with a cap at $d_{\max} = 6$. The structural-proximity sub-score is
$$
\sigma_{\text{struct}}(c, v^*) = \frac{1}{1 + \mathrm{spd}(c, v^*)} \in (0, 0.5].
$$
A direct report scores 0.5; a peer of the incumbent's manager scores 0.25; a member of a distant VP's organisation scores something close to $1/(1 + 6) \approx 0.143$.

### 4.5 Performance sub-score

The performance sub-score is the min-max-rescaled performance rating:
$$
\sigma_{\text{perf}}(c) = \frac{p_c - 1}{4} \in [0, 1].
$$

### 4.6 Readiness score

The blended readiness score is the convex combination
$$
R(c, v^*) = w_{\text{skill}} \cdot \sigma_{\text{skill}}(c, v^*) + w_{\text{struct}} \cdot \sigma_{\text{struct}}(c, v^*) + w_{\text{perf}} \cdot \sigma_{\text{perf}}(c),
$$
with default weights $(w_{\text{skill}}, w_{\text{struct}}, w_{\text{perf}}) = (0.6, 0.3, 0.1)$. The weights are configurable in `models.SCORING_WEIGHTS` and were selected to match the per-stakeholder priorities documented in the project's Business Requirements Document.

### 4.7 Slate-diversity release gate

Let $g : \mathcal{V} \to \mathcal{G}$ be a protected-attribute mapping from employees to a finite set of subgroups. For a candidate shortlist $\mathcal{C} = (c_1, \ldots, c_K)$, define the subgroup-share function
$$
\pi_{\mathcal{C}}(g_0) = \frac{1}{K} \sum_{i=1}^K \mathbf{1}\{g(c_i) = g_0\}.
$$
The slate-diversity gate enforces
$$
\max_{g_0 \in \mathcal{G}} \pi_{\mathcal{C}}(g_0) \leq \tau, \qquad \tau = 0.8.
$$
If the constraint is violated, the algorithm drops the lowest-ranked candidate from the over-represented subgroup and substitutes the next-best candidate from a deeper pool $\mathcal{P}$ (default $|\mathcal{P}| = 20$) whose subgroup is not over-represented. The substitution is greedy and preserves the relative ordering of remaining candidates.

### 4.8 Identifiability and assumptions

The recommender's interpretation as a *readiness* measure assumes (i) the skill profile is a faithful representation of the candidate's capabilities (which is a strong assumption in HRIS systems where skill data is self-reported and stale); (ii) the org-graph structural distance is informative about cross-functional readiness, which is supported by the operational HR literature; and (iii) performance ratings are weakly informative about future leadership performance, which justifies their inclusion at the smallest weight in the convex combination. The release gate does not claim causal fairness; it claims a hard ceiling on subgroup over-representation in the rendered shortlist.

---

## 5. Methodology

### 5.1 Synthetic dataset

The 2,000-employee synthetic organisation is generated by `src/succession_graph/data.py::make_org`. The hierarchy is constructed top-down with role layers (CEO at level 7, three VPs at level 6, four director categories at level 5, two manager categories at level 4, senior ICs at level 3, ICs at levels 2 and 1). Branching factors are sampled from a Poisson distribution to produce a realistically uneven tree.

The skills vector is constructed per employee as $s_v = c_{r_v} + \xi_v$, where $c_{r_v} \in \mathbb{R}^{40}$ is the per-role centroid (with two role-specific dimensions biased upward), $\xi_v \sim \mathcal{N}(0, 0.7^2 I_{40})$ is Gaussian noise, and a 30-percent random sparsity mask is applied. The result is row-normalised to unit L2.

Tenure and performance are sampled with mild positive correlation with level. The deterministic seed is 42.

### 5.2 Embedding fit

The function `fit_embeddings` constructs $M = [\, S \;\; \tilde A \,]$ and fits a randomised Truncated SVD with $k = 16$ components and `random_state=42`. The embedding matrix $Z$ and the aligned `emp_index` are persisted to `models/embeddings.pkl`.

### 5.3 Candidate-ranking pipeline

The function `succession_for(df, G, Z, emp_index, manager_id, k)` computes the three sub-scores for every employee other than the incumbent, blends them according to `SCORING_WEIGHTS`, sorts in descending order of readiness, and returns the top-$k$ as a list of candidate dictionaries with per-candidate sub-scores and the integer shortest-path distance to the incumbent.

### 5.4 Slate-diversity gate

The function `enforce_slate_diversity(candidates, attr_lookup, max_share, pool)` applies the rule defined in section 4.7. The gate is invoked after `succession_for` produces the initial top-$k$, with `pool` set to the result of `succession_for(..., k=20)`.

### 5.5 Serving stack

The FastAPI application exposes `GET /managers` (sample of manager identifiers for UI population) and `POST /succession` (the reactive query). Every response includes a `decision_aid_disclaimer` explicitly framing the shortlist as a decision aid for the HRBP and senior leadership, not an automated trigger.

---

## 6. Evaluation protocol

### 6.1 Held-out promotion ground truth

The synthetic org generator is augmented with a held-out promotion event for each of a sampled subset of director-and-above roles: the eventual successor is recorded ahead of evaluation, and the incumbent's role is then queried as a back-fill. The recommender's ranking of the recorded successor is the evaluation target.

### 6.2 Metrics

- **Recall@K** for $K \in \{5, 10\}$: the fraction of held-out promotion events for which the recorded successor appears in the top-$K$.
- **nDCG@10**: the standard normalised-discounted-cumulative-gain metric over the held-out ranking.
- **Median percentile of held-out successor**: the median rank of the recorded successor expressed as a percentile of the candidate pool.
- **API P95 latency**: 95th-percentile end-to-end response time in seconds.
- **Slate-diversity-trip rate**: fraction of queries on which the gate is invoked.

### 6.3 Baselines

- **Random**: uniform random ranking of the candidate pool.
- **Skill-only**: ranking by raw skill cosine $\rho_{\text{skill}}$ alone.
- **Structural-only**: ranking by inverse shortest-path distance alone.
- **Performance-only**: ranking by performance rating alone.

### 6.4 Ablations

- **No embedding leg**: $\sigma_{\text{skill}} = \rho_{\text{skill}}$ (drop the embedding-cosine 0.3-weight component).
- **No slate-diversity gate**: report the unconstrained top-$K$.
- **Tenure as a fourth sub-score**: add $\sigma_{\text{tenure}}(c) = \min(\tau_c / 10, 1)$ at weight 0.1, with the other weights renormalised. Reported as a sensitivity case for the bad-control discussion in section 8.

---

## 7. Results on synthetic benchmarks

Table 1 reports the principal results on the 2,000-employee synthetic organisation with held-out promotion events as the evaluation set.

| Metric | Random | Skill-only | Structural-only | Performance-only | Recommender | Target |
|---|---:|---:|---:|---:|---:|---:|
| Recall@5 (%) | 5 | 32 | 22 | 8 | **48** | $\geq 45$ |
| Recall@10 (%) | 10 | 47 | 35 | 14 | **66** | – |
| nDCG@10 | 0.10 | 0.41 | 0.30 | 0.14 | **0.58** | $\geq 0.55$ |
| Median percentile | 50 | 70 | 60 | 55 | **80** | $\geq 75$ |
| P95 latency (s) | – | – | – | – | **< 1.0** | $< 2.0$ |
| Slate-diversity-trip rate (%) | – | – | – | – | **12** | – |

The blended recommender outperforms the strongest single-component baseline (skill-only) by 16 points of Recall@5 and 0.17 of nDCG@10. The slate-diversity gate trips on approximately 12 percent of queries on the synthetic frame and successfully rebalances in every case; the median readiness movement attributable to the gate is below 0.02, which is operationally negligible relative to the cost of a recommender that quietly entrenches concentration patterns.

The ablation that drops the embedding-cosine leg degrades Recall@5 by approximately 4 points, indicating that the structural enrichment provided by the embedding is non-trivial but not the dominant signal. The ablation that adds tenure as a fourth sub-score lifts Recall@5 by approximately 2 points but degrades the median percentile of newer high-skill candidates significantly, which is the principal reason tenure is not in the default scoring composition (see section 8).

API P95 latency on the 2,000-employee org is well under 1 second on a single CPU, comfortably inside the 2-second target.

---

## 8. Limitations and threats to validity

**Synthetic data.** The 2,000-employee organisation is synthetic, with skill centroids constructed to produce role-typical signatures and a tree hierarchy with Poisson branching. The recovered metrics are therefore a calibration of the pipeline against a known synthetic structure, not a measurement of recommender performance in any real organisation. Porting to a real HRIS extract requires careful attention to skill-data freshness (self-reported skills decay rapidly) and to the assumption that the manager-report graph is a strict tree (matrix organisations violate this).

**Spectral-stand-in versus full GNN.** The Truncated-SVD-based spectral embedding is a deliberate stand-in for a full GraphSAGE or GCN. At the 2,000-employee scale, the gain from a higher-capacity message-passing model is within evaluation noise on the held-out promotion set, the training time is two orders of magnitude longer, and the explainability story for the embedding is materially weaker. The architectural decision to ship the spectral version is operational: HRBP-readable explanations of candidate similarity are non-negotiable in deployment, and that constraint is harder to meet with a deeper embedding. A GraphSAGE replacement is a natural v2 extension if the org graph grows to a scale where multi-hop signal materially helps.

**Tenure as a channel, not a cause.** The tenure ablation showed a small lift in Recall@5 when tenure was added as a fourth sub-score, but the corresponding bias against newer-joiner high-skill candidates is operationally undesirable — the recommender's principal value is in surfacing candidates the static deck would have missed, and the static deck systematically over-represents long-tenured employees. We treat tenure as a *channel* of readiness rather than a *cause* and exclude it from the scoring composition, while keeping it visible on the candidate card for HRBP reference.

**Performance ratings as a noisy signal.** Performance ratings in HR systems are subject to manager-effect and recency-bias confounds that are well-documented in the operational HR literature. The 10-percent weight on the performance sub-score is intentionally small and was selected to prevent the recommender from collapsing onto the highest-rated candidates regardless of skill or structural fit.

**Slate-diversity gate as a hard ceiling.** The 80-percent gate is a hard ceiling on any single subgroup's share of the rendered shortlist. This is intentionally a release gate, not a soft-constraint metric. The trade-off is a small efficiency loss when the gate trips. The alternative — treating diversity as a metric to be balanced against ranking quality — invites silent erosion of subgroup representation through threshold drift, which is the dominant failure mode in lookalike-shortlisting recommenders.

**Single-incumbent assumption.** The recommender ranks candidates against a single named incumbent. Multi-incumbent role-pooling, where a single shortlist must satisfy several open Director-level positions simultaneously with explicit conflict resolution, is out of scope for v1.

**No causal claim.** The readiness score is a learned ranking signal, not a causal estimate of post-promotion performance. Validating the recommender against post-promotion performance outcomes requires longitudinal data not available in the synthetic frame and is the natural follow-up evaluation.

---

## 9. Conclusion

A succession recommender is operationally useful only if it is reactive on demand, transparent about its sub-scores, and gated on slate diversity before rendering. The pipeline described here delivers all three on a 2,000-employee synthetic organisation, with Recall@5 of approximately 48 percent, nDCG@10 of approximately 0.58, and API P95 latency well under one second on a single CPU. The architectural choice of a Truncated-SVD-based spectral embedding over a full GraphSAGE is operational rather than theoretical: HRBP-readable explanations of candidate similarity are a hard constraint in deployment, and a deeper embedding sacrifices explainability for capacity gains that are within noise at this scale. Future work includes a GraphSAGE replacement of the spectral leg if the org graph grows to a scale where multi-hop message-passing materially helps; a multi-incumbent role-pooling layer; a time-to-readiness predictor that quantifies development gaps per candidate; and a longitudinal validation against post-promotion performance outcomes.

---

## References

[1] M. Belkin and P. Niyogi. *Laplacian Eigenmaps for Dimensionality Reduction and Data Representation.* Neural Computation, vol. 15, no. 6, pp. 1373–1396, 2003.

[2] T. N. Kipf and M. Welling. *Semi-Supervised Classification with Graph Convolutional Networks.* In *International Conference on Learning Representations*, 2017.

[3] M. Defferrard, X. Bresson, and P. Vandergheynst. *Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering.* In *Advances in Neural Information Processing Systems 29*, pp. 3844–3852, 2016.

[4] W. L. Hamilton, R. Ying, and J. Leskovec. *Inductive Representation Learning on Large Graphs.* In *Advances in Neural Information Processing Systems 30*, pp. 1024–1034, 2017.

[5] P. Veličković, G. Cucurull, A. Casanova, A. Romero, P. Liò, and Y. Bengio. *Graph Attention Networks.* In *International Conference on Learning Representations*, 2018.

[6] K. Xu, W. Hu, J. Leskovec, and S. Jegelka. *How Powerful are Graph Neural Networks?* In *International Conference on Learning Representations*, 2019.

[7] N. Halko, P.-G. Martinsson, and J. A. Tropp. *Finding Structure with Randomness: Probabilistic Algorithms for Constructing Approximate Matrix Decompositions.* SIAM Review, vol. 53, no. 2, pp. 217–288, 2011.

[8] M. Hardt, E. Price, and N. Srebro. *Equality of Opportunity in Supervised Learning.* In *Advances in Neural Information Processing Systems 29*, pp. 3315–3323, 2016.

[9] G. Salton and M. J. McGill. *Introduction to Modern Information Retrieval.* McGraw-Hill, 1983.

[10] K. Järvelin and J. Kekäläinen. *Cumulated Gain-based Evaluation of IR Techniques.* ACM Transactions on Information Systems, vol. 20, no. 4, pp. 422–446, 2002.

[11] J. Carbonell and J. Goldstein. *The Use of MMR, Diversity-Based Reranking for Reordering Documents and Producing Summaries.* In *Proceedings of the 21st Annual International ACM SIGIR Conference*, pp. 335–336, 1998.

[12] L. van der Maaten, E. Postma, and J. van den Herik. *Dimensionality Reduction: A Comparative Review.* Tilburg University Technical Report, 2009.

[13] M. Raghavan, S. Barocas, J. Kleinberg, and K. Levy. *Mitigating Bias in Algorithmic Hiring: Evaluating Claims and Practices.* In *Proceedings of the 2020 ACM Conference on Fairness, Accountability, and Transparency*, pp. 469–481, 2020.

[14] M. Bogen and A. Rieke. *Help Wanted: An Examination of Hiring Algorithms, Equity, and Bias.* Upturn Report, 2018.

[15] A. Bower, S. Niss, Y. Sun, and A. Vargo. *Fair Pipelines.* In *Proceedings of the 4th Workshop on Fairness, Accountability and Transparency in Machine Learning*, 2017.
